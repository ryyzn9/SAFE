"""PPO Baseline Trainer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Callable

from safe.config import SAFEConfig


class PPOTrainer:
    """Standard PPO trainer as baseline comparison."""
    
    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        reward_fn: Callable[[str], float],
        config: SAFEConfig,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.cfg = config
        self.device = next(policy_model.parameters()).device
        
        hidden_size = policy_model.config.hidden_size
        
        # Simple value head
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        ).to(self.device).to(torch.bfloat16)
        
        self.optimizer = torch.optim.AdamW(
            list(policy_model.parameters()) + list(self.v_head.parameters()),
            lr=config.policy_lr,
        )
        
        self.step = 0
    
    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]) -> Dict[str, Any]:
        """Generate rollouts for PPO."""
        self.policy.eval()
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        responses = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        rewards = torch.tensor(
            [self.reward_fn(t) for t in full_texts],
            device=self.device,
            dtype=torch.float32,
        )
        
        completion_mask = outputs[:, prompt_len:] != self.tokenizer.pad_token_id
        completion_length = completion_mask.sum(dim=1).float().mean().item()
        
        attention_mask = (outputs != self.tokenizer.pad_token_id).long()
        
        policy_out = self.policy(outputs, attention_mask=attention_mask)
        ref_out = self.ref(outputs, attention_mask=attention_mask)
        
        policy_logits = policy_out.logits[:, prompt_len - 1:-1, :]
        ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
        
        tokens = outputs[:, prompt_len:]
        policy_lp = torch.gather(F.log_softmax(policy_logits, -1), -1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_lp = torch.gather(F.log_softmax(ref_logits, -1), -1, tokens.unsqueeze(-1)).squeeze(-1)
        
        mask = (tokens != self.tokenizer.pad_token_id).float()
        kl = ((policy_lp - ref_lp) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return {
            "ids": outputs,
            "mask": attention_mask,
            "rewards": rewards,
            "kl": kl.mean().item(),
            "completion_length": completion_length,
            "policy_logprobs": policy_lp,
            "ref_logprobs": ref_lp,
            "prompt_len": prompt_len,
        }
    
    def train_step(self, rollouts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one PPO training step."""
        self.policy.train()
        
        ids = rollouts["ids"]
        mask = rollouts["mask"]
        rewards = rollouts["rewards"]
        old_lp = rollouts["policy_logprobs"]
        ref_lp = rollouts["ref_logprobs"]
        prompt_len = rollouts["prompt_len"]
        
        response_mask = (ids[:, prompt_len:] != self.tokenizer.pad_token_id).float()
        
        with torch.no_grad():
            hidden = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            old_v = self.v_head(hidden[:, -1, :]).squeeze(-1)
            advantages = rewards - old_v
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_kl = 0
        
        for _ in range(self.cfg.epochs_per_batch):
            outputs = self.policy(ids, attention_mask=mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            logits = outputs.logits[:, prompt_len - 1:-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            
            tokens = ids[:, prompt_len:]
            current_lp = torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            
            ratio = torch.exp(current_lp - old_lp.float())
            adv_expanded = advantages.unsqueeze(1).expand_as(ratio)
            
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * adv_expanded
            
            policy_loss = -(torch.min(surr1, surr2) * response_mask).sum() / response_mask.sum().clamp(min=1)
            
            kl = ((current_lp - ref_lp.float()) * response_mask).sum() / response_mask.sum().clamp(min=1)
            
            v = self.v_head(hidden[:, -1, :]).squeeze(-1)
            value_loss = 0.5 * ((v - rewards) ** 2).mean()
            
            loss = policy_loss + 0.05 * kl + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            total_kl += kl.item()
        
        self.step += 1
        
        return {
            "step": self.step,
            "reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "value_loss": value_loss.item(),
            "kl": total_kl / self.cfg.epochs_per_batch,
            "completion_length": rollouts["completion_length"],
        }
