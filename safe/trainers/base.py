"""Base trainer class with common functionality."""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional

from safe.config import SAFEConfig
from safe.models.critic import DoubleCritic
from safe.reward.reward_model import RewardNormalizer


class BaseTrainer(ABC):
    """Base class for all trainers with common rollout and training logic."""
    
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
        
        # Initialize critic
        self.critic = DoubleCritic(
            hidden_size, 
            alpha=config.alpha_softmin,
            use_layernorm=True,
        ).to(self.device).to(torch.bfloat16)
        
        # Target critic for Polyak averaging
        self.critic_target = DoubleCritic(
            hidden_size,
            alpha=config.alpha_softmin,
            use_layernorm=True,
        ).to(self.device).to(torch.bfloat16)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.opt_policy = torch.optim.AdamW(
            policy_model.parameters(), lr=config.policy_lr
        )
        self.opt_critic = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr
        )
        
        self.step = 0
        self.reward_normalizer = RewardNormalizer()
    
    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]) -> Dict[str, Any]:
        """Generate rollouts for a batch of prompts."""
        self.policy.eval()
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate responses
        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode responses
        responses = self.tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        # Compute rewards
        raw_rewards = torch.tensor(
            [self.reward_fn(t) for t in full_texts],
            device=self.device,
            dtype=torch.float32,
        )
        
        self.reward_normalizer.update(raw_rewards)
        normalized_rewards = self.reward_normalizer.normalize(raw_rewards)
        
        # Compute completion length
        completion_mask = outputs[:, prompt_len:] != self.tokenizer.pad_token_id
        completion_length = completion_mask.sum(dim=1).float().mean().item()
        
        # Get logprobs from policy and reference
        attention_mask = (outputs != self.tokenizer.pad_token_id).long()
        
        policy_out = self.policy(outputs, attention_mask=attention_mask)
        ref_out = self.ref(outputs, attention_mask=attention_mask)
        
        policy_logits = policy_out.logits[:, prompt_len - 1:-1, :]
        ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
        
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        tokens = outputs[:, prompt_len:]
        policy_lp = torch.gather(policy_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_lp = torch.gather(ref_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        
        # Compute KL and entropy
        mask = (tokens != self.tokenizer.pad_token_id).float()
        kl = ((policy_lp - ref_lp) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        probs = torch.exp(policy_logprobs)
        entropy = -(probs * policy_logprobs).sum(dim=-1)
        entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return {
            "ids": outputs,
            "mask": attention_mask,
            "rewards": normalized_rewards,
            "raw_rewards": raw_rewards,
            "kl": kl.mean().item(),
            "completion_length": completion_length,
            "policy_logprobs": policy_lp,
            "ref_logprobs": ref_lp,
            "entropy": entropy.mean().item(),
            "prompt_len": prompt_len,
        }
    
    @abstractmethod
    def train_step(self, rollouts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one training step. Must be implemented by subclasses."""
        pass
    
    def update_target_critic(self):
        """Polyak averaging for target critic."""
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.cfg.tau_polyak).add_(self.cfg.tau_polyak * p.data)
