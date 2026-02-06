"""Asymmetric KL Trainer - Double Soft-Min Critics with Asymmetric KL."""

import torch
import torch.nn.functional as F
from typing import Dict, Any

from safe.trainers.base import BaseTrainer
from safe.controllers.asymmetric import AsymmetricKLController
from safe.config import AsymmetricKLConfig


class AsymmetricKLTrainer(BaseTrainer):
    """
    Asymmetric KL Trainer with Double Soft-Min Critics
    
    Key features from main27.py:
    1. Asymmetric KL penalty (only penalizes positive KL)
    2. Momentum penalty for early collapse detection
    3. Double Soft-Min critic for pessimistic value estimation
    """
    
    def __init__(self, policy_model, ref_model, tokenizer, reward_fn, config: AsymmetricKLConfig):
        super().__init__(policy_model, ref_model, tokenizer, reward_fn, config)
        
        # Initialize Asymmetric KL Controller
        self.kl_controller = AsymmetricKLController(
            threshold=config.kl_threshold,
            lambda_asymmetric=config.kl_lambda_asymmetric,
            lambda_momentum=config.kl_lambda_momentum,
            momentum_window=config.kl_momentum_window,
        )
    
    def train_step(self, rollouts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one training step with Asymmetric KL algorithm."""
        self.policy.train()
        
        ids = rollouts["ids"]
        mask = rollouts["mask"]
        rewards = rollouts["rewards"]
        old_lp = rollouts["policy_logprobs"]
        ref_lp = rollouts["ref_logprobs"]
        prompt_len = rollouts["prompt_len"]
        
        response_mask = (ids[:, prompt_len:] != self.tokenizer.pad_token_id).float()
        
        # Compute old value estimates and advantages
        with torch.no_grad():
            hidden = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(hidden)
            old_v = self.critic.soft_min(v1, v2)
            advantages = rewards - old_v
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0
        
        for _ in range(self.cfg.epochs_per_batch):
            # Forward pass
            outputs = self.policy(ids, attention_mask=mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            logits = outputs.logits[:, prompt_len - 1:-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            
            tokens = ids[:, prompt_len:]
            current_lp = torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            
            # PPO clipped objective
            ratio = torch.exp(current_lp - old_lp.float())
            adv_expanded = advantages.unsqueeze(1).expand_as(ratio)
            
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * adv_expanded
            
            policy_loss = -(torch.min(surr1, surr2) * response_mask).sum() / response_mask.sum().clamp(min=1)
            
            # KL divergence
            kl = ((current_lp - ref_lp.float()) * response_mask).sum() / response_mask.sum().clamp(min=1)
            
            # Critic loss (MSE with clipping)
            v1, v2 = self.critic(hidden)
            v = self.critic.soft_min(v1, v2)
            v_clipped = old_v + torch.clamp(v - old_v, -self.cfg.clip_range_vf, self.cfg.clip_range_vf)
            
            value_loss = 0.5 * torch.max((v - rewards) ** 2, (v_clipped - rewards) ** 2).mean()
            
            # Entropy
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Asymmetric KL Penalty
            kl_info = self.kl_controller.compute_penalty(kl.item())
            kl_penalty = kl_info["penalty"]
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss + kl_penalty - self.cfg.entropy_coef * entropy
            
            # Optimization step
            self.opt_policy.zero_grad()
            self.opt_critic.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            
            self.opt_policy.step()
            self.opt_critic.step()
            
            # Update target critic
            self.update_target_critic()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += kl.item()
        
        self.step += 1
        num_epochs = self.cfg.epochs_per_batch
        
        return {
            "step": self.step,
            "reward": rollouts["raw_rewards"].mean().item(),
            "reward_std": rollouts["raw_rewards"].std().item(),
            "policy_loss": total_policy_loss / num_epochs,
            "value_loss": total_value_loss / num_epochs,
            "kl": total_kl / num_epochs,
            "completion_length": rollouts["completion_length"],
            "kl_penalty": kl_info["penalty"],
            "kl_momentum": kl_info["momentum"],
            "kl_asymmetric_active": kl_info["asymmetric_active"],
        }
