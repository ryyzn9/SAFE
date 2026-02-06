"""Reward model loading and normalization utilities."""

import math
import torch
from typing import Callable, Optional


class RewardNormalizer:
    """Running mean/variance normalizer for rewards using Welford's algorithm."""
    
    def __init__(self, epsilon: float = 1e-8):
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        """Update running statistics with batch of values."""
        batch_mean = x.mean().item()
        batch_var = x.var().item() if x.shape[0] > 1 else 0.0
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize values using running statistics."""
        return (x - self.mean) / (math.sqrt(self.var) + self.epsilon)
    
    def reset(self):
        """Reset statistics."""
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0


class RewardModel:
    """Wrapper for reward model with caching and batching support."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 1024,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
    
    @torch.no_grad()
    def compute_reward(self, text: str) -> float:
        """Compute reward for a single text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        if hasattr(outputs, "score"):
            return outputs.score.item()
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
            return logits[0].item() if logits.dim() == 1 else logits[0, 0].item()
        return 0.0
    
    @torch.no_grad()
    def compute_rewards_batch(self, texts: list[str]) -> torch.Tensor:
        """Compute rewards for a batch of texts."""
        rewards = [self.compute_reward(t) for t in texts]
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)


def load_reward_model(
    model_name: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    device: str = "cuda",
    torch_dtype = torch.bfloat16,
) -> tuple[RewardModel, Callable[[str], float]]:
    """
    Load reward model and return wrapper + simple compute function.
    
    Returns:
        Tuple of (RewardModel instance, compute_reward function)
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        
        rm = RewardModel(model, tokenizer, device)
        print(f"✓ Loaded reward model: {model_name}")
        return rm, rm.compute_reward
        
    except Exception as e:
        print(f"⚠ Failed to load {model_name}: {e}")
        print("⚠ Using fallback heuristic reward")
        
        def fallback_reward(text: str) -> float:
            r = 0.0
            words = text.lower().split()
            if 20 < len(words) < 200:
                r += 0.3
            if words:
                r += len(set(words)) / len(words) * 0.3
            for w in ["help", "sure", "here"]:
                if w in text.lower():
                    r += 0.05
            return min(max(r, -1.0), 1.0)
        
        return None, fallback_reward
