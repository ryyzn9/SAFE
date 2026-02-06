"""Asymmetric KL Controller - Only penalizes positive KL (over-confidence)."""

from typing import Dict, Any
from safe.controllers.base import BaseKLController


class AsymmetricKLController(BaseKLController):
    """
    Asymmetric KL controller that only penalizes POSITIVE KL (over-confidence).
    Allows negative KL (exploration) without penalty.
    
    This prevents the late-training collapse where KL explodes, causing reward drops.
    
    Key insight: Negative KL means the policy is more uncertain than reference,
    which is fine for exploration. Only positive KL (overconfident divergence)
    needs penalization.
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        lambda_asymmetric: float = 0.1,
        lambda_momentum: float = 5.0,
        momentum_window: int = 50,
    ):
        """
        Args:
            threshold: Only penalize KL above this value (default 0.3)
            lambda_asymmetric: Strength of the quadratic penalty (default 0.1)
            lambda_momentum: Strength of the momentum penalty (default 5.0)
            momentum_window: Steps to compute KL velocity (default 50)
        """
        self.threshold = threshold
        self.lambda_asymmetric = lambda_asymmetric
        self.lambda_momentum = lambda_momentum
        self.momentum_window = momentum_window
        self.kl_history = []
    
    def compute_penalty(self, kl: float, **kwargs) -> Dict[str, Any]:
        """
        Compute asymmetric KL penalty.
        
        Args:
            kl: Current KL divergence value (can be negative)
            
        Returns:
            dict with 'penalty', 'momentum', 'asymmetric_penalty', 'momentum_penalty'
        """
        # Track KL history for momentum calculation
        self.kl_history.append(kl)
        if len(self.kl_history) > self.momentum_window * 2:
            self.kl_history.pop(0)
        
        # === ASYMMETRIC PENALTY ===
        # Only penalize POSITIVE KL above threshold
        # Negative KL (exploration) gets ZERO penalty
        if kl > self.threshold:
            asymmetric_penalty = self.lambda_asymmetric * (kl - self.threshold) ** 2
            asymmetric_active = True
        else:
            asymmetric_penalty = 0.0
            asymmetric_active = False
        
        # === MOMENTUM PENALTY ===
        # Penalize rapidly increasing KL (early collapse detection)
        if len(self.kl_history) >= self.momentum_window:
            kl_old = self.kl_history[-self.momentum_window]
            kl_new = self.kl_history[-1]
            momentum = (kl_new - kl_old) / self.momentum_window
            
            # Only penalize POSITIVE momentum (KL increasing)
            if momentum > 0:
                momentum_penalty = self.lambda_momentum * momentum ** 2
            else:
                momentum_penalty = 0.0
        else:
            momentum = 0.0
            momentum_penalty = 0.0
        
        total_penalty = asymmetric_penalty + momentum_penalty
        
        return {
            "penalty": total_penalty,
            "asymmetric_penalty": asymmetric_penalty,
            "momentum_penalty": momentum_penalty,
            "momentum": momentum,
            "asymmetric_active": asymmetric_active,
            "kl": kl,
        }
    
    def reset(self):
        """Reset KL history."""
        self.kl_history = []
