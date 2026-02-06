"""Entropy-Aware Predictive Controller for SAFE algorithm."""

from typing import Dict, Any
from safe.controllers.base import BaseKLController
from safe.controllers.pid import PIDController, PhaseDetector


class EntropyAwarePredictiveController(BaseKLController):
    """
    SAFE: Entropy-Aware Predictive Controller
    
    Components:
    1. Entropy-Gated Penalty (modulates by entropy)
    2. Dual-Timescale Tracking (short + long EMA)
    3. PID Tuning (dynamic threshold)
    4. Phase Detection (context-aware)
    
    Key insight: Low entropy indicates the model is becoming overconfident,
    so the KL penalty should be amplified. High entropy means exploration
    is healthy, so relax the penalty.
    """
    
    def __init__(
        self,
        base_threshold: float = 0.3,
        entropy_floor: float = 2.0,
        max_preview_kl: float = 0.5,
        pid_kp: float = 0.3,
        pid_ki: float = 0.01,
        pid_kd: float = 0.05,
    ):
        """
        Args:
            base_threshold: Base KL threshold
            entropy_floor: Below this entropy, amplify penalty
            max_preview_kl: Hard ceiling for gradient preview
            pid_kp, pid_ki, pid_kd: PID controller gains
        """
        # Dual-Timescale trackers
        self.kl_short = 0.0
        self.kl_long = 0.0
        
        # PID Controller
        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)
        
        # Phase Detector
        self.phase_detector = PhaseDetector()
        
        # Settings
        self.base_threshold = base_threshold
        self.entropy_floor = entropy_floor
        self.max_preview_kl = max_preview_kl
    
    def compute_penalty(self, kl: float, **kwargs) -> Dict[str, Any]:
        """
        Compute entropy-gated KL penalty with phase awareness.
        
        Args:
            kl: Current KL divergence
            entropy: Current policy entropy (required kwarg)
            reward: Current reward (required kwarg)
        
        Returns:
            dict with penalty info
        """
        entropy = kwargs.get("entropy", self.entropy_floor)
        reward = kwargs.get("reward", 0.0)
        
        # Update trackers
        self.kl_short = 0.9 * self.kl_short + 0.1 * kl
        self.kl_long = 0.99 * self.kl_long + 0.01 * kl
        self.phase_detector.update(reward)
        
        # Compute dynamic threshold
        pid_adjustment = self.pid.update(reward)
        phase_mult = self.phase_detector.get_threshold_multiplier()
        threshold = (self.base_threshold + pid_adjustment) * phase_mult
        threshold = max(0.1, min(0.6, threshold))
        
        # Determine penalty
        if self.kl_short > threshold:
            base_penalty = 0.1 * (self.kl_short - threshold) ** 2
            
            # Entropy Gate: Low entropy amplifies penalty
            entropy_factor = max(0.5, self.entropy_floor / (entropy + 0.1))
            penalty = base_penalty * entropy_factor
            active = True
        else:
            penalty = 0.0
            entropy_factor = 1.0
            active = False
        
        return {
            "penalty": penalty,
            "threshold": threshold,
            "kl_short": self.kl_short,
            "kl_long": self.kl_long,
            "entropy_factor": entropy_factor,
            "phase": self.phase_detector.phase,
            "active": active,
        }
    
    def should_scale_step(self, preview_kl: float) -> bool:
        """Check if gradient step should be scaled based on preview KL."""
        return preview_kl > self.max_preview_kl
    
    def get_scale_factor(self, preview_kl: float) -> float:
        """Get scaling factor for gradient step."""
        if preview_kl <= self.max_preview_kl:
            return 1.0
        return self.max_preview_kl / preview_kl
    
    def reset(self):
        """Reset controller state."""
        self.kl_short = 0.0
        self.kl_long = 0.0
        self.pid.reset()
        self.phase_detector.reset()
