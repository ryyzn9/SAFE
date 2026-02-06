"""PID Controller and Phase Detector for dynamic threshold adjustment."""

from typing import Dict
import numpy as np


class PIDController:
    """Simple PID controller for threshold adjustment."""
    
    def __init__(self, kp: float = 0.3, ki: float = 0.01, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.reward_ema = None
    
    def update(self, reward: float) -> float:
        """Update PID controller with current reward.
        
        Args:
            reward: Current reward value
            
        Returns:
            PID adjustment value
        """
        # Target: want positive reward velocity
        if self.reward_ema is None:
            self.reward_ema = reward
            return 0.0
        
        old_ema = self.reward_ema
        self.reward_ema = 0.95 * self.reward_ema + 0.05 * reward
        
        # Error = actual velocity - target velocity
        velocity = self.reward_ema - old_ema
        target_velocity = 0.001
        error = velocity - target_velocity
        
        # PID terms
        p_out = self.kp * error
        self.integral = max(-1.0, min(1.0, self.integral + error))  # Anti-windup
        i_out = self.ki * self.integral
        d_out = self.kd * (error - self.prev_error)
        self.prev_error = error
        
        return p_out + i_out + d_out
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.reward_ema = None


class PhaseDetector:
    """Classifies training phase: WARMUP, CLIMBING, PLATEAU, CONVERGED."""
    
    def __init__(self):
        self.reward_history = []
        self.phase = "WARMUP"
    
    def update(self, reward: float):
        """Update phase detection with current reward."""
        self.reward_history.append(reward)
        if len(self.reward_history) < 50:
            self.phase = "WARMUP"
            return
        
        recent = self.reward_history[-50:]
        old = self.reward_history[-100:-50] if len(self.reward_history) >= 100 else recent
        
        recent_mean = sum(recent) / len(recent)
        old_mean = sum(old) / len(old)
        recent_std = np.std(recent)
        
        if recent_mean > old_mean + 0.01:
            self.phase = "CLIMBING"
        elif recent_std < 0.02 and recent_mean > 0.7:
            self.phase = "CONVERGED"
        else:
            self.phase = "PLATEAU"
    
    def get_threshold_multiplier(self) -> float:
        """Get threshold multiplier based on current phase."""
        return {
            "WARMUP": 1.5,    # Very relaxed
            "CLIMBING": 1.2,  # Relaxed
            "PLATEAU": 0.8,   # Tight
            "CONVERGED": 1.0  # Normal
        }[self.phase]
    
    def reset(self):
        """Reset phase detector."""
        self.reward_history = []
        self.phase = "WARMUP"
