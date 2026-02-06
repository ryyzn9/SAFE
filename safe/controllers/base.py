"""Base KL Controller interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseKLController(ABC):
    """Abstract base class for KL controllers."""
    
    @abstractmethod
    def compute_penalty(self, kl: float, **kwargs) -> Dict[str, Any]:
        """Compute KL penalty.
        
        Args:
            kl: Current KL divergence value
            **kwargs: Additional arguments (entropy, reward, etc.)
            
        Returns:
            Dictionary with 'penalty' and additional info
        """
        pass
    
    def reset(self):
        """Reset controller state."""
        pass
