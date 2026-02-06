"""Double Soft-Min Critic for pessimistic value estimation."""

import math
import torch
import torch.nn as nn


class DoubleCritic(nn.Module):
    """
    Double Critic with Soft-Min aggregation and optional LayerNorm.
    
    Uses two independent value heads and combines them with soft-min,
    which provides pessimistic value estimates to prevent overestimation
    (similar to TD3/SAC in continuous control).
    
    The soft-min operation is:
        V = -alpha * (logsumexp(-v1/alpha, -v2/alpha) - log(2))
    
    As alpha -> 0, this approaches min(v1, v2).
    As alpha -> inf, this approaches mean(v1, v2).
    """
    
    def __init__(
        self,
        hidden_size: int,
        alpha: float = 0.3,
        use_layernorm: bool = True,
    ):
        """
        Args:
            hidden_size: Size of input hidden states
            alpha: Soft-min temperature (lower = more pessimistic)
            use_layernorm: Whether to apply LayerNorm to inputs (SAFE uses this)
        """
        super().__init__()
        self.alpha = alpha
        self.use_layernorm = use_layernorm
        
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)
        
        self.v1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )
        self.v2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute value estimates from both heads.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size) from language model
            
        Returns:
            Tuple of (v1, v2) each of shape (batch,)
        """
        # Take last token's hidden state
        h = hidden_states[:, -1, :]
        
        if self.use_layernorm:
            h = self.ln(h)
        
        return self.v1(h).squeeze(-1), self.v2(h).squeeze(-1)
    
    def soft_min(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-min of two value estimates.
        
        Args:
            v1, v2: Value estimates of shape (batch,)
            
        Returns:
            Soft-min values of shape (batch,)
        """
        stacked = torch.stack([-v1 / self.alpha, -v2 / self.alpha], dim=-1)
        return -self.alpha * (torch.logsumexp(stacked, dim=-1) - math.log(2))
    
    def get_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convenience method to get soft-min value directly."""
        v1, v2 = self.forward(hidden_states)
        return self.soft_min(v1, v2)


class SimpleCritic(nn.Module):
    """Simple single-head critic for baseline comparison."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.v_head(hidden_states[:, -1, :]).squeeze(-1)
