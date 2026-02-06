"""Configuration dataclasses for S3-KLQ training."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class S3KLQConfig:
    """Base configuration for S3-KLQ training."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B"
    reward_model_name: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    
    # LoRA settings
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
    batch_size: int = 16
    gradient_accumulation: int = 2
    max_steps: int = 2000
    max_new_tokens: int = 128
    
    # Learning rates
    policy_lr: float = 1e-5
    critic_lr: float = 1e-5
    
    # PPO settings
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    epochs_per_batch: int = 2
    entropy_coef: float = 0.01
    
    # Critic settings
    alpha_softmin: float = 0.3
    tau_polyak: float = 0.05
    
    # Logging
    log_every: int = 50
    save_every: int = 200
    
    # Distributed training
    num_gpus: int = 8
    rollout_workers_per_gpu: int = 1
    use_placement_groups: bool = True
    
    # Data
    dataset_name: str = "Anthropic/hh-rlhf"
    train_split: str = "train[:5000]"
    eval_split: str = "test[:500]"


@dataclass
class SAFEConfig(S3KLQConfig):
    """Configuration for SAFE (Entropy-Aware Predictive Controller)."""
    
    # Synchronized LRs (key to SAFE stability)
    policy_lr: float = 1e-5
    critic_lr: float = 1e-5  # MATCHED with policy
    
    # Entropy-Aware Settings
    kl_base_threshold: float = 0.3
    entropy_floor: float = 2.0  # Below this, amplify penalty
    max_preview_kl: float = 0.5  # Hard ceiling for gradient preview
    
    # PID Controller Gains
    pid_kp: float = 0.3
    pid_ki: float = 0.01
    pid_kd: float = 0.05
    
    target_kl: float = 0.05


@dataclass
class AsymmetricKLConfig(S3KLQConfig):
    """Configuration for Asymmetric KL with Double Soft-Min Critics."""
    
    # Learning rates (from v3 experiments)
    policy_lr: float = 1e-5
    critic_lr: float = 5e-5  # Faster critic
    
    # Asymmetric KL settings
    kl_threshold: float = 0.3  # Only penalize KL above this
    kl_lambda_asymmetric: float = 0.1  # Penalty strength for positive KL
    kl_lambda_momentum: float = 5.0  # Penalty for rapidly rising KL
    kl_momentum_window: int = 50  # Steps to compute KL velocity
    
    target_kl: float = 0.05


def load_config(config_path: str) -> S3KLQConfig:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    algorithm = data.pop("algorithm", "safe")
    
    if algorithm == "safe":
        return SAFEConfig(**data)
    elif algorithm == "asymmetric_kl":
        return AsymmetricKLConfig(**data)
    else:
        return S3KLQConfig(**data)
