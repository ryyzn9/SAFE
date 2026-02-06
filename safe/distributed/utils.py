"""Utility functions for distributed training."""

import torch
import numpy as np
from typing import Dict, Any, List


def shard_batch(batch: List[str], num_shards: int) -> List[List[str]]:
    """
    Shard a batch of prompts across workers.
    
    Args:
        batch: List of prompts
        num_shards: Number of shards (workers)
        
    Returns:
        List of prompt shards
    """
    return [list(shard) for shard in np.array_split(batch, num_shards)]


def merge_rollouts(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge rollouts from multiple workers.
    
    Args:
        rollouts: List of rollout dicts from workers
        
    Returns:
        Merged rollout dict
    """
    if len(rollouts) == 1:
        return rollouts[0]
    
    merged = {}
    
    # Concatenate tensor fields
    tensor_keys = ["ids", "mask", "rewards", "raw_rewards", "policy_logprobs", "ref_logprobs"]
    for key in tensor_keys:
        if key in rollouts[0]:
            tensors = [r[key] for r in rollouts]
            merged[key] = torch.cat(tensors, dim=0)
    
    # Average scalar fields
    scalar_keys = ["kl", "completion_length", "entropy"]
    for key in scalar_keys:
        if key in rollouts[0]:
            values = [r[key] for r in rollouts]
            merged[key] = sum(values) / len(values)
    
    # Take first value for metadata
    if "prompt_len" in rollouts[0]:
        merged["prompt_len"] = rollouts[0]["prompt_len"]
    
    return merged


def aggregate_gradients(gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Average gradients from multiple workers.
    
    Args:
        gradients: List of gradient dicts
        
    Returns:
        Averaged gradients
    """
    if len(gradients) == 1:
        return gradients[0]
    
    averaged = {}
    for key in gradients[0]:
        stacked = torch.stack([g[key] for g in gradients])
        averaged[key] = stacked.mean(dim=0)
    
    return averaged


def sync_model_weights(source_model, target_models: list):
    """
    Synchronize model weights from source to targets.
    
    Args:
        source_model: Source model to copy from
        target_models: List of target models to copy to
    """
    state_dict = source_model.state_dict()
    for target in target_models:
        target.load_state_dict(state_dict)
