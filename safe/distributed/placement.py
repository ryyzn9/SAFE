"""Ray placement group utilities for multi-GPU training."""

import ray
from typing import Optional


def create_placement_group(
    num_gpus: int,
    strategy: str = "PACK",
    name: Optional[str] = None,
) -> ray.util.placement_group.PlacementGroup:
    """
    Create a Ray placement group for distributed training.
    
    Args:
        num_gpus: Number of GPUs to allocate
        strategy: Placement strategy ("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD")
        name: Optional name for the placement group
        
    Returns:
        Ray PlacementGroup object
    """
    # Create bundles - one per GPU
    bundles = [{"GPU": 1, "CPU": 4} for _ in range(num_gpus)]
    
    pg = ray.util.placement_group(
        bundles,
        strategy=strategy,
        name=name or f"safe_pg_{num_gpus}gpu",
    )
    
    # Wait for placement group to be ready
    ray.get(pg.ready())
    print(f"✓ Created placement group with {num_gpus} GPUs (strategy: {strategy})")
    
    return pg


def get_available_gpus() -> int:
    """Get number of available GPUs in Ray cluster."""
    if not ray.is_initialized():
        ray.init()
    
    resources = ray.available_resources()
    return int(resources.get("GPU", 0))
