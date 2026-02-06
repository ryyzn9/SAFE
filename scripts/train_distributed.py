#!/usr/bin/env python
"""Distributed training script using Ray."""

import argparse

from safe.config import load_config
from safe.data.dataset import load_hh_rlhf
from safe.distributed.ray_trainer import RayDistributedTrainer


def main():
    parser = argparse.ArgumentParser(description="Distributed SAFE training with Ray")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--algorithm", type=str, choices=["safe", "asymmetric_kl"], default="safe")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs (overrides config)")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--ray_address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--output", type=str, default="results_distributed.json")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.num_gpus:
        config.num_gpus = args.num_gpus
    if args.max_steps:
        config.max_steps = args.max_steps
    
    print(f"="*60)
    print(f"Distributed Training Configuration")
    print(f"="*60)
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Num GPUs: {config.num_gpus}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Batch Size (per GPU): {config.batch_size}")
    print(f"  Total Batch Size: {config.batch_size * config.num_gpus}")
    print(f"="*60)
    
    # Load dataset
    train_data, eval_data = load_hh_rlhf(config.train_split, config.eval_split)
    
    # Create distributed trainer
    trainer = RayDistributedTrainer(
        config=config,
        algorithm=args.algorithm,
        address=args.ray_address,
    )
    
    # Train
    results = trainer.train(
        dataset=train_data,
        max_steps=config.max_steps,
        log_every=config.log_every,
        save_every=config.save_every,
    )
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {args.output}")
    
    trainer.shutdown()


if __name__ == "__main__":
    main()
