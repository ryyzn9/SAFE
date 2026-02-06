#!/usr/bin/env python
"""Main training script for SAFE."""

import argparse
import json
import time
from itertools import cycle

import torch
from tqdm.auto import tqdm

from safe.config import load_config, SAFEConfig, AsymmetricKLConfig
from safe.data.dataset import load_hh_rlhf, create_dataloader
from safe.reward.reward_model import load_reward_model


def create_models(config):
    """Load policy, reference, and create trainer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from safe.trainers.safe import SAFETrainer
    from safe.trainers.asymmetric_kl import AsymmetricKLTrainer
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_config)
    policy.print_trainable_parameters()
    
    print("Loading reference model...")
    ref = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    
    print("Loading reward model...")
    _, reward_fn = load_reward_model(config.reward_model_name)
    
    # Create trainer
    if isinstance(config, SAFEConfig):
        trainer = SAFETrainer(policy, ref, tokenizer, reward_fn, config)
        print("✓ SAFE trainer initialized")
    else:
        trainer = AsymmetricKLTrainer(policy, ref, tokenizer, reward_fn, config)
        print("✓ Asymmetric KL trainer initialized")
    
    return trainer, tokenizer


def train(config, trainer, dataset, args):
    """Run training loop."""
    dataloader = create_dataloader(dataset, config.batch_size)
    data_iter = cycle(iter(dataloader))
    
    results = {
        "steps": [],
        "rewards": [],
        "reward_std": [],
        "kl": [],
        "value_loss": [],
    }
    
    start_time = time.time()
    
    for step in tqdm(range(config.max_steps), desc="Training"):
        batch = next(data_iter)
        rollouts = trainer.generate_rollouts(batch["prompt"])
        metrics = trainer.train_step(rollouts)
        
        results["steps"].append(step)
        results["rewards"].append(metrics["reward"])
        results["reward_std"].append(metrics["reward_std"])
        results["kl"].append(metrics["kl"])
        results["value_loss"].append(metrics["value_loss"])
        
        if step % config.log_every == 0:
            print(f"Step {step}: reward={metrics['reward']:.3f}, kl={metrics['kl']:.4f}")
        
        if step > 0 and step % config.save_every == 0:
            with open(f"results_step{step}.json", "w") as f:
                json.dump(results, f)
    
    total_time = time.time() - start_time
    print(f"✓ Training complete. Total time: {total_time / 60:.1f} min")
    
    # Save final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {args.output}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train SAFE alignment model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--algorithm", type=str, choices=["safe", "asymmetric_kl"], default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--dry-run", action="store_true", help="Just load config and exit")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.max_steps:
        config.max_steps = args.max_steps
    
    if args.dry_run:
        print(f"Config loaded: {config}")
        return
    
    # Load dataset
    train_data, eval_data = load_hh_rlhf(config.train_split, config.eval_split)
    
    # Create trainer
    trainer, tokenizer = create_models(config)
    
    # Train
    train(config, trainer, train_data, args)


if __name__ == "__main__":
    main()
