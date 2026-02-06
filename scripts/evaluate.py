#!/usr/bin/env python
"""Evaluation script for trained models."""

import argparse
import json
import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safe.reward.reward_model import load_reward_model
from safe.data.dataset import load_hh_rlhf


def evaluate(model_path: str, config_path: str = None, num_samples: int = 100):
    """Evaluate a trained model."""
    from safe.config import load_config
    
    if config_path:
        config = load_config(config_path)
    else:
        # Default config
        model_name = "Qwen/Qwen2.5-3B"
        reward_model_name = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("Loading reward model...")
    _, reward_fn = load_reward_model(config.reward_model_name)
    
    print("Loading eval dataset...")
    _, eval_data = load_hh_rlhf(train_split="train[:100]", eval_split=f"test[:{num_samples}]")
    
    device = next(model.parameters()).device
    rewards = []
    
    print(f"Evaluating on {num_samples} samples...")
    for example in tqdm(eval_data):
        prompt = example["prompt"]
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        full_text = prompt + response
        
        reward = reward_fn(full_text)
        rewards.append(reward)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {mean_reward:.4f}")
    print(f"  Std Reward: {std_reward:.4f}")
    print(f"  Min Reward: {min(rewards):.4f}")
    print(f"  Max Reward: {max(rewards):.4f}")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "rewards": rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SAFE model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved LoRA weights")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    
    results = evaluate(args.model_path, args.config, args.num_samples)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
