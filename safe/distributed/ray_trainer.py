"""Ray Distributed Trainer for multi-GPU training."""

import ray
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from itertools import cycle

from safe.config import SAFEConfig, AsymmetricKLConfig, load_config
from safe.distributed.placement import create_placement_group
from safe.distributed.utils import shard_batch, merge_rollouts


@ray.remote(num_gpus=1)
class RolloutWorker:
    """
    Ray remote actor for generating rollouts on a single GPU.
    
    Each worker holds its own copy of the policy, reference, and reward models,
    and generates rollouts independently. This is the core of data parallelism.
    """
    
    def __init__(self, config: SAFEConfig, worker_id: int):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        from safe.reward.reward_model import load_reward_model
        
        self.config = config
        self.worker_id = worker_id
        self.device = torch.device("cuda")
        
        print(f"[Worker {worker_id}] Initializing on GPU {torch.cuda.current_device()}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load policy model with LoRA
        self.policy = AutoModelForCausalLM.from_pretrained(
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
        self.policy = get_peft_model(self.policy, lora_config)
        
        # Load reference model (frozen)
        self.ref = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False
        
        # Load reward model
        _, self.reward_fn = load_reward_model(config.reward_model_name)
        
        print(f"[Worker {worker_id}] ✓ Initialization complete")
    
    def generate_rollouts(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate rollouts for a batch of prompts."""
        import torch
        import torch.nn.functional as F
        
        self.policy.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            prompt_len = inputs.input_ids.shape[1]
            
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            responses = self.tokenizer.batch_decode(
                outputs[:, prompt_len:], skip_special_tokens=True
            )
            full_texts = [p + r for p, r in zip(prompts, responses)]
            
            raw_rewards = torch.tensor(
                [self.reward_fn(t) for t in full_texts],
                device=self.device,
                dtype=torch.float32,
            )
            
            completion_mask = outputs[:, prompt_len:] != self.tokenizer.pad_token_id
            completion_length = completion_mask.sum(dim=1).float().mean().item()
            
            attention_mask = (outputs != self.tokenizer.pad_token_id).long()
            
            policy_out = self.policy(outputs, attention_mask=attention_mask)
            ref_out = self.ref(outputs, attention_mask=attention_mask)
            
            policy_logits = policy_out.logits[:, prompt_len - 1:-1, :]
            ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
            
            policy_logprobs = F.log_softmax(policy_logits, dim=-1)
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            
            tokens = outputs[:, prompt_len:]
            policy_lp = torch.gather(policy_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            ref_lp = torch.gather(ref_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            
            mask = (tokens != self.tokenizer.pad_token_id).float()
            kl = ((policy_lp - ref_lp) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            probs = torch.exp(policy_logprobs)
            entropy = -(probs * policy_logprobs).sum(dim=-1)
            entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Move to CPU for transfer
        return {
            "ids": outputs.cpu(),
            "mask": attention_mask.cpu(),
            "rewards": raw_rewards.cpu(),
            "raw_rewards": raw_rewards.cpu(),
            "kl": kl.mean().item(),
            "completion_length": completion_length,
            "policy_logprobs": policy_lp.cpu(),
            "ref_logprobs": ref_lp.cpu(),
            "entropy": entropy.mean().item(),
            "prompt_len": prompt_len,
        }
    
    def update_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Update policy weights from central trainer."""
        self.policy.load_state_dict(state_dict)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current policy weights."""
        return {k: v.cpu() for k, v in self.policy.state_dict().items()}


class RayDistributedTrainer:
    """
    Coordinates distributed training across N GPUs using Ray.
    
    Architecture:
    - N RolloutWorkers, each on its own GPU, generate rollouts in parallel
    - Central trainer aggregates rollouts and computes gradients
    - Weights are synchronized back to workers after each step
    
    This gives near-linear speedup for rollout generation (the bottleneck).
    """
    
    def __init__(
        self,
        config: SAFEConfig,
        algorithm: str = "safe",
        address: Optional[str] = None,
    ):
        """
        Initialize distributed trainer.
        
        Args:
            config: Training configuration
            algorithm: "safe" or "asymmetric_kl"
            address: Ray cluster address (None for local)
        """
        self.config = config
        self.algorithm = algorithm
        
        # Initialize Ray
        if not ray.is_initialized():
            if address:
                ray.init(address=address)
            else:
                ray.init()
        
        print(f"Ray initialized. Available GPUs: {ray.available_resources().get('GPU', 0)}")
        
        # Create placement group
        if config.use_placement_groups:
            self.pg = create_placement_group(config.num_gpus)
        else:
            self.pg = None
        
        # Create workers
        self.workers = []
        for i in range(config.num_gpus):
            if self.pg:
                worker = RolloutWorker.options(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=self.pg,
                        placement_group_bundle_index=i,
                    )
                ).remote(config, i)
            else:
                worker = RolloutWorker.remote(config, i)
            self.workers.append(worker)
        
        print(f"✓ Created {config.num_gpus} rollout workers")
        
        # Initialize local trainer for gradient computation
        self._init_local_trainer()
    
    def _init_local_trainer(self):
        """Initialize local trainer for gradient updates."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        from safe.reward.reward_model import load_reward_model
        from safe.trainers.safe import SAFETrainer
        from safe.trainers.asymmetric_kl import AsymmetricKLTrainer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load policy model on main GPU
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.policy = get_peft_model(self.policy, lora_config)
        
        # Load reference (frozen)
        self.ref = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False
        
        _, reward_fn = load_reward_model(self.config.reward_model_name)
        
        # Create appropriate trainer
        if self.algorithm == "safe":
            self.trainer = SAFETrainer(
                self.policy, self.ref, self.tokenizer, reward_fn,
                SAFEConfig(**{k: getattr(self.config, k) for k in self.config.__dataclass_fields__})
            )
        else:
            self.trainer = AsymmetricKLTrainer(
                self.policy, self.ref, self.tokenizer, reward_fn,
                AsymmetricKLConfig(**{k: getattr(self.config, k) for k in self.config.__dataclass_fields__})
            )
        
        print(f"✓ Local trainer initialized ({self.algorithm})")
    
    def train_step(self, batch: List[str]) -> Dict[str, Any]:
        """
        Execute one distributed training step.
        
        1. Shard prompts across workers
        2. Generate rollouts in parallel
        3. Merge rollouts
        4. Compute gradients and update policy
        5. Sync weights to workers
        """
        # Shard batch across workers
        shards = shard_batch(batch, len(self.workers))
        
        # Generate rollouts in parallel
        futures = [
            worker.generate_rollouts.remote(shard)
            for worker, shard in zip(self.workers, shards)
        ]
        rollouts = ray.get(futures)
        
        # Merge rollouts
        merged = merge_rollouts(rollouts)
        
        # Move back to GPU
        for key in ["ids", "mask", "rewards", "raw_rewards", "policy_logprobs", "ref_logprobs"]:
            if key in merged:
                merged[key] = merged[key].to(self.device)
        
        # Train step
        metrics = self.trainer.train_step(merged)
        
        # Sync weights to workers
        state_dict = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        sync_futures = [worker.update_weights.remote(state_dict) for worker in self.workers]
        ray.get(sync_futures)
        
        return metrics
    
    def train(
        self,
        dataset,
        max_steps: int,
        log_every: int = 50,
        save_every: int = 200,
    ):
        """
        Run full training loop.
        
        Args:
            dataset: Dataset with 'prompt' field
            max_steps: Maximum training steps
            log_every: Steps between logging
            save_every: Steps between checkpoints
        """
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        import json
        import time
        
        from safe.data.dataset import collate_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size * len(self.workers),  # Scale batch size
            shuffle=True,
            collate_fn=collate_fn,
        )
        data_iter = cycle(iter(dataloader))
        
        results = {
            "steps": [],
            "rewards": [],
            "reward_std": [],
            "kl": [],
            "value_loss": [],
        }
        
        start_time = time.time()
        
        for step in tqdm(range(max_steps), desc=f"Training ({self.algorithm})"):
            batch = next(data_iter)
            metrics = self.train_step(batch["prompt"])
            
            # Record metrics
            results["steps"].append(step)
            results["rewards"].append(metrics["reward"])
            results["reward_std"].append(metrics["reward_std"])
            results["kl"].append(metrics["kl"])
            results["value_loss"].append(metrics["value_loss"])
            
            if step % log_every == 0:
                print(f"Step {step}: reward={metrics['reward']:.3f}, kl={metrics['kl']:.4f}")
            
            if step > 0 and step % save_every == 0:
                with open(f"results_step{step}.json", "w") as f:
                    json.dump(results, f)
        
        total_time = time.time() - start_time
        print(f"Training complete. Total time: {total_time / 60:.1f} min")
        
        return results
    
    def shutdown(self):
        """Shutdown Ray workers."""
        ray.shutdown()
