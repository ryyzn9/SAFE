# mainD.py - S3-KLQ-v3 with Entropy-Aware Predictive Controller
# Based on: main27.py + Plan D (pland28.md)
# Key Features:
#   1. Synchronized LRs (Policy = Critic = 1e-5)
#   2. LayerNorm in DoubleCritic
#   3. Entropy-Gated KL Penalty
#   4. Phase-Aware Thresholding
#   5. PID-based Dynamic Threshold

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from itertools import cycle

print("Installing dependencies...")
os.system("pip install -q transformers accelerate peft datasets matplotlib")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

# =============== CONFIG ===============
@dataclass  
class Config:
    model_name: str = "Qwen/Qwen2.5-3B"
    reward_model_name: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    
    lora_r: int = 128
    lora_alpha: int = 128
    
    batch_size: int = 16
    gradient_accumulation: int = 2
    max_steps: int = 2000
    
    # SYNCHRONIZED LEARNING RATES (Plan D Stability)
    policy_lr: float = 1e-5
    critic_lr: float = 1e-5  # MATCHED with policy
    
    max_new_tokens: int = 128
    
    # Double Critic settings
    alpha_softmin: float = 0.3
    tau_polyak: float = 0.05
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    
    # Plan D: Entropy-Aware Settings
    kl_base_threshold: float = 0.3
    entropy_floor: float = 2.0  # Below this, amplify penalty
    max_preview_kl: float = 0.5  # Hard ceiling for gradient preview
    
    # PID Controller Gains
    pid_kp: float = 0.3
    pid_ki: float = 0.01
    pid_kd: float = 0.05
    
    entropy_coef: float = 0.01
    epochs_per_batch: int = 2
    target_kl: float = 0.05
    
    log_every: int = 50
    save_every: int = 200

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============== TOKENIZER ===============
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =============== REWARD MODEL ===============
print("Loading reward model...")
try:
    from transformers import AutoModelForSequenceClassification
    reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name, trust_remote_code=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    reward_model.eval()
    
    def compute_reward(text: str) -> float:
        inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = reward_model(**inputs)
        if hasattr(outputs, 'score'):
            return outputs.score.item()
        elif hasattr(outputs, 'logits'):
            return outputs.logits[0].item() if outputs.logits.dim() == 1 else outputs.logits[0, 0].item()
        return 0.0
    print("✓ ArmoRM loaded")
except Exception as e:
    print(f"⚠ Fallback reward: {e}")
    def compute_reward(text: str) -> float:
        r = 0.0
        words = text.lower().split()
        if 20 < len(words) < 200: r += 0.3
        if words: r += len(set(words))/len(words) * 0.3
        for w in ["help", "sure", "here"]: 
            if w in text.lower(): r += 0.05
        return min(max(r, -1.0), 1.0)

# =============== REWARD NORMALIZATION ===============
class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = x.mean().item()
        batch_var = x.var().item() if x.shape[0] > 1 else 0.0
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x):
        return (x - self.mean) / (math.sqrt(self.var) + self.epsilon)

# =============== PID CONTROLLER (Plan D) ===============
class PIDController:
    """Simple PID controller for threshold adjustment."""
    def __init__(self, kp=0.3, ki=0.01, kd=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.reward_ema = None
    
    def update(self, reward):
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

# =============== PHASE DETECTOR (Plan D) ===============
class PhaseDetector:
    """Classifies training phase: WARMUP, CLIMBING, PLATEAU, CONVERGED."""
    def __init__(self):
        self.reward_history = []
        self.phase = "WARMUP"
    
    def update(self, reward):
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
    
    def get_threshold_multiplier(self):
        return {
            "WARMUP": 1.5,    # Very relaxed
            "CLIMBING": 1.2,  # Relaxed
            "PLATEAU": 0.8,   # Tight
            "CONVERGED": 1.0  # Normal
        }[self.phase]

# =============== ENTROPY-AWARE PREDICTIVE CONTROLLER (Plan D) ===============
class EntropyAwarePredictiveController:
    """
    Plan D: The Ultimate KL Controller
    
    Components:
    1. Entropy-Gated Penalty (modulates by entropy)
    2. Dual-Timescale Tracking (short + long EMA)
    3. PID Tuning (dynamic threshold)
    4. Phase Detection (context-aware)
    """
    
    def __init__(self, cfg):
        # Dual-Timescale trackers
        self.kl_short = 0.0
        self.kl_long = 0.0
        
        # PID Controller
        self.pid = PIDController(kp=cfg.pid_kp, ki=cfg.pid_ki, kd=cfg.pid_kd)
        
        # Phase Detector
        self.phase_detector = PhaseDetector()
        
        # Settings
        self.base_threshold = cfg.kl_base_threshold
        self.entropy_floor = cfg.entropy_floor
        self.max_preview_kl = cfg.max_preview_kl
    
    def compute_penalty(self, kl, entropy, reward):
        """
        Compute entropy-gated KL penalty with phase awareness.
        
        Args:
            kl: Current KL divergence
            entropy: Current policy entropy
            reward: Current reward
        
        Returns:
            dict with penalty info
        """
        # Update trackers
        self.kl_short = 0.9 * self.kl_short + 0.1 * kl
        self.kl_long = 0.99 * self.kl_long + 0.01 * kl
        self.phase_detector.update(reward)
        
        # Compute dynamic threshold
        pid_adjustment = self.pid.update(reward)
        phase_mult = self.phase_detector.get_threshold_multiplier()
        threshold = (self.base_threshold + pid_adjustment) * phase_mult
        threshold = max(0.1, min(0.6, threshold))
        
        # Determine penalty
        if self.kl_short > threshold:
            base_penalty = 0.1 * (self.kl_short - threshold) ** 2
            
            # Entropy Gate: Low entropy amplifies penalty
            entropy_factor = max(0.5, self.entropy_floor / (entropy + 0.1))
            penalty = base_penalty * entropy_factor
            active = True
        else:
            penalty = 0.0
            entropy_factor = 1.0
            active = False
        
        return {
            'penalty': penalty,
            'threshold': threshold,
            'kl_short': self.kl_short,
            'kl_long': self.kl_long,
            'entropy_factor': entropy_factor,
            'phase': self.phase_detector.phase,
            'active': active
        }
    
    def should_scale_step(self, preview_kl):
        """Called after gradient preview."""
        return preview_kl > self.max_preview_kl
    
    def get_scale_factor(self, preview_kl):
        if preview_kl <= self.max_preview_kl:
            return 1.0
        return self.max_preview_kl / preview_kl

# =============== LORA ===============
lora_config = LoraConfig(
    r=config.lora_r, lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# =============== MODELS ===============
print("Loading policy model...")
policy_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
policy_model = get_peft_model(policy_model, lora_config)
policy_model.print_trainable_parameters()

print("Loading ref model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# =============== DATASET ===============
print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:500]")

def extract_prompt(ex):
    t = ex["chosen"]
    return {"prompt": t.split("Assistant:")[0] + "Assistant:" if "Assistant:" in t else t[:300]}

dataset = dataset.map(extract_prompt)
eval_dataset = eval_dataset.map(extract_prompt)
print(f"Dataset: {len(dataset)} train, {len(eval_dataset)} eval")

# =============== DOUBLE SOFT-MIN CRITIC WITH LAYERNORM ===============
class DoubleCritic(nn.Module):
    """Double Critic with LayerNorm for stability (Plan D)."""
    def __init__(self, hidden_size, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ln = nn.LayerNorm(hidden_size)  # NEW: Stabilize inputs
        self.v1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//4), nn.GELU(), nn.Linear(hidden_size//4, 1))
        self.v2 = nn.Sequential(nn.Linear(hidden_size, hidden_size//4), nn.GELU(), nn.Linear(hidden_size//4, 1))
    
    def forward(self, h):
        h = self.ln(h[:,-1,:])  # Normalize before value heads
        return self.v1(h).squeeze(-1), self.v2(h).squeeze(-1)
    
    def soft_min(self, v1, v2):
        s = torch.stack([-v1/self.alpha, -v2/self.alpha], -1)
        return -self.alpha * (torch.logsumexp(s, -1) - math.log(2))

# =============== S3-KLQ-v3 TRAINER WITH PLAN D CONTROLLER ===============
class S3KLQTrainerD:
    """S3-KLQ-v3 Trainer with Entropy-Aware Predictive Controller."""
    
    def __init__(self, policy, ref, tok, reward_fn, cfg):
        self.policy = policy
        self.ref = ref
        self.tok = tok
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.device = next(policy.parameters()).device
        hs = policy.config.hidden_size
        
        # Critic with LayerNorm
        self.critic = DoubleCritic(hs, cfg.alpha_softmin).to(self.device).to(torch.bfloat16)
        self.critic_target = DoubleCritic(hs, cfg.alpha_softmin).to(self.device).to(torch.bfloat16)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters(): 
            p.requires_grad = False
        
        # SYNCHRONIZED LEARNING RATES
        self.opt_p = torch.optim.AdamW(policy.parameters(), lr=cfg.policy_lr)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        
        self.step = 0
        self.reward_normalizer = RewardNormalizer()
        
        # Plan D: Entropy-Aware Predictive Controller
        self.kl_controller = EntropyAwarePredictiveController(cfg)
        
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        inp = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        plen = inp.input_ids.shape[1]
        out = self.policy.generate(**inp, max_new_tokens=self.cfg.max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=self.tok.pad_token_id)
        resp = self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        texts = [p+r for p,r in zip(prompts, resp)]
        rew = torch.tensor([self.reward_fn(t) for t in texts], device=self.device, dtype=torch.float32)
        
        # Normalize rewards
        self.reward_normalizer.update(rew)
        rew_norm = self.reward_normalizer.normalize(rew)
        
        comp_len = (out[:, plen:] != self.tok.pad_token_id).sum(dim=1).float().mean().item()
        
        policy_out = self.policy(out, attention_mask=(out != self.tok.pad_token_id).long())
        ref_out = self.ref(out, attention_mask=(out != self.tok.pad_token_id).long())
        
        policy_logits = policy_out.logits[:, plen-1:-1, :]
        ref_logits = ref_out.logits[:, plen-1:-1, :]
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        tokens = out[:, plen:]
        policy_lp = torch.gather(policy_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_lp = torch.gather(ref_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        
        mask = (tokens != self.tok.pad_token_id).float()
        kl = ((policy_lp - ref_lp) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Compute entropy for Plan D
        probs = torch.exp(policy_logprobs)
        entropy = -(probs * policy_logprobs).sum(-1)
        entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return {"ids": out, "mask": (out != self.tok.pad_token_id).long(), "rewards": rew_norm, "raw_rewards": rew,
                "kl": kl.mean().item(), "completion_length": comp_len,
                "policy_logprobs": policy_lp, "ref_logprobs": ref_lp,
                "entropy": entropy.mean().item()}
    
    def train_step(self, roll):
        self.policy.train()
        ids, mask, rew = roll["ids"], roll["mask"], roll["rewards"]
        plen = ids.shape[1] - roll["policy_logprobs"].shape[1]
        old_lp, ref_lp = roll["policy_logprobs"], roll["ref_logprobs"]
        response_mask = (ids[:, plen:] != self.tok.pad_token_id).float()
        
        with torch.no_grad():
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            old_v = self.critic.soft_min(v1, v2)
            adv = (rew - old_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        total_p, total_v, total_kl = 0, 0, 0
        for _ in range(self.cfg.epochs_per_batch):
            outputs = self.policy(ids, attention_mask=mask, output_hidden_states=True)
            h = outputs.hidden_states[-1]
            logits = outputs.logits[:, plen-1:-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tokens = ids[:, plen:]
            current_lp = torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            
            ratio = torch.exp(current_lp - old_lp.float())
            adv_exp = adv.unsqueeze(1).expand_as(ratio)
            surr1, surr2 = ratio * adv_exp, torch.clamp(ratio, 1-self.cfg.clip_range, 1+self.cfg.clip_range) * adv_exp
            p_loss = -(torch.min(surr1, surr2) * response_mask).sum() / response_mask.sum().clamp(min=1)
            
            kl = ((current_lp - ref_lp.float()) * response_mask).sum() / response_mask.sum().clamp(min=1)
            
            v1, v2 = self.critic(h)
            v = self.critic.soft_min(v1, v2)
            vc = old_v + torch.clamp(v - old_v, -self.cfg.clip_range_vf, self.cfg.clip_range_vf)
            
            # Use Huber Loss for robustness (cast to float32 to avoid dtype mismatch)
            v_loss = F.huber_loss(v.float(), rew.float(), delta=0.5) + F.huber_loss(vc.float(), rew.float(), delta=0.5)
            v_loss = 0.5 * v_loss
            
            # Calculate entropy
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(-1).mean()
            
            # Plan D: Compute Entropy-Gated Penalty
            kl_info = self.kl_controller.compute_penalty(
                kl.item(), 
                roll["entropy"],
                roll["raw_rewards"].mean().item()
            )
            kl_penalty = kl_info['penalty']
            
            # Final loss: PPO + Value + Entropy-Gated KL Penalty - Entropy
            loss = p_loss + 0.5 * v_loss + kl_penalty - self.cfg.entropy_coef * entropy
            
            self.opt_p.zero_grad()
            self.opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # Tighter clip for critic
            self.opt_p.step()
            self.opt_c.step()
            
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1-self.cfg.tau_polyak).add_(self.cfg.tau_polyak * p.data)
            
            total_p += p_loss.item()
            total_v += v_loss.item()
            total_kl += kl.item()
        
        avg_kl = total_kl / self.cfg.epochs_per_batch
        
        self.step += 1
        return {
            "step": self.step, 
            "reward": roll["raw_rewards"].mean().item(), 
            "reward_std": roll["raw_rewards"].std().item(),
            "policy_loss": total_p / self.cfg.epochs_per_batch, 
            "value_loss": total_v / self.cfg.epochs_per_batch,
            "kl": avg_kl, 
            "completion_length": roll["completion_length"],
            "entropy": roll["entropy"],
            "kl_penalty": kl_info['penalty'],
            "kl_threshold": kl_info['threshold'],
            "phase": kl_info['phase'],
            "kl_active": kl_info['active']
        }

# =============== PLOTTING ===============
def plot_results(results, save_path="training_plots_mainD.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.plot(results["steps"], results["rewards"], label="S3-KLQ-v3", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.set_title("Reward"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(results["steps"], results["kl"], label="KL", alpha=0.7)
    ax.plot(results["steps"], results["kl_threshold"], label="Dynamic Threshold", alpha=0.7, linestyle='--')
    ax.set_xlabel("Step"); ax.set_ylabel("KL"); ax.set_title("KL vs Dynamic Threshold"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(results["steps"], results["value_loss"], label="Value Loss", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Value Loss"); ax.set_title("Value Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(results["steps"], results["entropy"], label="Entropy", alpha=0.7)
    ax.axhline(y=config.entropy_floor, color='r', linestyle='--', label=f'Entropy Floor ({config.entropy_floor})')
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy"); ax.set_title("Policy Entropy"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(results["steps"], results["completion_length"], label="Completion Length", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Tokens"); ax.set_title("Completion Length"); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    window = min(100, len(results["rewards"])//10)
    if window > 1:
        smooth = np.convolve(results["rewards"], np.ones(window)/window, mode='valid')
        ax.plot(range(len(smooth)), smooth, label="Smoothed Reward", linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward (smoothed)"); ax.set_title("Smoothed Reward"); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✓ Plots saved to {save_path}")

# =============== INITIALIZE ===============
print("\n" + "="*60)
print("Initializing S3-KLQ-v3 Trainer (Plan D: Entropy-Aware Predictive)")
print("="*60)

trainer = S3KLQTrainerD(policy_model, ref_model, tokenizer, compute_reward, config)
print("✓ S3-KLQ-v3 trainer initialized:")
print(f"  - Policy LR: {config.policy_lr}")
print(f"  - Critic LR: {config.critic_lr} (SYNCHRONIZED)")
print(f"  - Base KL Threshold: {config.kl_base_threshold}")
print(f"  - Entropy Floor: {config.entropy_floor}")
print(f"  - PID Gains: Kp={config.pid_kp}, Ki={config.pid_ki}, Kd={config.pid_kd}")

results = {
    "steps": [], "rewards": [], "reward_std": [], "kl": [], 
    "value_loss": [], "completion_length": [], "entropy": [],
    "kl_penalty": [], "kl_threshold": [], "phase": []
}

def collate_fn(batch):
    return {"prompt": [item["prompt"] for item in batch]}

# =============== TRAIN ===============
print(f"\n{'='*60}")
print(f"Training S3-KLQ-v3 (Plan D) - {config.max_steps} steps")
print("="*60)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
infinite_loader = cycle(train_loader)
start_time = time.time()

for step in tqdm(range(config.max_steps)):
    batch = next(infinite_loader)
    rollouts = trainer.generate_rollouts(batch["prompt"])
    metrics = trainer.train_step(rollouts)
    
    results["steps"].append(step)
    results["rewards"].append(metrics["reward"])
    results["reward_std"].append(metrics["reward_std"])
    results["kl"].append(metrics["kl"])
    results["value_loss"].append(metrics["value_loss"])
    results["completion_length"].append(metrics["completion_length"])
    results["entropy"].append(metrics["entropy"])
    results["kl_penalty"].append(metrics["kl_penalty"])
    results["kl_threshold"].append(metrics["kl_threshold"])
    results["phase"].append(metrics["phase"])
    
    if step % config.log_every == 0:
        kl_status = "⚠️PENALIZED" if metrics["kl_active"] else "✓OK"
        print(f"Step {step}: reward={metrics['reward']:.3f}, kl={metrics['kl']:.4f} {kl_status}, "
              f"threshold={metrics['kl_threshold']:.3f}, phase={metrics['phase']}, "
              f"entropy={metrics['entropy']:.2f}, v_loss={metrics['value_loss']:.3f}")
    
    # Save checkpoint periodically
    if step > 0 and step % config.save_every == 0:
        with open(f"results_mainD_step{step}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Checkpoint saved: results_mainD_step{step}.json")

train_time = time.time() - start_time

# =============== SUMMARY ===============
print("\n" + "="*60)
print("EXPERIMENT SUMMARY (mainD.py - Plan D: Entropy-Aware Predictive)")
print("="*60)

print(f"\nTraining Results:")
print(f"  Final reward: {np.mean(results['rewards'][-200:]):.3f} ± {np.mean(results['reward_std'][-200:]):.3f}")
print(f"  Max reward: {max(results['rewards']):.3f}")
print(f"  Final KL: {np.mean(results['kl'][-200:]):.4f}")
print(f"  Final Entropy: {np.mean(results['entropy'][-200:]):.2f}")
print(f"  Value Loss Spikes (>0.1): {sum(1 for v in results['value_loss'] if v > 0.1)}")
print(f"  KL Penalty Activations: {sum(1 for r in results['kl_penalty'] if r > 0)}")
print(f"  Training time: {train_time/60:.1f} min")

# =============== PLOT & SAVE ===============
plot_results(results, "training_plots_mainD.png")

with open("results_mainD.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Results saved to results_mainD.json")
print("✓ EXPERIMENT COMPLETE")
