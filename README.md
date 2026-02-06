# SAFE: Ray Distributed Alignment Training

A modular implementation of alignment algorithms with Ray-based distributed training for multi-GPU setups (8-N H100s).

## Algorithms

- **SAFE**: Entropy-Aware Predictive Controller with synchronized LRs and LayerNorm critics
- **Asymmetric KL**: Double Soft-Min Critics with asymmetric KL penalty

## Installation

```bash
cd notebooks/SAFE/safe
pip install -e .
```

## Usage

### Single GPU Training
```bash
python scripts/train.py --config configs/base.yaml --algorithm safe
```

### Multi-GPU Distributed Training
```bash
# Start Ray (if not using existing cluster)
ray start --head --num-gpus=8

# Launch training
python scripts/train_distributed.py \
    --config configs/h100_8gpu.yaml \
    --algorithm safe \
    --num_gpus 8
```

### Scaling to More GPUs
Simply change `num_gpus` in config or CLI:
```bash
python scripts/train_distributed.py --config configs/h100_16gpu.yaml --num_gpus 16
```

## Project Structure

```
safe/
├── safe/
│   ├── config.py           # Configuration dataclasses
│   ├── controllers/        # KL controllers (asymmetric, entropy-aware, PID)
│   ├── models/             # Critic networks
│   ├── reward/             # Reward model utilities
│   ├── data/               # Dataset loaders
│   ├── trainers/           # SAFE, Asymmetric KL, PPO trainers
│   └── distributed/        # Ray distributed training
├── scripts/                # Training & evaluation scripts
└── configs/                # YAML configuration files
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Ray >= 2.9.0
- transformers, peft, accelerate
