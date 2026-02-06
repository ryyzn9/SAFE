"""Dataset loading and processing utilities."""

from typing import Callable
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


def extract_prompt(example: dict) -> dict:
    """Extract prompt from HH-RLHF format."""
    text = example["chosen"]
    if "Assistant:" in text:
        prompt = text.split("Assistant:")[0] + "Assistant:"
    else:
        prompt = text[:300]
    return {"prompt": prompt}


def load_hh_rlhf(
    train_split: str = "train[:5000]",
    eval_split: str = "test[:500]",
) -> tuple[Dataset, Dataset]:
    """
    Load Anthropic HH-RLHF dataset.
    
    Args:
        train_split: Training data split specification
        eval_split: Evaluation data split specification
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_data = load_dataset("Anthropic/hh-rlhf", split=train_split)
    eval_data = load_dataset("Anthropic/hh-rlhf", split=eval_split)
    
    train_data = train_data.map(extract_prompt)
    eval_data = eval_data.map(extract_prompt)
    
    print(f"✓ Dataset loaded: {len(train_data)} train, {len(eval_data)} eval")
    return train_data, eval_data


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader."""
    return {"prompt": [item["prompt"] for item in batch]}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
