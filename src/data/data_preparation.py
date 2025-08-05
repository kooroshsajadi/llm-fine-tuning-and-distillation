import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import Dataset
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, input_path: str, transform: Optional[callable] = None):
        self.input_path = Path(input_path)
        self.transform = transform
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> List[str]:
        prompts = []
        if self.input_path.is_file() and self.input_path.suffix == '.txt':
            with open(self.input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        prompts.append(clean_line)
            logger.info(f"Loaded {len(prompts)} prompts from {self.input_path}")
        elif self.input_path.is_dir():
            txt_files = sorted(self.input_path.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(f"No .txt files found in {self.input_path}")
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        prompts.append(content)
            logger.info(f"Loaded {len(prompts)} prompts from {len(txt_files)} files in {self.input_path}")
        else:
            raise ValueError(f"Invalid input path: {self.input_path}")
        if not prompts:
            raise ValueError(f"No valid prompts found in {self.input_path}")
        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        if self.transform:
            prompt = self.transform(prompt)
        return prompt

def prepare_tokenized_dataset(
    input_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    logger: Optional[logging.Logger] = None,
    model_type: str = "causal_lm"
) -> Dataset:
    """
    Prepare a tokenized dataset for fine-tuning.

    Args:
        input_path (str): Path to directory of .txt files or a single .txt file.
        tokenizer (AutoTokenizer): Tokenizer for processing prompts.
        max_length (int): Maximum sequence length.
        logger (logging.Logger, optional): Logger for info and warnings.
        model_type (str): Model type ("causal_lm" or "masked_lm").

    Returns:
        Dataset: Hugging Face Dataset with input_ids, attention_mask, and labels.
    """
    if logger is None:
        logger = setup_logger(__name__)

    logger.info(f"Preparing dataset from {input_path} for {model_type}")
    prompt_dataset = TextPromptDataset(input_path, transform=None)
    prompts = prompt_dataset.prompts

    if len(prompts) == 0:
        raise ValueError("No valid prompts in dataset")
    if len(prompts) < 4:
        logger.warning(f"Dataset size ({len(prompts)}) is smaller than typical batch size (4)")

    try:
        if model_type == "masked_lm":
            # For MLM, add random masking to labels
            tokenized = tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_special_tokens_mask=True
            )
            labels = tokenized['input_ids'].clone()
            special_tokens_mask = tokenized['special_tokens_mask'].bool()
            probability_matrix = torch.full(labels.shape, 0.15)
            probability_matrix.masked_fill_(special_tokens_mask, 0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Ignore non-masked tokens in loss
            labels[masked_indices] = tokenized['input_ids'][masked_indices]
        else:
            # For CLM, use input_ids as labels
            tokenized = tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            labels = tokenized['input_ids'].clone()

        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        })
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise

def data_collator(
    features: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
    model_type: str = "causal_lm"
) -> Dict[str, torch.Tensor]:
    """
    Collate dataset features into a batch for training.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries.
        logger (logging.Logger, optional): Logger for debugging.
        model_type (str): Model type ("causal_lm" or "masked_lm").

    Returns:
        Dict[str, torch.Tensor]: Batched tensors for input_ids, attention_mask, labels, and teacher_logits.
    """
    if logger is None:
        logger = setup_logger(__name__)

    required_keys = ['input_ids', 'attention_mask', 'labels']
    if not all(all(key in f for key in required_keys) for f in features):
        raise ValueError("Invalid dataset features: missing input_ids, attention_mask, or labels")

    batch = {
        'input_ids': torch.stack([torch.tensor(f['input_ids'], dtype=torch.long) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask'], dtype=torch.long) for f in features]),
        'labels': torch.stack([torch.tensor(f['labels'], dtype=torch.long) for f in features])
    }
    if all('teacher_logits' in f for f in features):
        batch['teacher_logits'] = torch.stack([torch.tensor(f['teacher_logits'], dtype=torch.float32) for f in features])

    logger.debug(f"Batch shapes: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}, labels={batch['labels'].shape}" +
                 (f", teacher_logits={batch['teacher_logits'].shape}" if 'teacher_logits' in batch else ""))
    return batch