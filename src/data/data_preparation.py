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

# Configure logging
logger = setup_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, input_path: str, transform: Optional[callable] = None):
        """
        Initialize dataset from a directory or single file containing text prompts.

        Args:
            input_path (str): Path to a directory of .txt files or a single .txt file.
            transform (callable, optional): Optional transform to apply to prompts.
        """
        self.input_path = Path(input_path)
        self.transform = transform
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> List[str]:
        """
        Load prompts from a directory of .txt files or a single file.

        Returns:
            List[str]: List of prompt strings.
        """
        prompts = []
        
        if self.input_path.is_file() and self.input_path.suffix == '.txt':
            # Handle single file for backward compatibility
            with open(self.input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        prompts.append(clean_line)
            logger.info(f"Loaded {len(prompts)} prompts from {self.input_path}")
        elif self.input_path.is_dir():
            # Handle directory of .txt files
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
            raise ValueError(f"Invalid input path: {self.input_path} is neither a .txt file nor a directory")

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
    logger: Optional[logging.Logger] = None
) -> Dataset:
    """
    Prepare a tokenized dataset for fine-tuning from a directory or file.

    Args:
        input_path (str): Path to a directory of .txt files or a single .txt file.
        tokenizer (AutoTokenizer): Tokenizer for processing prompts.
        max_length (int): Maximum sequence length for tokenization.
        logger (logging.Logger, optional): Logger for info and warnings.

    Returns:
        Dataset: Hugging Face Dataset with input_ids, attention_mask, and labels.
    """
    if logger is None:
        logger = setup_logger(__name__)

    logger.info(f"Preparing dataset from {input_path}")
    prompt_dataset = TextPromptDataset(input_path, transform=None)
    prompts = prompt_dataset.prompts

    if len(prompts) == 0:
        raise ValueError("No valid prompts in dataset")
    if len(prompts) < 4:
        logger.warning(f"Dataset size ({len(prompts)}) is smaller than typical batch size (4)")

    try:
        tokenized = tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone()
    })
    logger.info(f"Dataset prepared with {len(dataset)} examples")
    return dataset

def load_kd_dataset(
    input_path: str,
    logits_dir: str,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    logger: Optional[logging.Logger] = None
) -> Dataset:
    """
    Load a dataset for knowledge distillation, including teacher logits.

    Args:
        input_path (str): Path to a directory of .txt files or a single .txt file.
        logits_dir (str): Directory containing teacher logits (.npy files).
        tokenizer (AutoTokenizer): Tokenizer for processing prompts.
        max_length (int): Maximum sequence length for tokenization.
        logger (logging.Logger, optional): Logger for info and warnings.

    Returns:
        Dataset: Hugging Face Dataset with input_ids, attention_mask, labels, and teacher_logits.
    """
    if logger is None:
        logger = setup_logger(__name__)

    logger.info(f"Preparing KD dataset from {input_path} with logits from {logits_dir}")
    # Load prompts and tokenize
    prompt_dataset = TextPromptDataset(input_path, transform=None)
    prompts = prompt_dataset.prompts

    if len(prompts) == 0:
        raise ValueError("No valid prompts in dataset")
    if len(prompts) < 4:
        logger.warning(f"Dataset size ({len(prompts)}) is smaller than typical batch size (4)")

    try:
        tokenized = tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise

    # Load teacher logits
    logits_dir = Path(logits_dir)
    logit_files = sorted(logits_dir.glob("logits_batch*.npy"))
    if not logit_files:
        raise FileNotFoundError(f"No logit files found in {logits_dir}")

    teacher_logits = []
    expected_vocab_size = tokenizer.vocab_size
    for logit_file in logit_files:
        try:
            logits = np.load(logit_file)
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            if logits_tensor.size(1) != max_length or logits_tensor.size(2) != expected_vocab_size:
                logger.error(f"Invalid logits shape in {logit_file}: {logits_tensor.shape}, expected [batch, {max_length}, {expected_vocab_size}]")
                raise ValueError(f"Invalid logits shape in {logit_file}")
            teacher_logits.append(logits_tensor)
        except Exception as e:
            logger.error(f"Failed to load logits from {logit_file}: {str(e)}")
            raise

    try:
        teacher_logits = torch.cat(teacher_logits, dim=0)
    except RuntimeError as e:
        logger.error(f"Failed to concatenate logits: {str(e)}")
        for idx, t in enumerate(teacher_logits):
            logger.error(f"Logits tensor {idx} shape: {t.shape}")
        raise

    if len(teacher_logits) != len(prompts):
        raise ValueError(f"Mismatch between prompts ({len(prompts)}) and teacher logits ({len(teacher_logits)})")

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone(),
        'teacher_logits': teacher_logits
    })
    logger.info(f"KD dataset prepared with {len(dataset)} examples")
    return dataset

def data_collator(
    features: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> Dict[str, torch.Tensor]:
    """
    Collate dataset features into a batch for training.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries with input_ids, attention_mask, labels, and optionally teacher_logits.
        logger (logging.Logger, optional): Logger for debugging batch shapes.

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