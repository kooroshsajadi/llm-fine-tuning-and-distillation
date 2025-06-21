import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer
from datasets import Dataset
from pathlib import Path

class TextPromptDataset(Dataset):
    def __init__(self, file_path: str, transform: Optional[callable] = None):
        self.file_path = file_path
        self.transform = transform
        self.prompts = self._load_prompts(file_path)
    
    def _load_prompts(self, file_path: str) -> List[str]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    prompts.append(clean_line)
        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        if self.transform:
            prompt = self.transform(prompt)
        return prompt

def prepare_tokenized_dataset(
    file_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    logger: Optional[logging.Logger] = None
) -> Dataset:
    """
    Prepare a tokenized dataset for fine-tuning.

    Args:
        file_path (str): Path to the prompt file.
        tokenizer (AutoTokenizer): Tokenizer for processing prompts.
        max_length (int): Maximum sequence length for tokenization.
        logger (logging.Logger, optional): Logger for info and warnings.

    Returns:
        Dataset: Hugging Face Dataset with input_ids, attention_mask, and labels.
    """
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)

    prompt_dataset = TextPromptDataset(file_path, transform=None)
    prompts = prompt_dataset.prompts

    if len(prompts) == 0:
        raise ValueError("No valid prompts in dataset")
    if len(prompts) < 4:
        logger.warning(f"Dataset size ({len(prompts)}) is smaller than typical batch size (4)")

    tokenized = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone()
    })
    logger.info(f"Dataset prepared with {len(dataset)} examples")
    return dataset

def load_kd_dataset(
    prompt_file: str,
    logits_dir: str,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    logger: Optional[logging.Logger] = None
) -> Dataset:
    """
    Load a dataset for knowledge distillation, including teacher logits.

    Args:
        prompt_file (str): Path to the prompt file.
        logits_dir (str): Directory containing teacher logits (.npy files).
        tokenizer (AutoTokenizer): Tokenizer for processing prompts.
        max_length (int): Maximum sequence length for tokenization.
        logger (logging.Logger, optional): Logger for info and warnings.

    Returns:
        Dataset: Hugging Face Dataset with input_ids, attention_mask, labels, and teacher_logits.
    """
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)

    # Load prompts and tokenize
    prompt_dataset = TextPromptDataset(prompt_file, transform=None)
    prompts = prompt_dataset.prompts

    if len(prompts) == 0:
        raise ValueError("No valid prompts in dataset")
    if len(prompts) < 4:
        logger.warning(f"Dataset size ({len(prompts)}) is smaller than typical batch size (4)")

    tokenized = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Load teacher logits
    logits_dir = Path(logits_dir)
    logit_files = sorted(logits_dir.glob("logits_batch*.npy"))
    if not logit_files:
        raise FileNotFoundError(f"No logit files found in {logits_dir}")

    teacher_logits = []
    expected_vocab_size = tokenizer.vocab_size
    for logit_file in logit_files:
        logits = np.load(logit_file)
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        if logits_tensor.size(1) != max_length or logits_tensor.size(2) != expected_vocab_size:
            logger.error(f"Invalid logits shape in {logit_file}: {logits_tensor.shape}, expected [batch, {max_length}, {expected_vocab_size}]")
            raise ValueError(f"Invalid logits shape in {logit_file}")
        teacher_logits.append(logits_tensor)

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
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)

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