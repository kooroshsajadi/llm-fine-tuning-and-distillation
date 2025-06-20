import os
import logging
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer
from datasets import Dataset

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

def data_collator(
    features: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> Dict[str, torch.Tensor]:
    """
    Collate dataset features into a batch for training.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries with input_ids, attention_mask, and labels.
        logger (logging.Logger, optional): Logger for debugging batch shapes.

    Returns:
        Dict[str, torch.Tensor]: Batched tensors for input_ids, attention_mask, and labels.
    """
    if logger is None:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)

    if not all('input_ids' in f and 'attention_mask' in f and 'labels' in f for f in features):
        raise ValueError("Invalid dataset features")

    batch = {
        'input_ids': torch.stack([torch.tensor(f['input_ids'], dtype=torch.long) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask'], dtype=torch.long) for f in features]),
        'labels': torch.stack([torch.tensor(f['labels'], dtype=torch.long) for f in features])
    }
    logger.debug(f"Batch shapes: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}, labels={batch['labels'].shape}")
    return batch