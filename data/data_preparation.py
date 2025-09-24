import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase, TensorType
from datasets import Dataset, DatasetDict, load_dataset
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

SPLIT_CONFIG = {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42
}

class TextDataset(TorchDataset):
    def __init__(self, input_path: str, transform: Optional[Callable] = None):
        self.input_path = Path(input_path)
        self.transform = transform
        self.texts = self._load_texts()

    def _load_texts(self) -> List[str]:
        texts = []
        if self.input_path.is_file() and self.input_path.suffix == '.txt':
            with open(self.input_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(texts)} texts from {self.input_path}")
        elif self.input_path.is_dir():
            txt_files = sorted(self.input_path.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(f"No .txt files found in {self.input_path}")
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
            logger.info(f"Loaded {len(texts)} texts from {len(txt_files)} files in {self.input_path}")
        else:
            raise ValueError(f"Invalid input path: {self.input_path}")
        if not texts:
            raise ValueError(f"No valid texts found in {self.input_path}")
        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.transform:
            text = self.transform(text)
        return text

def prepare_tokenized_dataset(
    input_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    logger: Optional[logging.Logger] = None,
    model_type: str = "causal_lm",
    masking_probability: float = 0.15
) -> Dataset:
    if logger is None:
        logger = setup_logger('src.data.data_preparation')

    logger.info(f"Preparing dataset from {input_path} for {model_type}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist")
    try:
        prompt_dataset = TextDataset(input_path, transform=None)
        texts = prompt_dataset.texts
    except Exception as e:
        logger.error(f"Failed to load dataset from {input_path}: {str(e)}")
        raise

    if len(texts) == 0:
        raise ValueError("No valid texts in dataset")
    if len(texts) < 4:
        logger.warning(f"Dataset size ({len(texts)}) is smaller than typical batch size (4)")

    try:
        if model_type == "causal_lm":
            tokenized = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=TensorType.PYTORCH,
                add_special_tokens=True
            )
            input_ids = tokenized['input_ids']
            labels = input_ids.clone()
            # Shift labels for next-token prediction
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            labels[tokenized['attention_mask'] == 0] = -100
            logger.info(f"Tokenizer selected for causal_lm with shifted labels")
        elif model_type == "masked_lm":
            tokenized = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=TensorType.PYTORCH,
                return_special_tokens_mask=True
            )
            labels = tokenized['input_ids'].clone()
            special_tokens_mask = tokenized['special_tokens_mask'].bool()
            probability_matrix = torch.full(labels.shape, masking_probability)
            probability_matrix.masked_fill_(special_tokens_mask, 0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100
            labels[masked_indices] = tokenized['input_ids'][masked_indices]
            logger.info(f"Tokenizer selected for masked_lm with {masking_probability*100}% masking")
        elif model_type == "seq2seq_lm":
            tokenized = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=TensorType.PYTORCH
            )
            labels = tokenized['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            logger.info(f"Tokenizer selected for seq2seq_lm")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose from ['causal_lm', 'masked_lm', 'seq2seq_lm']")

        if 'input_ids' not in tokenized or 'attention_mask' not in tokenized:
            logger.error("Tokenization did not produce input_ids or attention_mask")
            raise ValueError("Invalid tokenizer output")
        dataset_dict = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
        if model_type == "masked_lm" and 'special_tokens_mask' in tokenized:
            dataset_dict['special_tokens_mask'] = tokenized['special_tokens_mask']
        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    except tokenizer.TokenizerError as e:
        logger.error(f"Tokenizer error: {str(e)}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during tokenization (e.g., out of memory): {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during tokenization: {str(e)}")
        raise

def data_collator(
    features: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
    model_type: str = "causal_lm"
) -> Dict[str, torch.Tensor]:
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

def prepare_dataset_dict(
    input_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    model_type: str = "causal_lm",
    split_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> DatasetDict:
    """
    Prepare and split into train/validation/test sets using ratios and seed in SPLIT_CONFIG.
    Always returns reproducible splits given the same SPLIT_CONFIG and input.
    """
    if logger is None:
        logger = setup_logger('src.data.data_preparation')
    cfg = split_config or SPLIT_CONFIG

    logger.info(f"Splitting dataset using config: {cfg}")
    # Tokenize all texts first
    full_dataset = prepare_tokenized_dataset(
        input_path,
        tokenizer,
        max_length=max_length,
        logger=logger,
        model_type=model_type
    )
    # Shuffle deterministically
    full_dataset = full_dataset.shuffle(seed=cfg["seed"])
    # First: Train/Test split
    train_test = full_dataset.train_test_split(test_size=cfg["test_ratio"], seed=cfg["seed"])
    train = train_test['train']
    test = train_test['test']
    # Second: Train/Val split within the train set
    train_val = train.train_test_split(test_size=cfg["val_ratio"] / (1.0 - cfg["test_ratio"]), seed=cfg["seed"])
    train = train_val["train"]
    val = train_val["test"]
    split_dict = DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })
    logger.info(f"Split sizes: train={len(split_dict['train'])}, val={len(split_dict['validation'])}, test={len(split_dict['test'])}")
    return split_dict

# Example usage in scripts:
# from src.data.data_preparation import prepare_dataset_dict, SPLIT_CONFIG
# dataset_dict = prepare_dataset_dict("data/all_texts/", tokenizer, max_length=128)
