import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
from transformers import PreTrainedModel, TrainerCallback

def sha256sum(filename: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def save_adapter(
    model: PreTrainedModel,
    base_model: str,
    lora_config: Any,
    output_dir: str
) -> None:
    """
    Save LoRA adapter with metadata.

    Args:
        model: The model with LoRA adapters.
        base_model: Name of the base model.
        lora_config: LoRA configuration.
        output_dir: Directory to save the adapter.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    metadata = {
        "base_model": base_model,
        "lora_config": lora_config.to_dict() if hasattr(lora_config, 'to_dict') else vars(lora_config)
    }
    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def generate_checksum(output_dir: str) -> None:
    """
    Generate SHA-256 checksums for all non-checksum files in output_dir.

    Args:
        output_dir: Directory containing files to checksum.
    """
    output_path = Path(output_dir)
    for file_path in output_path.iterdir():
        if file_path.is_file() and not file_path.suffix == '.sha256':
            checksum = sha256sum(file_path)
            with open(file_path.with_suffix(file_path.suffix + '.sha256'), 'w') as f:
                f.write(checksum)

def batch_iterable(iterable: Iterable, batch_size: int) -> List:
    """
    Yield batches of size batch_size from an iterable.

    Args:
        iterable (Iterable): The input iterable to batch.
        batch_size (int): Number of items per batch.

    Yields:
        List: A batch of items from the iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class LogEpochLossCallback(TrainerCallback):
    """
    Callback to log the final training loss after each epoch.

    Args:
        logger (logging.Logger, optional): Logger for printing loss values.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Log the final training loss at the end of each epoch.

        Args:
            args: TrainingArguments.
            state: TrainerState containing log history.
            control: TrainerControl.
            **kwargs: Additional arguments.
        """
        if state.log_history:
            for log in reversed(state.log_history):
                if 'loss' in log:
                    self.logger.info(f"Epoch {state.epoch:.0f}: Final Loss = {log['loss']:.4f}")
                    break