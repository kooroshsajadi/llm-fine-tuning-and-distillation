import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from peft import PeftModel
from transformers import PreTrainedModel, TrainerCallback
import json

def sha256sum(filename: Path) -> str:
    """
    Compute SHA-256 checksum for a file.

    Args:
        filename (Path): Path to the file.

    Returns:
        str: Hexadecimal SHA-256 checksum.
    """
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_adapter(model: PeftModel, base_model: str, lora_config, output_path: str) -> None:
    """
    Save LoRA adapter with metadata.

    Args:
        model (PeftModel): Fine-tuned model with LoRA adapters.
        base_model (str): Name of the base model (e.g., "openai-community/gpt2-medium").
        lora_config: LoRA configuration object.
        output_path (str): Directory to save adapter files.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    model.save_pretrained(output_path)
    metadata = {
        "base_model": base_model,
        "lora_rank": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "architecture": "GPT-2"
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def generate_checksum(output_dir: str) -> None:
    """
    Generate SHA-256 checksums for all files in a directory.

    Args:
        output_dir (str): Directory containing files to checksum.
    """
    output_path = Path(output_dir)
    for file in output_path.glob("*"):
        if file.is_file() and not file.suffix.endswith(".sha256"):
            checksum = sha256sum(file)
            with open(file.with_suffix(file.suffix + ".sha256"), "w") as f:
                f.write(checksum)

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
            # Get the most recent log entry with a loss value
            for log in reversed(state.log_history):
                if 'loss' in log:
                    self.logger.info(f"Epoch {state.epoch:.0f}: Final Loss = {log['loss']:.4f}")
                    break