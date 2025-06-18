#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from more_itertools import chunked
from tqdm import tqdm
from src.core.model_loader import ModelLoader
from src.data.data_preparation import TextPromptDataset
import numpy as np
import hashlib

def save_logits(outputs: dict, output_dir: Path, batch_idx: int):
    """
    Secure output handler with checksum and memory-mapped storage.
    Stores logits and metadata for each batch.
    """
    logits = outputs['logits'].cpu().numpy()
    hidden_states = outputs['hidden_states'].cpu().numpy()
    attention_mask = outputs['attention_mask'].cpu().numpy()

    # Save logits using numpy's memory-mapped format
    logits_path = output_dir / f"logits_batch{batch_idx:04d}.npy"
    np.save(logits_path, logits)

    # Save hidden states
    hs_path = output_dir / f"hidden_states_batch{batch_idx:04d}.npy"
    np.save(hs_path, hidden_states)

    # Save attention mask
    am_path = output_dir / f"attention_mask_batch{batch_idx:04d}.npy"
    np.save(am_path, attention_mask)

    # Generate and save checksums for integrity verification
    for file_path in [logits_path, hs_path, am_path]:
        checksum = sha256sum(file_path)
        with open(str(file_path) + ".sha256", "w") as f:
            f.write(checksum)

def sha256sum(filename: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h  = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    teacher_model = "openai-community/gpt2-medium"
    batch_size = 32  # Adjust based on available GPU memory

    try:
        loader = ModelLoader(model_name=teacher_model)
        input_texts = TextPromptDataset("data/synthetic/prompts_v1.txt")

        # Create output directory if it doesn't exist
        output_dir = Path("data/synthetic/v1_t0.7")
        output_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(chunked(input_texts, batch_size))):
            outputs = loader.generate_logits(batch)
            save_logits(outputs, output_dir, batch_idx)

    except Exception as e:
        logging.error(f"Logit generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
