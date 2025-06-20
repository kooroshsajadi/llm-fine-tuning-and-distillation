#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from src.core.model_loader import ModelLoader
from src.data.data_preparation import TextPromptDataset
from src.utils import generate_checksum, batch_iterable
import numpy as np

def save_logits(outputs: dict, output_dir: Path, batch_idx: int):
    """
    Save logits, hidden states, and attention masks with checksums.

    Args:
        outputs (dict): Model outputs containing logits, hidden_states, and attention_mask.
        output_dir (Path): Directory to save output files.
        batch_idx (int): Batch index for file naming.
    """
    # Save logits, hidden states, and attention mask as NumPy arrays
    logits_path = output_dir / f"logits_batch{batch_idx:04d}.npy"
    hidden_states_path = output_dir / f"hidden_states_batch{batch_idx:04d}.npy"
    attention_mask_path = output_dir / f"attention_mask_batch{batch_idx:04d}.npy"

    np.save(logits_path, outputs['logits'].cpu().numpy())
    np.save(hidden_states_path, outputs['hidden_states'].cpu().numpy())
    np.save(attention_mask_path, outputs['attention_mask'].cpu().numpy())

    # Generate checksums for saved files
    generate_checksum(output_dir)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    teacher_model = "openai-community/gpt2-medium"
    adapter_path = "data/adapters/final_adapters/gpt2_lora"  # Use fine-tuned LoRA adapter
    batch_size = 4  # Aligned with fine-tuning effective batch size

    try:
        # Initialize ModelLoader with LoRA adapter
        loader = ModelLoader(
            model_name=teacher_model,
            adapter_path=adapter_path,
            max_length=128,  # Match fine-tuning
            train_mode=False,
            use_4bit=False  # Disable quantization on CPU
        )
        input_texts = TextPromptDataset("data/synthetic/prompts_v1.txt")

        # Create output directory
        output_dir = Path("data/synthetic/v1_t0.7")
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Process dataset in batches
        for batch_idx, batch in enumerate(tqdm(batch_iterable(input_texts.prompts, batch_size), desc="Generating logits")):
            outputs = loader.generate_logits(batch)
            save_logits(outputs, output_dir, batch_idx)

        logger.info(f"Logit generation completed. Outputs saved to {output_dir}")

    except Exception as e:
        logger.error(f"Logit generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()