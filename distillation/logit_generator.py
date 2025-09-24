#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from model.model_loader import ModelLoader
from src.data.data_preparation import TextDataset
from src.utils.utils import generate_checksum, batch_iterable
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def save_logits(outputs: dict, output_dir: Path, batch_idx: int):
    """
    Save logits, hidden states, and attention masks with checksums.

    Args:
        outputs (dict): Model outputs containing logits, hidden_states, and attention_mask.
        output_dir (Path): Directory to save output files.
        batch_idx (int): Batch index for file naming.
    """
    # Save logits and attention mask as NumPy arrays
    logits_path = output_dir / f"logits_batch{batch_idx:04d}.npy"
    attention_mask_path = output_dir / f"attention_mask_batch{batch_idx:04d}.npy"

    np.save(logits_path, outputs['logits'].cpu().numpy())
    np.save(attention_mask_path, outputs['attention_mask'].cpu().numpy())

    # Generate checksums for saved files
    generate_checksum(output_dir)

def main(
    adapter_path: str = "data/lora_fine_tuned",
    output_path: str = "data/synthetic/v1_t0.7",
    model_name: str = "openai-community/gpt2-medium"
):
    """
    Generate logits for prompts using a teacher model with a LoRA adapter.

    Args:
        adapter_path (str): Path to the LoRA adapter directory.
        output_path (str): Directory to save generated logits.
        model_name (str): Hugging Face model name (e.g., "openai-community/gpt2-medium").
    """
    try:
        # Initialize ModelLoader with LoRA adapter
        loader = ModelLoader(
            model_name=model_name,
            adapter_path=adapter_path,
            max_length=128,  # Match fine-tuning and inference
            train_mode=False,
            device_map="cpu",
            use_qlora=False  # Disable quantization for inference
        )
        # Load prompts
        input_texts = TextDataset("data/synthetic/prompts_v1.txt")
        if len(input_texts) == 0:
            raise ValueError("No prompts found in data/synthetic/prompts_v1.txt")

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        logger.info(f"Output directory created: {output_dir}")

        # Process dataset in batches
        batch_size = 4  # Aligned with fine-tuning
        for batch_idx, batch in enumerate(tqdm(
            batch_iterable(input_texts.prompts, batch_size),
            desc="Generating logits",
            total=(len(input_texts.prompts) + batch_size - 1) // batch_size
        )):
            outputs = loader.generate_logits(batch)
            save_logits(outputs, output_dir, batch_idx)

        logger.info(f"Logit generation completed. Outputs saved to {output_dir}")

    except Exception as e:
        logger.error(f"Logit generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()