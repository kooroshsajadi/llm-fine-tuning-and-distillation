import argparse
from pathlib import Path
import logging
from peft import PeftModel
from transformers import AutoTokenizer
from model.model_loader import ModelLoader
import utils.utils as utils
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def merge_adapter(config_path: str):
    """
    Loads a base model, applies the fine-tuned adapter, merges them,
    and saves the full merged model and tokenizer.
    
    Args:
        config_path (str): Path to the YAML config file.
    """
    config = utils.return_config(config_path)
    tuner_config = config.get('fine_tuning', {})
    
    base_model = tuner_config.get('base_model')
    model_type = tuner_config.get('model_type', 'causal_lm')
    use_qlora = tuner_config.get('use_qlora', False)
    device_map = tuner_config.get('device_map', 'auto')
    max_length = tuner_config.get('max_length', 128)
    output_dir = Path(tuner_config.get('output_dir'))
    adapter_path = output_dir / "model"
    tokenizer_path = output_dir / "tokenizer"
    
    if not adapter_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"Adapter or tokenizer not found in {output_dir}. Run fine-tuning first.")
    
    logger.info(f"Loading base model: {base_model}")
    model_loader = ModelLoader(
        model_name=base_model,
        model_type=model_type,
        adapter_path=str(adapter_path),
        use_qlora=use_qlora,
        device_map=device_map,
        max_length=max_length,
        train_mode=False  # Inference mode for merging
    )
    
    model = model_loader.model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))  # Load saved tokenizer
    
    # Ensure it's a PEFT model
    if not isinstance(model, PeftModel):
        raise ValueError("Loaded model is not a PEFT model. Check adapter loading.")
    
    logger.info("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model and tokenizer
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info(f"Merged model and tokenizer saved to {merged_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge fine-tuned adapter into base model.")
    parser.add_argument("--config", type=str, default="configs/fine_tuning/Meta-Llama-3-8B-Instruct.yaml", help="Path to the fine-tuning config YAML file.")
    args = parser.parse_args()
    
    merge_adapter(args.config)