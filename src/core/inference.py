#!/usr/bin/env python3
import logging
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, tokenizer_name: str = "openai-community/gpt2") -> tuple:
    """
    Load the trained model and tokenizer.

    Args:
        model_path (str): Path to the saved model directory (e.g., adapters/gpt2_small_kd).
        tokenizer_name (str): Hugging Face tokenizer name (default: openai-community/gpt2).

    Returns:
        tuple: (model, tokenizer)

    Raises:
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If model or tokenizer loading fails.
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        logger.error(f"Model directory does not exist: {model_path}")
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side='right',
            trust_remote_code=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer loaded: {tokenizer_name}")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU-based
            device_map={"": device},
            trust_remote_code=False
        )
        model.eval()
        logger.info(f"Model loaded from {model_path} on {device}")

        # Ensure model config aligns with tokenizer
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.debug(f"Set model.config.pad_token_id to {model.config.pad_token_id}")

        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {str(e)}")
        raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")

def generate_text(model, tokenizer, prompt: str, max_length: int = 50, **generation_kwargs) -> str:
    """
    Generate text using the model for a given prompt.

    Args:
        model: Loaded model for inference.
        tokenizer: Loaded tokenizer.
        prompt (str): Input prompt for generation.
        max_length (int): Maximum length of generated text (default: 50).
        **generation_kwargs: Additional generation parameters (e.g., temperature, top_p).

    Returns:
        str: Generated text.

    Raises:
        RuntimeError: If generation fails.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Generated text for prompt '{prompt}': {generated_text}")
        return generated_text
    except Exception as e:
        logger.error(f"Generation failed for prompt '{prompt}': {str(e)}")
        raise RuntimeError(f"Generation failed: {str(e)}")

def evaluate_model(model, tokenizer, prompts: list, max_length: int = 50, **generation_kwargs):
    """
    Evaluate the model by generating text for a list of prompts.

    Args:
        model: Loaded model for inference.
        tokenizer: Loaded tokenizer.
        prompts (list): List of input prompts for evaluation.
        max_length (int): Maximum length of generated text (default: 50).
        **generation_kwargs: Additional generation parameters.
    """
    logger.info("Starting model evaluation...")
    for prompt in prompts:
        try:
            generated_text = generate_text(model, tokenizer, prompt, max_length, **generation_kwargs)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}\n")
        except RuntimeError as e:
            logger.error(f"Skipping prompt '{prompt}' due to error: {str(e)}")

def main():
    """
    Main function to evaluate a trained model via text generation.

    Example:
        python -m src.scripts.inference --model_path adapters/gpt2_small_kd --prompt "Hello, world!"
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model via text generation.")
    parser.add_argument("--model_path", default="adapters/gpt2_small_kd",
                        help="Path to the saved model directory")
    parser.add_argument("--tokenizer_name", default="openai-community/gpt2",
                        help="Hugging Face tokenizer name")
    parser.add_argument("--prompt", default=None,
                        help="Single prompt for generation (if not provided, uses default prompts)")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling probability")
    args = parser.parse_args()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_name)

        # Define prompts
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = [
                "The sun sets slowly behind the mountain, casting a warm glow over the valley.",
                "In a world where AI governs all decisions, one human discovers a hidden truth.",
                "What is the meaning of life in a universe without boundaries?"
            ]
            logger.info("Using default prompts for evaluation")

        # Generation parameters
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True
        }

        # Evaluate model
        evaluate_model(model, tokenizer, prompts, args.max_length, **generation_kwargs)

    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during inference: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()