#!/usr/bin/env python3
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from src.data.data_preparation import prepare_tokenized_dataset

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
        model_path (str): Path to the saved model directory (e.g., data/gpt2_small_kd).
        tokenizer_name (str): Hugging Face tokenizer name (default: openai-community/gpt2).

    Returns:
        tuple: (model, tokenizer)

    Raises:
        FileNotFoundError: If model_path or required files do not exist.
        RuntimeError: If model or tokenizer loading fails.
    """
    model_path = Path(model_path)
    required_files = ['config.json', 'model.safetensors']
    if not model_path.is_dir():
        logger.error(f"Model directory does not exist: {model_path}")
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    
    for file in required_files:
        if not (model_path / file).exists():
            logger.error(f"Required file missing in {model_path}: {file}")
            raise FileNotFoundError(f"Required file missing: {file}")

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
            trust_remote_code=False,
            output_hidden_states=False  # Explicitly disable to avoid warning
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

def compute_perplexity(model, tokenizer, texts: list, max_length: int = 128) -> float:
    """
    Compute perplexity for a list of texts.

    Args:
        model: Loaded model for inference.
        tokenizer: Loaded tokenizer.
        texts (list): List of texts to evaluate.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        float: Average perplexity across texts.
    """
    total_loss = 0.0
    total_count = 0
    model.eval()
    for text in texts:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss
            total_count += 1
            logger.debug(f"Loss for text '{text[:50]}...': {loss:.4f}")
        except Exception as e:
            logger.warning(f"Skipping text '{text[:50]}...' due to error: {str(e)}")
    
    if total_count == 0:
        logger.error("No valid texts for perplexity computation")
        raise RuntimeError("No valid texts for perplexity computation")
    
    avg_loss = total_loss / total_count
    perplexity = np.exp(avg_loss)
    logger.info(f"Average Perplexity: {perplexity:.2f} (Loss: {avg_loss:.4f})")
    return perplexity

def generate_text(model, tokenizer, prompt: str, max_length: int = 50, **generation_kwargs) -> str:
    """
    Generate text using the model for a given prompt.

    Args:
        model: Loaded model for inference.
        tokenizer: Loaded tokenizer.
        prompt (str): Input prompt for generation.
        max_length (int): Maximum length of generated text.
        **generation_kwargs: Additional generation parameters (e.g., temperature, top_p).

    Returns:
        str: Generated text.

    Raises:
        RuntimeError: If generation fails.
    """
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)
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

def evaluate_model(
    model,
    tokenizer,
    dataset,
    max_length: int = 50,
    perplexity_max_length: int = 128,
    **generation_kwargs
):
    """
    Evaluate the model by generating text and computing perplexity.

    Args:
        model: Loaded model for inference.
        tokenizer: Loaded tokenizer.
        dataset: Hugging Face Dataset with input_ids for prompts.
        max_length (int): Maximum length for text generation.
        perplexity_max_length (int): Maximum length for perplexity computation.
        **generation_kwargs: Additional generation parameters.
    """
    logger.info("Starting model evaluation...")

    # Decode prompts from dataset
    prompts = [tokenizer.decode(item['input_ids'], skip_special_tokens=True) for item in dataset]

    # Generate text for prompts
    for prompt in prompts:
        try:
            generated_text = generate_text(model, tokenizer, prompt, max_length, **generation_kwargs)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}\n")
        except RuntimeError as e:
            logger.error(f"Skipping prompt '{prompt}' due to error: {str(e)}")

    # Compute perplexity on prompts
    if prompts:
        try:
            compute_perplexity(model, tokenizer, prompts, perplexity_max_length)
        except RuntimeError as e:
            logger.error(f"Perplexity computation failed: {str(e)}")
    else:
        logger.warning("No prompts available for perplexity computation")

def main():
    """
    Main function to evaluate a trained model via text generation and perplexity.

    Example:
        python -m src.core.inference
    """
    try:
        # Hardcoded parameters
        model_path = "data/gpt2_small_kd"
        test_file = "data/synthetic/prompts_v1.txt"
        tokenizer_name = "openai-community/gpt2"
        max_length = 50
        perplexity_max_length = 128
        generation_kwargs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_name)

        # Load prompts using prepare_tokenized_dataset
        dataset = prepare_tokenized_dataset(
            file_path=test_file,
            tokenizer=tokenizer,
            max_length=perplexity_max_length,
            logger=logger
        )
        logger.info(f"Loaded dataset with {len(dataset)} prompts from {test_file}")

        # Evaluate model
        evaluate_model(
            model,
            tokenizer,
            dataset,
            max_length=max_length,
            perplexity_max_length=perplexity_max_length,
            **generation_kwargs
        )

    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during inference: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()