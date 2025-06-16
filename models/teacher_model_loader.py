from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_teacher_model(
    model_name="openai-community/gpt2-medium",
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    device_map="auto",         # Auto-detect GPU if available
    cache_dir="./my_model_cache"
):
    """
    Load a pre-trained teacher model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): Name of the pre-trained model on Hugging Face Hub.
        torch_dtype (torch.dtype): Data type for model weights.
        device_map (str): Device placement strategy.
        cache_dir (str): Custom directory for caching model files.
    
    Returns:
        tuple: (model, tokenizer) - Loaded model and tokenizer objects.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print(f"Successfully loaded teacher model: {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading teacher model {model_name}: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = load_teacher_model()
        prompt = "Hello, I'm a language model,"
        # Encode the prompt to input IDs
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        # Get raw outputs for distillation purposes
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # Available for distillation loss
            print(f"Logits shape: {logits.shape}")  # For debugging
        # Generate text for testing
        generated_outputs = model.generate(
            input_ids,
            max_new_tokens=30,  # Limit new tokens generated
            num_return_sequences=5,  # Generate 5 sequences
            do_sample=True,  # Enable sampling for diversity
            temperature=0.7,  # Control creativity
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id  # Handle padding for open-end generation
        )
        # Decode and print generated sequences
        for i, output in enumerate(generated_outputs):
            generated_text = tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"Sequence {i+1}: {prompt}{generated_text}")
    except Exception as e:
        print(f"Failed to run the model: {str(e)}")
