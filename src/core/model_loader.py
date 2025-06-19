import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ModelLoader:
    def __init__(self, model_name="openai-community/gpt2-medium", adapter_path=None):
        # Configuration for mixed precision
        self.use_fp16 = torch.cuda.is_available()
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Model initialization with distillation-specific parameters
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,  # Required for hidden state distillation
            torch_dtype=self.dtype
        ).eval()  # Always in evaluation mode for logit generation

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()  # Crucial for consistent logit generation
        
        # Tokenizer configuration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Critical for batch processing

    def generate_logits(self, texts, **generation_kwargs):
        """Generate model outputs with hidden states for distillation"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Prevent OOM errors
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states[-1],  # Final transformer layer
            'attention_mask': inputs.attention_mask
        }
