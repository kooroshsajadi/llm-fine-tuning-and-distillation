import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(
        self,
        model_name="openai-community/gpt2-medium",
        adapter_path=None,
        lora_config=None,
        use_4bit=False,
        device_map="auto",
        max_length=128,
        train_mode=False
    ):
        """
        Initialize model and tokenizer for fine-tuning or KD.

        Args:
            model_name (str): Hugging Face model name (e.g., "openai-community/gpt2").
            adapter_path (str, optional): Path to LoRA adapter (None for base model).
            lora_config (LoraConfig, optional): LoRA configuration for fine-tuning.
            use_4bit (bool): Enable 4-bit quantization (requires GPU and bitsandbytes).
            device_map (str): Device placement strategy ("auto" or specific device).
            max_length (int): Maximum sequence length for tokenization.
            train_mode (bool): If True, set model to training mode (for fine-tuning or student KD).
        """
        self.model_name = model_name  # Store model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = torch.cuda.is_available() and not use_4bit
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.max_length = max_length
        self.train_mode = train_mode

        # Validate model source (specific to GPT-2 for fine-tuning)
        if lora_config and not self._validate_model_source(model_name):
            raise ValueError(f"Untrusted model source: {model_name}")

        # Configure quantization
        bnb_config = None
        if use_4bit:
            if not torch.cuda.is_available():
                logger.warning("No GPU detected; disabling 4-bit quantization")
                use_4bit = False
            else:
                try:
                    import bitsandbytes
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    logger.info("4-bit quantization enabled")
                except ImportError:
                    logger.warning("bitsandbytes unavailable - using FP32 precision")
                    use_4bit = False

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map if use_4bit else None,
            output_hidden_states=True,  # For KD
            torch_dtype=self.dtype if not use_4bit else None,
            trust_remote_code=False
        ).to(self.device)

        # Log model configuration
        logger.debug(f"Model config: {self.model.config}")

        # Apply quantization preparation for LoRA
        if use_4bit and lora_config:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

        # Apply LoRA if provided
        if lora_config:
            self.model = get_peft_model(self.model, lora_config)
            # Freeze non-LoRA parameters
            for name, param in self.model.named_parameters():
                param.requires_grad = "lora" in name.lower()
            self.model.print_trainable_parameters()
            trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
            logger.info(f"Trainable parameters: {trainable_params}")
            if not trainable_params:
                raise RuntimeError("No trainable parameters detected after LoRA injection")
        elif adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path).to(self.device)

        # Set mode
        self.model.train(train_mode)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='right',
            trust_remote_code=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        logger.info(f"Set model.config.pad_token_id to {self.model.config.pad_token_id}")

        # Clear any invalid config attributes
        if hasattr(self.model.config, 'loss_type') and self.model.config.loss_type is None:
            self.model.config.loss_type = 'ForCausalLMLoss'
            logger.info("Set model.config.loss_type to 'ForCausalLMLoss'")

    def _validate_model_source(self, model_name: str) -> bool:
        """Verify model is GPT-2 variant."""
        return "gpt2" in model_name.lower()

    def generate_logits(self, texts):
        """Generate model outputs with hidden states for distillation."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad() if not self.model.training else torch.enable_grad():
            outputs = self.model(**inputs)

        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states[-1],
            'attention_mask': inputs.attention_mask
        }