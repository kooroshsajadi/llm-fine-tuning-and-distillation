from pathlib import Path
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
        use_qlora=False,
        device_map="auto",
        max_length=128,
        train_mode=False
    ):
        """
        Initialize model and tokenizer for fine-tuning or inference.

        Args:
            model_name (str): Hugging Face model name or path (e.g., "data/gpt2_small_kd").
            adapter_path (str, optional): Path to LoRA adapter.
            lora_config (LoraConfig, optional): LoRA configuration for fine-tuning.
            use_qlora (bool): Enable QLoRA with 4-bit NF4 quantization (requires GPU and bitsandbytes).
            device_map (str): Device placement strategy ("auto" or "cpu").
            max_length (int): Maximum sequence length for tokenization.
            train_mode (bool): Set model to training mode (for fine-tuning).
        """
        self.model_name = model_name
        self.use_gpu = torch.cuda.is_available() and device_map != "cpu"
        self.max_length = max_length
        self.train_mode = train_mode

        # Device and dtype setup
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.use_fp16 = self.use_gpu and not use_qlora
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        logger.info(f"Using {'GPU' if self.use_gpu else 'CPU'} with dtype {self.dtype}")

        # Quantization config
        bnb_config = None
        if use_qlora:
            if not self.use_gpu:
                logger.warning("QLoRA requires GPU; disabling quantization")
                use_qlora = False
            elif not self._check_bitsandbytes():
                logger.warning("bitsandbytes unavailable; disabling QLoRA")
                use_qlora = False
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                logger.info("QLoRA enabled with 4-bit NF4 quantization")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map if not bnb_config else None,
            output_hidden_states=False,  # Disable to avoid warning
            torch_dtype=self.dtype if not bnb_config else None,
            trust_remote_code=False
        ).to(self.device)

        # Prepare for k-bit training (QLoRA)
        if use_qlora and lora_config:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        # Apply LoRA or adapter
        if lora_config:
            self.model = get_peft_model(self.model, lora_config)
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

    def _check_bitsandbytes(self) -> bool:
        """Check if bitsandbytes is available."""
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False

    def _validate_model_source(self, model_name: str) -> bool:
        """Verify model is GPT-2 variant or local path."""
        return "gpt2" in model_name.lower() or Path(model_name).exists()

    def generate_logits(self, texts):
        """Generate model outputs for inference or distillation."""
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
            'attention_mask': inputs.attention_mask
        }