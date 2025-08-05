from pathlib import Path
import torch
import logging
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class ModelLoader:
    def __init__(
        self,
        model_name: str = "openai-community/gpt2-medium",
        model_type: str = "causal_lm",
        adapter_path: str | None = None,
        lora_config=None,
        use_qlora: bool = False,
        device_map: str = "auto",
        max_length: int = 128,
        train_mode: bool = False
    ):
        """
        Initialize model and tokenizer for fine-tuning or inference.

        Args:
            model_name (str): Hugging Face model name or path.
            model_type (str): Model type ("causal_lm" or "masked_lm").
            adapter_path (str, optional): Path to LoRA adapter.
            lora_config (LoraConfig, optional): LoRA configuration.
            use_qlora (bool): Enable QLoRA (requires CUDA GPU and bitsandbytes).
            device_map (str): Device placement strategy ("auto", "cpu", "xpu").
            max_length (int): Maximum sequence length.
            train_mode (bool): Set model to training mode.
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.use_gpu = torch.cuda.is_available() or torch.xpu.is_available()
        self.use_xpu = torch.xpu.is_available()
        self.max_length = max_length
        self.train_mode = train_mode

        # Device and dtype setup
        if device_map == "auto":
            self.device = torch.device(
                "xpu" if self.use_xpu else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device_map)
        self.use_fp16 = self.use_gpu and not use_qlora and not self.use_xpu
        self.dtype = torch.float16 if self.use_fp16 else torch.bfloat16 if self.use_xpu else torch.float32
        logger.info(f"Using {'XPU' if self.use_xpu else 'CUDA' if torch.cuda.is_available() else 'CPU'} with dtype {self.dtype}")

        # Quantization config
        bnb_config = None
        if use_qlora:
            if not torch.cuda.is_available():
                logger.warning("QLoRA requires CUDA GPU; disabling quantization")
                use_qlora = False
            elif self.use_xpu:
                logger.warning("QLoRA not supported on Intel ARC; disabling quantization")
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
        try:
            if self.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map=device_map if not bnb_config else None,
                    output_hidden_states=False,
                    torch_dtype=self.dtype if not bnb_config else None,
                    trust_remote_code=False
                ).to(self.device)
            elif self.model_type == "masked_lm":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map=device_map if not bnb_config else None,
                    output_hidden_states=False,
                    torch_dtype=self.dtype if not bnb_config else None,
                    trust_remote_code=False
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

        # Optimize for Intel ARC
        if self.use_xpu:
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                logger.info("Applied IPEX optimization for Intel ARC")
            except ImportError:
                logger.warning("intel-extension-for-pytorch not installed; skipping IPEX optimization")

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
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path).to(self.device)
            except Exception as e:
                logger.error(f"Failed to load adapter from {adapter_path}: {str(e)}")
                raise

        # Set mode
        self.model.train(train_mode)

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='right' if self.model_type == "causal_lm" else 'left',
                trust_remote_code=False
            )
            if self.model_type == "causal_lm":
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            elif self.model_type == "masked_lm":
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.info(f"Set model.config.pad_token_id to {self.model.config.pad_token_id}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
            raise

    def _check_bitsandbytes(self) -> bool:
        """Check if bitsandbytes is available."""
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False

    def _validate_model_source(self, model_name: str) -> bool:
        """Verify model is a supported variant or local path."""
        supported_models = ["gpt2", "bert-base-italian", "electra-base-italian"]
        return any(model in model_name.lower() for model in supported_models) or Path(model_name).exists()

    def generate_logits(self, texts: list[str]) -> dict:
        """Generate model outputs for inference or distillation."""
        try:
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
        except Exception as e:
            logger.error(f"Failed to generate logits: {str(e)}")
            raise