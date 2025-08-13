from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class ModelLoader:
    MODEL_TYPE_MAPPING = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "SEQ_2_SEQ_LM": AutoModelForSeq2SeqLM
    }

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
        self.model_name = model_name
        self.model_type = model_type
        self.use_gpu = torch.cuda.is_available() or torch.xpu.is_available()
        self.use_xpu = torch.xpu.is_available()
        self.max_length = max_length
        self.train_mode = train_mode

        if self.model_type not in self.MODEL_TYPE_MAPPING:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose from {list(self.MODEL_TYPE_MAPPING.keys())}")

        if device_map == "auto":
            self.device = torch.device(
                "xpu" if self.use_xpu else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device_map)
        logger.info(f"Using device: {self.device}")

        self.use_fp16 = self.use_gpu and not use_qlora and not self.use_xpu
        self.dtype = torch.float16 if self.use_fp16 else torch.bfloat16 if self.use_xpu else torch.float32
        logger.info(f"Using dtype: {self.dtype}")

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

        try:
            model_class = self.MODEL_TYPE_MAPPING[self.model_type]
            self.model = model_class.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map if not bnb_config else None,
                output_hidden_states=False,
                torch_dtype=self.dtype if not bnb_config else None,
                trust_remote_code=False
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

        if self.use_xpu:
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                logger.info("Applied IPEX optimization for Intel ARC")
            except ImportError:
                logger.warning("intel-extension-for-pytorch not installed; skipping IPEX optimization")

        if use_qlora and lora_config:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        if lora_config:
            self.model = get_peft_model(self.model, lora_config)
            for name, param in self.model.named_parameters():
                param.requires_grad = "lora" in name.lower()
            self.model.print_trainable_parameters()
            trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
            # logger.info(f"Trainable parameters: {trainable_params}")
            if not trainable_params:
                raise RuntimeError("No trainable parameters detected after LoRA injection")
        elif adapter_path:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path).to(self.device)
            except Exception as e:
                logger.error(f"Failed to load adapter from {adapter_path}: {str(e)}")
                raise

        # Check if adapters are loaded correctly
        trainable_params = [(n, p.requires_grad) for n, p in self.model.named_parameters()]
        print("\n--- Trainable Parameters ---")
        for n, req_grad in trainable_params:
            if req_grad:
                print(f"{n}: requires_grad={req_grad}")
        print(f"Total trainable params: {sum(x[1] for x in trainable_params)}")
        assert any(x[1] for x in trainable_params), "No parameters require grad!"
        
        self.model.train(train_mode)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='right' if self.model_type == "causal_lm" else 'left',
                trust_remote_code=False
            )
            # Ensure pad_token is set
            if not self.tokenizer.pad_token:
                if self.model_type == "masked_lm":
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added [PAD] as pad_token for tokenizer")
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
                    logger.info(f"Set pad_token to {self.tokenizer.pad_token}")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            # Resize model embeddings if pad_token was added
            if self.tokenizer.pad_token_id >= self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized model embeddings to {len(self.tokenizer)} to accommodate pad_token")
            logger.info(f"Set model.config.pad_token_id to {self.model.config.pad_token_id}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
            raise

    def _check_bitsandbytes(self) -> bool:
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False

    def _validate_model_source(self, model_name: str) -> bool:
        supported_models = ["gpt2", "bert-base-italian", "electra-base-italian", "t5"]
        return any(model in model_name.lower() for model in supported_models) or Path(model_name).exists()

    def generate_logits(self, texts: list[str]) -> dict:
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