from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig
from src.utils.logging_utils import setup_logger
from typing import Type

logger = setup_logger('src.core.model_loader')

class ModelLoader:
    MODEL_TYPE_MAPPING = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM
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
        self.use_cuda = torch.cuda.is_available()
        self.use_xpu = torch.xpu.is_available()
        self.use_gpu = self.use_cuda or self.use_xpu  # Maintain for backward compatibility
        self.max_length = max_length
        self.train_mode = train_mode
        self.use_qlora = use_qlora

        self._validate_inputs(max_length=self.max_length, model_name=model_name)

        self._select_device(device_map=device_map)

        cuda_capability = self._select_data_type(use_qlora=self.use_qlora)

        # Configure QLoRA and BitStandBytes.
        bnb_config = None
        if self.use_qlora:
            if not self.use_cuda:
                logger.warning("QLoRA requires CUDA GPU; disabling quantization")
                self.use_qlora = False
            elif self.use_xpu:
                logger.warning("QLoRA not supported on Intel XPU; disabling quantization")
                self.use_qlora = False
            elif not self._check_bitsandbytes():
                logger.warning("bitsandbytes unavailable; disabling QLoRA")
                self.use_qlora = False
            else:
                # Dynamically select compute dtype for QLoRA
                bnb_compute_dtype = (
                    torch.bfloat16 if cuda_capability[0] >= 8 else
                    torch.float16 if cuda_capability[0] >= 7 else
                    torch.float32
                )
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=bnb_compute_dtype,
                    bnb_4bit_use_double_quant=True
                )
                logger.info(f"QLoRA enabled with 4-bit NF4 quantization, compute dtype: {bnb_compute_dtype}")

        try:
            if self.model_type not in self.MODEL_TYPE_MAPPING:
                raise ValueError(
                    f"Unsupported model_type: {self.model_type}. "
                    f"Choose from {list(self.MODEL_TYPE_MAPPING.keys())}"
                )
            model_class: Type = self.MODEL_TYPE_MAPPING[self.model_type]
            if self.model_type == "causal_lm" and model_class != AutoModelForCausalLM:
                logger.warning(f"Expected AutoModelForCausalLM for causal_lm, got {model_class.__name__}")

            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": device_map if not bnb_config else "auto",  # Let bitsandbytes handle placement
                "output_hidden_states": False,
                "torch_dtype": self.dtype if not bnb_config else None,
                "trust_remote_code": False
            }

            self.model = model_class.from_pretrained(model_name, **model_kwargs)
            if not bnb_config:  # Only move to device if not quantized
                self.model = self.model.to(self.device)

            # # Set loss_type based on model_type
            # loss_type_map = {
            #     "causal_lm": "ForCausalLMLoss",
            #     "seq2seq_lm": "ForSeq2SeqLMLoss",  # Placeholder; adjust based on actual loss
            #     "masked_lm": "ForMaskedLMLoss"     # Placeholder; adjust based on actual loss
            # }
            # if self.model_type in loss_type_map:
            #     self.model.config.loss_type = loss_type_map[self.model_type]
            #     logger.info(f"Set model.config.loss_type to {self.model.config.loss_type}")
            # else:
            #     logger.warning(f"No loss_type defined for {self.model_type}; keeping default")
            
        except (ValueError, OSError) as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_name}: {str(e)}")
            raise

        if self.use_xpu:
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16)  # Explicitly use BF16
                logger.info("Applied IPEX optimization for Intel XPU with BF16")
            except ImportError:
                logger.warning("intel-extension-for-pytorch not installed; skipping IPEX optimization")
            except Exception as e:
                logger.error(f"IPEX optimization failed: {str(e)}")
                raise

        if self.use_qlora and lora_config:
            if not isinstance(lora_config, LoraConfig):
                raise TypeError("lora_config must be a peft.LoraConfig object")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True  # Could be a parameter
            )
            logger.info("Prepared model for QLoRA training with gradient checkpointing")

        if lora_config:
            if not isinstance(lora_config, LoraConfig):
                raise TypeError("lora_config must be a peft.LoraConfig object")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
            logger.info(f"Trainable LoRA parameters: {trainable_params}")
            if not trainable_params:
                raise RuntimeError("No trainable parameters detected after LoRA injection")
        elif adapter_path:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                if not self.use_qlora:  # Only move if not quantized
                    self.model = self.model.to(self.device)
                logger.info(f"Loaded PEFT adapter from {adapter_path}")
            except Exception as e:
                logger.error(f"Failed to load adapter from {adapter_path}: {str(e)}")
                raise
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='right' if self.model_type == "causal_lm" else 'left',
                trust_remote_code=False
            )
            if not self.tokenizer.pad_token:
                if self.model_type == "masked_lm":
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added [PAD] as pad_token for tokenizer")
                elif self.model_type == "causal_lm" and self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
                else:
                    logger.warning("No eos_token available; setting pad_token to [PAD]")
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.tokenizer.pad_token = '[PAD]'
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id >= self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized model embeddings to {len(self.tokenizer)} to accommodate pad_token")
            logger.info(f"Set model.config.pad_token_id to {self.tokenizer.pad_token_id}")
        except (ValueError, OSError) as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading tokenizer for {model_name}: {str(e)}")
            raise

        self.model.train(self.train_mode)

    def _validate_inputs(self, max_length: int, model_name: str) -> None:
                # Max length validation.
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        
        # Model name validation.
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        
        # Model type validation.
        if self.model_type not in self.MODEL_TYPE_MAPPING:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Supported types: {list(self.MODEL_TYPE_MAPPING.keys())}"
            )
    
    def _select_device(self, device_map: str) -> None:
        if device_map == "auto":
            self.device = torch.device(
                "xpu" if self.use_xpu else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device_map)
        logger.info(f"Using device: {self.device}")
    
    def _select_data_type(self, use_qlora) -> tuple[int, int]:
        cuda_capability = (0, 0)
        if self.use_cuda:
            cuda_capability = torch.cuda.get_device_capability()
            logger.debug(f"CUDA capability: {cuda_capability}")

        # Enable FP16 only for CUDA with Tensor Cores (SM 7.0+)
        self.use_fp16 = (
            self.use_cuda and not use_qlora and not self.use_xpu
            and cuda_capability[0] >= 7
        )

        # Enable BF16 for CUDA (SM 8.0+) or XPU
        self.use_bf16 = (
            (self.use_cuda and cuda_capability[0] >= 8 and not self.use_fp16)
            or self.use_xpu
        )

        # Data type selection.
        self.dtype = (
            torch.float16 if self.use_fp16 else
            torch.bfloat16 if self.use_bf16 else
            torch.float32
        )
        logger.info(f"Using dtype: {self.dtype}")
        return cuda_capability

    def _check_bitsandbytes(self) -> bool:
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False

    def _validate_model_source(self, model_name: str) -> bool:
        supported_models = ["gpt2", "bert-base-italian", "electra-base-italian", "t5"]
        return any(model in model_name.lower() for model in supported_models) or Path(model_name).exists()

    def _format_count(self, n: int) -> str:
        # Human-friendly: 1.23M, 456.7M, 13.4B
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n/1_000:.2f}K"
        return str(n)

    def _dtype_num_bytes(self, dtype: torch.dtype) -> int:
        if dtype == torch.float32:
            return 4
        if dtype in (torch.float16, torch.bfloat16):
            return 2
        if dtype in (torch.int8,):
            return 1
        # Conservative default
        return 4

    def _estimate_param_memory_bytes(self, model: torch.nn.Module) -> int:
        # Parameter memory footprint only (excludes optimizer states, activation buffers)
        total_bytes = 0
        for p in model.parameters():
            # On quantized models (bnb), param.dtype may not reflect storage exactly,
            # but it’s the best cheap proxy without bnb internals.
            total_bytes += p.numel() * self._dtype_num_bytes(p.dtype)
        return total_bytes

    def _log_model_profile(self, title: str = "Loaded model profile") -> None:
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        # Estimate memory (parameters only; excludes gradients/optimizer/activations)
        approx_param_mem = self._estimate_param_memory_bytes(self.model)
        # Convert to MB/GB
        mem_mb = approx_param_mem / (1024**2)
        mem_gb = approx_param_mem / (1024**3)

        # PEFT/adapters info (best-effort)
        peft_attached = False
        try:
            from peft import PeftModel
            peft_attached = isinstance(self.model, PeftModel) or any("lora" in n.lower() for n, _ in self.model.named_parameters())
        except Exception:
            pass

        # Quantization hint
        quantized_hint = "Yes (bnb 4-bit)" if any(getattr(m, "weight_bit_width", None) in (4, "4") for m in self.model.modules()) else "Unknown/No"
        # Fallback heuristic: check for bitsandbytes modules by name
        if quantized_hint == "Unknown/No":
            if any("bnb" in type(m).__name__.lower() or "bitsandbytes" in str(type(m)).lower() for m in self.model.modules()):
                quantized_hint = "Likely (bitsandbytes detected)"

        logger.info(
            f"\n{title}\n"
            f"- Model name: {self.model_name}\n"
            f"- Model type: {self.model_type}\n"
            f"- Device: {self.device}\n"
            f"- Dtype: {self.dtype}\n"
            f"- Quantized: {quantized_hint}\n"
            f"- PEFT/Adapters attached: {peft_attached}\n"
            f"- Total params: {self._format_count(total_params)} ({total_params:,})\n"
            f"- Trainable params: {self._format_count(trainable_params)} ({trainable_params:,})\n"
            f"- Non-trainable params: {self._format_count(non_trainable_params)} ({non_trainable_params:,})\n"
            f"- Approx parameter memory: {mem_mb:.2f}MB ({mem_gb:.3f}GB) [parameters only]\n"
        )

    def get_model_profile(self) -> dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        approx_param_mem = self._estimate_param_memory_bytes(self.model)
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "approx_param_mem_bytes": approx_param_mem,
        }

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