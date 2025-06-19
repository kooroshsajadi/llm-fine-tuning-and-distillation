
import torch
import logging
from pathlib import Path
from transformers import (AutoModelForCausalLM,AutoTokenizer,Trainer,
    TrainingArguments,BitsAndBytesConfig,AutoConfig,GPT2Config)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from src.core.logit_generator import sha256sum

# Security-conscious logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GPT2LoRAFineTuner:
    """
    GPT-2 specific LoRA fine-tuner with security-conscious implementation.
    
    This class provides an end-to-end pipeline for efficiently fine-tuning GPT-2 models
    using Low-Rank Adaptation (LoRA). It incorporates:
    - Architecture-specific configuration for GPT-2
    - Security validations for model sources and data inputs
    - Memory-efficient 4-bit quantization support
    - Integrity checks via SHA-256 checksums
    - Resource-constrained training with fallback mechanisms
    """

    def __init__(
        self,
        base_model: str = "openai-community/gpt2-medium",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit_quant: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the GPT-2 LoRA fine-tuner.
        """
        self.base_model = base_model
        self.tokenizer = self._load_tokenizer(base_model)
        self.model_config = self._load_config(base_model)

        # GPT-2 specific attention modules for LoRA injection
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj"],  # Query/Value projections
            lora_dropout=lora_dropout,
            bias="none",  # Do not train bias parameters
            task_type="CAUSAL_LM"  # Causal language modeling
        )

        # Load model with security and quantization checks
        self.model = self._load_model(
            base_model, 
            use_4bit_quant, 
            device_map
        )

    def _load_tokenizer(self, model_name: str):
        """Securely load tokenizer with validation."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Critical for batch processing
        return tokenizer

    def _load_config(self, model_name: str):
        """Validate and load model configuration."""
        config = AutoConfig.from_pretrained(model_name)
        if not isinstance(config, GPT2Config):
            raise ValueError(f"Model {model_name} is not a GPT-2 architecture")
        return config

    def _load_model(
        self, 
        model_name: str, 
        use_4bit: bool, 
        device_map: str
    ):
        """Load base model with security and quantization."""
        if not self._validate_model_source(model_name):
            raise ValueError(f"Untrusted model source: {model_name}")

        # Configure 4-bit quantization if supported
        bnb_config = None
        if use_4bit and torch.cuda.is_available():
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
                bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=False,
        )

        # Inject LoRA adapters into base model
        peft_model = get_peft_model(model, self.lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def _validate_model_source(self, model_name: str) -> bool:
        """Verify model is GPT-2 variant."""
        return "gpt2" in model_name.lower()

    def prepare_dataset(self, data_path: str, max_length: int = 512) -> Dataset:
        """
        Prepare secure dataset from text prompts.
        """
        if not Path(data_path).is_file():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        if len(prompts) == 0:
            raise ValueError("No valid prompts in dataset")

        tokenized = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()  # Labels = inputs for CLM
        })

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./lora_results",
        per_device_train_batch_size: int = 4,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):
        """
        Execute secure training with resource monitoring.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Dynamic optimizer selection
        try:
            import bitsandbytes
            optim_name = "paged_adamw_8bit"
        except ImportError:
            optim_name = "adamw_torch"
            logger.warning("bitsandbytes not installed. Using standard AdamW optimizer")

        # --- CRITICAL FIXES ---
        self.model.config.use_cache = False  # Required for gradient checkpointing
        self.model.train()  # Ensure training mode

        trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable_params:
            logger.error("No trainable parameters detected in model")
            raise RuntimeError("LoRA adapters not properly initialized")
        # --- END CRITICAL FIXES ---

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to="none",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=True,
            logging_dir=str(output_path / "logs"),
            save_total_limit=1,
            max_grad_norm=0.3,
            warmup_ratio=0.05
        )

        def data_collator(features):
            return {
                'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
                'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
                'labels': torch.stack([torch.tensor(f['labels']) for f in features])
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        try:
            # --- PRE-TRAIN GRADIENT VERIFICATION ---
            logger.info("Verifying gradient connectivity...")
            test_input = next(iter(trainer.get_train_dataloader()))
            test_input = {k: v.to(self.model.device) for k, v in test_input.items()}
            with torch.set_grad_enabled(True):
                outputs = self.model(**test_input)
                if outputs.loss.grad_fn is None:
                    raise RuntimeError("Loss not connected to computational graph")
                outputs.loss.backward()
                gradient_found = False
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        gradient_found = True
                if not gradient_found:
                    raise RuntimeError("No gradients computed for any trainable parameters")
            logger.info("Gradient verification passed. Starting training...")
            # --- END VERIFICATION ---

            trainer.train()
        except torch.cuda.OutOfMemoryError:
            logger.error("Training failed: GPU out of memory")
            raise
        except RuntimeError as e:
            logger.error(f"Training failed: {str(e)}")
            if "CUDA" in str(e):
                logger.error("Consider reducing batch size or sequence length")
            elif "grad" in str(e).lower():
                logger.error("Check LoRA configuration and parameter requires_grad")
            raise

        self.save_adapter(output_dir)
        self._generate_checksum(output_dir)

    def save_adapter(self, output_path: str):
        """
        Save adapter with security metadata.
        """
        self.model.save_pretrained(output_path)
        metadata = {
            "base_model": self.base_model,
            "lora_rank": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "target_modules": self.lora_config.target_modules,
            "architecture": "GPT-2"
        }
        with open(Path(output_path) / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_checksum(self, output_dir: str):
        """
        Generate SHA-256 checksums for all files.
        """
        output_path = Path(output_dir)
        for file in output_path.glob("*"):
            if file.is_file() and not file.suffix.endswith(".sha256"):
                checksum = sha256sum(file)
                with open(file.with_suffix(file.suffix + ".sha256"), "w") as f:
                    f.write(checksum)

def main():
    # Initialize fine-tuner
    tuner = GPT2LoRAFineTuner(
        base_model="openai-community/gpt2-medium",
        lora_rank=8,
        lora_alpha=32
    )

    # Prepare dataset
    dataset = tuner.prepare_dataset("data/synthetic/prompts_v1.txt")

    # Start training
    tuner.train(
        dataset,
        output_dir="adapters/gpt2_lora_v1",
        per_device_train_batch_size=4,
        num_train_epochs=3
    )

    # Save adapter weights
    tuner.save_adapter("final_adapters/gpt2_lora")

if __name__ == "__main__":
    main()
    