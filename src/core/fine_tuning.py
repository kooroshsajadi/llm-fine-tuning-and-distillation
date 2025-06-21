import torch
import logging
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from src.core.model_loader import ModelLoader
from src.utils.utils import save_adapter, generate_checksum, LogEpochLossCallback
from src.data.data_preparation import prepare_tokenized_dataset, data_collator

# Logging setup unchanged
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GPT2LoRAFineTuner:
    def __init__(
        self,
        base_model: str = "openai-community/gpt2-medium",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit_quant: bool = True,
        device_map: str = "auto"
    ):
        self.base_model = base_model
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj", "mlp.c_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=True
        )
        self.loader = ModelLoader(
            model_name=base_model,
            lora_config=self.lora_config,
            use_4bit=use_4bit_quant,
            device_map=device_map,
            train_mode=True
        )
        self.model = self.loader.model
        self.tokenizer = self.loader.tokenizer

    def prepare_dataset(self, data_path: str, max_length: int = 128) -> Dataset:
        """
        Prepare dataset for fine-tuning using data preparation utilities.

        Args:
            data_path (str): Path to the prompt file.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            Dataset: Hugging Face Dataset with input_ids, attention_mask, and labels.
        """
        return prepare_tokenized_dataset(
            file_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            logger=logger
        )

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./lora_training",
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            import bitsandbytes
            optim_name = "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
            logger.warning("Bitsandbytes detected, using 8-bit optimizer")
        except ImportError:
            optim_name = "adamw_torch"
            logger.warning("bitsandbytes not installed. Using standard AdamW optimizer")
        self.model.config.use_cache = False
        self.model.train()
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
            gradient_checkpointing=False,
            logging_dir=str(output_path / "logs"),
            save_total_limit=1,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            dataloader_pin_memory=torch.cuda.is_available(),
            gradient_accumulation_steps=4
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda features: data_collator(features, logger=logger),
            callbacks=[LogEpochLossCallback(logger=logger)]
        )
        try:
            import psutil
            process = psutil.Process(os.getpid())
            logger.info(f"Initial memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
        except ImportError:
            logger.warning("psutil not installed; memory usage logging disabled")
        logger.info("Verifying gradient connectivity...")
        test_input = next(iter(trainer.get_train_dataloader()))
        test_input = {k: v.to(self.model.device) for k, v in test_input.items()}
        with torch.set_grad_enabled(True):
            self.model.zero_grad()
            outputs = self.model(**test_input)
            if outputs.loss is None:
                raise RuntimeError("Loss is None; check input data and model configuration")
            if outputs.loss.grad_fn is None:
                raise RuntimeError("Loss not connected to computational graph")
            outputs.loss.backward()
            gradient_found = False
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    logger.info(f"Gradient found for {name}")
                    gradient_found = True
            if not gradient_found:
                raise RuntimeError("No gradients computed for any trainable parameters")
        logger.info("Gradient verification passed. Starting training...")
        self.model.zero_grad()
        trainer.train()
        if 'psutil' in locals():
            logger.info(f"Final memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
        save_adapter(self.model, self.base_model, self.lora_config, output_dir)
        generate_checksum(output_dir)

class QLoRAFineTuner:
    def __init__(
        self,
        base_model: str = "openai-community/gpt2-medium",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = ["c_attn", "c_proj"],
        device_map: str = "auto"
    ):
        """
        QLoRA Fine-Tuner with automatic device compatibility
        Implements QLoRA paper innovations with CPU/GPU flexibility
        """
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Detect execution environment
        self.use_gpu = torch.cuda.is_available()
        logger.info(f"Using {'GPU' if self.use_gpu else 'CPU'} execution device")
        
        # --- 4-bit NormalFloat (NF4) with Double Quantization ---
        # Paper Sec 3.1-3.2: Only applicable for GPU environments
        self.quant_config = None
        if self.use_gpu:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",          # NF4 data type
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,      # Double Quantization
            )
            logger.info("Enabled 4-bit NF4 quantization with double quantization")
        else:
            logger.warning("4-bit quantization disabled - requires GPU")
            device_map = "cpu"  # Force CPU mode
        
        # Load base model with device-aware configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=self.quant_config,
            device_map=device_map,
            trust_remote_code=False
        )
        
        # --- LoRA Configuration ---
        # Paper Sec 4: Apply LoRA to all linear layers
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA adapters
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        # Device placement confirmation
        logger.info(f"Model loaded on device: {self.model.device}")

    def prepare_dataset(self, data_path: str, max_length: int = 512) -> Dataset:
        """
        Prepare dataset with device-aware optimizations
        """
        # Security: Validate file path
        if not Path(data_path).is_file():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Read and sanitize prompts
        with open(data_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if len(prompts) == 0:
            raise ValueError("No valid prompts in dataset")
        
        # Tokenization with device-appropriate settings
        tokenized = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # CPU/GPU compatible dataset creation
        return Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()
        })

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./qlora_results",
        per_device_train_batch_size: int = 4,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):
        """
        Device-optimized training with QLoRA innovations
        """
        # Paper Sec 3.3: Paged Optimizers (GPU only)
        if self.use_gpu:
            try:
                import bitsandbytes
                optim_name = "paged_adamw_8bit"
                logger.info("Using paged optimizer for GPU memory management")
            except ImportError:
                optim_name = "adamw_torch"
                logger.warning("bitsandbytes unavailable - using standard AdamW")
        else:
            optim_name = "adamw_torch"
            logger.info("Using standard AdamW for CPU training")
        
        # Configure training with device-specific settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to="none",
            fp16=self.use_gpu and torch.cuda.is_available(),  # Mixed precision only on GPU
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=self.use_gpu,  # Memory optimization only on GPU
            logging_dir=str(Path(output_dir) / "logs"),
            save_total_limit=1,
            no_cuda=not self.use_gpu  # Explicitly disable CUDA if not available
        )
        
        # Data collator works on both CPU/GPU
        def data_collator(features):
            return {
                'input_ids': torch.stack([d['input_ids'] for d in features]),
                'attention_mask': torch.stack([d['attention_mask'] for d in features]),
                'labels': torch.stack([d['labels'] for d in features])
            }
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        # Critical QLoRA settings
        self.model.config.use_cache = False
        self.model.train()
        
        # Device-aware training execution
        logger.info(f"Starting training on {'GPU' if self.use_gpu else 'CPU'}")
        trainer.train()
        
        # Save adapter weights
        self.model.save_pretrained(output_dir)
        logger.info(f"Adapter saved to {output_dir}")

def main():
    tuner = GPT2LoRAFineTuner(
        base_model="openai-community/gpt2-medium",
        lora_rank=8,
        lora_alpha=32
    )
    dataset = tuner.prepare_dataset("data/synthetic/prompts_v1.txt")
    tuner.train(
        dataset,
        output_dir="adapters/gpt2_lora_v1",
        per_device_train_batch_size=1,
        num_train_epochs=3
    )
    save_adapter(tuner.model, tuner.base_model, tuner.lora_config, "final_adapters/gpt2_lora")

if __name__ == "__main__":
    main()