#!/usr/bin/env python3
import torch
import logging
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig
from datasets import Dataset
from src.core.model_loader import ModelLoader
from src.core.logit_generator import sha256sum

# Logging setup
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
        """
        Initialize the GPT-2 LoRA fine-tuner.
        """
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

        # Load model and tokenizer using ModelLoader
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
        # Unchanged
        if not Path(data_path).is_file():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        if len(prompts) == 0:
            raise ValueError("No valid prompts in dataset")
        if len(prompts) < 4:
            logger.warning(f"Dataset size ({len(prompts)}) is smaller than typical batch size (4)")

        tokenized = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()
        })
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./lora_results",
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):
        # Unchanged except for model reference
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        try:
            import bitsandbytes
            optim_name = "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
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

        def data_collator(features):
            if not all('input_ids' in f and 'attention_mask' in f and 'labels' in f for f in features):
                raise ValueError("Invalid dataset features")
            batch = {
                'input_ids': torch.stack([torch.tensor(f['input_ids'], dtype=torch.long) for f in features]),
                'attention_mask': torch.stack([torch.tensor(f['attention_mask'], dtype=torch.long) for f in features]),
                'labels': torch.stack([torch.tensor(f['labels'], dtype=torch.long) for f in features])
            }
            logger.debug(f"Batch shapes: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}, labels={batch['labels'].shape}")
            return batch

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
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

        self.save_adapter(output_dir)
        self._generate_checksum(output_dir)

    def save_adapter(self, output_path: str):
        """Save adapter with security metadata."""
        self.model.save_pretrained(output_path)
        metadata = {
            "base_model": self.base_model,
            "lora_rank": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "target_modules": list(self.lora_config.target_modules),
            "architecture": "GPT-2"
        }
        with open(Path(output_path) / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_checksum(self, output_dir: str):
        # Unchanged
        output_path = Path(output_dir)
        for file in output_path.glob("*"):
            if file.is_file() and not file.suffix.endswith(".sha256"):
                checksum = sha256sum(file)
                with open(file.with_suffix(file.suffix + ".sha256"), "w") as f:
                    f.write(checksum)

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
    tuner.save_adapter("final_adapters/gpt2_lora")

if __name__ == "__main__":
    main()