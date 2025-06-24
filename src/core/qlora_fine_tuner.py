import logging
import os
import torch
from pathlib import Path
from transformers import Trainer, TrainingArguments
from peft import LoraConfig
from datasets import Dataset
from src.data.data_preparation import prepare_tokenized_dataset, data_collator
from src.core.model_loader import ModelLoader

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class QLoRAFineTuner:
    def __init__(
        self,
        base_model: str = "openai-community/gpt2-medium",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = ["c_attn", "c_proj", "mlp.c_proj"],
        use_qlora: bool = True,
        device_map: str = "auto",
        max_length: int = 128
    ):
        """
        QLoRA/LoRA Fine-Tuner implementing:
        - QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
        - Automatic fallback to LoRA when QLoRA requirements not met

        Args:
            base_model (str): Model name or path.
            lora_rank (int): LoRA rank.
            lora_alpha (int): LoRA scaling factor.
            lora_dropout (float): LoRA dropout rate.
            target_modules (list): Modules to apply LoRA.
            use_qlora (bool): Enable QLoRA (requires GPU and bitsandbytes).
            device_map (str): Device placement strategy.
            max_length (int): Maximum sequence length.
        """
        self.base_model = base_model
        self.max_length = max_length
        self.use_qlora = use_qlora

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=True
        )

        # Load model and tokenizer via ModelLoader
        self.loader = ModelLoader(
            model_name=base_model,
            lora_config=self.lora_config,
            use_qlora=use_qlora,
            device_map=device_map,
            max_length=max_length,
            train_mode=True
        )
        self.model = self.loader.model
        self.tokenizer = self.loader.tokenizer
        self.use_gpu = self.loader.use_gpu

        # Log trainable parameters
        self.model.print_trainable_parameters()

    def prepare_dataset(self, data_path: str) -> Dataset:
        """
        Prepare dataset using project-standard function.

        Args:
            data_path (str): Path to prompt file.

        Returns:
            Dataset: Hugging Face Dataset with input_ids, attention_mask, labels.
        """
        return prepare_tokenized_dataset(
            file_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            logger=logger
        )

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "model/fine_tuned_model",
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):
        """
        Train the model with QLoRA or LoRA.

        Args:
            dataset (Dataset): Training dataset.
            output_dir (str): Directory to save model and logs.
            per_device_train_batch_size (int): Batch size per device.
            num_train_epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            logging_steps (int): Log every N steps.
            save_strategy (str): Save strategy (e.g., "epoch").
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Optimizer
        optim_name = "paged_adamw_8bit" if self.use_qlora and self.use_gpu else "adamw_torch"
        logger.info(f"Using optimizer: {optim_name}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to="none",
            fp16=self.use_gpu,
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=self.use_gpu,
            logging_dir=str(output_path / "logs"),
            save_total_limit=1,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            dataloader_pin_memory=self.use_gpu,
            gradient_accumulation_steps=4
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda features: data_collator(features, logger=logger)
        )

        # Gradient verification
        try:
            logger.info("Verifying gradient connectivity...")
            test_input = next(iter(trainer.get_train_dataloader()))
            test_input = {k: v.to(self.model.device) for k, v in test_input.items()}
            with torch.set_grad_enabled(True):
                self.model.zero_grad()
                outputs = self.model(**test_input)
                if outputs.loss is None:
                    raise RuntimeError("Loss is None; check input data")
                outputs.loss.backward()
                gradient_found = False
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        logger.info(f"Gradient found for {name}")
                        gradient_found = True
                if not gradient_found:
                    raise RuntimeError("No gradients computed")
            logger.info("Gradient verification passed")
        except Exception as e:
            logger.error(f"Gradient verification failed: {str(e)}")
            raise

        # Memory logging
        try:
            import psutil
            process = psutil.Process(os.getpid())
            logger.info(f"Initial memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
        except ImportError:
            logger.warning("psutil not installed; memory logging disabled")

        # Train
        self.model.config.use_cache = False
        self.model.train()
        trainer.train()

        # Final memory logging
        if 'psutil' in locals():
            logger.info(f"Final memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")

def main():
    # tuner = QLoRAFineTuner(base_model="openai-community/gpt2-medium", use_qlora=True)
    # dataset = tuner.prepare_dataset("data/synthetic/prompts_v1.txt")
    # tuner.train(dataset, output_dir="data/qlora_fine_tuned")

    tuner = QLoRAFineTuner(base_model="openai-community/gpt2-medium", use_qlora=False, device_map="cpu")
    dataset = tuner.prepare_dataset("data/synthetic/prompts_v1.txt")
    tuner.train(dataset, output_dir="data/lora_fine_tuned")

if __name__ == "__main__":
    main()