import logging
import os
import torch
from pathlib import Path
from transformers import Trainer, TrainingArguments
from peft import LoraConfig
from datasets import Dataset
from src.data.data_preparation import prepare_tokenized_dataset, data_collator
from src.core.model_loader import ModelLoader
from src.fine_tuning.fine_tuner import FineTuner
import yaml
from src.utils.logging_utils import setup_logger
import src.utils.utils as utils

# Logging setup
logger = setup_logger(__name__)

class QLoRAFineTuner(FineTuner):
    """QLoRA/LoRA Fine-Tuner supporting multiple model types."""

    DEFAULT_TARGET_MODULES = {
        "causal_lm": ["c_attn", "c_proj", "mlp.c_proj"],
        "masked_lm": ["query", "key", "value", "output.dense"],
        # "masked_lm": ["q", "k", "v", "dense"],
        "SEQ_2_SEQ_LM": ["self_attn.k", "self_attn.q", "self_attn.v", "self_attn.out", "encoder_attn.k",
                                       "encoder_attn.q", "encoder_attn.v", "encoder_attn.out", "fc1", "fc2"]
    }

    def __init__(
        self,
        base_model: str = "openai-community/gpt2-medium",
        model_type: str = "causal_lm",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list | None = None,
        use_qlora: bool = True,
        device_map: str = "auto",
        max_length: int = 128,
        logger: logging.Logger = logger
    ):
        """
        Initialize QLoRAFineTuner with configuration parameters.

        Args:
            base_model (str): Model name or path.
            model_type (str): Model type ("causal_lm", "masked_lm", "seq2seq").
            lora_rank (int): LoRA rank.
            lora_alpha (int): LoRA scaling factor.
            lora_dropout (float): LoRA dropout rate.
            target_modules (list, optional): Modules to apply LoRA (auto-detected if None).
            use_qlora (bool): Enable QLoRA (requires CUDA GPU).
            device_map (str): Device placement strategy.
            max_length (int): Maximum sequence length.
            logger (logging.Logger): Logger instance.
        """
        super().__init__(logger)
        self.base_model = base_model
        self.model_type = model_type
        self.max_length = max_length
        self.use_qlora = use_qlora

        if self.model_type not in self.DEFAULT_TARGET_MODULES:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose from {list(self.DEFAULT_TARGET_MODULES.keys())}")

        # Set target_modules
        target_modules = target_modules or self.DEFAULT_TARGET_MODULES[self.model_type]

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=self.model_type.upper(),
            fan_in_fan_out=(self.model_type == "causal_lm")
        )

        # Load model and tokenizer
        self.loader = ModelLoader(
            model_name=base_model,
            model_type=self.model_type,
            lora_config=self.lora_config,
            use_qlora=use_qlora,
            device_map=device_map,
            max_length=max_length,
            train_mode=True
        )
        self.model = self.loader.model
        self.tokenizer = self.loader.tokenizer
        self.use_gpu = self.loader.use_gpu

        self.model.print_trainable_parameters()

    def prepare_dataset(self, data_path: str) -> Dataset:
        self.logger.info("Preparing dataset from %s", data_path)
        return prepare_tokenized_dataset(
            input_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            logger=self.logger,
            model_type=self.model_type
        )

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "data/outputs/models/fine_tuned_model",
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        optim_name = "paged_adamw_8bit" if self.use_qlora and self.use_gpu and not self.loader.use_xpu else "adamw_torch"
        self.logger.info(f"Using optimizer: {optim_name}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to="none",
            fp16=self.use_gpu and not self.loader.use_xpu,
            bf16=self.loader.use_xpu,
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=False,
            logging_dir=str(output_path / "logs"),
            save_total_limit=1,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            dataloader_pin_memory=self.use_gpu,
            gradient_accumulation_steps=4
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda features: data_collator(features, logger=self.logger, model_type=self.model_type)
        )

        try:
            self.logger.info("Verifying gradient connectivity...")
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
                        self.logger.info(f"Gradient found for {name}")
                        gradient_found = True
                if not gradient_found:
                    raise RuntimeError("No gradients computed")
            self.logger.info("Gradient verification passed")
        except Exception as e:
            self.logger.error(f"Gradient verification failed: {str(e)}")
            raise

        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.logger.info(f"Initial memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
        except ImportError:
            self.logger.warning("psutil not installed; memory logging disabled")

        self.model.config.use_cache = False
        self.model.train()
        trainer.train()

        if 'psutil' in locals():
            self.logger.info(f"Final memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"Model and tokenizer saved to {output_dir}")

def main():
    config = utils.return_config("configs/fine_tuning/qlora_fine_tuner.yaml")

    tuner_config = config.get('fine_tuning', {})
    tuner = QLoRAFineTuner(
        base_model=tuner_config.get('base_model', 'Helsinki-NLP/opus-mt-it-en'),
        model_type=tuner_config.get('model_type', 'causal_lm'),
        lora_rank=tuner_config.get('lora_rank', 8),
        lora_alpha=tuner_config.get('lora_alpha', 32),
        lora_dropout=tuner_config.get('lora_dropout', 0.05),
        target_modules=tuner_config.get('target_modules', None),
        use_qlora=tuner_config.get('use_qlora', False),
        device_map=tuner_config.get('device_map', 'auto'),
        max_length=tuner_config.get('max_length', 128),
        logger=logger
    )
    dataset = tuner.prepare_dataset(config['datasets']['prefettura_v1_texts'])
    tuner.train(
        dataset=dataset,
        output_dir=tuner_config.get('output_dir', 'data/outputs/models/opus-mt-it-en-v1'),
        per_device_train_batch_size=tuner_config.get('per_device_train_batch_size', 1),
        num_train_epochs=tuner_config.get('num_train_epochs', 3),
        learning_rate=float(tuner_config.get('learning_rate', 1e-4)),
        logging_steps=tuner_config.get('logging_steps', 10),
        save_strategy=tuner_config.get('save_strategy', 'epoch')
    )

if __name__ == "__main__":
    main()