import torch.nn.functional as F
import logging
import os
from pathlib import Path
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from model.model_loader import ModelLoader
from utils.utils import save_model, generate_checksum, LogEpochLossCallback
from data.data_preparation import load_kd_dataset, data_collator

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class KDTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation with combined CE and KL loss.
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        """
        Initialize KDTrainer with temperature and alpha for loss computation.

        Args:
            temperature (float): Temperature for softening logits.
            alpha (float): Weight for KL divergence loss (1-alpha for CE loss).
            **kwargs: Other Trainer arguments.
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: alpha * KL_div(teacher_logits, student_logits) + (1-alpha) * CE_loss.

        Args:
            model: The student model.
            inputs: Batch with input_ids, attention_mask, labels, and teacher_logits.
            return_outputs (bool): If True, return model outputs.
            num_items_in_batch (int, optional): Number of items in the batch (ignored).

        Returns:
            torch.Tensor or (torch.Tensor, dict): Loss or (loss, outputs).
        """
        if 'teacher_logits' not in inputs:
            raise ValueError("teacher_logits missing in inputs")

        # Log pad_token_id
        pad_token_id = model.config.pad_token_id
        logger.debug(f"Using pad_token_id: {pad_token_id}")
        if pad_token_id is None:
            raise ValueError("model.config.pad_token_id is None; ensure it is set in ModelLoader")

        outputs = model(**{k: v for k, v in inputs.items() if k != 'teacher_logits'})
        student_logits = outputs.logits
        labels = inputs["labels"]
        teacher_logits = inputs["teacher_logits"]

        # Validate shapes
        if student_logits.shape != teacher_logits.shape:
            logger.error(f"Mismatch in logits shapes: student={student_logits.shape}, teacher={teacher_logits.shape}")
            raise ValueError("Logits shape mismatch")
        if labels.shape != student_logits.shape[:2]:
            logger.error(f"Mismatch in labels shape: labels={labels.shape}, expected={student_logits.shape[:2]}")
            raise ValueError("Labels shape mismatch")

        # Cross-entropy loss (hard labels)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=pad_token_id
        )

        # KL divergence loss (soft labels)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        logger.debug(f"CE Loss: {ce_loss:.4f}, KL Loss: {kl_loss:.4f}, Total Loss: {loss:.4f}")

        return (loss, outputs) if return_outputs else loss

class GPT2KDDistiller:
    def __init__(
        self,
        student_model: str = "openai-community/gpt2",
        max_length: int = 128
    ):
        """
        Initialize student model for knowledge distillation.

        Args:
            student_model (str): Hugging Face model name for student (e.g., "openai-community/gpt2").
            max_length (int): Maximum sequence length for tokenization.
        """
        self.max_length = max_length
        self.loader = ModelLoader(
            model_name=student_model,
            use_4bit=False,  # CPU-based
            device_map="auto",
            max_length=max_length,
            train_mode=True
        )
        self.model = self.loader.model
        self.tokenizer = self.loader.tokenizer

    def prepare_dataset(self, prompt_file: str, logits_dir: str) -> Dataset:
        """
        Prepare dataset for KD training.

        Args:
            prompt_file (str): Path to the prompt file.
            logits_dir (str): Directory containing teacher logits.

        Returns:
            Dataset: Hugging Face Dataset with input_ids, attention_mask, labels, and teacher_logits.
        """
        return load_kd_dataset(
            prompt_file=prompt_file,
            logits_dir=logits_dir,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            logger=logger
        )

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./adapters/gpt2_small_kd",
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Train the student model using knowledge distillation.

        Args:
            dataset (Dataset): KD dataset with teacher logits.
            output_dir (str): Directory to save the student model.
            per_device_train_batch_size (int): Batch size per device.
            num_train_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            logging_steps (int): Steps between logging.
            save_strategy (str): Save strategy ("epoch" or "steps").
            temperature (float): Temperature for KD loss.
            alpha (float): Weight for KL divergence loss.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        optim_name = "adamw_torch"
        self.model.train()
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to="none",
            fp16=False,  # CPU-based
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=False,
            logging_dir=str(output_path / "logs"),
            save_total_limit=1,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            dataloader_pin_memory=False,  # CPU-based
            gradient_accumulation_steps=4
        )

        trainer = KDTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda features: data_collator(features, logger=logger),
            callbacks=[LogEpochLossCallback(logger=logger)],
            temperature=temperature,
            alpha=alpha
        )

        try:
            import psutil
            process = psutil.Process(os.getpid())
            logger.info(f"Initial memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
        except ImportError:
            logger.warning("psutil not installed; memory usage logging disabled")

        logger.info("Starting knowledge distillation training...")
        trainer.train()

        if 'psutil' in locals():
            logger.info(f"Final memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        # Save the full student model
        save_model(self.model, output_dir)
        generate_checksum(output_dir)

def main():
    distiller = GPT2KDDistiller(
        student_model="openai-community/gpt2",
        max_length=128
    )
    dataset = distiller.prepare_dataset(
        prompt_file="data/synthetic/prompts_v1.txt",
        logits_dir="data/synthetic/v1_t0.7"
    )
    distiller.train(
        dataset,
        output_dir="adapters/gpt2_small_kd",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        temperature=2.0,
        alpha=0.5
    )

if __name__ == "__main__":
    main()