import logging
import os
import torch
from pathlib import Path
from peft import LoraConfig
from model.model_loader import ModelLoader
from fine_tuning.fine_tuner import FineTuner
from utils.logging_utils import setup_logger
import utils.utils as utils
from datasets import DatasetDict
from utils.metrics_utils import HFMetricHelper
from data.data_preparation import prepare_dataset_dict
from transformers import (
    Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments,
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, PreTrainedTokenizerBase
)


# Logging setup
logger = setup_logger(__name__)

class QLoRAFineTuner(FineTuner):
    """QLoRA/LoRA Fine-Tuner supporting multiple model types."""

    DEFAULT_TARGET_MODULES = {
        "causal_lm": ["c_attn", "c_proj", "mlp.c_proj"],
        "masked_lm": ["query", "key", "value", "output.dense"],
        # "masked_lm": ["q", "k", "v", "dense"],
        "SEQ_2_SEQ_LM": ["self_attn.k", "self_attn.q", "self_attn.v", "self_attn.out", "encoder_attn.k",
                                       "encoder_attn.q", "encoder_attn.v", "encoder_attn.out", "fc1", "fc2"],
        "distilgpt2": ["c_attn", "c_proj", "c_fc", "c_proj"],
        "llama3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "falcon7binstruct": ["query_key_value", "dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
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

        # Validate model_type
        if self.model_type not in self.DEFAULT_TARGET_MODULES:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose from {list(self.DEFAULT_TARGET_MODULES.keys())}")

        # Set target_modules
        self.target_modules = target_modules

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=self.model_type.upper(),
            fan_in_fan_out=(self.model_type == "causal_lm")
        )

        # Load model and tokenizer
        self.model_loader = ModelLoader(
            model_name=self.base_model,
            model_type=self.model_type,
            lora_config=self.lora_config,
            use_qlora=use_qlora,
            device_map=device_map,
            max_length=self.max_length,
            train_mode=True
        )
        self.model = self.model_loader.model
        self.model_loader._log_model_profile("Base model loaded")
        self.tokenizer = self.model_loader.tokenizer
        self.use_gpu = self.model_loader.use_gpu

        self.model.print_trainable_parameters()

    def _get_causal_lm_training_args(
        self,
        output_dir: str,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        num_train_epochs: int,
        learning_rate: float,
        logging_steps: int, 
        save_strategy: str,
        optim_name: str
    ) -> TrainingArguments:
        """Returns TrainingArguments for causal language models."""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_strategy="epoch",
            report_to="none",
            fp16=self.model_loader.use_fp16,
            bf16=self.model_loader.use_bf16,
            save_total_limit=1,
            dataloader_pin_memory=self.model_loader.use_cuda or self.model_loader.use_xpu,
            gradient_accumulation_steps=4,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=False,
            logging_dir='logs',
        )

    def _get_seq2seq_training_args(
        self,
        output_dir: str,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        num_train_epochs: int,
        learning_rate: float,
        logging_steps: int,
        save_strategy: str,
        optim_name: str
    ) -> Seq2SeqTrainingArguments:
        """Returns Seq2SeqTrainingArguments for sequence-to-sequence models."""
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=128,
            eval_accumulation_steps=2,
            generation_num_beams=4,
            report_to="none",
            fp16=self.model_loader.use_fp16,
            bf16=self.model_loader.use_bf16,
            save_total_limit=1,
            dataloader_pin_memory=self.model_loader.use_cuda or self.model_loader.use_xpu,
            gradient_accumulation_steps=4,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=False,
            logging_dir='logs',
        )

    def _get_trainer(
        self,
        model_type: str,
        dataset_dict: DatasetDict,
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        model
    ):
        """Factory method to return the appropriate trainer based on model_type."""
        if model_type not in self.model_loader.MODEL_TYPE_MAPPING:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from {list(self.model_loader.MODEL_TYPE_MAPPING.keys())}"
            )

        if model_type == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # No masked language modeling for causal LM
                pad_to_multiple_of=8 if self.use_qlora else None
            )
            compute_metrics = None  # Perplexity/loss computed by default
            return Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_dict['train'],
                eval_dataset=dataset_dict['validation'] if 'validation' in dataset_dict else None,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
        elif model_type == "seq2seq_lm":
            metric_helper = HFMetricHelper(tokenizer=tokenizer, bertscore_model_type="bert-base-multilingual-cased")
            return Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_dict['train'],
                eval_dataset=dataset_dict['validation'] if 'validation' in dataset_dict else None,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                compute_metrics=metric_helper.compute
            )
        else:
            raise NotImplementedError(f"Trainer for model_type {model_type} not implemented")

    def train(
        self,
        dataset_dict: DatasetDict,
        output_dir: str,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 1,
        num_train_epochs: int = 2,
        learning_rate: float = 1e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ):

        # Validate dataset.
        if 'train' not in dataset_dict:
            raise ValueError("DatasetDict must contain a 'train' split.")
        
        # Optimizer selection
        optim_name = "paged_adamw_8bit" if self.use_qlora and self.model_loader.use_cuda and not self.model_loader.use_xpu else "adamw_torch"
        self.logger.info(f"Using optimizer: {optim_name}")

        # Select training arguments based on model type.
        training_args_map = {
            "causal_lm": self._get_causal_lm_training_args,
            "seq2seq_lm": self._get_seq2seq_training_args
        }
        if self.model_type not in training_args_map:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Choose from {list(training_args_map.keys())}"
            )
        
        training_args = training_args_map[self.model_type](
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            optim_name=optim_name
        )
        
        # Initialize trainer
        trainer = self._get_trainer(
            model_type=self.model_type,
            dataset_dict=dataset_dict,
            training_args=training_args,
            tokenizer=self.tokenizer,
            model=self.model
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

        model_dir = Path(output_dir) / "model"
        tok_dir = Path(output_dir) / "tokenizer"

        self.model.save_pretrained(str(model_dir))
        self.tokenizer.save_pretrained(tok_dir)
        self.logger.info(f"Model and tokenizer saved to {output_dir}")

def main():
    config = utils.return_config("configs/fine_tuning/Meta-Llama-3-8B-Instruct.yaml")

    tuner_config = config.get('fine_tuning', {})
    tuner = QLoRAFineTuner(
        base_model=tuner_config.get('base_model', 'Helsinki-NLP/opus-mt-it-en'),
        model_type=tuner_config.get('model_type', 'causal_lm'),
        lora_rank=tuner_config.get('lora_rank', 8),
        lora_alpha=tuner_config.get('lora_alpha', 32),
        lora_dropout=tuner_config.get('lora_dropout', 0.05),
        target_modules=tuner_config.get('target_modules'),
        use_qlora=tuner_config.get('use_qlora', False),
        device_map=tuner_config.get('device_map', 'auto'),
        max_length=tuner_config.get('max_length', 128),
        logger=logger
    )

    dataset_dict = prepare_dataset_dict(input_path=config['datasets']['leggi_area_3_text'],
                                        tokenizer=tuner.model_loader.tokenizer,
                                        max_length=tuner_config.get('max_length', 128),
                                        model_type=tuner_config.get('model_type', 'causal_lm'))
    
    tuner.train(
        dataset_dict=dataset_dict,
        output_dir='artifacts/models/fine_tuned_models/Meta-Llama-3-8B-Instruct', #falcon-7b-instruct
        num_train_epochs=tuner_config.get('num_train_epochs', 10),
        learning_rate=float(tuner_config.get('learning_rate', 1e-5)),
        logging_steps=tuner_config.get('logging_steps', 10),
        save_strategy=tuner_config.get('save_strategy', 'epoch')
    )

if __name__ == "__main__":
    main()