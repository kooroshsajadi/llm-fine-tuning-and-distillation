import logging
import os
import torch
from pathlib import Path
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from peft import LoraConfig
from datasets import Dataset
from src.data.data_preparation import prepare_tokenized_dataset, data_collator
from src.core.model_loader import ModelLoader
from src.fine_tuning.fine_tuner import FineTuner
from src.utils.logging_utils import setup_logger
import src.utils.utils as utils
# from src.data.data_preparation import split_train_val
from datasets import DatasetDict
from src.utils.metrics_utils import HFMetricHelper
import numpy as np
import nltk
from evaluate import load
from src.data.data_preparation import prepare_dataset_dict


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

        # Validate model_type
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

    # def prepare_dataset(self, data_path: str, val_ratio=0.1) -> DatasetDict:
    #     self.logger.info("Preparing dataset from %s", data_path)
    #     full_dataset = prepare_tokenized_dataset(
    #         input_path=data_path,
    #         tokenizer=self.tokenizer,
    #         max_length=self.max_length,
    #         logger=self.logger,
    #         model_type=self.model_type
    #     )
    #     # Split into train and validation sets
    #     dataset_dict = split_train_val(full_dataset, val_ratio=val_ratio)
    #     self.logger.info(
    #         f"Dataset split: train size: {len(dataset_dict['train'])}, validation size: {len(dataset_dict['validation'])}"
    #     )
    #     return dataset_dict

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
        output_path = Path(output_dir)
        # output_path.mkdir(parents=True, exist_ok=True)

        optim_name = "paged_adamw_8bit" if self.use_qlora and self.use_gpu and not self.loader.use_xpu else "adamw_torch"
        self.logger.info(f"Using optimizer: {optim_name}")

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_strategy=IntervalStrategy.EPOCH,
            predict_with_generate=True,
            generation_max_length=128,
            eval_accumulation_steps=2,
            generation_num_beams=4, # For beam search decoding
            report_to="none",
            fp16=self.use_gpu and not self.loader.use_xpu,
            bf16=self.loader.use_xpu,
            save_total_limit=1,
            dataloader_pin_memory=self.use_gpu,
            gradient_accumulation_steps=4,
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            remove_unused_columns=False,
            optim=optim_name,
            gradient_checkpointing=False,
            logging_dir='logs',
        )

        # training_args = TrainingArguments(
        #     output_dir=output_dir,
        #     per_device_train_batch_size=per_device_train_batch_size,
        #     num_train_epochs=num_train_epochs,
        #     learning_rate=learning_rate,
        #     logging_steps=logging_steps,
        #     save_strategy=save_strategy, # When to save model checkpoints ('epoch', 'steps', etc.)
        #     report_to="none", # Disable reporting to external experiment trackers
        #     fp16=self.use_gpu and not self.loader.use_xpu, # Use mixed precision (float16) training if using GPU without XPU
        #     bf16=self.loader.use_xpu, # Use bfloat16 precision when using XPU device
        #     remove_unused_columns=False,
        #     optim=optim_name,
        #     gradient_checkpointing=False,
        #     logging_dir=str(output_path / "logs"),
        #     eval_strategy="epoch",
        #     eval_accumulation_steps=2,
        #     save_total_limit=1,
        #     predic=True,
        #     max_grad_norm=0.3, # Maximum norm for gradient clipping to stabilize training
        #     warmup_ratio=0.05, # Fraction of total steps for learning rate warmup at start
        #     dataloader_pin_memory=self.use_gpu, # Pin memory for DataLoader for faster host-to-device transfer if using GPU
        #     gradient_accumulation_steps=4 # Number of forward passes before backward update to simulate larger batch size
        # )

        # metric_helper = HFMetricHelper(tokenizer=self.tokenizer)
        metric = load("rouge")
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Rouge expects a newline after each sentence
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip(), language='italian')) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip(), language='italian')) for label in decoded_labels]

            # Note that other metrics may not have a `use_aggregator` parameter
            # and thus will return a list, computing a metric for each sentence.
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
            # Extract a few results
            result = {key: value * 100 for key, value in result.items()}

            # Add mean generated length
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)

            return {k: round(v, 4) for k, v in result.items()}

        seq2seq_trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            data_collator=lambda features: DataCollatorForSeq2Seq(self.tokenizer, model=self.model)(features),
            compute_metrics=compute_metrics
        )

        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=dataset_dict['train'],
        #     eval_dataset=dataset_dict['validation'],
        #     data_collator=lambda features: data_collator(features, logger=self.logger, model_type=self.model_type),
        #     compute_metrics=metric_helper.compute
        # )

        try:
            self.logger.info("Verifying gradient connectivity...")
            test_input = next(iter(seq2seq_trainer.get_train_dataloader()))
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
        seq2seq_trainer.train()

        if 'psutil' in locals():
            self.logger.info(f"Final memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"Model and tokenizer saved to {output_dir}")

def main():
    config = utils.return_config("configs/fine_tuning/marian_mt.yaml")

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

    dataset_dict = prepare_dataset_dict(input_path=config['datasets']['prefettura_v1_texts'],
                                        tokenizer=tuner.loader.tokenizer,
                                        max_length=tuner_config.get('max_length', 128),
                                        model_type=tuner_config.get('model_type', 'causal_lm'))
    # dataset_dict = tuner.prepare_dataset(config['datasets']['prefettura_v1_texts'], val_ratio=0.1)
    tuner.train(
        dataset_dict=dataset_dict,
        output_dir='models/fine_tuned_models/opus-mt-it-en-v1',
        # per_device_train_batch_size=tuner_config.get('per_device_train_batch_size', 1),
        # per_device_eval_batch_size=tuner_config.get('per_device_eval_batch_size', 1),
        num_train_epochs=tuner_config.get('num_train_epochs', 3),
        learning_rate=float(tuner_config.get('learning_rate', 1e-4)),
        logging_steps=tuner_config.get('logging_steps', 10),
        save_strategy=tuner_config.get('save_strategy', 'epoch')
    )

if __name__ == "__main__":
    main()