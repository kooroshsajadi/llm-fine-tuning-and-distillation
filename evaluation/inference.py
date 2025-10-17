import torch
from pathlib import Path
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from data.data_preparation import prepare_dataset_dict
from utils import utils
from utils.logging_utils import setup_logger
from utils.metrics_utils import HFMetricHelper
from model.model_loader import ModelLoader

logger = setup_logger('evaluation.inference')

def inference(args):
    """
    Perform inference on a fine-tuned model with LoRA adapters and compute ROUGE, BLEU, and BERTScore metrics.

    Args:
        args (dict): Dictionary containing base_model_path, adapter_path, tokenizer_path, dataset_path,
                     max_length, batch_size, generation_max_length, num_beams, model_type,
                     offload_to_disk, and offload_dir.
    """
    # Map model_type to match ModelLoader expectations
    model_type_map = {
        "causal": "causal_lm",
        "seq2seq": "seq2seq_lm"
    }
    loader_model_type = model_type_map.get(args.get("model_type", "seq2seq"), "seq2seq_lm")

    # Set use_qlora based on offload_to_disk
    use_qlora = not args.get("offload_to_disk", True) and torch.cuda.is_available()

    # Load model and tokenizer using ModelLoader
    model_loader = ModelLoader(
        model_name=args["base_model_path"],
        model_type=loader_model_type,
        adapter_path=str(args["adapter_path"]),
        use_qlora=use_qlora,
        device_map="auto",
        max_length=args["max_length"],
        train_mode=False,
        offload_to_disk=args.get("offload_to_disk", True),
        offload_dir=args.get("offload_dir", "./offload_dir")
    )
    model = model_loader.model
    tokenizer = model_loader.tokenizer
    logger.info(f"Loaded model from {args['base_model_path']} with adapters from {args['adapter_path']} using ModelLoader")
    logger.info(f"QLoRA: {use_qlora}, Disk offloading: {model_loader.offload_to_disk} ({model_loader.offload_dir if model_loader.offload_to_disk else 'N/A'})")

    # Log model profile
    model_loader._log_model_profile("Inference model loaded")

    # Prepare test dataset
    dataset_dict = prepare_dataset_dict(
        input_path=args["dataset_path"],
        tokenizer=tokenizer,
        max_length=args["max_length"],
        model_type=loader_model_type,
        logger=logger
    )
    tokenized_test = dataset_dict["test"]
    logger.info(f"Loaded test dataset with {len(tokenized_test)} examples")

    # Setup trainer
    if args.get("model_type", "seq2seq") == "causal":
        training_args = TrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=args["batch_size"],
            predict_with_generate=True,
            generation_max_length=args["generation_max_length"],
            generation_num_beams=args["num_beams"],
            do_predict=True,
            report_to="none",
            fp16=model_loader.use_fp16,
            bf16=model_loader.use_bf16,
            remove_unused_columns=False
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=args["batch_size"],
            predict_with_generate=True,
            generation_max_length=args["generation_max_length"],
            generation_num_beams=args["num_beams"],
            do_predict=True,
            fp16=model_loader.use_fp16,
            bf16=model_loader.use_bf16,
            remove_unused_columns=False
        )

    metric_helper = HFMetricHelper(
        tokenizer=tokenizer,
        bertscore_model_type="bert-base-multilingual-cased",
        model=model
    )

    if args.get("model_type", "seq2seq") == "causal":
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            compute_metrics=lambda eval_pred: metric_helper.compute(eval_pred, compute_ppl=True)
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=metric_helper.compute
        )

    # Generate predictions and compute metrics
    logger.info("Starting inference...")
    predictions = trainer.predict(tokenized_test)
    metrics = predictions.metrics
    logger.info(f"Evaluation Metrics: {metrics}")

    # Save predictions
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    output_path = Path("data/predictions.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in decoded_preds:
            f.write(pred + "\n")
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    config = utils.return_config("configs/fine_tuning/Meta-Llama-3-8B-Instruct.yaml")
    args = {
        "base_model_path": config['fine_tuning']["base_model"],
        "adapter_path": Path(config['fine_tuning']['output_dir']) / "model",
        "tokenizer_path": Path(config['fine_tuning']['output_dir']) / "tokenizer",
        "dataset_path": "data/leggi_area_3_text",
        "max_length": 256,
        "batch_size": 8,
        "generation_max_length": 256,
        "num_beams": 4,
        "model_type": "causal",
        "offload_to_disk": True,  # Default to disk offloading
        "offload_dir": "./artifacts/offload_dir",  # Default offload directory
        "execution_device": None
    }

    inference(args)