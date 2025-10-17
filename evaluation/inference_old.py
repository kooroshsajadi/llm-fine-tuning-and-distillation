import torch
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForLanguageModeling, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM, Trainer,
                          TrainingArguments)
from peft import PeftModel
from data.data_preparation import prepare_dataset_dict
from utils import utils
from utils.logging_utils import setup_logger
from utils.metrics_utils import HFMetricHelper
from model.model_loader import ModelLoader


logger = setup_logger('evaluation.inference')

def inference(args):
    """
    Perform inference on a fine-tuned seq2seq model with LoRA adapters and compute ROUGE, BLEU, and BERTScore metrics.

    Args:
        args (dict): Dictionary containing base_model_path, adapter_path, tokenizer_path, dataset_path,
                     max_length, batch_size, generation_max_length, num_beams, and model_type.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_path"])
    logger.info(f"Loaded tokenizer from {args['tokenizer_path']}")

    # Load base model and LoRA adapters
    if args.get("model_type", "seq2seq") == "causal":
        base_model = AutoModelForCausalLM.from_pretrained(args["base_model_path"])
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args["base_model_path"])
    model = PeftModel.from_pretrained(base_model, args["adapter_path"])
    logger.info(f"Loaded model from {args['base_model_path']} with adapters from {args['adapter_path']}")

    # Set device
    device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model moved to {device}")

    # Prepare test dataset
    dataset_dict = prepare_dataset_dict(
        input_path=args["dataset_path"],
        tokenizer=tokenizer,
        max_length=args["max_length"],
        model_type=args.get("model_type", "seq2seq"),
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
            fp16=(device.type == "cuda"),  # Enable mixed precision for CUDA
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
        )

    metric_helper = HFMetricHelper(tokenizer=tokenizer,
                                   bertscore_model_type="bert-base-multilingual-cased",
                                   model=model)

    if args.get("model_type", "seq2seq") == "causal":
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            # processing_class=tokenizer,  # Updated to avoid FutureWarning
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            compute_metrics=lambda eval_pred: metric_helper.compute(eval_pred, compute_ppl=True)
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,  # Updated to avoid FutureWarning
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=metric_helper.compute
    )

    # Generate predictions and compute metrics
    logger.info("Starting inference...")
    predictions = trainer.predict(tokenized_test)
    metrics = metric_helper.compute(predictions, compute_ppl=True, input_ids=predictions.label_ids)
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
        "model_type": "causal"
    }

    inference(args)