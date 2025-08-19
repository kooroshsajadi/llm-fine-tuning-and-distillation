import argparse
import nltk
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import PeftModel
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from src.data.data_preparation import prepare_dataset_dict
from src.utils import utils

def inference(args):
    """Perform inference on a fine-tuned seq2seq model with LoRA adapters and compute ROUGE scores.

    Args:
        args (dict): Dictionary containing base_model_path, adapter_path, tokenizer_path, dataset_path,
                     max_length, batch_size, generation_max_length, num_beams, and model_type.
    """
    # Ensure NLTK punkt tokenizer for Italian is available
    nltk.download('punkt')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_path"])

    # Load base model from Hugging Face
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args["base_model_path"])
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, args["adapter_path"])
    
    # Set device (xpu, cuda, or cpu)
    device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Prepare data splits using the same config as during training
    dataset_dict = prepare_dataset_dict(
        input_path=args["dataset_path"],
        tokenizer=tokenizer,
        max_length=args["max_length"],
        model_type=args.get("model_type", "seq2seq")
    )

    # Use the pre-tokenized test split
    tokenized_test = dataset_dict["test"]

    # Setup trainer for generation
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_inference",  # Temporary directory for inference outputs
        per_device_eval_batch_size=args["batch_size"],
        predict_with_generate=True,
        generation_max_length=args["generation_max_length"],
        generation_num_beams=args["num_beams"],
        do_predict=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),  # Data collator for batching
    )

    # Generate predictions & evaluate
    predictions = trainer.predict(tokenized_test)
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids

    # Replace -100 in labels with pad_token_id
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Tokenize sentences for ROUGE evaluation (matching training code)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip(), language='italian')) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip(), language='italian')) for label in decoded_labels]

    # Compute ROUGE scores
    rouge = load_metric("rouge")
    results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print("ROUGE results:", {k: round(v * 100, 4) for k, v in results.items()})  # Scale to percentage

    # Save predictions to file
    output_path = Path("predictions.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)  # Create directory with owner-only permissions
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in decoded_preds:
            f.write(pred + "\n")

if __name__ == "__main__":
    config = utils.return_config("configs/fine_tuning/marian_mt.yaml") 
    args = {
        "base_model_path": "Helsinki-NLP/opus-mt-it-en",
        "adapter_path": "models/fine_tuned_models/opus-mt-it-en-v1/model",
        "tokenizer_path": "models/fine_tuned_models/opus-mt-it-en-v1/tokenizer",
        "dataset_path": f"{config['datasets']['prefettura_v1_texts']}",
        "max_length": 128,
        "batch_size": 8,
        "generation_max_length": 128,
        "num_beams": 4,
        "model_type": "seq2seq"
    }

    inference(args)