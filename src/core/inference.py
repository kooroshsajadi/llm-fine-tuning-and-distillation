import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
from evaluate import load as load_metric
import torch
import numpy as np

def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)

    # Load and prepare test dataset
    test_dataset = load_dataset(args.dataset_name, split="test")
    # Tokenization logic (similar to training)
    def preprocess(example):
        return tokenizer(example['text'], truncation=True, padding="max_length", max_length=args.max_length)
    tokenized_test = test_dataset.map(preprocess, batched=True)

    # Setup trainer for generation
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_inference",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.num_beams,
        do_predict=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        # No train_dataset
    )

    # Generate predictions & evaluate
    predictions = trainer.predict(tokenized_test)
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    rouge = load_metric("rouge")
    results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print("ROUGE results:", results)

    # Optionally save predictions & results
    with open("predictions.txt", "w", encoding="utf-8") as f:
        for pred in decoded_preds:
            f.write(pred + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of test dataset")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    main(args)
