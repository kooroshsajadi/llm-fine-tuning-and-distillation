from evaluate import load
import numpy as np
import nltk

class HFMetricHelper:
    """
    A modular helper class to compute Hugging Face evaluation metrics such as ROUGE and BERTScore.
    """

    def __init__(self, tokenizer, bertscore_model_type="itcharo/flaubert-base-cased-sentiment"):
        self.tokenizer = tokenizer
        self.rouge = load("rouge")  # Existing ROUGE metric
        self.bertscore = load("bertscore")  # BERTScore metric
        self.bertscore_model_type = bertscore_model_type

    def compute_rouge(self, predictions, references):
        results = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        return {
            "rouge1": results["rouge1"].mid.fmeasure,
            "rouge2": results["rouge2"].mid.fmeasure,
            "rougeL": results["rougeL"].mid.fmeasure,
        }

    def compute_bertscore(self, predictions, references):
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="it",  # Italian language code
            model_type=self.bertscore_model_type,
            rescale_with_baseline=True
        )
        # BERTScore returns Precision, Recall, and F1 lists, we report mean F1
        return {
            "bertscore_f1": float(np.mean(results["f1"]))
        }

    def compute(self, eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        if isinstance(predictions, tuple) or (len(predictions.shape) == 3):
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_metrics = self.compute_rouge(decoded_preds, decoded_labels)
        bertscore_metrics = self.compute_bertscore(decoded_preds, decoded_labels)

        # Combine all metrics to return
        metrics = {**rouge_metrics, **bertscore_metrics}
        return metrics
