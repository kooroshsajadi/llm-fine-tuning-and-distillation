from evaluate import load
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class HFMetricHelper:
    """
    A modular helper class to compute evaluation metrics: ROUGE, BLEU, and BERTScore.

    Args:
        tokenizer: The tokenizer used for decoding predictions and labels.
        bertscore_model_type (str): The model type for BERTScore (default: "bert-base-multilingual-cased").
    """

    def __init__(self, tokenizer, bertscore_model_type="bert-base-multilingual-cased"):
        self.tokenizer = tokenizer
        self.rouge = load("rouge")  # Load ROUGE metric
        self.bertscore = load("bertscore")  # Load BERTScore metric
        self.bertscore_model_type = bertscore_model_type
        nltk.download('punkt', quiet=True)  # Ensure NLTK punkt tokenizer is available

    def compute_rouge(self, predictions, references):
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

        Args:
            predictions (list): List of predicted texts.
            references (list): List of reference texts.

        Returns:
            dict: ROUGE scores (rouge1, rouge2, rougeL) as percentages.
        """
        predictions = ["\n".join(nltk.sent_tokenize(pred.strip(), language='italian')) for pred in predictions]
        references = ["\n".join(nltk.sent_tokenize(ref.strip(), language='italian')) for ref in references]
        
        results = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        return {
            "rouge1": round(results["rouge1"] * 100, 4),
            "rouge2": round(results["rouge2"] * 100, 4),
            "rougeL": round(results["rougeL"] * 100, 4),
        }

    def compute_bleu(self, predictions, references):
        """
        Compute corpus-level BLEU score.

        Args:
            predictions (list): List of predicted texts.
            references (list): List of reference texts.

        Returns:
            dict: BLEU score (bleu) as a percentage.
        """
        tokenized_preds = [nltk.word_tokenize(pred, language='italian') for pred in predictions]
        tokenized_refs = [[nltk.word_tokenize(ref, language='italian')] for ref in references]
        
        smoothing = SmoothingFunction().method1
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
        return {"bleu": round(bleu_score * 100, 4)}

    def compute_bertscore(self, predictions, references):
        """
        Compute BERTScore F1 score.

        Args:
            predictions (list): List of predicted texts.
            references (list): List of reference texts.

        Returns:
            dict: BERTScore F1 score (bertscore_f1) as a percentage.
        """
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="it",
            model_type=self.bertscore_model_type,
            rescale_with_baseline=True
        )
        return {"bertscore_f1": round(float(np.mean(results["f1"])) * 100, 4)}

    def compute(self, eval_pred):
        """
        Compute ROUGE, BLEU, and BERTScore metrics for evaluation predictions.

        Args:
            eval_pred: An object containing predictions and label_ids (e.g., from Seq2SeqTrainer.predict).

        Returns:
            dict: Combined metrics (rouge1, rouge2, rougeL, bleu, bertscore_f1).
        """
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        if isinstance(predictions, tuple) or (len(predictions.shape) == 3):
            predictions = np.argmax(predictions, axis=-1)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_metrics = self.compute_rouge(decoded_preds, decoded_labels)
        bleu_metrics = self.compute_bleu(decoded_preds, decoded_labels)
        bertscore_metrics = self.compute_bertscore(decoded_preds, decoded_labels)

        return {**rouge_metrics, **bleu_metrics, **bertscore_metrics}