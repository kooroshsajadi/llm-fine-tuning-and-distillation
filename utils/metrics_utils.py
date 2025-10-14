from evaluate import load
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu.metrics.chrf import CHRF  # For chrF; install if needed: pip install sacrebleu
from nltk.translate.meteor_score import meteor_score  # For METEOR


class HFMetricHelper:
    """
    A modular helper class to compute evaluation metrics: ROUGE, BLEU, and BERTScore.

    Args:
        tokenizer: The tokenizer used for decoding predictions and labels.
        bertscore_model_type (str): The model type for BERTScore (default: "bert-base-multilingual-cased").
    """

    def __init__(self, tokenizer, bertscore_model_type="bert-base-multilingual-cased", model = None):
        self.tokenizer = tokenizer
        self.rouge = load("rouge")  # Load ROUGE metric
        self.bertscore = load("bertscore")  # Load BERTScore metric
        self.bertscore_model_type = bertscore_model_type
        nltk.download('punkt', quiet=True)  # Ensure NLTK punkt tokenizer is available
        self.model = model  # Optional: Pass causal model for PPL computation
        self.perplexity = None  # Placeholder for PPL
        self.accuracy = load("accuracy")  # For token-level accuracy
        self.meteor = None  # Lazy-load METEOR
        self.chrf = CHRF()  # chrF metric

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


    def compute_perplexity(self, input_ids, labels=None, stride=512, max_length=None):
        """
        Compute perplexity for causal LM using sliding-window approximation.

        Args:
            input_ids (torch.Tensor or np.array): Input token IDs for validation sequences.
            labels (torch.Tensor or np.array, optional): Target IDs; if None, use input_ids shifted.
            stride (int): Window stride to balance speed and context (default: 512 for efficiency).
            max_length (int, optional): Model's max context; auto-detected if model provided.

        Returns:
            dict: Perplexity score (ppl) as a float.
        
        Notes:
            Requires model in eval mode. For cybersecurity fine-tuning, low PPL (< 10-20) indicates
            strong next-token prediction on domain data like logs or reports.
        """
        if self.model is None:
            raise ValueError("Model required for perplexity computation.")
        self.model.eval()
        max_length = max_length or self.model.config.max_position_embeddings
        input_ids = torch.tensor(input_ids) if not torch.is_tensor(input_ids) else input_ids
        if labels is None:
            labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore pads for loss

        total_loss = 0.0
        total_tokens = 0
        for i in range(0, input_ids.size(1), stride):
            end_idx = min(i + max_length, input_ids.size(1))
            segment_ids = input_ids[:, i:end_idx]
            segment_labels = labels[:, i:end_idx]
            
            with torch.no_grad():
                outputs = self.model(segment_ids, labels=segment_labels)
                loss = outputs.loss
                total_loss += loss.item() * (end_idx - i)
                total_tokens += (end_idx - i)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        ppl = np.exp(avg_loss)
        return {"ppl": round(ppl, 4)}

    def compute_exact_match_and_accuracy(self, predictions, references):
        """
        Compute exact match and token-level accuracy post-decoding.

        Args:
            predictions (list): Decoded predictions.
            references (list): Decoded references.

        Returns:
            dict: EM and accuracy as percentages.
        
        Notes:
            EM is strict; use for factual tasks. Accuracy operates on tokenized level for causal eval.
        """
        em_score = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(predictions)
        tokenized_preds = [nltk.word_tokenize(p, language='italian') for p in predictions]
        tokenized_refs = [[nltk.word_tokenize(r, language='italian')] for r in references]
        # Flatten for accuracy; assumes 1:1 token alignment
        flat_preds = [token for seq in tokenized_preds for token in seq]
        flat_refs = [token for seq in tokenized_refs for token in seq[0]]
        if len(flat_preds) == len(flat_refs):
            acc_results = self.accuracy.compute(predictions=flat_preds, references=flat_refs)
        else:
            acc_results = {"accuracy": 0.0}  # Fallback if misaligned
        return {
            "em": round(em_score * 100, 4),
            "accuracy": round(acc_results["accuracy"] * 100, 4)
        }

    def compute_meteor_and_chrf(self, predictions, references):
        """
        Compute METEOR and chrF scores for Italian generation.

        Args:
            predictions (list): Predicted texts.
            references (list): Reference texts.

        Returns:
            dict: METEOR and chrF scores as percentages.
        
        Notes:
            METEOR handles stemming/synonyms; chrF is character-based for morphology.
            Lazy-loads NLTK for METEOR.
        """
        if self.meteor is None:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)  # For Italian support
            self.meteor = True  # Flag for computation

        tokenized_preds = [nltk.word_tokenize(p, language='italian') for p in predictions]
        tokenized_refs = [[nltk.word_tokenize(r, language='italian')] for r in references]

        meteor_scores = [meteor_score(ref, pred) for pred, ref in zip(tokenized_preds, [tr[0] for tr in tokenized_refs])]
        meteor_avg = np.mean(meteor_scores)

        chrf_score = self.chrf.corpus_score(tokenized_preds, [[t] for t in tokenized_refs]).score / 100  # Normalize to [0,1]

        return {
            "meteor": round(meteor_avg * 100, 4),
            "chrf": round(chrf_score * 100, 4)
        }


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