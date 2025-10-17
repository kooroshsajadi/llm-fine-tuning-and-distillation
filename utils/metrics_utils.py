from evaluate import load
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu.metrics.chrf import CHRF  # Install if needed: pip install sacrebleu
from nltk.translate.meteor_score import meteor_score
import torch


class HFMetricHelper:
    """
        A modular helper class to compute evaluation metrics for text generation tasks: ROUGE, BLEU, BERTScore,
        Exact Match (EM), Accuracy, METEOR, chrF, and Perplexity (PPL). Tailored for Italian text evaluation.

        Args:
            tokenizer: The tokenizer used for decoding predictions and labels.
            bertscore_model_type (str): The model type for BERTScore (default: "bert-base-multilingual-cased").
            model: Optional pretrained model for perplexity computation (e.g., causal LM).
    """

    def __init__(self, tokenizer, bertscore_model_type="bert-base-multilingual-cased", model=None):
        self.tokenizer = tokenizer
        self.rouge = load("rouge")  # Load ROUGE metric from the `evaluate` library
        self.bertscore = load("bertscore")  # Load BERTScore metric from the `evaluate` library
        self.bertscore_model_type = bertscore_model_type
        nltk.download('punkt', quiet=True)  # Download NLTK punkt tokenizer for sentence splitting
        self.model = model  # Optional: Pretrained model for perplexity computation
        self.perplexity = None  # Placeholder for PPL (not used directly)
        self.accuracy = load("accuracy")  # Load accuracy metric for token-level evaluation
        self.meteor = None  # Flag for lazy-loading METEOR dependencies
        self.chrf = CHRF()  # Initialize chrF metric for character-based evaluation

    def compute_rouge(self, predictions, references):
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L scores to evaluate text similarity.

        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between predicted and reference texts:
        - ROUGE-1: Unigram overlap (word-level similarity).
        - ROUGE-2: Bigram overlap (phrase-level similarity).
        - ROUGE-L: Longest Common Subsequence (captures sentence structure).

        Args:
            predictions (list): List of predicted text strings.
            references (list): List of reference text strings.

        Returns:
            dict: ROUGE scores (rouge1, rouge2, rougeL) as percentages (0-100).
        """
        # Sentence-tokenize predictions and references for ROUGE, joining with newlines as expected by the metric
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
        Compute corpus-level BLEU score to evaluate translation or generation quality.

        BLEU (Bilingual Evaluation Understudy) measures n-gram precision between predicted and reference texts,
        with a brevity penalty to discourage overly short outputs. It is commonly used for machine translation
        and text generation, focusing on exact word matches (1-4 grams).

        Args:
            predictions (list): List of predicted text strings.
            references (list): List of reference text strings.

        Returns:
            dict: BLEU score (bleu) as a percentage (0-100).
        """
        # Tokenize predictions and references for BLEU, ensuring Italian-specific word tokenization
        tokenized_preds = [nltk.word_tokenize(pred, language='italian') for pred in predictions]
        tokenized_refs = [[nltk.word_tokenize(ref, language='italian')] for ref in references]
        
        # Apply smoothing to handle short sequences or zero n-gram overlaps
        smoothing = SmoothingFunction().method1
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
        return {"bleu": round(bleu_score * 100, 4)}

    def compute_bertscore(self, predictions, references):
        """
        Compute BERTScore F1 to evaluate semantic similarity.

        BERTScore uses contextual embeddings from a pretrained language model (e.g., BERT) to compute
        cosine similarity between predicted and reference texts. It captures semantic meaning beyond
        exact word matches, making it robust for paraphrasing and semantic equivalence.

        Args:
            predictions (list): List of predicted text strings.
            references (list): List of reference text strings.

        Returns:
            dict: BERTScore F1 score (bertscore_f1) as a percentage (0-100).
        """
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="it",  # Specify Italian language for better tokenization
            model_type=self.bertscore_model_type,
            rescale_with_baseline=True  # Rescale scores for interpretability
        )
        return {"bertscore_f1": round(float(np.mean(results["f1"])) * 100, 4)}


    def compute_perplexity(self, input_ids, labels=None, stride=512, max_length=None):
        """
        Compute perplexity for causal language models to measure next-token prediction quality.

        Perplexity (PPL) quantifies how well a language model predicts a sequence, calculated as the
        exponentiated average negative log-likelihood of the sequence. Lower PPL indicates better
        prediction confidence. Low PPL (<10-20) suggests the model is well-adapted to
        domain-specific data (e.g., logs, reports).

        Args:
            input_ids (torch.Tensor or np.array): Input token IDs for validation sequences.
            labels (torch.Tensor or np.array, optional): Target token IDs; defaults to input_ids.
            stride (int): Sliding window stride for efficient processing of long sequences (default: 512).
            max_length (int, optional): Model's maximum context length; auto-detected if model provided.

        Returns:
            dict: Perplexity score (ppl) as a float.
        """
        if self.model is None:
            raise ValueError("Model required for perplexity computation.")
        self.model.eval()
        max_length = max_length or self.model.config.max_position_embeddings
        input_ids = torch.tensor(input_ids) if not torch.is_tensor(input_ids) else input_ids
        if labels is None:
            labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss

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
        Compute Exact Match (EM) and token-level accuracy for evaluating factual correctness.

        - Exact Match (EM): Measures the percentage of predictions that exactly match the reference
          text (case-insensitive, ignoring leading/trailing whitespace). Ideal for tasks requiring
          precise outputs, like question answering or factual generation.
        - Token-level Accuracy: Measures the percentage of correctly predicted tokens after
          tokenization, useful for evaluating partial correctness in causal language models.

        Args:
            predictions (list): List of predicted text strings.
            references (list): List of reference text strings.

        Returns:
            dict: EM and accuracy scores as percentages (0-100).
        """
        # Compute Exact Match score
        em_score = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(predictions)
        # Tokenize for accuracy computation
        tokenized_preds = [nltk.word_tokenize(p, language='italian') for p in predictions]
        tokenized_refs = [[nltk.word_tokenize(r, language='italian')] for r in references]
        # Flatten tokenized sequences for token-level accuracy
        flat_preds = [token for seq in tokenized_preds for token in seq]
        flat_refs = [token for seq in tokenized_refs for token in seq[0]]
        # Compute accuracy if token counts align; otherwise, return 0
        if len(flat_preds) == len(flat_refs):
            acc_results = self.accuracy.compute(predictions=flat_preds, references=flat_refs)
        else:
            acc_results = {"accuracy": 0.0}  # Fallback for misaligned sequences
        return {
            "em": round(em_score * 100, 4),
            "accuracy": round(acc_results["accuracy"] * 100, 4)
        }

    def compute_meteor_and_chrf(self, predictions, references):
        """
        Compute METEOR and chrF scores for advanced text generation evaluation.

        - METEOR (Metric for Evaluation of Translation with Explicit ORdering): Measures text similarity
          by considering unigram matches, stemming, and synonyms (via WordNet for Italian). It balances
          precision, recall, and word order, making it robust for translation and generation tasks.
        - chrF (Character n-gram F-score): A character-based metric that evaluates n-gram overlap at the
          character level, effective for morphologically rich languages like Italian where word-level
          metrics may miss nuances.

        Args:
            predictions (list): List of predicted text strings.
            references (list): List of reference text strings.

        Returns:
            dict: METEOR and chrF scores as percentages (0-100).
        """
        if self.meteor is None:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)  # For Italian WordNet support
            self.meteor = True  # Flag to avoid redundant downloads

        # Tokenize predictions and references for METEOR
        tokenized_preds = [nltk.word_tokenize(p, language='italian') for p in predictions]
        tokenized_refs = [[nltk.word_tokenize(r, language='italian')] for r in references]

        # Compute METEOR score for each prediction-reference pair
        meteor_scores = [meteor_score(ref[0], pred) for pred, ref in zip(tokenized_preds, tokenized_refs)]
        meteor_avg = np.mean(meteor_scores)

        # Compute chrF score using sacrebleu (expects tokenized input for consistency)
        chrf_score = self.chrf.corpus_score(tokenized_preds, [[t[0]] for t in tokenized_refs]).score / 100  # Normalize to [0,1]

        return {
            "meteor": round(meteor_avg * 100, 4),
            "chrf": round(chrf_score * 100, 4)
        }

    def compute(self, eval_pred, compute_ppl=False, input_ids=None):
        """
        Compute multiple evaluation metrics (ROUGE, BLEU, BERTScore, EM, Accuracy, METEOR, chrF, and
        optionally Perplexity) for model predictions.

        Args:
            eval_pred: A tuple or object containing predictions (logits or token IDs) and label_ids
                       (e.g., from Seq2SeqTrainer or Trainer).
            compute_ppl (bool): Whether to compute perplexity (requires input_ids and model).
            input_ids (torch.Tensor, optional): Input token IDs for perplexity computation.

        Returns:
            dict: Combined metrics including rouge1, rouge2, rougeL, bleu, bertscore_f1, em, accuracy,
                  meteor, chrf, and optionally ppl.

        Notes:
            - Handles both sequence-to-sequence (logits) and causal LM (token IDs) predictions.
            - Decodes predictions and labels using the tokenizer, skipping special tokens.
            - Perplexity requires a model and input_ids; disabled by default.
        """
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # Handle sequence-to-sequence model outputs (logits) or causal LM outputs (token IDs)
        if isinstance(predictions, tuple) or (len(predictions.shape) == 3):
            # Convert logits to token IDs by taking the argmax
            predictions = np.argmax(predictions, axis=-1)

        # Replace -100 in labels (used for ignoring loss) with pad_token_id for decoding
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode predictions and labels to text, removing special tokens
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute all metrics
        rouge_metrics = self.compute_rouge(decoded_preds, decoded_labels)
        bleu_metrics = self.compute_bleu(decoded_preds, decoded_labels)
        bertscore_metrics = self.compute_bertscore(decoded_preds, decoded_labels)
        em_acc_metrics = self.compute_exact_match_and_accuracy(decoded_preds, decoded_labels)
        meteor_chrf_metrics = self.compute_meteor_and_chrf(decoded_preds, decoded_labels)

        # Combine all metrics into a single dictionary
        all_metrics = {
            **rouge_metrics,
            **bleu_metrics,
            **bertscore_metrics,
            **em_acc_metrics,
            **meteor_chrf_metrics
        }

        # Compute perplexity if requested and input_ids/model are provided
        if compute_ppl and input_ids is not None and self.model is not None:
            ppl_metrics = self.compute_perplexity(input_ids)
            all_metrics.update(ppl_metrics)

        return all_metrics