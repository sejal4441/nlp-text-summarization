from typing import List, Dict, Any, Union
import re
from collections import Counter

class RougeEvaluator:
    """
    Evaluates generated summaries against reference gold-standard summaries.
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics (Precision, Recall, F1).
    Supports either the standard `rouge_score` package or built-in exact formula computation.
    """
    def __init__(self, use_stemmer: bool = True):
        self.use_stemmer = use_stemmer
        self._rouge_scorer = None
        try:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=use_stemmer
            )
        except ImportError:
            self._rouge_scorer = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple clean tokenization: lowercase alphanumeric tokens."""
        return re.findall(r'\b\w+\b', text.lower())

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Extract n-grams and counts."""
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Longest Common Subsequence length using dynamic programming."""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if seq1[i - 1] == seq2[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def _calc_prf(self, hits: int, pred_len: int, ref_len: int) -> Dict[str, float]:
        precision = hits / pred_len if pred_len > 0 else 0.0
        recall = hits / ref_len if ref_len > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fmeasure": round(f1, 4)
        }

    def evaluate_single(self, reference: str, prediction: str) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L for a single (reference, prediction) pair.
        Returns precision, recall, and fmeasure for each metric.
        """
        if self._rouge_scorer is not None:
            scores = self._rouge_scorer.score(reference, prediction)
            return {
                "rouge1": {
                    "precision": round(scores["rouge1"].precision, 4),
                    "recall": round(scores["rouge1"].recall, 4),
                    "fmeasure": round(scores["rouge1"].fmeasure, 4),
                },
                "rouge2": {
                    "precision": round(scores["rouge2"].precision, 4),
                    "recall": round(scores["rouge2"].recall, 4),
                    "fmeasure": round(scores["rouge2"].fmeasure, 4),
                },
                "rougeL": {
                    "precision": round(scores["rougeL"].precision, 4),
                    "recall": round(scores["rougeL"].recall, 4),
                    "fmeasure": round(scores["rougeL"].fmeasure, 4),
                }
            }

        # Fallback pure-python ROUGE calculation
        ref_tokens = self._tokenize(reference)
        pred_tokens = self._tokenize(prediction)

        # ROUGE-1
        ref_uni = self._get_ngrams(ref_tokens, 1)
        pred_uni = self._get_ngrams(pred_tokens, 1)
        overlap_1 = sum((ref_uni & pred_uni).values())
        r1 = self._calc_prf(overlap_1, len(pred_tokens), len(ref_tokens))

        # ROUGE-2
        ref_bi = self._get_ngrams(ref_tokens, 2)
        pred_bi = self._get_ngrams(pred_tokens, 2)
        overlap_2 = sum((ref_bi & pred_bi).values())
        r2 = self._calc_prf(overlap_2, max(0, len(pred_tokens) - 1), max(0, len(ref_tokens) - 1))

        # ROUGE-L
        lcs_len = self._lcs_length(ref_tokens, pred_tokens)
        rl = self._calc_prf(lcs_len, len(pred_tokens), len(ref_tokens))

        return {"rouge1": r1, "rouge2": r2, "rougeL": rl}

    def evaluate_batch(
        self,
        references: List[str],
        predictions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean ROUGE-1, ROUGE-2, and ROUGE-L across a batch of documents.
        """
        if len(references) != len(predictions):
            raise ValueError("Number of references and predictions must match.")

        if not references:
            return {}

        accum = {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }

        n = len(references)
        for ref, pred in zip(references, predictions):
            score = self.evaluate_single(ref, pred)
            for metric in ["rouge1", "rouge2", "rougeL"]:
                for key in ["precision", "recall", "fmeasure"]:
                    accum[metric][key] += score[metric][key]

        mean_scores = {}
        for metric, vals in accum.items():
            mean_scores[metric] = {
                k: round(v / n, 4) for k, v in vals.items()
            }

        return mean_scores

    @staticmethod
    def format_scores_table(results_dict: Dict[str, Dict[str, Dict[str, float]]]) -> str:
        """
        Formats a dictionary of {model_name: rouge_results} into an ASCII/Markdown table.
        """
        header = "| Method / Model | ROUGE-1 F1 | ROUGE-1 Rec | ROUGE-2 F1 | ROUGE-2 Rec | ROUGE-L F1 | ROUGE-L Rec |"
        separator = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
        lines = [header, separator]

        for model_name, metrics in results_dict.items():
            r1 = metrics.get("rouge1", {})
            r2 = metrics.get("rouge2", {})
            rl = metrics.get("rougeL", {})
            row = (
                f"| **{model_name}** | "
                f"{r1.get('fmeasure', 0.0):.4f} | {r1.get('recall', 0.0):.4f} | "
                f"{r2.get('fmeasure', 0.0):.4f} | {r2.get('recall', 0.0):.4f} | "
                f"{rl.get('fmeasure', 0.0):.4f} | {rl.get('recall', 0.0):.4f} |"
            )
            lines.append(row)

        return "\n".join(lines)
