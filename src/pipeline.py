import time
import logging
from typing import Dict, Any, List, Optional
from src.summarizer.abstractive import TransformerSummarizer
from src.summarizer.extractive import ExtractiveSummarizer
from src.evaluation.rouge_eval import RougeEvaluator

logger = logging.getLogger(__name__)

class SummarizationPipeline:
    """
    End-to-End Summarization and Comparison Pipeline.
    Supports:
    - Pretrained BART & T5 models
    - Extractive algorithms (TextRank, Lead-K, TF-IDF)
    - Long document handling
    - ROUGE-1, ROUGE-2, ROUGE-L evaluation
    - Comparative analysis (Compression ratio, word count, latency, ROUGE metrics)
    """
    def __init__(
        self,
        abstractive_model_name: str = "sshleifer/distilbart-cnn-12-6",
        enable_t5: bool = False,
        t5_model_name: str = "t5-small"
    ):
        self.extractive_summarizer = ExtractiveSummarizer()
        self.rouge_evaluator = RougeEvaluator()

        self.bart_model_name = abstractive_model_name
        self.bart_summarizer: Optional[TransformerSummarizer] = None

        self.enable_t5 = enable_t5
        self.t5_model_name = t5_model_name
        self.t5_summarizer: Optional[TransformerSummarizer] = None

    def load_models(self):
        """Lazy load or initialize transformer models."""
        if self.bart_summarizer is None:
            logger.info(f"Initializing BART summarizer ({self.bart_model_name})...")
            self.bart_summarizer = TransformerSummarizer(model_name=self.bart_model_name)

        if self.enable_t5 and self.t5_summarizer is None:
            logger.info(f"Initializing T5 summarizer ({self.t5_model_name})...")
            self.t5_summarizer = TransformerSummarizer(model_name=self.t5_model_name)

    def summarize_and_compare(
        self,
        article: str,
        reference_summary: Optional[str] = None,
        abstractive_max_length: int = 142,
        extractive_sentences: int = 3
    ) -> Dict[str, Any]:
        """
        Runs both Extractive and Abstractive summarizers on an input document,
        computes evaluation metrics if reference is provided, and returns full comparative breakdown.
        """
        self.load_models()

        orig_word_count = len(article.split())
        results = {
            "document_stats": {
                "word_count": orig_word_count,
                "character_count": len(article),
            },
            "summaries": {},
            "evaluation": {},
            "reference": reference_summary
        }

        # 1. Extractive: Lead-3
        t0 = time.time()
        lead3_sum = self.extractive_summarizer.summarize(article, method="lead_k", num_sentences=extractive_sentences)
        t_lead3 = round(time.time() - t0, 4)
        results["summaries"]["Extractive (Lead-3)"] = {
            "text": lead3_sum,
            "type": "extractive",
            "word_count": len(lead3_sum.split()),
            "compression_ratio": round(len(lead3_sum.split()) / max(1, orig_word_count), 3),
            "latency_seconds": t_lead3
        }

        # 2. Extractive: TextRank
        t0 = time.time()
        textrank_sum = self.extractive_summarizer.summarize(article, method="textrank", num_sentences=extractive_sentences)
        t_textrank = round(time.time() - t0, 4)
        results["summaries"]["Extractive (TextRank)"] = {
            "text": textrank_sum,
            "type": "extractive",
            "word_count": len(textrank_sum.split()),
            "compression_ratio": round(len(textrank_sum.split()) / max(1, orig_word_count), 3),
            "latency_seconds": t_textrank
        }

        # 3. Abstractive: BART
        if self.bart_summarizer is not None:
            t0 = time.time()
            bart_sum = self.bart_summarizer.summarize(
                article,
                max_length=abstractive_max_length,
                hierarchical_long_text=True
            )
            t_bart = round(time.time() - t0, 3)
            results["summaries"][f"Abstractive ({self.bart_model_name.split('/')[-1]})"] = {
                "text": bart_sum,
                "type": "abstractive",
                "word_count": len(bart_sum.split()),
                "compression_ratio": round(len(bart_sum.split()) / max(1, orig_word_count), 3),
                "latency_seconds": t_bart
            }

        # 4. Abstractive: T5 (if enabled)
        if self.t5_summarizer is not None:
            t0 = time.time()
            t5_sum = self.t5_summarizer.summarize(
                article,
                max_length=abstractive_max_length,
                hierarchical_long_text=True
            )
            t_t5 = round(time.time() - t0, 3)
            results["summaries"][f"Abstractive ({self.t5_model_name.split('/')[-1]})"] = {
                "text": t5_sum,
                "type": "abstractive",
                "word_count": len(t5_sum.split()),
                "compression_ratio": round(len(t5_sum.split()) / max(1, orig_word_count), 3),
                "latency_seconds": t_t5
            }

        # Compute ROUGE scores if reference summary is available
        if reference_summary and reference_summary.strip():
            for method_name, sum_data in results["summaries"].items():
                pred_text = sum_data["text"]
                rouge_scores = self.rouge_evaluator.evaluate_single(
                    reference=reference_summary,
                    prediction=pred_text
                )
                results["evaluation"][method_name] = rouge_scores

        return results
