import logging
from typing import List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from src.summarizer.chunker import TextChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerSummarizer:
    """
    Abstractive summarizer using pretrained Transformer models (BART / T5).
    Features:
    - Support for BART (facebook/bart-large-cnn, sshleifer/distilbart-cnn-12-6)
    - Support for T5 (t5-small, t5-base, t5-large)
    - Smart hierarchical map-reduce chunking for long text documents exceeding model context limits
    - Tunable beam search parameters
    """
    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        device: Optional[str] = None,
        max_chunk_tokens: int = 768
    ):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading summarization model '{self.model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.is_t5 = "t5" in self.model_name.lower()
        self.chunker = TextChunker(max_tokens=max_chunk_tokens)
        logger.info(f"Model '{self.model_name}' loaded successfully.")

    def summarize_chunk(
        self,
        text: str,
        min_length: int = 30,
        max_length: int = 142,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """Summarize a single text chunk within token limits."""
        if self.is_t5 and not text.strip().startswith("summarize:"):
            input_text = "summarize: " + text.strip()
        else:
            input_text = text.strip()

        inputs = self.tokenizer(
            input_text,
            max_length=1024 if not self.is_t5 else 512,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=num_beams,
                min_length=min_length,
                max_length=max_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary.strip()

    def summarize(
        self,
        text: str,
        min_length: int = 40,
        max_length: int = 150,
        num_beams: int = 4,
        hierarchical_long_text: bool = True
    ) -> str:
        """
        Generate summary for short or long documents.
        If document exceeds chunk token threshold and hierarchical_long_text is True,
        chunks the document, summarizes each chunk (Map step), and then summarizes
        the combined chunk summaries (Reduce step) if still too long.
        """
        text = text.strip()
        if not text:
            return ""

        tokens = self.tokenizer.encode(text, truncation=False)
        model_limit = 1024 if not self.is_t5 else 512

        # Direct summarization if short enough
        if len(tokens) <= model_limit:
            return self.summarize_chunk(
                text,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams
            )

        if not hierarchical_long_text:
            # Fallback to simple truncation
            return self.summarize_chunk(
                text,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams
            )

        # Map-Reduce Hierarchical Summarization for Long Documents
        chunks = self.chunker.chunk_document(text, tokenizer=self.tokenizer)
        logger.info(f"Document has {len(tokens)} tokens. Split into {len(chunks)} chunks for hierarchical summarization.")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # Target summary per chunk
            chunk_summary = self.summarize_chunk(
                chunk,
                min_length=max(20, min_length // len(chunks)),
                max_length=max(50, max_length // len(chunks) * 2),
                num_beams=num_beams
            )
            chunk_summaries.append(chunk_summary)

        combined_summary = " ".join(chunk_summaries)
        combined_tokens = self.tokenizer.encode(combined_summary, truncation=False)

        # If combined chunk summaries are still long, run a final Reduce pass
        if len(combined_tokens) > max_length:
            logger.info("Executing Reduce summarization pass on combined chunk summaries.")
            return self.summarize_chunk(
                combined_summary,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams
            )

        return combined_summary
