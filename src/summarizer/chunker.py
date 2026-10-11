import re
from typing import List

class TextChunker:
    """
    Sentence-boundary aware chunker for handling long documents in Transformer models.
    Prevents truncation artifacts and maintains semantic coherence.
    """
    def __init__(self, max_tokens: int = 700, overlap_sentences: int = 1):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex boundary matching."""
        # Handles periods, question marks, exclamation marks followed by whitespace and capital letter or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, text: str, tokenizer=None) -> List[str]:
        """
        Split a long document into chunks that fit within tokenizer limits.
        If tokenizer is provided, use actual token counts; otherwise estimate with words (1 word ~= 1.3 tokens).
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            if tokenizer:
                token_count = len(tokenizer.encode(sentence, truncation=False, add_special_tokens=False))
            else:
                token_count = int(len(sentence.split()) * 1.3)

            if current_len + token_count > self.max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Add overlap from previous chunk if requested
                if self.overlap_sentences > 0:
                    current_chunk = current_chunk[-self.overlap_sentences:]
                    if tokenizer:
                        current_len = sum(len(tokenizer.encode(s, truncation=False, add_special_tokens=False)) for s in current_chunk)
                    else:
                        current_len = sum(int(len(s.split()) * 1.3) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0

            current_chunk.append(sentence)
            current_len += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]
