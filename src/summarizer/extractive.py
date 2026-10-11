import re
import numpy as np
from typing import List, Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.summarizer.chunker import TextChunker

class ExtractiveSummarizer:
    """
    Extractive summarization using:
    - TextRank (Graph-based PageRank on sentence similarity matrix)
    - TF-IDF / LSA scoring (Term frequency - inverse document frequency sentence ranking)
    - Lead-K Baseline (Standard benchmark selecting first K sentences)
    """

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        return TextChunker.split_sentences(text)

    def summarize_lead_k(self, text: str, k: int = 3) -> str:
        """Lead-K: Select the first K sentences (classic strong news baseline)."""
        sentences = self.split_sentences(text)
        return " ".join(sentences[:k])

    def summarize_tfidf(self, text: str, num_sentences: int = 3) -> str:
        """TF-IDF sentence ranking: ranks sentences by aggregate term importance."""
        sentences = self.split_sentences(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Fallback if vocabulary is empty
            return " ".join(sentences[:num_sentences])

        sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        top_indices = sorted(np.argsort(sentence_scores)[-num_sentences:])
        selected_sentences = [sentences[i] for i in top_indices]
        return " ".join(selected_sentences)

    def summarize_textrank(
        self,
        text: str,
        num_sentences: int = 3,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-4
    ) -> str:
        """
        TextRank algorithm: builds sentence graph using cosine similarity of TF-IDF vectors,
        runs PageRank until convergence, and extracts top scoring sentences in original order.
        """
        sentences = self.split_sentences(text)
        n = len(sentences)
        if n <= num_sentences:
            return " ".join(sentences)

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return " ".join(sentences[:num_sentences])

        # Compute pairwise cosine similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Remove self-loops
        np.fill_diagonal(sim_matrix, 0)

        # Normalize rows to form transition probability matrix
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated sentences
        with np.errstate(divide='ignore', invalid='ignore'):
            transition_matrix = np.where(row_sums > 0, sim_matrix / row_sums, 1.0 / n)

        # Power iteration for PageRank
        scores = np.ones(n) / n
        for _ in range(max_iter):
            prev_scores = scores.copy()
            scores = (1 - damping) / n + damping * transition_matrix.T.dot(prev_scores)
            if np.linalg.norm(scores - prev_scores, ord=1) < tol:
                break

        # Select top sentences and preserve original document flow
        top_indices = sorted(np.argsort(scores)[-num_sentences:])
        selected_sentences = [sentences[i] for i in top_indices]
        return " ".join(selected_sentences)

    def summarize(
        self,
        text: str,
        method: Literal["textrank", "lead_k", "tfidf"] = "textrank",
        num_sentences: int = 3
    ) -> str:
        """Unified extractive summarizer entrypoint."""
        if method == "textrank":
            return self.summarize_textrank(text, num_sentences=num_sentences)
        elif method == "lead_k":
            return self.summarize_lead_k(text, k=num_sentences)
        elif method == "tfidf":
            return self.summarize_tfidf(text, num_sentences=num_sentences)
        else:
            raise ValueError(f"Unknown extractive method: {method}")
