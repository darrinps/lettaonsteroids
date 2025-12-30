"""
Cross-encoder reranking for retrieved documents.
"""
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from .index import SearchResult


class CrossEncoderReranker:
    """Rerank search results using cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """Initialize cross-encoder reranker."""
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Query string
            results: List of search results to rerank
            top_k: Return top-k results (None = return all)

        Returns:
            Reranked list of search results
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result.doc.get_text()) for result in results]

        # Get reranking scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Create new results with updated scores
        reranked = []
        for result, score in zip(results, scores):
            reranked_result = SearchResult(
                doc_id=result.doc_id,
                score=float(score),
                doc=result.doc,
                source=f"{result.source}_reranked"
            )
            reranked.append(reranked_result)

        # Sort by new scores
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Return top-k if specified
        if top_k is not None:
            return reranked[:top_k]

        return reranked

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores
        """
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist()
