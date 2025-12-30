"""
Hybrid retrieval index combining BM25 (sparse) and FAISS (dense) search.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation."""
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def get_text(self) -> str:
        """Get full text for indexing."""
        return f"{self.title} {self.content}"


@dataclass
class SearchResult:
    """Search result with score."""
    doc_id: str
    score: float
    doc: Document
    source: str  # 'bm25', 'faiss', or 'hybrid'


class HybridIndex:
    """Hybrid retrieval index combining BM25 and FAISS."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """Initialize hybrid index."""
        self.embedding_model_name = embedding_model
        self.device = device
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Storage
        self.documents: List[Document] = []
        self.doc_id_to_idx: Dict[str, int] = {}

        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        # FAISS index
        self.faiss_index: Optional[faiss.Index] = None
        self.doc_embeddings: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        start_idx = len(self.documents)

        for idx, doc in enumerate(documents):
            self.documents.append(doc)
            self.doc_id_to_idx[doc.id] = start_idx + idx

        # Build BM25 index
        self.tokenized_corpus = [self._tokenize(doc.get_text()) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build FAISS index
        texts = [doc.get_text() for doc in self.documents]
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        self.doc_embeddings = embeddings.astype('float32')

        # Create FAISS index (using flat L2 for CPU-friendly exact search)
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(self.doc_embeddings)

    def search_bm25(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using BM25 sparse retrieval."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return docs with positive scores
                results.append(SearchResult(
                    doc_id=self.documents[idx].id,
                    score=float(scores[idx]),
                    doc=self.documents[idx],
                    source='bm25'
                ))

        return results

    def search_faiss(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using FAISS dense retrieval."""
        if self.faiss_index is None:
            return []

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                # Convert L2 distance to similarity score (inverse)
                score = 1.0 / (1.0 + dist)
                results.append(SearchResult(
                    doc_id=self.documents[idx].id,
                    score=float(score),
                    doc=self.documents[idx],
                    source='faiss'
                ))

        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 12,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
    ) -> List[SearchResult]:
        """Hybrid search combining BM25 and FAISS with score fusion."""
        # Get results from both retrievers (fetch more to allow for fusion)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        faiss_results = self.search_faiss(query, top_k=top_k * 2)

        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score
            if score_range == 0:
                return {r.doc_id: 1.0 for r in results}
            return {
                r.doc_id: (r.score - min_score) / score_range
                for r in results
            }

        bm25_norm = normalize_scores(bm25_results)
        faiss_norm = normalize_scores(faiss_results)

        # Combine scores
        all_doc_ids = set(bm25_norm.keys()) | set(faiss_norm.keys())
        combined_scores = {}

        for doc_id in all_doc_ids:
            bm25_score = bm25_norm.get(doc_id, 0.0)
            faiss_score = faiss_norm.get(doc_id, 0.0)
            combined_scores[doc_id] = (
                bm25_weight * bm25_score + faiss_weight * faiss_score
            )

        # Sort by combined score
        sorted_doc_ids = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Build results
        results = []
        for doc_id, score in sorted_doc_ids:
            idx = self.doc_id_to_idx[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                doc=self.documents[idx],
                source='hybrid'
            ))

        return results

    def save(self, index_dir: Path) -> None:
        """Save index to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save documents
        with open(index_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save doc ID mapping
        with open(index_dir / "doc_id_map.pkl", "wb") as f:
            pickle.dump(self.doc_id_to_idx, f)

        # Save BM25 tokenized corpus
        with open(index_dir / "tokenized_corpus.pkl", "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))

        # Save embeddings
        if self.doc_embeddings is not None:
            np.save(index_dir / "embeddings.npy", self.doc_embeddings)

        # Save config
        config = {
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
        }
        with open(index_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, index_dir: Path) -> None:
        """Load index from disk."""
        # Load documents
        with open(index_dir / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        # Load doc ID mapping
        with open(index_dir / "doc_id_map.pkl", "rb") as f:
            self.doc_id_to_idx = pickle.load(f)

        # Load BM25 tokenized corpus
        with open(index_dir / "tokenized_corpus.pkl", "rb") as f:
            self.tokenized_corpus = pickle.load(f)

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

        # Load embeddings
        self.doc_embeddings = np.load(index_dir / "embeddings.npy")

        # Load config (for verification)
        with open(index_dir / "config.json", "r") as f:
            config = json.load(f)

        print(f"Loaded index with {config['num_documents']} documents")
