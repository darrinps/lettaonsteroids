"""
Memory systems: baseline (naive keyword) vs augmented (hybrid retrieval).
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class MemoryMode(str, Enum):
    """Memory operation mode."""
    BASELINE = "baseline"
    MEM0 = "mem0"
    MEM0_ENHANCED = "mem0_enhanced"
    LETTA = "letta"  # Renamed from AUGMENTED for clarity


@dataclass
class MemoryResult:
    """Result from memory query."""
    mode: MemoryMode
    results: List[Dict[str, Any]]
    query: str
    metadata: Dict[str, Any]


class BaselineMemory:
    """
    Baseline memory system using naive keyword matching.
    Simulates Mem0-style simple retrieval.
    """

    def __init__(self):
        """Initialize baseline memory."""
        self.documents: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to baseline memory."""
        self.documents = documents

    def query(self, query: str, top_k: int = 5) -> MemoryResult:
        """
        Simple keyword-based retrieval.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            MemoryResult with matched documents
        """
        # Tokenize query
        query_tokens = set(self._tokenize(query.lower()))

        # Score documents by keyword overlap
        scored_docs = []
        for doc in self.documents:
            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            doc_tokens = set(self._tokenize(text))

            # Simple overlap score
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                scored_docs.append((overlap, doc))

        # Sort by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Get top-k
        results = [
            {
                "doc_id": doc.get('id', doc.get('doc_id')),  # Ensure doc_id key exists
                "title": doc.get('title', ''),
                "content": doc.get('content', ''),
                "score": score,
                "match_type": "keyword"
            }
            for score, doc in scored_docs[:top_k]
        ]

        return MemoryResult(
            mode=MemoryMode.BASELINE,
            results=results,
            query=query,
            metadata={
                "num_results": len(results),
                "total_candidates": len(scored_docs)
            }
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]  # Filter short tokens

    def add_to_conversation(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_context(self, n: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation messages."""
        return self.conversation_history[-n:] if self.conversation_history else []


class Mem0Memory:
    """
    Mem0 memory system using actual Mem0 library.
    Provides vector-based semantic search with conversation context.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize Mem0 memory."""
        try:
            from mem0 import Memory
            import os

            # Use provided key or get from environment
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

            # Configure Mem0 with OpenAI embeddings
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "letta_eval",
                        "embedding_model_dims": 1536,
                        "path": "data/qdrant"  # Local storage
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": api_key
                    }
                }
            }

            self.memory = Memory.from_config(config)
            self.conversation_history: List[Dict[str, str]] = []
            self.documents_added = False

        except ImportError:
            raise RuntimeError("mem0ai library not installed. Run: poetry add mem0ai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mem0: {e}")

    def add_documents(self, documents: List[Dict[str, Any]], progress_callback=None) -> None:
        """Add documents to Mem0 memory."""
        if self.documents_added:
            return

        total_docs = len(documents)
        for i, doc in enumerate(documents, 1):
            if progress_callback:
                progress_callback(f"Adding document {i}/{total_docs} to Mem0: {doc.get('title', 'Untitled')}")

            # Add each document as a memory
            text = f"Title: {doc.get('title', '')}\n\n{doc.get('content', '')}"
            self.memory.add(
                text,
                user_id="system",
                metadata={
                    "doc_id": doc.get('id', doc.get('doc_id')),
                    "title": doc.get('title', ''),
                    "source": "corpus"
                }
            )

        self.documents_added = True
        if progress_callback:
            progress_callback(f"Mem0 initialization complete: {total_docs} documents added")

    def query(self, query: str, top_k: int = 5) -> MemoryResult:
        """
        Search using Mem0's semantic search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            MemoryResult with matched documents
        """
        # Search memories using Mem0
        memories = self.memory.search(
            query,
            user_id="system",
            limit=top_k
        )

        # Format results
        results = []
        for mem in memories:
            # Extract doc_id from metadata
            metadata = mem.get('metadata', {})
            doc_id = metadata.get('doc_id', 'unknown')
            title = metadata.get('title', '')

            # Parse content (Title: ...\n\nContent)
            content = mem.get('memory', '')
            if '\n\n' in content:
                _, content = content.split('\n\n', 1)

            results.append({
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "score": mem.get('score', 0.0),
                "match_type": "mem0_semantic"
            })

        return MemoryResult(
            mode=MemoryMode.MEM0,
            results=results,
            query=query,
            metadata={
                "num_results": len(results),
                "provider": "mem0"
            }
        )

    def add_to_conversation(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_context(self, n: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation messages."""
        return self.conversation_history[-n:] if self.conversation_history else []


class EnhancedMem0Memory:
    """
    Enhanced Mem0 memory system with cross-encoder reranking and retrieval optimization.
    Uses Mem0 for retrieval + applies reranking + heuristic retrieval decision for fair comparison with Letta.
    """

    def __init__(self, mem0_memory, reranker, policy=None):
        """
        Initialize enhanced Mem0 memory.

        Args:
            mem0_memory: Base Mem0Memory instance
            reranker: CrossEncoderReranker instance
            policy: Optional SelfRAGPolicy for retrieval decisions
        """
        self.mem0_memory = mem0_memory
        self.reranker = reranker
        self.policy = policy
        self.conversation_history: List[Dict[str, str]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to Mem0 memory."""
        self.mem0_memory.add_documents(documents)

    def query(self, query: str, top_k: int = 5, use_rerank: bool = True, force_retrieve: bool = False) -> MemoryResult:
        """
        Search using Mem0 with optional reranking and heuristic retrieval decision.

        Args:
            query: Search query
            top_k: Number of results to return
            use_rerank: Whether to apply cross-encoder reranking
            force_retrieve: Skip retrieval decision gate

        Returns:
            MemoryResult with matched documents
        """
        # Check if retrieval is needed (Self-RAG gate) - same as Letta
        should_retrieve = force_retrieve
        decision_info = {}

        if not force_retrieve and self.policy:
            decision = self.policy.should_retrieve(
                query,
                [msg["content"] for msg in self.conversation_history[-3:]]
            )
            should_retrieve = decision.should_retrieve
            decision_info = {
                "should_retrieve": decision.should_retrieve,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            }

        if not should_retrieve:
            # Skip retrieval
            return MemoryResult(
                mode=MemoryMode.MEM0_ENHANCED,
                results=[],
                query=query,
                metadata={
                    "num_results": 0,
                    "retrieval_skipped": True,
                    **decision_info
                }
            )

        # Get more candidates from Mem0 for reranking
        retrieval_k = 8 if use_rerank else top_k
        mem0_result = self.mem0_memory.query(query, top_k=retrieval_k)

        if not use_rerank or not self.reranker or len(mem0_result.results) <= top_k:
            # No reranking needed or not enough results
            return MemoryResult(
                mode=MemoryMode.MEM0_ENHANCED,
                results=mem0_result.results[:top_k],
                query=query,
                metadata={
                    **mem0_result.metadata,
                    "reranking_applied": False,
                    **decision_info
                }
            )

        # Apply cross-encoder reranking
        try:
            # Prepare documents for reranking
            docs_to_rerank = [
                {
                    "doc_id": doc["doc_id"],
                    "content": doc["content"]
                }
                for doc in mem0_result.results
            ]

            # Rerank using cross-encoder
            reranked = self.reranker.rerank(query, docs_to_rerank, top_k=top_k)

            # Format results
            results = []
            for doc in reranked:
                results.append({
                    "doc_id": doc["doc_id"],
                    "title": next((d["title"] for d in mem0_result.results if d["doc_id"] == doc["doc_id"]), ""),
                    "content": doc["content"],
                    "score": doc["score"],
                    "match_type": "mem0_enhanced_reranked"
                })

            return MemoryResult(
                mode=MemoryMode.MEM0_ENHANCED,
                results=results,
                query=query,
                metadata={
                    "num_results": len(results),
                    "provider": "mem0_enhanced",
                    "reranking_applied": True,
                    "initial_candidates": len(mem0_result.results),
                    **decision_info
                }
            )

        except Exception as e:
            # Fallback to non-reranked results on error
            return MemoryResult(
                mode=MemoryMode.MEM0_ENHANCED,
                results=mem0_result.results[:top_k],
                query=query,
                metadata={
                    **mem0_result.metadata,
                    "reranking_applied": False,
                    "reranking_error": str(e),
                    **decision_info
                }
            )

    def add_to_conversation(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        self.mem0_memory.add_to_conversation(role, content)

    def get_conversation_context(self, n: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation messages."""
        return self.conversation_history[-n:] if self.conversation_history else []


class AugmentedMemory:
    """
    Augmented memory system using hybrid retrieval + reranking.
    Integrates with HybridIndex and CrossEncoderReranker.
    """

    def __init__(self, retrieval_tool, policy=None):
        """
        Initialize augmented memory.

        Args:
            retrieval_tool: RetrievalTool instance
            policy: Optional SelfRAGPolicy for retrieval decisions
        """
        self.retrieval_tool = retrieval_tool
        self.policy = policy
        self.conversation_history: List[Dict[str, str]] = []
        self.retrieval_cache: Dict[str, Any] = {}

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        force_retrieve: bool = False
    ) -> MemoryResult:
        """
        Augmented retrieval with optional Self-RAG gating.

        Args:
            query: Search query
            top_k: Number of results
            use_rerank: Whether to use reranking
            force_retrieve: Skip retrieval decision gate

        Returns:
            MemoryResult with retrieved documents
        """
        # Check if retrieval is needed (Self-RAG gate)
        should_retrieve = force_retrieve
        decision_info = {}

        if not force_retrieve and self.policy:
            decision = self.policy.should_retrieve(
                query,
                [msg["content"] for msg in self.conversation_history[-3:]]
            )
            should_retrieve = decision.should_retrieve
            decision_info = {
                "should_retrieve": decision.should_retrieve,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            }

        if not should_retrieve:
            # Skip retrieval
            return MemoryResult(
                mode=MemoryMode.LETTA,
                results=[],
                query=query,
                metadata={
                    "num_results": 0,
                    "retrieval_skipped": True,
                    **decision_info
                }
            )

        # Check cache
        cache_key = f"{query}:{top_k}:{use_rerank}"
        if cache_key in self.retrieval_cache:
            cached = self.retrieval_cache[cache_key]
            return MemoryResult(
                mode=MemoryMode.LETTA,
                results=cached["results"],
                query=query,
                metadata={
                    **cached["metadata"],
                    "from_cache": True,
                    **decision_info
                }
            )

        # Perform retrieval
        tool_result = self.retrieval_tool.retrieve(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            retrieval_k=8
        )

        if not tool_result.success:
            return MemoryResult(
                mode=MemoryMode.LETTA,
                results=[],
                query=query,
                metadata={
                    "error": tool_result.error,
                    **decision_info
                }
            )

        results = tool_result.data

        # Cache results
        self.retrieval_cache[cache_key] = {
            "results": results,
            "metadata": tool_result.metadata
        }

        return MemoryResult(
            mode=MemoryMode.LETTA,
            results=results,
            query=query,
            metadata={
                **tool_result.metadata,
                "from_cache": False,
                **decision_info
            }
        )

    def add_to_conversation(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_context(self, n: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation messages."""
        return self.conversation_history[-n:] if self.conversation_history else []

    def clear_cache(self) -> None:
        """Clear retrieval cache."""
        self.retrieval_cache.clear()
