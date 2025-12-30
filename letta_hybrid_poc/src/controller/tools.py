"""
Tools for Letta controller to interact with retrieval and memory systems.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..retrieval.index import HybridIndex, SearchResult
from ..retrieval.rerank import CrossEncoderReranker


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RetrievalTool:
    """Tool for hybrid retrieval with reranking."""

    def __init__(
        self,
        index: HybridIndex,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        """Initialize retrieval tool."""
        self.index = index
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        retrieval_k: int = 12
    ) -> ToolResult:
        """
        Retrieve relevant documents for query.

        Args:
            query: Search query
            top_k: Number of final results
            use_rerank: Whether to use cross-encoder reranking
            retrieval_k: Number of results before reranking

        Returns:
            ToolResult with retrieved documents
        """
        try:
            # Hybrid retrieval
            results = self.index.search_hybrid(query, top_k=retrieval_k)

            # Smart reranking: only rerank if needed
            if use_rerank and self.reranker and self._should_rerank(results):
                results = self.reranker.rerank(query, results, top_k=top_k)
            else:
                results = results[:top_k]

            # Format results
            documents = [
                {
                    "doc_id": r.doc_id,
                    "title": r.doc.title,
                    "content": r.doc.content,
                    "score": r.score,
                    "source": r.source
                }
                for r in results
            ]

            return ToolResult(
                success=True,
                data=documents,
                metadata={
                    "query": query,
                    "num_results": len(documents),
                    "reranked": use_rerank
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                error=str(e)
            )

    def _should_rerank(self, results: List[SearchResult]) -> bool:
        """
        Decide if reranking is worth the cost.

        Skip reranking when BM25 and FAISS already agree (balanced sources).

        Args:
            results: Search results from hybrid retrieval

        Returns:
            True if reranking should be performed
        """
        if len(results) < 5:
            return True  # Too few results, rerank

        # Check source distribution in top-5
        top_5 = results[:5]
        sources = [r.source for r in top_5]

        # Count how many are primarily from BM25 vs FAISS
        bm25_count = sum(1 for s in sources if 'bm25' in s.lower())
        faiss_count = sum(1 for s in sources if 'faiss' in s.lower())

        # If sources are balanced (2-3 from each), they already agree
        # Reranking won't help much
        if 2 <= bm25_count <= 3:
            return False  # Balanced - skip reranking

        # If very imbalanced (all from one source), reranking can help
        return True

    def retrieve_by_id(self, doc_ids: List[str]) -> ToolResult:
        """
        Retrieve specific documents by ID.

        Args:
            doc_ids: List of document IDs

        Returns:
            ToolResult with documents
        """
        try:
            documents = []
            for doc_id in doc_ids:
                if doc_id in self.index.doc_id_to_idx:
                    idx = self.index.doc_id_to_idx[doc_id]
                    doc = self.index.documents[idx]
                    documents.append({
                        "doc_id": doc.id,
                        "title": doc.title,
                        "content": doc.content
                    })

            return ToolResult(
                success=True,
                data=documents,
                metadata={"num_found": len(documents)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                error=str(e)
            )


class MemoryTool:
    """Tool for interacting with conversation memory."""

    def __init__(self):
        """Initialize memory tool."""
        self.short_term: List[Dict[str, str]] = []
        self.working_context: Dict[str, Any] = {}

    def add_message(self, role: str, content: str) -> ToolResult:
        """Add message to short-term memory."""
        try:
            self.short_term.append({"role": role, "content": content})
            return ToolResult(success=True, data={"added": True})
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    def get_recent(self, n: int = 5) -> ToolResult:
        """Get recent messages from memory."""
        try:
            recent = self.short_term[-n:] if self.short_term else []
            return ToolResult(success=True, data=recent)
        except Exception as e:
            return ToolResult(success=False, data=[], error=str(e))

    def set_context(self, key: str, value: Any) -> ToolResult:
        """Set working context variable."""
        try:
            self.working_context[key] = value
            return ToolResult(success=True, data={"set": key})
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    def get_context(self, key: str) -> ToolResult:
        """Get working context variable."""
        try:
            value = self.working_context.get(key)
            return ToolResult(
                success=True,
                data={"key": key, "value": value},
                metadata={"found": key in self.working_context}
            )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    def clear_short_term(self) -> ToolResult:
        """Clear short-term memory."""
        try:
            count = len(self.short_term)
            self.short_term.clear()
            return ToolResult(success=True, data={"cleared": count})
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Any] = {}

    def register(self, name: str, tool: Any) -> None:
        """Register a tool."""
        self.tools[name] = tool

    def get(self, name: str) -> Optional[Any]:
        """Get a registered tool."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
