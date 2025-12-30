"""
Letta controller - orchestrates memory, retrieval, and LLM generation.
"""
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
import time
from .memory import BaselineMemory, Mem0Memory, EnhancedMem0Memory, AugmentedMemory, MemoryMode
from ..adapters.llm_ollama import OllamaLLM
from ..controller.policies import SelfRAGPolicy


@dataclass
class ChatResponse:
    """Response from Letta controller."""
    answer: str
    mode: MemoryMode
    retrieved_docs: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    latency_ms: float


class LettaController:
    """
    Main controller orchestrating baseline vs augmented chat modes.
    """

    def __init__(
        self,
        llm: OllamaLLM,
        baseline_memory: BaselineMemory,
        augmented_memory: AugmentedMemory,
        mem0_memory: Optional[Mem0Memory] = None,
        mem0_enhanced_memory: Optional[EnhancedMem0Memory] = None,
        policy: Optional[SelfRAGPolicy] = None,
        mode: MemoryMode = MemoryMode.LETTA
    ):
        """
        Initialize Letta controller.

        Args:
            llm: LLM adapter
            baseline_memory: Baseline memory system
            augmented_memory: Augmented/Letta memory system
            mem0_memory: Mem0 memory system (optional)
            mem0_enhanced_memory: Enhanced Mem0 with reranking (optional)
            policy: Self-RAG policy (optional)
            mode: Operating mode (baseline, mem0, mem0_enhanced, or letta)
        """
        self.llm = llm
        self.baseline_memory = baseline_memory
        self.mem0_memory = mem0_memory
        self.mem0_enhanced_memory = mem0_enhanced_memory
        self.augmented_memory = augmented_memory
        self.policy = policy
        self.mode = mode

    def set_mode(self, mode: MemoryMode) -> None:
        """Switch between baseline and augmented modes."""
        self.mode = mode

    def chat(
        self,
        query: str,
        top_k: int = 5,
        use_critique: bool = True,
        temperature: float = 0.7,
        force_retrieve: bool = False
    ) -> ChatResponse:
        """
        Process chat query using current mode.

        Args:
            query: User query
            top_k: Number of retrieved docs
            use_critique: Whether to use Self-RAG critique
            temperature: LLM temperature
            force_retrieve: Skip heuristic retrieval decision (for evaluation)

        Returns:
            ChatResponse with answer and metadata
        """
        start_time = time.time()

        # Select memory system based on mode
        if self.mode == MemoryMode.BASELINE:
            memory_result = self.baseline_memory.query(query, top_k=top_k)
            memory = self.baseline_memory
        elif self.mode == MemoryMode.MEM0:
            if not self.mem0_memory:
                raise ValueError("Mem0 memory not initialized")
            memory_result = self.mem0_memory.query(query, top_k=top_k)
            memory = self.mem0_memory
        elif self.mode == MemoryMode.MEM0_ENHANCED:
            if not self.mem0_enhanced_memory:
                raise ValueError("Enhanced Mem0 memory not initialized")
            memory_result = self.mem0_enhanced_memory.query(
                query,
                top_k=top_k,
                use_rerank=True,
                force_retrieve=force_retrieve
            )
            memory = self.mem0_enhanced_memory
        else:  # LETTA (or AUGMENTED for backward compatibility)
            memory_result = self.augmented_memory.query(
                query,
                top_k=top_k,
                use_rerank=True,
                force_retrieve=force_retrieve
            )
            memory = self.augmented_memory

        # Get conversation context
        conv_context = memory.get_conversation_context(n=3)

        # Generate answer
        answer = self._generate_answer(
            query,
            memory_result.results,
            conv_context,
            temperature
        )

        # Critique answer if enabled (for enhanced modes)
        critique_result = None
        enhanced_modes = [MemoryMode.MEM0_ENHANCED, MemoryMode.LETTA]
        if use_critique and self.mode in enhanced_modes and self.policy:
            if memory_result.results and self._needs_critique(answer, memory_result.results):
                retrieved_texts = [
                    f"{doc['title']}: {doc['content'][:200]}"
                    for doc in memory_result.results[:3]
                ]
                critique_result = self.policy.critique_answer(
                    query,
                    answer,
                    retrieved_texts
                )

                # Use corrected answer if provided
                if critique_result.corrected_answer and critique_result.has_errors:
                    answer = critique_result.corrected_answer

        # Update conversation history
        memory.add_to_conversation("user", query)
        memory.add_to_conversation("assistant", answer)

        latency_ms = (time.time() - start_time) * 1000

        # Build metadata
        metadata = {
            "mode": self.mode.value,
            "num_retrieved": len(memory_result.results),
            "retrieval_metadata": memory_result.metadata,
        }

        if critique_result:
            metadata["critique"] = {
                "is_supported": critique_result.is_supported,
                "is_useful": critique_result.is_useful,
                "has_errors": critique_result.has_errors,
                "feedback": critique_result.feedback
            }

        return ChatResponse(
            answer=answer,
            mode=self.mode,
            retrieved_docs=memory_result.results,
            metadata=metadata,
            latency_ms=latency_ms
        )

    def _generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        conversation: List[Dict[str, str]],
        temperature: float
    ) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            conversation: Recent conversation
            temperature: LLM temperature

        Returns:
            Generated answer
        """
        # Build context from retrieved docs
        if retrieved_docs:
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:5], 1):
                title = doc.get('title', 'Untitled')
                content = doc.get('content', '')
                context_parts.append(f"[{i}] {title}\n{content}\n")
            context = "\n".join(context_parts)
        else:
            context = "No relevant documents found."

        # Build conversation context
        conv_text = ""
        if conversation:
            conv_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in conversation[-3:]
            ])

        # System prompt
        system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.

Instructions:
- Use information from the context documents to answer
- Be specific and cite relevant details
- If the context doesn't contain the answer, say so clearly
- Be concise but complete
- If multiple documents are relevant, synthesize the information"""

        # User prompt
        user_prompt = f"""Context Documents:
{context}

{f"Previous conversation:{conv_text}" if conv_text else ""}

Question: {query}

Answer:"""

        try:
            answer = self.llm.generate(
                user_prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=512
            )
            return answer.strip()

        except Exception as e:
            return f"Error generating answer: {e}"

    def _needs_critique(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
        """
        Decide if critique is needed for this answer.

        Skip critique for:
        - Short factual answers (< 50 words)
        - High-confidence retrieval (top doc score > 0.8)

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents with scores

        Returns:
            True if critique is needed
        """
        # Short answers are usually safe
        word_count = len(answer.split())
        if word_count < 50:
            return False

        # High confidence retrieval doesn't need critique
        if retrieved_docs and len(retrieved_docs) >= 3:
            top_score = retrieved_docs[0].get('score', 0)
            if top_score > 0.8:
                return False

        return True  # Otherwise critique

    def reset_conversation(self) -> None:
        """Reset conversation history in both memory systems."""
        self.baseline_memory.conversation_history.clear()
        self.augmented_memory.conversation_history.clear()
        self.augmented_memory.clear_cache()
