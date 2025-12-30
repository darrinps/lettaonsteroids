"""
Self-RAG policies for retrieval control and answer critique.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel
from ..adapters.llm_ollama import OllamaLLM


class RetrievalDecision(BaseModel):
    """Decision on whether to retrieve."""
    should_retrieve: bool
    confidence: float
    reasoning: str


class CritiqueResult(BaseModel):
    """Critique of generated answer."""
    is_supported: bool  # Is the answer supported by context?
    is_useful: bool  # Is the answer useful for the query?
    has_errors: bool  # Does the answer have factual errors?
    feedback: str
    corrected_answer: Optional[str] = None


class SelfRAGPolicy:
    """Self-RAG policy controller for retrieval decisions and answer critique."""

    def __init__(self, llm: OllamaLLM):
        """Initialize Self-RAG policy."""
        self.llm = llm

    def should_retrieve(self, query: str, conversation_history: List[str] = None) -> RetrievalDecision:
        """
        Decide whether retrieval is needed for this query.

        Args:
            query: User query
            conversation_history: Optional conversation context

        Returns:
            RetrievalDecision with reasoning
        """
        # Fast heuristic path: Check for common question patterns
        query_lower = query.lower()
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'describe', 'define']

        # Check for question words or factual queries
        if any(word in query_lower for word in question_words):
            return RetrievalDecision(
                should_retrieve=True,
                confidence=0.9,
                reasoning="Question word detected - likely requires external knowledge retrieval"
            )

        # Check for conversational patterns that don't need retrieval
        conversational = ['hello', 'hi ', 'thanks', 'thank you', 'bye', 'goodbye', 'okay', 'ok']
        if any(word in query_lower for word in conversational) and len(query.split()) < 5:
            return RetrievalDecision(
                should_retrieve=False,
                confidence=0.9,
                reasoning="Conversational query - no external knowledge needed"
            )

        # Fall back to LLM for edge cases
        system_prompt = """You are a retrieval decision system. Determine if external knowledge retrieval is needed to answer the query.

Retrieve if:
- Query asks for specific facts, dates, numbers, or technical details
- Query requires domain knowledge (medical, legal, technical)
- Query asks about entities, people, places, or events

Don't retrieve if:
- Query is conversational or casual (greetings, thanks)
- Query asks for opinions or creative writing
- Query is about simple reasoning or math that doesn't need external facts

Respond in this exact format:
DECISION: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""

        context = ""
        if conversation_history:
            context = f"\n\nConversation history:\n" + "\n".join(conversation_history[-3:])

        prompt = f"""Query: {query}{context}

Should we retrieve external knowledge?"""

        try:
            response = self.llm.generate(prompt, system=system_prompt, temperature=0.3, max_tokens=150)

            # Parse response
            lines = response.strip().split('\n')
            decision_line = next((l for l in lines if l.startswith('DECISION:')), 'DECISION: YES')
            confidence_line = next((l for l in lines if l.startswith('CONFIDENCE:')), 'CONFIDENCE: 0.7')
            reasoning_line = next((l for l in lines if l.startswith('REASONING:')), 'REASONING: Unknown')

            should_retrieve = 'YES' in decision_line.upper()
            confidence = float(confidence_line.split(':')[1].strip())
            reasoning = reasoning_line.split(':', 1)[1].strip()

            return RetrievalDecision(
                should_retrieve=should_retrieve,
                confidence=confidence,
                reasoning=reasoning
            )

        except Exception as e:
            # Default to retrieving on error
            return RetrievalDecision(
                should_retrieve=True,
                confidence=0.5,
                reasoning=f"Error in decision: {e}, defaulting to retrieve"
            )

    def critique_answer(
        self,
        query: str,
        answer: str,
        retrieved_context: List[str]
    ) -> CritiqueResult:
        """
        Critique generated answer for factual accuracy and usefulness.

        Args:
            query: Original query
            answer: Generated answer
            retrieved_context: Retrieved document snippets used

        Returns:
            CritiqueResult with feedback
        """
        system_prompt = """You are an answer quality critic. Evaluate if the answer:
1. Is supported by the provided context (no hallucinations)
2. Is useful and relevant to the query
3. Contains factual errors or contradictions

Be strict but fair. Check for consistency with context."""

        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(retrieved_context[:5])])

        prompt = f"""Query: {query}

Retrieved Context:
{context_text}

Generated Answer:
{answer}

Evaluate the answer:
SUPPORTED: [YES/NO] - Is the answer supported by context?
USEFUL: [YES/NO] - Is the answer useful for the query?
ERRORS: [YES/NO] - Does it have factual errors?
FEEDBACK: [explanation]
CORRECTION: [corrected answer if needed, or 'N/A']"""

        try:
            response = self.llm.generate(prompt, system=system_prompt, temperature=0.2, max_tokens=300)

            # Parse response
            lines = response.strip().split('\n')
            supported_line = next((l for l in lines if l.startswith('SUPPORTED:')), 'SUPPORTED: YES')
            useful_line = next((l for l in lines if l.startswith('USEFUL:')), 'USEFUL: YES')
            errors_line = next((l for l in lines if l.startswith('ERRORS:')), 'ERRORS: NO')
            feedback_line = next((l for l in lines if l.startswith('FEEDBACK:')), 'FEEDBACK: Good answer')
            correction_line = next((l for l in lines if l.startswith('CORRECTION:')), 'CORRECTION: N/A')

            is_supported = 'YES' in supported_line.upper()
            is_useful = 'YES' in useful_line.upper()
            has_errors = 'YES' in errors_line.upper()
            feedback = feedback_line.split(':', 1)[1].strip()
            correction = correction_line.split(':', 1)[1].strip()

            return CritiqueResult(
                is_supported=is_supported,
                is_useful=is_useful,
                has_errors=has_errors,
                feedback=feedback,
                corrected_answer=correction if correction.upper() != 'N/A' else None
            )

        except Exception as e:
            # Default to accepting answer on error
            return CritiqueResult(
                is_supported=True,
                is_useful=True,
                has_errors=False,
                feedback=f"Critique error: {e}",
                corrected_answer=None
            )

    def select_best_answer(
        self,
        query: str,
        candidate_answers: List[str],
        retrieved_context: List[str]
    ) -> int:
        """
        Select best answer from multiple candidates.

        Args:
            query: Original query
            candidate_answers: List of candidate answers
            retrieved_context: Retrieved context

        Returns:
            Index of best answer
        """
        if len(candidate_answers) == 1:
            return 0

        system_prompt = """You are an answer selection system. Choose the best answer based on:
1. Factual accuracy (supported by context)
2. Completeness
3. Relevance to query"""

        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(retrieved_context[:3])])

        candidates_text = "\n\n".join([
            f"ANSWER {i+1}:\n{ans}"
            for i, ans in enumerate(candidate_answers)
        ])

        prompt = f"""Query: {query}

Context:
{context_text}

Candidate Answers:
{candidates_text}

Select the best answer. Respond with only the number (1, 2, 3, etc.)"""

        try:
            response = self.llm.generate(prompt, system=system_prompt, temperature=0.1, max_tokens=10)

            # Extract number
            import re
            match = re.search(r'\d+', response)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(candidate_answers):
                    return idx

        except Exception:
            pass

        # Default to first answer
        return 0
