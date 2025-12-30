"""
Async evaluation runner with streaming support.
"""
import asyncio
import time
import queue
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from ..eval.eval import (
    load_system,
    evaluate_query,
    EVAL_QUERIES,
    calculate_recall_at_k
)


class EvalStatus(str, Enum):
    """Evaluation status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EvalState:
    """Current evaluation state."""
    status: EvalStatus
    current_query: Optional[int] = None
    total_queries: int = 0
    results: List[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "current_query": self.current_query,
            "total_queries": self.total_queries,
            "completed_queries": len(self.results),
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class EvalRunner:
    """Async evaluation runner with streaming callbacks."""

    def __init__(self):
        """Initialize evaluation runner."""
        self.state = EvalState(status=EvalStatus.IDLE)
        self.callbacks: List[Callable] = []
        self._stop_flag = False
        self._task: Optional[asyncio.Task] = None

    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for streaming events."""
        self.callbacks.append(callback)

    async def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all callbacks."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                print(f"Callback error: {e}")

    async def start_evaluation(
        self,
        provider: str = "openai",
        index_dir: Path = Path("data/index"),
        openai_model: str = "gpt-4o-mini",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mistral:latest",
        top_k: int = 5
    ):
        """
        Start evaluation in background.

        Args:
            provider: LLM provider (openai/ollama)
            index_dir: Index directory
            openai_model: OpenAI model name
            ollama_url: Ollama API URL
            ollama_model: Ollama model name
            top_k: Number of documents to retrieve
        """
        if self.state.status == EvalStatus.RUNNING:
            raise RuntimeError("Evaluation already running")

        self._stop_flag = False
        self._task = asyncio.create_task(
            self._run_evaluation(
                provider=provider,
                index_dir=index_dir,
                openai_model=openai_model,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                top_k=top_k
            )
        )

        return self._task

    async def _run_evaluation(
        self,
        provider: str,
        index_dir: Path,
        openai_model: str,
        ollama_url: str,
        ollama_model: str,
        top_k: int
    ):
        """Run evaluation with streaming."""
        try:
            # Update state
            self.state = EvalState(
                status=EvalStatus.RUNNING,
                total_queries=len(EVAL_QUERIES),
                started_at=time.time()
            )

            await self._emit("evaluation_started", {
                "total_queries": len(EVAL_QUERIES),
                "provider": provider,
                "model": openai_model if provider == "openai" else ollama_model
            })

            # Load system (run in thread pool to not block)
            await self._emit("system_loading", {})
            await self._emit("log", {"message": "Starting system initialization..."})
            await self._emit("log", {"message": "This may take 1-2 minutes on first run. Please wait..."})

            # Create queue for progress messages from load_system
            progress_queue = queue.Queue()

            # Start load_system in background thread
            load_task = asyncio.create_task(
                asyncio.to_thread(
                    load_system,
                    index_dir=index_dir,
                    provider=provider,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    openai_model=openai_model,
                    progress_callback=lambda msg: progress_queue.put(msg)
                )
            )

            # Poll queue and emit messages while load_system is running
            while not load_task.done():
                try:
                    msg = progress_queue.get_nowait()
                    await self._emit("log", {"message": msg})
                except queue.Empty:
                    await asyncio.sleep(0.1)

            # Get result
            baseline_controller, mem0_controller, mem0_enhanced_controller, letta_controller = await load_task

            # Emit any remaining messages
            while not progress_queue.empty():
                msg = progress_queue.get_nowait()
                await self._emit("log", {"message": msg})

            await self._emit("log", {"message": "✓ All system components loaded successfully!"})
            await self._emit("system_loaded", {})

            # Run queries
            results = []
            for i, query_data in enumerate(EVAL_QUERIES):
                if self._stop_flag:
                    self.state.status = EvalStatus.STOPPED
                    await self._emit("evaluation_stopped", {
                        "completed_queries": len(results)
                    })
                    return

                # Update state
                self.state.current_query = i + 1

                await self._emit("query_started", {
                    "index": i + 1,
                    "total": len(EVAL_QUERIES),
                    "query": query_data["query"],
                    "category": query_data["category"]
                })

                # Evaluate query (run in thread pool)
                await self._emit("log", {"message": f"Starting evaluation for query {i+1}: {query_data['query']}"})
                try:
                    # Create queue for progress messages from evaluate_query
                    eval_progress_queue = queue.Queue()

                    # Start evaluate_query in background thread
                    eval_task = asyncio.create_task(
                        asyncio.to_thread(
                            evaluate_query,
                            query_data=query_data,
                            baseline_controller=baseline_controller,
                            mem0_controller=mem0_controller,
                            mem0_enhanced_controller=mem0_enhanced_controller,
                            letta_controller=letta_controller,
                            top_k=top_k,
                            progress_callback=lambda msg: eval_progress_queue.put(msg)
                        )
                    )

                    # Poll queue and emit messages while evaluate_query is running
                    while not eval_task.done():
                        try:
                            msg = eval_progress_queue.get_nowait()
                            await self._emit("log", {"message": msg})
                        except queue.Empty:
                            await asyncio.sleep(0.05)

                    # Get result
                    query_results = await eval_task

                    # Emit any remaining messages
                    while not eval_progress_queue.empty():
                        msg = eval_progress_queue.get_nowait()
                        await self._emit("log", {"message": msg})

                    await self._emit("log", {"message": f"Completed evaluation for query {i+1}"})
                except Exception as e:
                    print(f"DEBUG: Error evaluating query {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

                result_entry = {
                    "query": query_data["query"],
                    "category": query_data["category"],
                    "relevant_docs": query_data["relevant_docs"],
                    **query_results
                }

                results.append(result_entry)
                self.state.results = results

                # Prepare emit data with all available systems
                emit_data = {
                    "index": i + 1,
                    "total": len(EVAL_QUERIES),
                    "query": query_data["query"],
                    "baseline": query_results["baseline"],
                    "letta": query_results["letta"]
                }

                if query_results.get("mem0") is not None:
                    emit_data["mem0"] = query_results["mem0"]

                if query_results.get("mem0_enhanced") is not None:
                    emit_data["mem0_enhanced"] = query_results["mem0_enhanced"]

                await self._emit("query_completed", emit_data)

            # Calculate aggregate metrics
            await self._emit("log", {"message": f"Calculating aggregate metrics for {len(results)} results"})
            try:
                baseline_latencies = [r["baseline"]["latency_ms"] for r in results]
                letta_latencies = [r["letta"]["latency_ms"] for r in results]
                baseline_recalls = [r["baseline"]["recall_at_k"] for r in results]
                letta_recalls = [r["letta"]["recall_at_k"] for r in results]

                aggregate_metrics = {
                    "baseline": {
                        "avg_latency_ms": sum(baseline_latencies) / len(baseline_latencies),
                        "avg_recall_at_k": sum(baseline_recalls) / len(baseline_recalls),
                        "total_queries": len(results)
                    },
                    "letta": {
                        "avg_latency_ms": sum(letta_latencies) / len(letta_latencies),
                        "avg_recall_at_k": sum(letta_recalls) / len(letta_recalls),
                        "total_queries": len(results)
                    }
                }
            except Exception as e:
                print(f"DEBUG: Error calculating baseline/letta metrics: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Add Mem0 Basic metrics if available
            if mem0_controller and all(r.get("mem0") is not None for r in results):
                await self._emit("log", {"message": "Calculating Mem0 Basic metrics"})
                mem0_latencies = [r["mem0"]["latency_ms"] for r in results]
                mem0_recalls = [r["mem0"]["recall_at_k"] for r in results]
                aggregate_metrics["mem0"] = {
                    "avg_latency_ms": sum(mem0_latencies) / len(mem0_latencies),
                    "avg_recall_at_k": sum(mem0_recalls) / len(mem0_recalls),
                    "total_queries": len(results)
                }

            # Add Mem0 Enhanced metrics if available
            if mem0_enhanced_controller and all(r.get("mem0_enhanced") is not None for r in results):
                await self._emit("log", {"message": "Calculating Mem0 Enhanced metrics"})
                mem0_enh_latencies = [r["mem0_enhanced"]["latency_ms"] for r in results]
                mem0_enh_recalls = [r["mem0_enhanced"]["recall_at_k"] for r in results]
                aggregate_metrics["mem0_enhanced"] = {
                    "avg_latency_ms": sum(mem0_enh_latencies) / len(mem0_enh_latencies),
                    "avg_recall_at_k": sum(mem0_enh_recalls) / len(mem0_enh_recalls),
                    "total_queries": len(results)
                }

            # Update state
            self.state.status = EvalStatus.COMPLETED
            self.state.completed_at = time.time()

            await self._emit("log", {"message": "Finalizing results..."})
            await self._emit("evaluation_completed", {
                "aggregate_metrics": aggregate_metrics,
                "total_queries": len(results),
                "duration_seconds": self.state.completed_at - self.state.started_at
            })

        except Exception as e:
            self.state.status = EvalStatus.ERROR
            self.state.error = str(e)
            await self._emit("evaluation_error", {
                "error": str(e)
            })

    def stop_evaluation(self):
        """Stop running evaluation."""
        if self.state.status == EvalStatus.RUNNING:
            self._stop_flag = True

    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return self.state.to_dict()

    def get_results(self) -> List[Dict[str, Any]]:
        """Get completed results."""
        return self.state.results
