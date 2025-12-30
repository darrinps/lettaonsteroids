"""
Evaluation script comparing baseline vs augmented modes.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from dotenv import load_dotenv
from ..adapters.llm_ollama import OllamaLLM, OllamaConfig
from ..adapters.llm_openai import OpenAILLM, OpenAIConfig
from ..retrieval.index import HybridIndex
from ..retrieval.rerank import CrossEncoderReranker
from ..controller.tools import RetrievalTool
from ..controller.policies import SelfRAGPolicy
from ..core.memory import BaselineMemory, Mem0Memory, EnhancedMem0Memory, AugmentedMemory, MemoryMode
from ..core.letta_controller import LettaController

# Load environment variables
load_dotenv()

console = Console()


# Evaluation queries with expected relevant doc IDs
EVAL_QUERIES = [
    {
        "query": "What is Cedar policy language used for?",
        "relevant_docs": ["doc_001", "doc_002", "doc_011"],
        "category": "technical"
    },
    {
        "query": "How do I ensure my website is ADA compliant?",
        "relevant_docs": ["doc_003", "doc_004", "doc_013"],
        "category": "compliance"
    },
    {
        "query": "What are the different types of dermal fillers?",
        "relevant_docs": ["doc_012"],
        "category": "medical"
    },
    {
        "query": "How long does Botox treatment last?",
        "relevant_docs": ["doc_006"],
        "category": "medical"
    },
    {
        "query": "What is the difference between BM25 and FAISS?",
        "relevant_docs": ["doc_014", "doc_015"],
        "category": "technical"
    },
    {
        "query": "What are WCAG 2.1 Level AA contrast requirements?",
        "relevant_docs": ["doc_004"],
        "category": "compliance"
    },
    {
        "query": "How does CoolSculpting work?",
        "relevant_docs": ["doc_007"],
        "category": "medical"
    },
    {
        "query": "What laser types are used for hair removal?",
        "relevant_docs": ["doc_009"],
        "category": "medical"
    },
]


def load_system(
    index_dir: Path,
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "mistral:latest",
    openai_model: str = "gpt-4o-mini",
    progress_callback=None
):
    """Load system components."""
    def log(message: str):
        """Log to console and optionally to callback."""
        console.print(f"[dim]{message}[/dim]")
        if progress_callback:
            progress_callback(message)

    # Load index
    log("Loading document index...")
    index = HybridIndex()
    index.load(index_dir)
    log(f"Loaded index with {len(index.documents)} documents")

    # Initialize LLM based on provider
    if provider == "openai":
        log(f"Using OpenAI ({openai_model})...")
        llm_config = OpenAIConfig(model=openai_model)
        try:
            llm = OpenAILLM(config=llm_config)
            if not llm.is_available():
                raise RuntimeError("OpenAI API not available")
        except Exception as e:
            console.print(f"[bold red]Error: Cannot connect to OpenAI: {e}[/bold red]")
            console.print(f"[yellow]Make sure OPENAI_API_KEY is set in .env file[/yellow]")
            raise typer.Exit(1)
    else:  # ollama
        llm_config = OllamaConfig(base_url=ollama_url, model=ollama_model)
        llm = OllamaLLM(config=llm_config)

        if not llm.is_available():
            console.print(f"[bold red]Error: Cannot connect to Ollama at {ollama_url}[/bold red]")
            raise typer.Exit(1)

    # Initialize reranker
    log("Initializing cross-encoder reranker...")
    try:
        reranker = CrossEncoderReranker()
    except:
        reranker = None
        console.print("[yellow]Warning: Reranker not available[/yellow]")

    # Initialize retrieval tool
    log("Setting up retrieval tools...")
    retrieval_tool = RetrievalTool(index=index, reranker=reranker)

    # Initialize policy
    log("Initializing Self-RAG policy...")
    policy = SelfRAGPolicy(llm=llm)

    # Initialize memory systems
    log("Initializing memory systems...")
    baseline_memory = BaselineMemory()
    baseline_memory.add_documents([
        {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content
        }
        for doc in index.documents
    ])

    augmented_memory = AugmentedMemory(
        retrieval_tool=retrieval_tool,
        policy=policy
    )

    # Initialize Mem0 memory systems
    log("Initializing Mem0 (Please wait...slow!)...")
    mem0_memory = None
    mem0_enhanced_memory = None

    try:
        mem0_memory = Mem0Memory()
        log(f"Adding {len(index.documents)} documents to Mem0 (this involves API calls for embeddings)...")
        mem0_memory.add_documents(
            [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content
                }
                for doc in index.documents
            ],
            progress_callback=log
        )
        log("Mem0 memory initialized successfully")

        # Create enhanced Mem0 with reranking and policy (for fair comparison with Letta)
        mem0_enhanced_memory = EnhancedMem0Memory(
            mem0_memory=mem0_memory,
            reranker=reranker,
            policy=policy
        )
        log("Enhanced Mem0 memory initialized successfully")
    except Exception as e:
        log(f"Warning: Could not initialize Mem0: {e}")
        console.print("[yellow]Continuing without Mem0 comparisons[/yellow]")

    # Create controllers for all four modes
    baseline_controller = LettaController(
        llm=llm,
        baseline_memory=baseline_memory,
        augmented_memory=augmented_memory,
        mem0_memory=None,
        mem0_enhanced_memory=None,
        policy=policy,
        mode=MemoryMode.BASELINE
    )

    mem0_controller = None
    if mem0_memory:
        mem0_controller = LettaController(
            llm=llm,
            baseline_memory=BaselineMemory(),
            augmented_memory=AugmentedMemory(retrieval_tool=retrieval_tool, policy=policy),
            mem0_memory=mem0_memory,
            mem0_enhanced_memory=None,
            policy=policy,
            mode=MemoryMode.MEM0
        )

    mem0_enhanced_controller = None
    if mem0_enhanced_memory:
        mem0_enhanced_controller = LettaController(
            llm=llm,
            baseline_memory=BaselineMemory(),
            augmented_memory=AugmentedMemory(retrieval_tool=retrieval_tool, policy=policy),
            mem0_memory=None,
            mem0_enhanced_memory=mem0_enhanced_memory,
            policy=policy,
            mode=MemoryMode.MEM0_ENHANCED
        )

    letta_controller = LettaController(
        llm=llm,
        baseline_memory=BaselineMemory(),
        augmented_memory=AugmentedMemory(retrieval_tool=retrieval_tool, policy=policy),
        mem0_memory=None,
        mem0_enhanced_memory=None,
        policy=policy,
        mode=MemoryMode.LETTA
    )

    # Add documents to baseline memories
    letta_controller.baseline_memory.add_documents([
        {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content
        }
        for doc in index.documents
    ])

    return baseline_controller, mem0_controller, mem0_enhanced_controller, letta_controller


def calculate_recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """Calculate recall@k metric."""
    if not relevant_doc_ids:
        return 0.0

    retrieved_set = set(retrieved_doc_ids[:k])
    relevant_set = set(relevant_doc_ids)

    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def evaluate_query(
    query_data: Dict[str, Any],
    baseline_controller: LettaController,
    mem0_controller: Optional[LettaController],
    mem0_enhanced_controller: Optional[LettaController],
    letta_controller: LettaController,
    top_k: int = 5,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Evaluate single query on all four modes.

    Note: force_retrieve=True is used for all systems to ensure consistent retrieval
    for recall measurement. The heuristic retrieval decision (skip for simple queries)
    is active in production/chat mode but bypassed during evaluation.
    """
    query = query_data["query"]
    relevant_docs = query_data["relevant_docs"]

    def log(message: str):
        """Log to console and optionally to callback."""
        print(f"DEBUG: {message}")
        if progress_callback:
            progress_callback(message)

    results = {}

    # Evaluate baseline (force_retrieve=True to skip heuristic decision during eval)
    log(f"Evaluating baseline for: {query}")
    baseline_response = baseline_controller.chat(query, top_k=top_k, use_critique=False, force_retrieve=True)
    log("Baseline complete")
    baseline_doc_ids = [doc["doc_id"] for doc in baseline_response.retrieved_docs]
    baseline_recall = calculate_recall_at_k(baseline_doc_ids, relevant_docs, top_k)

    results["baseline"] = {
        "answer": baseline_response.answer,
        "latency_ms": baseline_response.latency_ms,
        "num_retrieved": len(baseline_response.retrieved_docs),
        "recall_at_k": baseline_recall,
        "retrieved_doc_ids": baseline_doc_ids[:top_k]
    }

    # Evaluate Mem0 Basic (if available)
    if mem0_controller:
        log("Evaluating Mem0 Basic")
        mem0_response = mem0_controller.chat(query, top_k=top_k, use_critique=False, force_retrieve=True)
        log("Mem0 Basic complete")
        mem0_doc_ids = [doc["doc_id"] for doc in mem0_response.retrieved_docs]
        mem0_recall = calculate_recall_at_k(mem0_doc_ids, relevant_docs, top_k)

        results["mem0"] = {
            "answer": mem0_response.answer,
            "latency_ms": mem0_response.latency_ms,
            "num_retrieved": len(mem0_response.retrieved_docs),
            "recall_at_k": mem0_recall,
            "retrieved_doc_ids": mem0_doc_ids[:top_k]
        }
    else:
        results["mem0"] = None

    # Evaluate Mem0 Enhanced (if available)
    if mem0_enhanced_controller:
        log("Evaluating Mem0 Enhanced")
        mem0_enh_response = mem0_enhanced_controller.chat(query, top_k=top_k, use_critique=True, force_retrieve=True)
        log("Mem0 Enhanced complete")
        mem0_enh_doc_ids = [doc["doc_id"] for doc in mem0_enh_response.retrieved_docs]
        mem0_enh_recall = calculate_recall_at_k(mem0_enh_doc_ids, relevant_docs, top_k)

        results["mem0_enhanced"] = {
            "answer": mem0_enh_response.answer,
            "latency_ms": mem0_enh_response.latency_ms,
            "num_retrieved": len(mem0_enh_response.retrieved_docs),
            "recall_at_k": mem0_enh_recall,
            "retrieved_doc_ids": mem0_enh_doc_ids[:top_k],
            "metadata": mem0_enh_response.metadata
        }
    else:
        results["mem0_enhanced"] = None

    # Evaluate Letta (Hybrid + Enhancements)
    log("Evaluating Letta")
    letta_response = letta_controller.chat(query, top_k=top_k, use_critique=True, force_retrieve=True)
    log("Letta complete")
    letta_doc_ids = [doc["doc_id"] for doc in letta_response.retrieved_docs]
    letta_recall = calculate_recall_at_k(letta_doc_ids, relevant_docs, top_k)

    results["letta"] = {
        "answer": letta_response.answer,
        "latency_ms": letta_response.latency_ms,
        "num_retrieved": len(letta_response.retrieved_docs),
        "recall_at_k": letta_recall,
        "retrieved_doc_ids": letta_doc_ids[:top_k],
        "metadata": letta_response.metadata
    }

    return results


def main(
    provider: str = typer.Option(
        "openai",
        "--provider",
        "-p",
        help="LLM provider: openai or ollama"
    ),
    index_dir: Path = typer.Option(
        "data/index",
        "--index-dir",
        "-i",
        help="Index directory"
    ),
    openai_model: str = typer.Option(
        "gpt-4o-mini",
        "--openai-model",
        help="OpenAI model name"
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434",
        "--ollama-url",
        help="Ollama API URL"
    ),
    ollama_model: str = typer.Option(
        "mistral:latest",
        "--ollama-model",
        help="Ollama model name"
    ),
    output: Path = typer.Option(
        "eval_results.json",
        "--output",
        "-o",
        help="Output JSON file"
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of documents to retrieve"
    ),
):
    """
    Run evaluation comparing baseline vs augmented modes.

    Example:
        poetry run python -m src.eval.eval run --provider openai
        poetry run python -m src.eval.eval run --provider ollama
    """
    model_name = openai_model if provider == "openai" else ollama_model

    console.print("[bold blue]Starting Evaluation[/bold blue]")
    console.print(f"[dim]Provider: {provider.upper()}[/dim]")
    console.print(f"[dim]Model: {model_name}[/dim]")
    console.print(f"[dim]Queries: {len(EVAL_QUERIES)}[/dim]")
    console.print()

    # Load system
    console.print("[yellow]Loading system...[/yellow]")
    baseline_controller, mem0_controller, mem0_enhanced_controller, letta_controller = load_system(
        index_dir,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        openai_model=openai_model
    )
    console.print("[green]SUCCESS: System loaded[/green]")
    console.print()

    # Run evaluation
    results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(
            "Evaluating queries...",
            total=len(EVAL_QUERIES)
        )

        for query_data in EVAL_QUERIES:
            query_results = evaluate_query(
                query_data,
                baseline_controller,
                mem0_controller,
                mem0_enhanced_controller,
                letta_controller,
                top_k=top_k
            )

            results.append({
                "query": query_data["query"],
                "category": query_data["category"],
                "relevant_docs": query_data["relevant_docs"],
                **query_results
            })

            progress.update(task, advance=1)

    # Calculate aggregate metrics
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

    # Add Mem0 Basic metrics if available
    if mem0_controller and all(r["mem0"] is not None for r in results):
        mem0_latencies = [r["mem0"]["latency_ms"] for r in results]
        mem0_recalls = [r["mem0"]["recall_at_k"] for r in results]
        aggregate_metrics["mem0"] = {
            "avg_latency_ms": sum(mem0_latencies) / len(mem0_latencies),
            "avg_recall_at_k": sum(mem0_recalls) / len(mem0_recalls),
            "total_queries": len(results)
        }

    # Add Mem0 Enhanced metrics if available
    if mem0_enhanced_controller and all(r.get("mem0_enhanced") is not None for r in results):
        mem0_enh_latencies = [r["mem0_enhanced"]["latency_ms"] for r in results]
        mem0_enh_recalls = [r["mem0_enhanced"]["recall_at_k"] for r in results]
        aggregate_metrics["mem0_enhanced"] = {
            "avg_latency_ms": sum(mem0_enh_latencies) / len(mem0_enh_latencies),
            "avg_recall_at_k": sum(mem0_enh_recalls) / len(mem0_enh_recalls),
            "total_queries": len(results)
        }

    # Display results table
    console.print()
    console.print("[bold blue]Evaluation Results[/bold blue]")
    console.print()

    # Create table based on available systems
    if "mem0_enhanced" in aggregate_metrics:
        # Four-way comparison: Baseline vs Mem0 (Basic) vs Mem0 (Enhanced) vs Letta
        table = Table(title="Four-Way Fair Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline\n(Keyword)", style="yellow", justify="center")
        table.add_column("Mem0\n(Basic)", style="blue", justify="center")
        table.add_column("Mem0\n(+Rerank+Critique)", style="magenta", justify="center")
        table.add_column("Letta\n(Hybrid+All)", style="green", justify="center")

        # Latency row
        baseline_latency = aggregate_metrics["baseline"]["avg_latency_ms"]
        mem0_latency = aggregate_metrics["mem0"]["avg_latency_ms"]
        mem0_enh_latency = aggregate_metrics["mem0_enhanced"]["avg_latency_ms"]
        letta_latency = aggregate_metrics["letta"]["avg_latency_ms"]

        table.add_row(
            "Avg Latency (ms)",
            f"{baseline_latency:.1f}",
            f"{mem0_latency:.1f}",
            f"{mem0_enh_latency:.1f}",
            f"{letta_latency:.1f}"
        )

        # Recall row
        baseline_recall = aggregate_metrics["baseline"]["avg_recall_at_k"]
        mem0_recall = aggregate_metrics["mem0"]["avg_recall_at_k"]
        mem0_enh_recall = aggregate_metrics["mem0_enhanced"]["avg_recall_at_k"]
        letta_recall = aggregate_metrics["letta"]["avg_recall_at_k"]

        table.add_row(
            f"Avg Recall@{top_k}",
            f"{baseline_recall:.3f}",
            f"{mem0_recall:.3f}",
            f"{mem0_enh_recall:.3f}",
            f"{letta_recall:.3f}"
        )
    elif "mem0" in aggregate_metrics:
        # Three-way: Baseline vs Mem0 vs Letta (no enhanced)
        table = Table(title="Three-Way Comparison: Baseline vs Mem0 vs Letta")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("Mem0", style="blue")
        table.add_column("Letta", style="green")

        baseline_latency = aggregate_metrics["baseline"]["avg_latency_ms"]
        mem0_latency = aggregate_metrics["mem0"]["avg_latency_ms"]
        letta_latency = aggregate_metrics["letta"]["avg_latency_ms"]

        table.add_row(
            "Avg Latency (ms)",
            f"{baseline_latency:.1f}",
            f"{mem0_latency:.1f}",
            f"{letta_latency:.1f}"
        )

        baseline_recall = aggregate_metrics["baseline"]["avg_recall_at_k"]
        mem0_recall = aggregate_metrics["mem0"]["avg_recall_at_k"]
        letta_recall = aggregate_metrics["letta"]["avg_recall_at_k"]

        table.add_row(
            f"Avg Recall@{top_k}",
            f"{baseline_recall:.3f}",
            f"{mem0_recall:.3f}",
            f"{letta_recall:.3f}"
        )
    else:
        # Two-way: Just Baseline vs Letta
        table = Table(title="Baseline vs Letta Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("Letta", style="green")
        table.add_column("Improvement", style="magenta")

        baseline_latency = aggregate_metrics["baseline"]["avg_latency_ms"]
        letta_latency = aggregate_metrics["letta"]["avg_latency_ms"]
        latency_diff = ((letta_latency - baseline_latency) / baseline_latency) * 100

        table.add_row(
            "Avg Latency (ms)",
            f"{baseline_latency:.1f}",
            f"{letta_latency:.1f}",
            f"{latency_diff:+.1f}%"
        )

        baseline_recall = aggregate_metrics["baseline"]["avg_recall_at_k"]
        letta_recall = aggregate_metrics["letta"]["avg_recall_at_k"]
        recall_improvement = ((letta_recall - baseline_recall) / baseline_recall) * 100 if baseline_recall > 0 else 0

        table.add_row(
            f"Avg Recall@{top_k}",
            f"{baseline_recall:.3f}",
            f"{letta_recall:.3f}",
            f"{recall_improvement:+.1f}%"
        )

    console.print(table)

    # Save results
    output_data = {
        "metadata": {
            "provider": provider,
            "model": model_name,
            "top_k": top_k,
            "num_queries": len(EVAL_QUERIES),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "aggregate_metrics": aggregate_metrics,
        "query_results": results
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print()
    console.print(f"[green]SUCCESS: Results saved to {output}[/green]")


if __name__ == "__main__":
    typer.run(main)
