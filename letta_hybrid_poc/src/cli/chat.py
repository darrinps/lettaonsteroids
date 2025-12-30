"""
CLI tool for interactive chat with baseline vs augmented modes.
"""
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from dotenv import load_dotenv
from ..adapters.llm_ollama import OllamaLLM, OllamaConfig
from ..adapters.llm_openai import OpenAILLM, OpenAIConfig
from ..retrieval.index import HybridIndex
from ..retrieval.rerank import CrossEncoderReranker
from ..controller.tools import RetrievalTool
from ..controller.policies import SelfRAGPolicy
from ..core.memory import BaselineMemory, AugmentedMemory, MemoryMode
from ..core.letta_controller import LettaController

# Load environment variables
load_dotenv()

app = typer.Typer(help="Chat with Letta using baseline or augmented mode")
console = Console()


def load_system(
    index_dir: Path,
    mode: str,
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "mistral:latest",
    openai_model: str = "gpt-4o-mini",
    use_reranker: bool = True
):
    """Load and initialize all system components."""
    console.print("[yellow]Initializing system...[/yellow]")

    # Load index
    console.print(f"[dim]Loading index from {index_dir}...[/dim]")
    index = HybridIndex()
    try:
        index.load(index_dir)
    except Exception as e:
        console.print(f"[bold red]Error loading index: {e}[/bold red]")
        console.print(f"[yellow]Run 'poetry run python -m src.cli.ingest build' first[/yellow]")
        raise typer.Exit(1)

    # Initialize LLM based on provider
    if provider == "openai":
        console.print(f"[dim]Using OpenAI ({openai_model})...[/dim]")
        llm_config = OpenAIConfig(model=openai_model)
        try:
            llm = OpenAILLM(config=llm_config)
            if not llm.is_available():
                raise RuntimeError("OpenAI API not available")
            console.print(f"[green]OpenAI connected successfully![/green]")
        except Exception as e:
            console.print(f"[bold red]Error: Cannot connect to OpenAI: {e}[/bold red]")
            console.print(f"[yellow]Make sure OPENAI_API_KEY is set in .env file[/yellow]")
            raise typer.Exit(1)
    else:  # ollama
        console.print(f"[dim]Connecting to Ollama ({ollama_url})...[/dim]")
        llm_config = OllamaConfig(base_url=ollama_url, model=ollama_model)
        llm = OllamaLLM(config=llm_config)

        if not llm.is_available():
            console.print(f"[bold red]Error: Cannot connect to Ollama at {ollama_url}[/bold red]")
            console.print(f"[yellow]Make sure Ollama is running: 'ollama serve'[/yellow]")
            raise typer.Exit(1)

    # Initialize reranker (optional)
    reranker = None
    if use_reranker:
        console.print(f"[dim]Loading cross-encoder reranker...[/dim]")
        try:
            reranker = CrossEncoderReranker()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load reranker: {e}[/yellow]")
            console.print(f"[yellow]Continuing without reranker...[/yellow]")

    # Initialize retrieval tool
    retrieval_tool = RetrievalTool(index=index, reranker=reranker)

    # Initialize policy
    policy = SelfRAGPolicy(llm=llm)

    # Initialize memory systems
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

    # Initialize controller
    controller = LettaController(
        llm=llm,
        baseline_memory=baseline_memory,
        augmented_memory=augmented_memory,
        policy=policy,
        mode=MemoryMode.BASELINE if mode == "baseline" else MemoryMode.LETTA
    )

    console.print("[bold green]SUCCESS: System initialized[/bold green]")
    return controller


@app.command("run")
def chat(
    mode: str = typer.Option(
        "augmented",
        "--mode",
        "-m",
        help="Chat mode: baseline or augmented"
    ),
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
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of documents to retrieve"
    ),
    use_reranker: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Use cross-encoder reranking"
    ),
    show_sources: bool = typer.Option(
        True,
        "--show-sources/--no-sources",
        help="Show retrieved sources"
    ),
):
    """
    Start interactive chat session.

    Example:
        poetry run python -m src.cli.chat run --mode augmented
        poetry run python -m src.cli.chat run --mode baseline
    """
    if mode not in ["baseline", "augmented"]:
        console.print("[bold red]Error: mode must be 'baseline' or 'augmented'[/bold red]")
        raise typer.Exit(1)

    if provider not in ["openai", "ollama"]:
        console.print("[bold red]Error: provider must be 'openai' or 'ollama'[/bold red]")
        raise typer.Exit(1)

    # Load system
    controller = load_system(
        index_dir,
        mode,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        openai_model=openai_model,
        use_reranker=use_reranker
    )

    # Display header
    model_name = openai_model if provider == "openai" else ollama_model
    console.print()
    console.print(Panel.fit(
        f"[bold]Letta Hybrid POC - {mode.upper()} Mode[/bold]\n"
        f"Provider: {provider.upper()}\n"
        f"Model: {model_name}\n"
        f"Type 'quit' to exit, '/reset' to clear conversation, '/mode' to switch modes",
        border_style="blue"
    ))
    console.print()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "quit":
                console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "/reset":
                controller.reset_conversation()
                console.print("[green]Conversation reset[/green]")
                continue

            if user_input.lower() == "/mode":
                current_mode = controller.mode
                new_mode = MemoryMode.LETTA if current_mode == MemoryMode.BASELINE else MemoryMode.BASELINE
                controller.set_mode(new_mode)
                console.print(f"[green]Switched to {new_mode.value} mode[/green]")
                continue

            # Process query
            console.print("[dim]Thinking...[/dim]")
            response = controller.chat(
                query=user_input,
                top_k=top_k,
                use_critique=True
            )

            # Display answer
            console.print()
            console.print(Panel(
                Markdown(response.answer),
                title="[bold green]Assistant[/bold green]",
                border_style="green"
            ))

            # Display metadata
            console.print(
                f"[dim]Mode: {response.mode.value} | "
                f"Retrieved: {response.metadata['num_retrieved']} docs | "
                f"Latency: {response.latency_ms:.0f}ms[/dim]"
            )

            # Show retrieved sources if enabled
            if show_sources and response.retrieved_docs:
                console.print()
                table = Table(title="Retrieved Sources", show_header=True)
                table.add_column("Doc ID", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("Score", style="green")

                for doc in response.retrieved_docs[:3]:
                    table.add_row(
                        doc['doc_id'],
                        doc['title'][:50] + "..." if len(doc['title']) > 50 else doc['title'],
                        f"{doc['score']:.3f}"
                    )

                console.print(table)

            # Show critique if available
            if "critique" in response.metadata:
                critique = response.metadata["critique"]
                if not critique["is_supported"] or critique["has_errors"]:
                    console.print(
                        f"[yellow]WARNING - Critique: {critique['feedback']}[/yellow]"
                    )

            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@app.command("query")
def single_query(
    query: str = typer.Argument(..., help="Query string"),
    mode: str = typer.Option(
        "augmented",
        "--mode",
        "-m",
        help="Chat mode: baseline or augmented"
    ),
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
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of documents to retrieve"
    ),
):
    """
    Run single query (useful for testing).

    Example:
        poetry run python -m src.cli.chat query "What is ABAC?" --mode augmented
        poetry run python -m src.cli.chat query "What is ABAC?" --provider openai
    """
    controller = load_system(
        index_dir,
        mode,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        openai_model=openai_model,
        use_reranker=True
    )

    response = controller.chat(query=query, top_k=top_k, use_critique=True)

    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title=f"[bold green]Answer ({response.mode.value})[/bold green]",
        border_style="green"
    ))

    console.print(
        f"[dim]Retrieved: {response.metadata['num_retrieved']} docs | "
        f"Latency: {response.latency_ms:.0f}ms[/dim]"
    )


if __name__ == "__main__":
    app()
