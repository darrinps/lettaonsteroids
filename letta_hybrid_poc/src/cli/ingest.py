"""
CLI tool for ingesting corpus and building hybrid index.
"""
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from ..retrieval.index import HybridIndex, Document

app = typer.Typer(help="Ingest corpus and build hybrid retrieval index")
console = Console()


@app.command("build")
def build_index(
    corpus_path: Path = typer.Option(
        "data/raw/corpus.json",
        "--corpus-path",
        "-c",
        help="Path to corpus JSON file"
    ),
    index_dir: Path = typer.Option(
        "data/index",
        "--index-dir",
        "-o",
        help="Output directory for index"
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        "-e",
        help="Sentence transformer model name"
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device for embeddings (cpu/cuda)"
    ),
):
    """
    Build hybrid retrieval index from corpus.

    Example:
        poetry run python -m src.cli.ingest build --corpus-path data/raw/corpus.json
    """
    console.print(f"[bold blue]Building hybrid index from {corpus_path}[/bold blue]")

    # Check if corpus exists
    if not corpus_path.exists():
        console.print(f"[bold red]Error: Corpus file not found at {corpus_path}[/bold red]")
        raise typer.Exit(1)

    # Load corpus
    console.print(f"[yellow]Loading corpus...[/yellow]")
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading corpus: {e}[/bold red]")
        raise typer.Exit(1)

    # Convert to Document objects
    documents = []
    for item in corpus_data:
        doc = Document(
            id=item["id"],
            title=item.get("title", ""),
            content=item.get("content", ""),
            metadata=item.get("metadata")
        )
        documents.append(doc)

    console.print(f"[green]Loaded {len(documents)} documents[/green]")

    # Initialize index
    console.print(f"[yellow]Initializing hybrid index (model: {embedding_model})...[/yellow]")
    try:
        index = HybridIndex(
            embedding_model=embedding_model,
            device=device
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing index: {e}[/bold red]")
        raise typer.Exit(1)

    # Build index
    console.print(f"[yellow]Building BM25 + FAISS index (this may take a moment)...[/yellow]")
    try:
        # Simplified progress without spinner to avoid Windows console encoding issues
        console.print("[dim]Indexing documents...[/dim]")
        index.add_documents(documents)
        console.print("[green]Done![/green]")
    except Exception as e:
        console.print(f"[bold red]Error building index: {e}[/bold red]")
        raise typer.Exit(1)

    # Save index
    console.print(f"[yellow]Saving index to {index_dir}...[/yellow]")
    try:
        index.save(Path(index_dir))
    except Exception as e:
        console.print(f"[bold red]Error saving index: {e}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]SUCCESS: Index built successfully![/bold green]")
    console.print(f"[dim]Index location: {index_dir}[/dim]")
    console.print(f"[dim]Documents indexed: {len(documents)}[/dim]")
    console.print(f"[dim]Embedding dimension: {index.embedding_dim}[/dim]")


@app.command("info")
def index_info(
    index_dir: Path = typer.Option(
        "data/index",
        "--index-dir",
        "-i",
        help="Index directory"
    ),
):
    """
    Display information about existing index.
    """
    config_path = index_dir / "config.json"

    if not config_path.exists():
        console.print(f"[bold red]Error: No index found at {index_dir}[/bold red]")
        raise typer.Exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        console.print("[bold blue]Index Information[/bold blue]")
        console.print(f"[dim]Location:[/dim] {index_dir}")
        console.print(f"[dim]Documents:[/dim] {config['num_documents']}")
        console.print(f"[dim]Embedding Model:[/dim] {config['embedding_model']}")
        console.print(f"[dim]Embedding Dimension:[/dim] {config['embedding_dim']}")

    except Exception as e:
        console.print(f"[bold red]Error reading index info: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
