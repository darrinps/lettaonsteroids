"""
Simple wrapper to run evaluation with OpenAI.
Usage: python -m poetry run python run_eval.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.eval.eval import main

if __name__ == "__main__":
    # Run evaluation with OpenAI
    main(
        provider="openai",
        index_dir=Path("data/index"),
        openai_model="gpt-4o-mini",
        ollama_url="http://localhost:11434",
        ollama_model="mistral:latest",
        output=Path("eval_results_final.json"),
        top_k=5
    )
