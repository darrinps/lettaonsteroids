"""
Quick test of OpenAI integration.
Run with: poetry run python test_openai.py
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing OpenAI integration...")
print()

try:
    from src.adapters.llm_openai import OpenAILLM, OpenAIConfig
    print("[OK] OpenAI adapter imported")

    # Test connection
    print("[INFO] Testing OpenAI API connection...")
    llm = OpenAILLM(config=OpenAIConfig(model="gpt-4o-mini"))

    if llm.is_available():
        print("[OK] OpenAI API connected successfully!")
        print()

        # Test generation
        print("[INFO] Testing generation...")
        response = llm.generate(
            "What is 2+2? Answer in one short sentence.",
            temperature=0.3,
            max_tokens=50
        )
        print(f"[OK] Response: {response}")
        print()

        print("=" * 50)
        print("OpenAI integration working!")
        print("=" * 50)
        print()
        print("You can now use:")
        print("  poetry run python -m src.cli.chat run --provider openai")
        print()
    else:
        print("[FAIL] Could not connect to OpenAI API")
        sys.exit(1)

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
