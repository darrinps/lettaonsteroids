#!/usr/bin/env bash
# Letta vs Mem0 Benchmark Runner
# Usage: ./run_benchmarks.sh [backend] [sessions] [noise]
#   backend: letta, mem0, or both (default: both)
#   sessions: number of test sessions (default: 1)
#   noise: noise ratio 0.0-1.0 (default: 0.0)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
BACKEND="${1:-both}"
SESSIONS="${2:-1}"
NOISE="${3:-0.0}"

echo "======================================"
echo "  Letta vs Mem0 Benchmark Runner"
echo "======================================"
echo "Backend:  $BACKEND"
echo "Sessions: $SESSIONS"
echo "Noise:    $NOISE"
echo "======================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Create and activate virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv .venv
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip and install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Install openai package (needed for embeddings)
if ! pip show openai &> /dev/null; then
    echo -e "${YELLOW}Installing openai package...${NC}"
    pip install openai --quiet
fi

# Create output directory
mkdir -p out

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not set. Similarity metrics will be disabled.${NC}"
fi

# Check if backend servers are running (optional warning)
if [ "$BACKEND" = "letta" ] || [ "$BACKEND" = "both" ]; then
    LETTA_URL="${LETTA_BASE_URL:-http://localhost:8283}"
    if ! curl -s --connect-timeout 2 "$LETTA_URL" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Cannot reach Letta server at $LETTA_URL${NC}"
        echo -e "${YELLOW}Make sure Letta server is running before testing Letta backend.${NC}"
    fi
fi

if [ "$BACKEND" = "mem0" ] || [ "$BACKEND" = "both" ]; then
    MEM0_URL="${MEM0_BASE_URL:-http://localhost:3000}"
    if ! curl -s --connect-timeout 2 "$MEM0_URL" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Cannot reach Mem0 server at $MEM0_URL${NC}"
        echo -e "${YELLOW}Make sure Mem0 server is running before testing Mem0 backend.${NC}"
    fi
fi

# Run benchmark
echo ""
echo -e "${GREEN}Running benchmarks...${NC}"
echo ""

python -m src.benchmark \
    --backend "$BACKEND" \
    --sessions "$SESSIONS" \
    --noise "$NOISE" \
    --out "out/results.json" \
    --csv "out/results.csv"

# Show results
echo ""
echo -e "${GREEN}======================================"
echo "  Benchmark Complete!"
echo "======================================${NC}"
echo "Results saved to:"
echo "  - out/results.json"
echo "  - out/results.csv"
echo ""
echo "To view results:"
echo "  cat out/results.csv"
echo ""
