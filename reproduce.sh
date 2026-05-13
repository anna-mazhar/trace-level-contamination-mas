#!/bin/bash
# Reproduce all statistics and figures from the paper
# Usage: ./reproduce.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Paper Artifact Reproduction"
echo "=============================================="

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

# Check dependencies
echo ""
echo "[1/4] Checking dependencies..."
python -c "import pandas, numpy, matplotlib" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}
echo "  OK: All dependencies available"

# Check data
echo ""
echo "[2/4] Checking data files..."
for model in gpt llama qwen; do
    dir="data/analysis_val_${model}"
    if [ ! -f "$dir/runs_flat.csv" ]; then
        echo "  ERROR: Missing $dir/runs_flat.csv"
        exit 1
    fi
    runs=$(tail -n +2 "$dir/runs_flat.csv" | wc -l | tr -d ' ')
    echo "  OK: $dir ($runs runs)"
done

# Generate statistics
echo ""
echo "[3/4] Generating statistics..."

echo "  Combined (3 models)..."
python scripts/paper_stats.py \
    data/analysis_val_gpt data/analysis_val_llama data/analysis_val_qwen \
    --output output/paper_stats_combined.json \
    > /dev/null

echo "  GPT only..."
python scripts/paper_stats.py \
    data/analysis_val_gpt \
    --output output/paper_stats_gpt.json \
    > /dev/null

echo "  OK: Statistics generated"

# Generate plots
echo ""
echo "[4/4] Generating plots..."

python scripts/paper_plots.py \
    --roots data/analysis_val_gpt data/analysis_val_llama data/analysis_val_qwen \
    --outdir output/plots \
    > /dev/null 2>&1

plot_count=$(ls -1 output/plots/*.pdf 2>/dev/null | wc -l | tr -d ' ')
echo "  OK: Generated $plot_count PDF plots"

# Summary
echo ""
echo "=============================================="
echo "Reproduction Complete"
echo "=============================================="
echo ""
echo "Generated outputs:"
echo "  output/paper_stats_combined.json"
echo "  output/paper_stats_gpt.json"
echo "  output/plots/*.pdf"
echo ""
echo "To verify: python verify.py"
