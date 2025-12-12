#!/bin/bash
#
# Run deep research and export to PDF
#
# Usage:
#   ./run_research.sh "Your research question here"
#   ./run_research.sh  # Will prompt for question
#
# Options (via environment variables):
#   ITERATIONS=300 ./run_research.sh "question"  # Custom iterations
#   PROVIDER=tavily ./run_research.sh "question" # Use Tavily provider
#

set -e

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Get research question
if [ -n "$1" ]; then
    QUESTION="$1"
else
    echo "Enter your research question:"
    read -r QUESTION
fi

if [ -z "$QUESTION" ]; then
    echo "Error: No research question provided"
    exit 1
fi

# Configuration with defaults
ITERATIONS="${ITERATIONS:-200}"
PROVIDER="${PROVIDER:-auto}"

echo ""
echo "=========================================="
echo "Starting Deep Research"
echo "=========================================="
echo "Question: $QUESTION"
echo "Iterations: $ITERATIONS"
echo "Provider: $PROVIDER"
echo ""

# Run research with CLI arguments (non-interactive mode)
python research.py --deep --iterations "$ITERATIONS" --provider "$PROVIDER" "$QUESTION"

echo ""
echo "=========================================="
echo "Exporting to PDF"
echo "=========================================="

# Export the latest report to PDF
python export_pdf.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
