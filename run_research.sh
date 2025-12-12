#!/bin/bash
#
# Run deep research and export to PDF
#
# Usage:
#   ./run_research.sh "Your research question here"
#   ./run_research.sh  # Will prompt for question
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

echo ""
echo "=========================================="
echo "Starting Deep Research"
echo "=========================================="
echo "Question: $QUESTION"
echo ""

# Run research with deep mode, auto provider selection
# Using heredoc to provide answers to prompts
python research.py << EOF
$QUESTION
deep
auto
200
EOF

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
