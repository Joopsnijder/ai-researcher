#!/bin/bash
#
# Convert markdown research reports to professional PDF using Eisvogel template
#
# Usage: ./scripts/convert-to-pdf.sh research/my-report.md
#        ./scripts/convert-to-pdf.sh research/my-report.md output.pdf
#
# Requirements:
#   - Pandoc 3.x: brew install pandoc
#   - LaTeX (BasicTeX or MacTeX): brew install --cask basictex
#   - Eisvogel template: installed in ~/.local/share/pandoc/templates/eisvogel.latex
#
# The script automatically adds YAML frontmatter for Eisvogel if not present.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
check_dependencies() {
    if ! command -v pandoc &> /dev/null; then
        echo -e "${RED}Error: pandoc is not installed${NC}"
        echo "Install with: brew install pandoc"
        exit 1
    fi

    if ! command -v pdflatex &> /dev/null; then
        echo -e "${RED}Error: pdflatex is not installed${NC}"
        echo "Install with: brew install --cask basictex"
        exit 1
    fi

    if [ ! -f ~/.local/share/pandoc/templates/eisvogel.latex ]; then
        echo -e "${RED}Error: Eisvogel template not found${NC}"
        echo "Download from: https://github.com/Wandmalfarbe/pandoc-latex-template/releases"
        echo "Place eisvogel.latex in: ~/.local/share/pandoc/templates/"
        exit 1
    fi
}

# Extract title from markdown (first # heading)
extract_title() {
    local input_file="$1"
    grep -m1 '^# ' "$input_file" | sed 's/^# //' || echo "Research Report"
}

# Check if file has YAML frontmatter
has_frontmatter() {
    local input_file="$1"
    head -1 "$input_file" | grep -q '^---$'
}

# Create temporary file with Eisvogel frontmatter
add_frontmatter() {
    local input_file="$1"
    local temp_file="$2"
    local title
    title=$(extract_title "$input_file")
    local date
    date=$(date +%Y-%m-%d)

    cat > "$temp_file" << EOF
---
title: "${title}"
author: "AI Research Agent"
date: "${date}"
titlepage: true
titlepage-color: "1a365d"
titlepage-text-color: "FFFFFF"
titlepage-rule-color: "FFFFFF"
titlepage-rule-height: 2
toc: true
toc-own-page: true
numbersections: true
colorlinks: false
disable-header-and-footer: true
lang: "nl"
geometry: "margin=2.5cm"
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyfoot[C]{\thepage}
  - \renewcommand{\headrulewidth}{0pt}
---

EOF

    # Append original content (skip existing frontmatter if present)
    # Also strip emojis for professional PDF output
    if has_frontmatter "$input_file"; then
        # Skip frontmatter (everything between first --- and second ---)
        sed -n '/^---$/,/^---$/d; p' "$input_file" | strip_emojis >> "$temp_file"
    else
        strip_emojis < "$input_file" >> "$temp_file"
    fi
}

# Remove emojis and special Unicode characters for cleaner PDF output
strip_emojis() {
    # Remove common emojis and symbols that don't render well in LaTeX
    sed -E 's/[🔍🤖💡✓✔️❌⚠️ℹ️📊📈📉🎯🚀💼📝✨🔥💪🏆⭐️🌟📌🔗💬🗣️👉👆👇📋🗂️📁💾🔧⚙️🛠️📢🎉🎊🔒🔓✅❎🆗🆕🆓🔴🟢🟡⬜️⬛️🟦🟩🟨🟥▶️⏸️⏹️🔄↗️➡️⬅️⬆️⬇️↩️↪️•]//g'
}

# Main conversion function
convert_to_pdf() {
    local input_file="$1"
    local output_file="$2"

    # Create temp file for processing
    local temp_file
    temp_file=$(mktemp /tmp/eisvogel_XXXXXX.md)
    trap "rm -f $temp_file" EXIT

    echo -e "${YELLOW}Preparing document...${NC}"
    add_frontmatter "$input_file" "$temp_file"

    echo -e "${YELLOW}Converting to PDF with Eisvogel template...${NC}"

    # Run pandoc with Eisvogel template
    # Using pdflatex (emojis are stripped for compatibility)
    pandoc "$temp_file" \
        --template=eisvogel \
        --pdf-engine=pdflatex \
        --listings \
        --highlight-style=tango \
        -V classoption=oneside \
        -o "$output_file"

    echo -e "${GREEN}PDF created: ${output_file}${NC}"
}

# Show usage
usage() {
    echo "Usage: $0 <input.md> [output.pdf]"
    echo ""
    echo "Convert markdown research reports to professional PDF using Eisvogel template."
    echo ""
    echo "Arguments:"
    echo "  input.md    Input markdown file"
    echo "  output.pdf  Output PDF file (optional, defaults to input filename with .pdf)"
    echo ""
    echo "Examples:"
    echo "  $0 research/my-report.md"
    echo "  $0 research/my-report.md ~/Desktop/report.pdf"
}

# Main entry point
main() {
    if [ $# -lt 1 ]; then
        usage
        exit 1
    fi

    local input_file="$1"
    local output_file="${2:-${input_file%.md}.pdf}"

    if [ ! -f "$input_file" ]; then
        echo -e "${RED}Error: Input file not found: ${input_file}${NC}"
        exit 1
    fi

    check_dependencies
    convert_to_pdf "$input_file" "$output_file"

    # Open PDF on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Opening PDF...${NC}"
        open "$output_file"
    fi
}

main "$@"
