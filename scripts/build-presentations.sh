#!/bin/bash

# Build Presentations Script
# Converts Mermaid diagrams to images and generates Marp presentations

set -e  # Exit on error

echo "🎨 Building Presentations with Mermaid Diagrams"
echo "================================================"
echo ""

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: ./build-presentations.sh <markdown-file>"
  echo ""
  echo "Example:"
  echo "  ./build-presentations.sh docs/producer-functionaliteit.md"
  exit 1
fi

INPUT_FILE="$1"
BASENAME=$(basename "$INPUT_FILE" .md)
DIRNAME=$(dirname "$INPUT_FILE")

echo "📄 Processing: $INPUT_FILE"
echo ""

# Step 1: Convert Mermaid to images
echo "🔄 Step 1: Converting Mermaid diagrams to SVG..."
node scripts/mermaid-to-images.js "$INPUT_FILE"
echo ""

# Step 2: Generate Marp HTML
echo "🌐 Step 2: Generating HTML presentation..."
marp "$DIRNAME/$BASENAME.build.md" --html --no-stdin -o "$DIRNAME/$BASENAME-final.html"
echo "   ✅ Generated: $DIRNAME/$BASENAME-final.html"
echo ""

# Step 3: Generate Marp PDF
echo "📄 Step 3: Generating PDF presentation..."
marp "$DIRNAME/$BASENAME.build.md" --pdf --allow-local-files -o "$DIRNAME/$BASENAME-final.pdf"
echo "   ✅ Generated: $DIRNAME/$BASENAME-final.pdf"
echo ""

echo "✨ Build complete!"
echo ""
echo "📦 Generated files:"
echo "   - $DIRNAME/$BASENAME.build.md (intermediate)"
echo "   - $DIRNAME/$BASENAME-final.html (Marp HTML)"
echo "   - $DIRNAME/$BASENAME-final.pdf (Marp PDF)"
echo "   - $DIRNAME/diagrams/*.svg (diagram images)"
echo ""
echo "🚀 Open presentation:"
echo "   open $DIRNAME/$BASENAME-final.html"
echo "   open $DIRNAME/$BASENAME-final.pdf"
