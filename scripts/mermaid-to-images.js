#!/usr/bin/env node

/**
 * Mermaid to Images Converter
 *
 * Converts Mermaid diagrams in Markdown to SVG images and generates
 * a new Markdown file with image references for Marp presentations.
 *
 * Usage: node scripts/mermaid-to-images.js <input.md>
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Get input file from command line
const inputFile = process.argv[2];

if (!inputFile) {
  console.error('Usage: node scripts/mermaid-to-images.js <input.md>');
  process.exit(1);
}

if (!fs.existsSync(inputFile)) {
  console.error(`Error: File ${inputFile} not found`);
  process.exit(1);
}

// Setup paths
const inputDir = path.dirname(inputFile);
const inputBasename = path.basename(inputFile, '.md');
const diagramsDir = path.join(inputDir, 'diagrams');
const outputFile = path.join(inputDir, `${inputBasename}.build.md`);
const tempDir = path.join(inputDir, '.mermaid-temp');

// Create directories
if (!fs.existsSync(diagramsDir)) {
  fs.mkdirSync(diagramsDir, { recursive: true });
}
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

console.log('🎨 Mermaid to Images Converter');
console.log('================================');
console.log(`📄 Input:  ${inputFile}`);
console.log(`📁 Output: ${outputFile}`);
console.log(`🖼️  Diagrams: ${diagramsDir}/`);
console.log('');

// Read markdown content
const markdown = fs.readFileSync(inputFile, 'utf-8');

// Extract all mermaid blocks
const mermaidRegex = /```mermaid\n([\s\S]*?)```/g;
const mermaidBlocks = [];
let match;
let counter = 1;

while ((match = mermaidRegex.exec(markdown)) !== null) {
  mermaidBlocks.push({
    index: counter,
    code: match[1].trim(),
    fullMatch: match[0],
    start: match.index
  });
  counter++;
}

console.log(`📊 Found ${mermaidBlocks.length} Mermaid diagrams`);
console.log('');

// Convert each mermaid block to SVG
let newMarkdown = markdown;

for (const block of mermaidBlocks) {
  const mermaidFile = path.join(tempDir, `diagram-${block.index}.mmd`);
  const svgFile = path.join(diagramsDir, `diagram-${block.index}.svg`);
  const relativeSvgPath = `diagrams/diagram-${block.index}.svg`;

  console.log(`🔄 Converting diagram ${block.index}...`);

  // Write mermaid code to temp file
  fs.writeFileSync(mermaidFile, block.code);

  try {
    // Convert to SVG using mmdc with larger dimensions
    execSync(
      `mmdc -i "${mermaidFile}" -o "${svgFile}" -b transparent -t default -w 1400 -H 1000`,
      { stdio: 'pipe' }
    );

    console.log(`   ✅ Generated: ${relativeSvgPath}`);

    // Replace mermaid block with styled image reference
    // Use HTML for better control over sizing
    const imageMarkdown = `<div style="text-align: center; margin: 10px auto;">
  <img src="${relativeSvgPath}" alt="Diagram ${block.index}" style="max-width: 95%; height: auto; max-height: 350px; object-fit: contain;" />
</div>`;
    newMarkdown = newMarkdown.replace(block.fullMatch, imageMarkdown);

  } catch (error) {
    console.error(`   ❌ Error converting diagram ${block.index}:`, error.message);
  }
}

// Write new markdown file
fs.writeFileSync(outputFile, newMarkdown);

// Cleanup temp directory
fs.rmSync(tempDir, { recursive: true, force: true });

console.log('');
console.log('✨ Conversion complete!');
console.log('');
console.log('📄 Generated files:');
console.log(`   - ${outputFile}`);
console.log(`   - ${mermaidBlocks.length} SVG diagrams in ${diagramsDir}/`);
console.log('');
console.log('🚀 Next steps:');
console.log(`   marp ${outputFile} --pdf -o ${inputBasename}.pdf`);
console.log(`   marp ${outputFile} --html -o ${inputBasename}.html`);
