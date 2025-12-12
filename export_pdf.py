#!/usr/bin/env python3
"""
Export research reports from Markdown to PDF.

Usage:
    python export_pdf.py                    # Export latest report from research/
    python export_pdf.py research/report.md # Export specific file

Configuration:
    Set PDF_OUTPUT_DIR in .env to specify where PDFs should be saved.
    If not set, PDFs are saved alongside the source .md file.

Requirements:
    pip install markdown

Note:
    This script generates an HTML file and opens it in the default browser.
    Use your browser's "Print to PDF" feature to save as PDF.
    For automated PDF generation, install: brew install basictex
    Then the script will use pandoc with pdflatex.
"""

import os
import sys
import glob
import shutil
import subprocess
import webbrowser
from pathlib import Path

import dotenv
import markdown

dotenv.load_dotenv()

# Configuration
RESEARCH_FOLDER = "research"
PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", "")

# CSS for HTML/PDF styling
HTML_CSS = """
<style>
@page {
    size: A4;
    margin: 2.5cm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 2em;
}
h1 { font-size: 24pt; margin-top: 0; color: #1a1a1a; }
h2 { font-size: 18pt; margin-top: 1.5em; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
h3 { font-size: 14pt; margin-top: 1.2em; color: #34495e; }
p { margin: 0.8em 0; text-align: justify; }
ul, ol { margin: 0.8em 0; padding-left: 2em; }
li { margin: 0.3em 0; }
code { background: #f4f4f4; padding: 0.2em 0.4em; border-radius: 3px; font-size: 10pt; font-family: "SF Mono", Monaco, monospace; }
pre { background: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }
pre code { padding: 0; background: none; }
blockquote { border-left: 4px solid #3498db; margin: 1em 0; padding-left: 1em; color: #666; font-style: italic; }
a { color: #3498db; text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 0.5em; text-align: left; }
th { background: #f4f4f4; }
@media print {
    body { max-width: none; padding: 0; }
    a { color: #333; }
    a::after { content: " (" attr(href) ")"; font-size: 9pt; color: #666; }
}
</style>
"""


def get_latest_report() -> str | None:
    """Find the most recently modified .md file in the research folder."""
    md_files = glob.glob(os.path.join(RESEARCH_FOLDER, "*.md"))

    if not md_files:
        print(f"No .md files found in {RESEARCH_FOLDER}/")
        return None

    # Sort by modification time, newest first
    md_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    # Skip final_report.md if there are other files (it's the temp file)
    for f in md_files:
        if os.path.basename(f) != "final_report.md":
            return f

    # If only final_report.md exists, use it
    return md_files[0]


def get_pdf_engine() -> str | None:
    """Get the best available PDF engine for pandoc.

    Prefers xelatex (better Unicode support) over pdflatex.
    """
    if shutil.which("xelatex"):
        return "xelatex"
    if shutil.which("pdflatex"):
        return "pdflatex"
    return None


def convert_with_pandoc(md_file: str, pdf_path: str) -> tuple[bool, str]:
    """Try to convert using pandoc + LaTeX engine.

    Returns:
        Tuple of (success, error_message)
    """
    pdf_engine = get_pdf_engine()
    if not pdf_engine:
        return False, "No LaTeX engine found (xelatex or pdflatex required)"

    try:
        cmd = [
            "pandoc",
            md_file,
            "-o",
            pdf_path,
            f"--pdf-engine={pdf_engine}",
            "-V",
            "geometry:margin=2.5cm",
            "-V",
            "fontsize=11pt",
            "-V",
            "documentclass=article",
            "-V",
            "colorlinks=true",
            "-V",
            "linkcolor=blue",
            "-V",
            "urlcolor=blue",
        ]

        # Add font settings for xelatex (better Unicode support)
        if pdf_engine == "xelatex":
            cmd.extend(["-V", "mainfont=Helvetica Neue"])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True, ""
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, error_msg
    except Exception as e:
        return False, str(e)


def convert_to_html(md_file: str, output_dir: str | None = None) -> str | None:
    """
    Convert a Markdown file to styled HTML.

    Args:
        md_file: Path to the Markdown file
        output_dir: Optional output directory

    Returns:
        Path to the generated HTML file
    """
    if not os.path.exists(md_file):
        print(f"Error: File not found: {md_file}")
        return None

    md_path = Path(md_file)
    html_name = md_path.stem + ".html"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, html_name)
    else:
        html_path = str(md_path.with_suffix(".html"))

    # Read and convert markdown
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_content = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "toc"],
    )

    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{md_path.stem}</title>
    {HTML_CSS}
</head>
<body>
    {html_content}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return html_path


def convert_to_pdf(md_file: str, output_dir: str | None = None) -> str | None:
    """
    Convert a Markdown file to PDF.

    Args:
        md_file: Path to the Markdown file
        output_dir: Optional output directory for the PDF

    Returns:
        Path to the generated PDF (or HTML if PDF not possible)
    """
    if not os.path.exists(md_file):
        print(f"Error: File not found: {md_file}")
        return None

    md_path = Path(md_file)
    pdf_name = md_path.stem + ".pdf"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, pdf_name)
    else:
        pdf_path = str(md_path.with_suffix(".pdf"))

    print(f"Converting: {md_file}")

    # Try pandoc + LaTeX engine first
    pdf_engine = get_pdf_engine()
    if shutil.which("pandoc") and pdf_engine:
        print(f"Using pandoc with {pdf_engine}...")
        success, error_msg = convert_with_pandoc(md_file, pdf_path)
        if success:
            print(f"\n✓ PDF created: {pdf_path}")
            return pdf_path
        print(f"Pandoc conversion failed: {error_msg}")
        print("Falling back to HTML...")

    # Fallback: generate HTML and open in browser for manual PDF export
    html_path = convert_to_html(md_file, output_dir)
    if html_path:
        print(f"\n✓ HTML created: {html_path}")
        print("\nOpening in browser - use Print (Cmd+P) → Save as PDF")
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
        return html_path

    return None


def main():
    """Main entry point."""
    # Determine which file to convert
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    else:
        md_file = get_latest_report()
        if not md_file:
            sys.exit(1)
        print(f"Using latest report: {md_file}")

    # Determine output directory
    output_dir = PDF_OUTPUT_DIR if PDF_OUTPUT_DIR else None

    if output_dir:
        print(f"Output directory: {output_dir}")

    # Convert to PDF
    result_path = convert_to_pdf(md_file, output_dir)

    if result_path:
        print(f"\nDone! Output saved to: {result_path}")
    else:
        print("\nConversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
