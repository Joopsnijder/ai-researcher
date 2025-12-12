#!/usr/bin/env python3
"""
Tijdelijk script om markdown te fixen in een rapport bestand.
Converteert:
- • bullets naar - bullets
- Platte URLs naar markdown links
- Verwijdert duplicate headers
"""

import re
import sys


def fix_markdown(content: str) -> str:
    """Fix markdown formatting issues in the content."""

    # 1. Vervang • bullets door - bullets
    content = content.replace("• ", "- ")
    content = content.replace("•", "-")

    # 2. Fix platte URLs in citation format: [Title, URL] -> [Title](URL)
    # Pattern: [Some Title, https://example.com/path]
    content = re.sub(
        r'\[([^\]]+),\s*(https?://[^\]]+)\]',
        r'[\1](\2)',
        content
    )

    # 3. Fix standalone URLs die niet al in markdown link format zijn
    # Maar skip URLs die al in () staan (markdown links)
    def fix_standalone_url(match):
        url = match.group(0)
        # Check of deze URL al deel is van een markdown link
        return url

    # 4. Fix citation format: [Title, https://url] wordt [Title](https://url)
    # Ook: "Title, https://url" in brackets
    content = re.sub(
        r'\[([^,\]]+),\s*(https?://[^\]\s]+)\]',
        r'[\1](\2)',
        content
    )

    # 5. Verwijder de emergency report wrapper als die er is
    if content.startswith("# Research Report (Auto-Generated)"):
        # Zoek naar de eerste echte content header
        lines = content.split('\n')
        new_lines = []
        skip_until_findings = True
        in_findings = False

        for line in lines:
            if skip_until_findings:
                if line.startswith("## Research Findings"):
                    skip_until_findings = False
                    in_findings = True
                    continue  # Skip deze header ook
                continue
            new_lines.append(line)

        content = '\n'.join(new_lines).strip()

    # 6. Fix inline URLs die na een ] komen maar niet in () staan
    # Pattern: ] https://... of ], https://...
    content = re.sub(
        r'\]\s*,?\s*(https?://[^\s\]]+)',
        r'](\1)',
        content
    )

    return content


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_markdown.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    # Lees het bestand
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix de markdown
    fixed_content = fix_markdown(content)

    # Schrijf terug
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✓ Markdown gefixed in {filename}")

    # Toon statistieken
    original_bullets = content.count("• ")
    original_len = len(content)
    new_len = len(fixed_content)

    print(f"  - {original_bullets} bullets geconverteerd naar markdown")
    print(f"  - Bestandsgrootte: {original_len} -> {new_len} bytes")


if __name__ == "__main__":
    main()
