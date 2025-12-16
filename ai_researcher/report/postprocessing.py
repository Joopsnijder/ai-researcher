"""Post-processing for research reports."""

import re
from datetime import datetime
from urllib.parse import urlparse

from ..ui.console import console


def postprocess_report(filepath: str, question: str) -> bool:
    """
    Post-process a research report to improve formatting.

    Improvements:
    1. Replace generic titles with the actual research question
    2. Fix source formatting: each source on its own line
    3. Make source URLs clickable with proper titles
    4. Convert inline [1] references to internal document links
    5. Replace {{DATE}} placeholder with current date

    Args:
        filepath: Path to the markdown report file
        question: The original research question

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Step 0: Fix date placeholder
        content = _fix_report_date(content)

        # Step 1: Fix title - use research question instead of generic titles
        content = _fix_report_title(content, question)

        # Step 2: Fix sources section - proper formatting
        content = _fix_sources_section(content)

        # Step 3: Convert inline [1] references to internal links
        content = _fix_inline_references(content)

        # Only write if changes were made
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            console.print("[dim]Rapport geoptimaliseerd (links, bronnen, titel)[/dim]")
            return True

        return True

    except Exception as e:
        console.print(f"[yellow]Post-processing failed: {e}[/yellow]")
        return False


def _fix_report_date(content: str) -> str:
    """Replace {{DATE}} placeholder with current date in Dutch format."""
    today = datetime.now().strftime("%d %B %Y")

    # Dutch month names
    month_translations = {
        "January": "januari",
        "February": "februari",
        "March": "maart",
        "April": "april",
        "May": "mei",
        "June": "juni",
        "July": "juli",
        "August": "augustus",
        "September": "september",
        "October": "oktober",
        "November": "november",
        "December": "december",
    }

    for eng, nl in month_translations.items():
        today = today.replace(eng, nl)

    return content.replace("{{DATE}}", today)


def _fix_report_title(content: str, question: str) -> str:
    """Replace generic titles with the research question."""
    # Common generic titles to replace
    generic_titles = [
        r"^# Onderzoeksrapport \(Automatisch Gegenereerd\)",
        r"^# Research Report \(Auto-Generated\)",
        r"^# Partial Research Report",
        r"^# Auto-generated Research Report",
        r"^# Onderzoeksrapport",
        r"^# Research Report",
    ]

    # Clean up the question for use as title
    title = question.strip()
    if not title.endswith("?"):
        # If it's not a question, capitalize first letter
        title = title[0].upper() + title[1:] if title else title

    for pattern in generic_titles:
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(
                pattern, f"# {title}", content, count=1, flags=re.MULTILINE
            )
            break

    return content


def _fix_sources_section(content: str) -> str:
    """
    Fix sources section formatting:
    - Each source on its own line
    - URLs become clickable markdown links
    - Add anchor IDs for internal linking
    """
    # Find the sources/bronnen section
    sources_match = re.search(
        r"(## (?:Bronnen|Sources|Referenties|References))\s*\n(.*?)(?=\n## |\n---|\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if not sources_match:
        return content

    sources_header = sources_match.group(1)
    sources_content = sources_match.group(2)

    # Extract all source references [n] with their URLs
    # Pattern matches: [n] followed by URL or markdown link
    fixed_sources = []

    # Split sources that are on the same line
    # Match patterns like: [1] URL [2] URL or [1] [Title](URL) [2] [Title](URL)
    source_pattern = r"\[(\d+)\]\s*(?:\[([^\]]*)\]\(([^)]+)\)|(\S+))"

    for match in re.finditer(source_pattern, sources_content):
        num = match.group(1)
        title = match.group(2)  # Title from [Title](URL)
        url_from_link = match.group(3)  # URL from [Title](URL)
        bare_url = match.group(4)  # Bare URL

        url = url_from_link or bare_url

        if url:
            # Clean URL
            url = url.strip()

            # If no title, extract domain as title
            if not title or title == url:
                title = _extract_title_from_url(url)

            # Format with anchor ID for internal linking
            fixed_sources.append(f'<a id="bron-{num}"></a>[{num}] [{title}]({url})')

    if fixed_sources:
        # Rebuild sources section with each source on its own line
        new_sources = sources_header + "\n\n" + "\n\n".join(fixed_sources) + "\n"

        # Replace old sources section
        content = (
            content[: sources_match.start()]
            + new_sources
            + content[sources_match.end() :]
        )

    return content


def _extract_title_from_url(url: str) -> str:
    """Extract a readable title from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")

        # Try to get a meaningful title from the path
        path = parsed.path.strip("/")
        if path:
            # Get the last segment of the path
            segments = path.split("/")
            last_segment = segments[-1] if segments else ""

            # Clean up the segment
            if last_segment and not last_segment.startswith("index"):
                # Remove file extensions
                title = re.sub(r"\.(html?|php|aspx?)$", "", last_segment)
                # Replace hyphens/underscores with spaces
                title = re.sub(r"[-_]", " ", title)
                # Capitalize words
                title = title.title()
                if len(title) > 5:  # Only use if meaningful
                    return f"{title} - {domain}"

        # Fallback to domain name
        return domain.title()
    except Exception:
        return url[:50] + "..." if len(url) > 50 else url


def _fix_inline_references(content: str) -> str:
    """
    Convert inline [n] references to internal links that jump to the source.

    [1] becomes [1](#bron-1)
    """
    # Find all inline references [n] that are NOT in the sources section
    # and NOT already links

    # First, find where sources section starts
    sources_start = None
    for pattern in [r"## Bronnen", r"## Sources", r"## Referenties", r"## References"]:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            sources_start = match.start()
            break

    if sources_start is None:
        sources_start = len(content)

    # Process only the content before sources section
    main_content = content[:sources_start]
    sources_section = content[sources_start:]

    # Replace [n] with [n](#bron-n) where n is a number
    # But avoid replacing already linked references like [n](...)
    def replace_ref(match):
        full_match = match.group(0)
        num = match.group(1)

        # Check if this is already a link (followed by '(')
        end_pos = match.end()
        if end_pos < len(main_content) and main_content[end_pos] == "(":
            return full_match

        return f"[{num}](#bron-{num})"

    # Pattern: [n] where n is 1-3 digits, not followed by ( or ]
    main_content = re.sub(r"\[(\d{1,3})\](?!\(|\])", replace_ref, main_content)

    return main_content + sources_section
