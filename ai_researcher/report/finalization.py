"""Report finalization and guarantee functions."""

import glob
import os
import re
import time

from ..config import RESEARCH_FOLDER, ensure_research_folder
from ..ui.console import console
from .extraction import extract_research_from_messages
from .emergency import create_emergency_report, refine_emergency_report_with_llm
from .language import detect_language
from .postprocessing import postprocess_report


def generate_report_filename(question: str) -> str:
    """
    Generate a safe filename based on the question.

    Args:
        question: The research question

    Returns:
        str: Safe filename like "what-is-quantum-computing.md"
    """
    # Take first 50 chars of question
    safe_question = question[:50]

    # Remove special characters and convert to lowercase
    safe_question = re.sub(r"[^\w\s-]", "", safe_question)
    safe_question = re.sub(r"[-\s]+", "-", safe_question)
    safe_question = safe_question.strip("-").lower()

    # Ensure it's not empty
    if not safe_question:
        safe_question = "research-report"

    return f"{safe_question}.md"


def ensure_report_exists(question, result, partial=False):
    """
    GUARANTEE: Ensures final_report.md exists after agent execution.

    This function is called deterministically (not by the LLM) to ensure
    a report file is always created, even if the agent failed to do so.

    Args:
        question: The research question
        result: Agent execution result (can be None)
        partial: Whether this is a partial report (error/interrupt case)
    """
    # Ensure research folder exists
    ensure_research_folder()
    final_report_path = os.path.join(RESEARCH_FOLDER, "final_report.md")

    # Check if report already exists
    if os.path.exists(final_report_path):
        console.print(
            f"[dim]{final_report_path} already exists (created by agent)[/dim]"
        )
        return

    # Check if agent wrote to a different file (common mistake)
    possible_reports = (
        glob.glob("*.md")
        + glob.glob("/tmp/*.md")
        + glob.glob(f"{RESEARCH_FOLDER}/*.md")
    )
    skip_files = ["README.md", "CLAUDE.md", "requirements.md", "final_report.md"]
    recent_md_files = [
        f
        for f in possible_reports
        if os.path.isfile(f)
        and os.path.getmtime(f) > time.time() - 300  # Last 5 minutes
        and os.path.basename(f) not in skip_files
        and not os.path.basename(f).startswith("test_")
    ]

    # If we found a recent .md file, use it instead of generating emergency report
    if recent_md_files:
        # Sort by modification time, newest first
        recent_md_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        found_report = recent_md_files[0]

        console.print(
            f"\n[bold yellow]Agent schreef naar verkeerde locatie: {found_report}[/bold yellow]"
        )
        console.print(f"[green]   Kopiëren naar {final_report_path}...[/green]")

        # Copy the found report to final_report.md
        with open(found_report, encoding="utf-8") as src:
            content = src.read()
        with open(final_report_path, "w", encoding="utf-8") as dst:
            dst.write(content)

        console.print(f"[green]Rapport gekopieerd van {found_report}[/green]")
        return

    # No recent file found - generate emergency report
    console.print(
        f"\n[bold yellow]Agent did not create {final_report_path} - generating emergency report...[/bold yellow]"
    )
    console.print("[yellow]   Mogelijke oorzaken:[/yellow]")
    console.print("[dim]   Agent dacht klaar te zijn zonder bestand te schrijven[/dim]")
    console.print(
        "[dim]   Recursion limit bereikt voordat rapport werd geschreven[/dim]"
    )

    # Extract any research from agent messages
    research_content = ""
    if result and "messages" in result:
        research_content = extract_research_from_messages(result["messages"])

    # Try LLM refinement first for better quality reports
    language = detect_language(question)
    refined_content = refine_emergency_report_with_llm(
        question, research_content, language
    )

    if refined_content:
        # Wrap refined content in report template
        report = f"""# Onderzoeksrapport (Automatisch Gegenereerd)

> **Note**: Dit rapport is automatisch gegenereerd en gestructureerd uit verzamelde
> research bevindingen omdat de agent het eindrapport niet zelf heeft geschreven.

{refined_content}

---
*Rapport gestructureerd met Claude op: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""
    else:
        # Fallback to old method if LLM fails
        report = create_emergency_report(question, research_content, partial)

    # Write report to file
    with open(final_report_path, "w", encoding="utf-8") as f:
        f.write(report)

    console.print("[green]Emergency report created from available research[/green]")
    if not research_content.strip():
        console.print("[dim]  (Note: Limited research content was available)[/dim]")


def rename_final_report(question: str) -> str | None:
    """
    Rename final_report.md to a question-based filename in research folder.

    Args:
        question: The research question

    Returns:
        str: The new filename (full path), or None if rename failed
    """
    ensure_research_folder()
    final_report_path = os.path.join(RESEARCH_FOLDER, "final_report.md")

    if not os.path.exists(final_report_path):
        return None

    base_filename = generate_report_filename(question)
    new_filename = os.path.join(RESEARCH_FOLDER, base_filename)

    # If file already exists, add a number
    base_name = new_filename[:-3]  # Remove .md
    counter = 1
    while os.path.exists(new_filename):
        new_filename = f"{base_name}-{counter}.md"
        counter += 1

    try:
        os.rename(final_report_path, new_filename)
        return new_filename
    except Exception as e:
        console.print(f"[yellow]Could not rename report: {e}[/yellow]")
        return final_report_path


def finalize_report(question: str) -> str | None:
    """
    Finalize report: rename and post-process.

    This combines rename_final_report and postprocess_report into one call.

    Args:
        question: The research question

    Returns:
        str: Final filename, or None if failed
    """
    final_filename = rename_final_report(question)
    if final_filename:
        postprocess_report(final_filename, question)
    return final_filename
