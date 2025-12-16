"""Helper functions for research runners."""

import os

from ..config import RESEARCH_FOLDER, REPORT_TRIGGER_THRESHOLD


def should_trigger_early_report(tracker) -> bool:
    """
    Check if we should trigger early report writing to guarantee completion.

    Returns True if:
    - We've used >= 85% of iterations AND
    - Report hasn't been triggered yet AND
    - Final report doesn't exist yet
    """
    if tracker.report_triggered:
        return False

    if tracker.recursion_limit <= 0:
        return False

    iterations_used_ratio = tracker.iteration_count / tracker.recursion_limit

    # Check if we're approaching the limit
    if iterations_used_ratio >= REPORT_TRIGGER_THRESHOLD:
        # Verify report doesn't exist yet
        final_report_path = os.path.join(RESEARCH_FOLDER, "final_report.md")
        if not os.path.exists(final_report_path):
            return True

    return False


def create_finalize_instruction() -> str:
    """
    Create an urgent instruction for the agent to finalize the report immediately.
    """
    return """
URGENT: You are approaching the iteration limit. You MUST write the final report NOW.

CRITICAL INSTRUCTIONS:
1. STOP all research activities immediately
2. Use all findings gathered so far (even if incomplete)
3. Write the final report to `research/final_report.md` using the write_file tool
4. Follow the standard report structure (metadata header, Management Samenvatting, etc.)
5. Include all citations gathered so far

Do NOT:
- Start new research-agent tasks
- Call the critique-agent
- Perform additional searches

BEGIN WRITING THE REPORT IMMEDIATELY.
"""
