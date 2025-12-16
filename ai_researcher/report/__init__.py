"""Report generation and processing module."""

from .language import detect_language
from .extraction import extract_research_from_messages
from .emergency import create_emergency_report, refine_emergency_report_with_llm
from .postprocessing import (
    postprocess_report,
    _fix_report_title,
    _fix_report_date,
    _fix_sources_section,
    _fix_inline_references,
    _extract_title_from_url,
)
from .finalization import (
    generate_report_filename,
    ensure_report_exists,
    rename_final_report,
    finalize_report,
)

__all__ = [
    # Language
    "detect_language",
    # Extraction
    "extract_research_from_messages",
    # Emergency
    "create_emergency_report",
    "refine_emergency_report_with_llm",
    # Postprocessing
    "postprocess_report",
    "_fix_report_title",
    "_fix_report_date",
    "_fix_sources_section",
    "_fix_inline_references",
    "_extract_title_from_url",
    # Finalization
    "generate_report_filename",
    "ensure_report_exists",
    "rename_final_report",
    "finalize_report",
]
