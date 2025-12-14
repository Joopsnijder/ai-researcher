"""
Unit tests for helper functions in research.py

These tests focus on pure functions that don't require API calls,
making them fast and free to run.
"""

import pytest
from unittest.mock import Mock

from research import (
    detect_language,
    generate_report_filename,
    load_prompt,
    create_emergency_report,
    extract_research_from_messages,
)


# =============================================================================
# Tests for detect_language()
# =============================================================================


class TestDetectLanguage:
    """Tests for Dutch vs English language detection."""

    def test_dutch_text_detected(self):
        """Dutch text with common Dutch words should return 'nl'."""
        dutch_text = "Dit is een test van de Nederlandse taal detectie"
        assert detect_language(dutch_text) == "nl"

    def test_english_text_detected(self):
        """English text should return 'en'."""
        english_text = "This is a test of the English language detection"
        assert detect_language(english_text) == "en"

    def test_mixed_text_dutch_dominant(self):
        """Text with more Dutch words should return 'nl'."""
        mixed_text = "De implementatie van het nieuwe systeem voor data processing"
        assert detect_language(mixed_text) == "nl"

    def test_empty_string_returns_english(self):
        """Empty string should default to 'en'."""
        assert detect_language("") == "en"

    def test_short_english_text(self):
        """Short English text without Dutch words should return 'en'."""
        short_text = "AI agents work"
        assert detect_language(short_text) == "en"


# =============================================================================
# Tests for generate_report_filename()
# =============================================================================


class TestGenerateReportFilename:
    """Tests for safe filename generation from questions."""

    def test_normal_question_to_slug(self):
        """Normal question should convert to lowercase slug."""
        question = "What is quantum computing"
        filename = generate_report_filename(question)
        assert filename == "what-is-quantum-computing.md"

    def test_special_characters_removed(self):
        """Special characters should be removed."""
        question = "What's the cost of AI? (2024)"
        filename = generate_report_filename(question)
        assert "?" not in filename
        assert "'" not in filename
        assert "(" not in filename
        assert filename.endswith(".md")

    def test_long_question_truncated(self):
        """Questions longer than 50 chars should be truncated."""
        long_question = (
            "This is a very long research question that should be truncated to fit"
        )
        filename = generate_report_filename(long_question)
        # Filename (without .md) should not exceed ~50 chars worth of content
        assert len(filename) < 60

    def test_unicode_normalized(self):
        """Unicode characters should be handled."""
        question = "Wat zijn de kosten van AI implementatie"
        filename = generate_report_filename(question)
        assert filename == "wat-zijn-de-kosten-van-ai-implementatie.md"

    def test_empty_string_fallback(self):
        """Empty string should return fallback filename."""
        assert generate_report_filename("") == "research-report.md"

    def test_only_special_chars_fallback(self):
        """String with only special chars should return fallback."""
        assert generate_report_filename("???!!!") == "research-report.md"


# =============================================================================
# Tests for load_prompt()
# =============================================================================


class TestLoadPrompt:
    """Tests for loading prompt files."""

    def test_load_existing_prompt(self):
        """Should load content from existing prompt file."""
        content = load_prompt("research_agent")
        assert isinstance(content, str)
        assert len(content) > 100
        assert "researcher" in content.lower()

    def test_load_nonexistent_prompt_raises(self):
        """Should raise FileNotFoundError for non-existent prompt."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")

    def test_prompt_with_placeholders_intact(self):
        """Prompts with {placeholders} should keep them intact."""
        content = load_prompt("emergency_refinement")
        assert "{lang_instruction}" in content
        assert "{question}" in content
        assert "{raw_findings}" in content

    def test_all_prompts_loadable(self):
        """All expected prompt files should be loadable."""
        expected_prompts = [
            "research_agent",
            "critique_agent",
            "quick_research",
            "deep_research",
            "emergency_refinement",
        ]
        for prompt_name in expected_prompts:
            content = load_prompt(prompt_name)
            assert isinstance(content, str)
            assert len(content) > 0


# =============================================================================
# Tests for extract_research_from_messages()
# =============================================================================


class TestExtractResearchFromMessages:
    """Tests for extracting research content from agent messages."""

    def test_extracts_long_content(self):
        """Should extract messages with substantial content."""
        long_content = "A" * 250 + "\n\n\n\nMore content here\nAnd more"
        messages = [Mock(content=long_content)]
        result = extract_research_from_messages(messages)
        assert long_content in result

    def test_skips_short_messages(self):
        """Messages under 200 chars should be skipped."""
        short_msg = Mock(content="Short message")
        long_msg = Mock(content="A" * 250 + "\n\n\n\nSubstantial research content here")
        messages = [short_msg, long_msg]
        result = extract_research_from_messages(messages)
        assert "Short message" not in result
        assert "Substantial research" in result

    def test_skips_system_prompts(self):
        """Messages starting with 'You are' should be skipped."""
        system_msg = Mock(
            content="You are an expert researcher. " + "A" * 200 + "\n\n\n\nMore"
        )
        messages = [system_msg]
        result = extract_research_from_messages(messages)
        assert result == ""

    def test_skips_error_messages(self):
        """Messages containing 'Error' at start should be skipped."""
        error_msg = Mock(
            content="Error occurred during search. " + "A" * 200 + "\n\n\n\n"
        )
        messages = [error_msg]
        result = extract_research_from_messages(messages)
        assert result == ""

    def test_empty_list_returns_empty_string(self):
        """Empty message list should return empty string."""
        assert extract_research_from_messages([]) == ""

    def test_messages_without_content_attribute(self):
        """Messages without content attribute should be skipped."""
        msg_no_content = Mock(spec=[])  # No content attribute
        msg_with_content = Mock(content="A" * 250 + "\n\n\n\nValid content")
        messages = [msg_no_content, msg_with_content]
        result = extract_research_from_messages(messages)
        assert "Valid content" in result


# =============================================================================
# Tests for create_emergency_report()
# =============================================================================


class TestCreateEmergencyReport:
    """Tests for emergency report generation."""

    def test_creates_report_with_content(self):
        """Should create formatted report with research content."""
        question = "What is AI?"
        content = "AI is artificial intelligence used in many applications."
        report = create_emergency_report(question, content)

        assert "# " in report  # Has title
        assert question in report
        assert content in report
        assert "Research Findings" in report

    def test_partial_flag_adds_warning(self):
        """Partial flag should add appropriate warning."""
        report = create_emergency_report("Test question", "Content", partial=True)
        assert "Partial" in report
        assert "Warning" in report or "interrupted" in report

    def test_non_partial_adds_note(self):
        """Non-partial report should add info note."""
        report = create_emergency_report("Test question", "Content", partial=False)
        assert "Auto-Generated" in report or "auto-generated" in report

    def test_empty_content_shows_message(self):
        """Empty content should show placeholder message."""
        report = create_emergency_report("Test question", "")
        assert "No research content" in report or "*No research" in report

    def test_whitespace_only_content(self):
        """Whitespace-only content should be treated as empty."""
        report = create_emergency_report("Test question", "   \n\t  ")
        assert "No research content" in report or "*No research" in report
