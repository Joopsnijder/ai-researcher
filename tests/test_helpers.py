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
    _fix_report_title,
    _fix_report_date,
    _fix_sources_section,
    _fix_inline_references,
    _extract_title_from_url,
    calculate_cost,
    AgentTracker,
    ANTHROPIC_PRICING,
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


# =============================================================================
# Tests for report post-processing functions
# =============================================================================


class TestFixReportTitle:
    """Tests for title replacement in reports."""

    def test_replaces_generic_title(self):
        """Should replace generic titles with the research question."""
        content = "# Onderzoeksrapport\n\nSome content here."
        question = "Wat is LangGraph?"
        result = _fix_report_title(content, question)
        assert "# Wat is LangGraph?" in result
        assert "# Onderzoeksrapport" not in result

    def test_replaces_auto_generated_title(self):
        """Should replace auto-generated titles."""
        content = "# Research Report (Auto-Generated)\n\nContent."
        question = "How do AI agents work?"
        result = _fix_report_title(content, question)
        assert "# How do AI agents work?" in result

    def test_keeps_custom_title(self):
        """Should keep titles that don't match generic patterns."""
        content = "# My Custom Research Title\n\nContent."
        question = "Some question"
        result = _fix_report_title(content, question)
        assert "# My Custom Research Title" in result


class TestFixReportDate:
    """Tests for date placeholder replacement."""

    def test_replaces_date_placeholder(self):
        """Should replace {{DATE}} with current date."""
        from datetime import date

        content = "| **Datum** | {{DATE}} |"
        result = _fix_report_date(content)
        expected_date = date.today().strftime("%Y-%m-%d")
        assert expected_date in result
        assert "{{DATE}}" not in result

    def test_replaces_multiple_placeholders(self):
        """Should replace all {{DATE}} occurrences."""
        from datetime import date

        content = "Date: {{DATE}}\nUpdated: {{DATE}}"
        result = _fix_report_date(content)
        expected_date = date.today().strftime("%Y-%m-%d")
        assert result.count(expected_date) == 2
        assert "{{DATE}}" not in result

    def test_no_placeholder_unchanged(self):
        """Content without placeholder should remain unchanged."""
        content = "| **Datum** | 2024-01-15 |"
        result = _fix_report_date(content)
        assert result == content


class TestFixSourcesSection:
    """Tests for sources section formatting."""

    def test_separates_inline_sources(self):
        """Should put each source on its own line."""
        content = """## Bronnen

[1] https://example.com/a [2] https://example.com/b [3] https://test.org
"""
        result = _fix_sources_section(content)
        # Each source should have its own anchor
        assert '<a id="bron-1"></a>' in result
        assert '<a id="bron-2"></a>' in result
        assert '<a id="bron-3"></a>' in result

    def test_handles_markdown_links(self):
        """Should preserve existing markdown link titles."""
        content = """## Sources

[1] [My Title](https://example.com/article)
"""
        result = _fix_sources_section(content)
        assert "[My Title]" in result
        assert "https://example.com/article" in result

    def test_no_sources_section(self):
        """Should return content unchanged if no sources section."""
        content = "# Report\n\nNo sources here."
        result = _fix_sources_section(content)
        assert result == content


class TestFixInlineReferences:
    """Tests for inline reference linking."""

    def test_converts_to_internal_links(self):
        """Should convert [1] to [1](#bron-1)."""
        content = "This fact is cited [1] and another [2].\n\n## Bronnen\n"
        result = _fix_inline_references(content)
        assert "[1](#bron-1)" in result
        assert "[2](#bron-2)" in result

    def test_preserves_sources_section(self):
        """Should not modify references in sources section."""
        content = "Text [1] here.\n\n## Bronnen\n\n[1] Some source"
        result = _fix_inline_references(content)
        # The [1] in sources should NOT become a link
        lines = result.split("## Bronnen")
        assert "[1](#bron-1)" in lines[0]  # In main content
        # In sources section, [1] should remain as-is or be in anchor format

    def test_ignores_existing_links(self):
        """Should not double-link already linked references."""
        content = "Already linked [1](#bron-1) reference.\n\n## Bronnen\n"
        result = _fix_inline_references(content)
        # Should not create [1](#bron-1)(#bron-1)
        assert "[1](#bron-1)(#bron-1)" not in result


class TestExtractTitleFromUrl:
    """Tests for URL title extraction."""

    def test_extracts_from_path(self):
        """Should extract readable title from URL path."""
        url = "https://example.com/articles/my-great-article"
        title = _extract_title_from_url(url)
        assert "My Great Article" in title or "example.com" in title.lower()

    def test_falls_back_to_domain(self):
        """Should use domain if path is not meaningful."""
        url = "https://docs.example.com/"
        title = _extract_title_from_url(url)
        assert "example" in title.lower()

    def test_handles_long_urls(self):
        """Should truncate very long URLs."""
        url = "https://example.com/" + "a" * 100
        title = _extract_title_from_url(url)
        assert len(title) <= 60 or "..." in title or "example" in title.lower()


# =============================================================================
# Tests for cost tracking functions
# =============================================================================


class TestCalculateCost:
    """Tests for LLM cost calculation."""

    def test_calculates_cost_for_known_model(self):
        """Should calculate cost using known model pricing."""
        # 1M input tokens at $3, 1M output tokens at $15
        cost = calculate_cost(1_000_000, 1_000_000, "claude-sonnet-4-5-20250929")
        assert cost == 18.0  # $3 + $15

    def test_calculates_cost_for_small_usage(self):
        """Should calculate cost for typical small usage."""
        # 10K input, 5K output
        cost = calculate_cost(10_000, 5_000, "claude-sonnet-4-5-20250929")
        expected = (10_000 / 1_000_000) * 3.0 + (5_000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 0.0001

    def test_uses_default_pricing_for_unknown_model(self):
        """Should use default pricing for unknown models."""
        cost = calculate_cost(1_000_000, 1_000_000, "unknown-model-xyz")
        assert cost == 18.0  # Same as default

    def test_zero_tokens_returns_zero_cost(self):
        """Zero tokens should return zero cost."""
        cost = calculate_cost(0, 0)
        assert cost == 0.0


class TestAgentTrackerCostTracking:
    """Tests for AgentTracker cost tracking methods."""

    def test_add_token_usage(self):
        """Should accumulate token usage."""
        tracker = AgentTracker()
        tracker.add_token_usage(1000, 500)
        tracker.add_token_usage(2000, 1000)
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500

    def test_get_total_cost(self):
        """Should calculate total cost from accumulated tokens."""
        tracker = AgentTracker()
        tracker.add_token_usage(100_000, 50_000)
        cost = tracker.get_total_cost()
        expected = (100_000 / 1_000_000) * 3.0 + (50_000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 0.0001

    def test_reset_token_tracking(self):
        """Should reset token counts to zero."""
        tracker = AgentTracker()
        tracker.add_token_usage(5000, 2500)
        tracker.reset_token_tracking()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0

    def test_initial_token_counts_are_zero(self):
        """New tracker should have zero token counts."""
        tracker = AgentTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.get_total_cost() == 0.0


class TestAnthropicPricing:
    """Tests for pricing constants."""

    def test_pricing_dict_has_required_keys(self):
        """Pricing dict should have input and output keys."""
        for model, prices in ANTHROPIC_PRICING.items():
            assert "input" in prices, f"Model {model} missing input price"
            assert "output" in prices, f"Model {model} missing output price"

    def test_default_pricing_exists(self):
        """Default pricing should exist for unknown models."""
        assert "default" in ANTHROPIC_PRICING

    def test_prices_are_positive(self):
        """All prices should be positive numbers."""
        for model, prices in ANTHROPIC_PRICING.items():
            assert prices["input"] > 0, f"Model {model} has non-positive input price"
            assert prices["output"] > 0, f"Model {model} has non-positive output price"
