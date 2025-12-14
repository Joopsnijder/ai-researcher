"""
Unit tests for deterministic report generation guarantee

These tests verify that final_report.md is ALWAYS created after agent execution,
regardless of whether the agent successfully creates it or not.
"""

import pytest
import os
from unittest.mock import Mock, patch

# Report path constants (must match research.py)
RESEARCH_FOLDER = "research"
FINAL_REPORT_PATH = os.path.join(RESEARCH_FOLDER, "final_report.md")


def setup_module():
    """Ensure research folder exists before tests."""
    os.makedirs(RESEARCH_FOLDER, exist_ok=True)


def test_ensure_report_exists_with_no_report():
    """Test that ensure_report_exists creates report when missing"""
    import glob
    from research import ensure_report_exists

    # Remove report and any test-related md files that could interfere
    cleanup_patterns = [
        FINAL_REPORT_PATH,
        os.path.join(RESEARCH_FOLDER, "*question*.md"),
        "*question*.md",
        "/tmp/*.md",
    ]
    for pattern in cleanup_patterns:
        for f in glob.glob(pattern):
            if os.path.exists(f):
                os.remove(f)

    # Mock result with some research content
    mock_result = {
        "messages": [
            Mock(
                content="This is research finding 1 about Python programming language."
            ),
            Mock(content="This is research finding 2 about data structures."),
        ]
    }

    # Call ensure_report_exists
    ensure_report_exists("What is Python?", mock_result, partial=False)

    # Verify report was created
    assert os.path.exists(FINAL_REPORT_PATH), "Report should be created"

    # Verify content
    with open(FINAL_REPORT_PATH, "r") as f:
        content = f.read()
        assert "What is Python?" in content
        assert "Research Report" in content
        assert "research finding" in content.lower()

    # Cleanup
    os.remove(FINAL_REPORT_PATH)


def test_ensure_report_exists_with_existing_report():
    """Test that ensure_report_exists doesn't overwrite existing report"""
    from research import ensure_report_exists

    # Create existing report
    original_content = "# Original Report\n\nThis is the original content."
    os.makedirs(RESEARCH_FOLDER, exist_ok=True)
    with open(FINAL_REPORT_PATH, "w") as f:
        f.write(original_content)

    # Call ensure_report_exists
    mock_result = {"messages": []}
    ensure_report_exists("Test question", mock_result, partial=False)

    # Verify original report is unchanged
    with open(FINAL_REPORT_PATH, "r") as f:
        content = f.read()
        assert content == original_content, "Existing report should not be overwritten"

    # Cleanup
    os.remove(FINAL_REPORT_PATH)


def test_ensure_report_exists_with_no_research():
    """Test emergency report when no research content available"""
    import glob
    from research import ensure_report_exists

    # Remove report and any test-related md files that could interfere
    cleanup_patterns = [
        FINAL_REPORT_PATH,
        os.path.join(RESEARCH_FOLDER, "*question*.md"),
        "*question*.md",
        "/tmp/*.md",
    ]
    for pattern in cleanup_patterns:
        for f in glob.glob(pattern):
            if os.path.exists(f):
                os.remove(f)

    # Call with None result (no research)
    ensure_report_exists("Test question", None, partial=True)

    # Verify report was created
    assert os.path.exists(FINAL_REPORT_PATH), (
        "Report should be created even without research"
    )

    # Verify it indicates no content
    with open(FINAL_REPORT_PATH, "r") as f:
        content = f.read()
        assert "Partial Research Report" in content
        assert "Test question" in content

    # Cleanup
    os.remove(FINAL_REPORT_PATH)


def test_extract_research_from_messages():
    """Test extraction of research content from messages"""
    from research import extract_research_from_messages

    # Messages need to be >200 chars and have >3 newlines to be included
    long_quantum_content = """This is a detailed research finding about quantum computing.

Quantum computing uses qubits instead of classical bits.
This enables parallel processing of multiple states.
Major players include IBM, Google, and Microsoft.
Applications include cryptography and drug discovery."""

    long_ml_content = """This is another detailed finding about machine learning algorithms.

Machine learning is a subset of artificial intelligence.
It includes supervised, unsupervised, and reinforcement learning.
Popular frameworks include TensorFlow and PyTorch.
Applications span from image recognition to natural language processing."""

    messages = [
        Mock(content="You are a research agent"),  # Should be skipped (system message)
        Mock(content="Short"),  # Should be skipped (too short)
        Mock(
            content=long_quantum_content
        ),  # Should be included (>200 chars, >3 newlines)
        Mock(
            content="Successfully wrote file"
        ),  # Should be skipped (tool confirmation)
        Mock(content=long_ml_content),  # Should be included
    ]

    result = extract_research_from_messages(messages)

    assert "quantum computing" in result
    assert "machine learning" in result
    assert "You are" not in result
    assert "Successfully" not in result


def test_create_emergency_report_normal():
    """Test emergency report creation for normal case"""
    from research import create_emergency_report

    research = "Some research findings"
    report = create_emergency_report("Test question", research, partial=False)

    assert "Research Report" in report
    assert "Test question" in report
    assert "Some research findings" in report
    assert "auto-generated" in report.lower()


def test_create_emergency_report_partial():
    """Test emergency report creation for partial/interrupted case"""
    from research import create_emergency_report

    research = "Incomplete research"
    report = create_emergency_report("Test question", research, partial=True)

    assert "Partial Research Report" in report
    assert "interrupted" in report.lower()
    assert "Incomplete research" in report


def test_report_guarantee_integration():
    """
    Integration test: Verify that run_research ALWAYS creates a report

    This is a smoke test - we mock the agent to avoid long execution times
    """
    from research import run_research, HybridSearchTool
    import research

    # Remove any existing report
    if os.path.exists(FINAL_REPORT_PATH):
        os.remove(FINAL_REPORT_PATH)

    # Initialize search tool
    research.search_tool = HybridSearchTool(provider="multi-search")

    # Mock agent to return immediately without creating report
    # Also mock rename_final_report to keep the file as research/final_report.md
    with patch("research.agent.stream", return_value=[]):
        with patch("research.rename_final_report", return_value=FINAL_REPORT_PATH):
            # Run research with mock
            run_research("Test question", recursion_limit=50)

    # CRITICAL: Verify report exists (this is the guarantee!)
    assert os.path.exists(FINAL_REPORT_PATH), (
        "Report MUST exist after run_research, regardless of agent behavior"
    )

    # Cleanup
    if os.path.exists(FINAL_REPORT_PATH):
        os.remove(FINAL_REPORT_PATH)


def test_should_trigger_early_report_below_threshold():
    """Test that early report trigger is False when below threshold"""
    from research import should_trigger_early_report, tracker, REPORT_TRIGGER_THRESHOLD

    # Reset tracker
    tracker.iteration_count = 50
    tracker.recursion_limit = 100
    tracker.report_triggered = False

    # Remove any existing report
    if os.path.exists(FINAL_REPORT_PATH):
        os.remove(FINAL_REPORT_PATH)

    # 50% is below 85% threshold - should not trigger
    assert not should_trigger_early_report(), (
        f"Should not trigger at {50 / 100:.0%}, threshold is {REPORT_TRIGGER_THRESHOLD:.0%}"
    )


def test_should_trigger_early_report_at_threshold():
    """Test that early report trigger is True when at or above threshold"""
    from research import should_trigger_early_report, tracker, REPORT_TRIGGER_THRESHOLD

    # Reset tracker
    tracker.iteration_count = 86  # 86% > 85% threshold
    tracker.recursion_limit = 100
    tracker.report_triggered = False

    # Remove any existing report
    if os.path.exists(FINAL_REPORT_PATH):
        os.remove(FINAL_REPORT_PATH)

    # 86% is above 85% threshold - should trigger
    assert should_trigger_early_report(), (
        f"Should trigger at {86 / 100:.0%}, threshold is {REPORT_TRIGGER_THRESHOLD:.0%}"
    )


def test_should_trigger_early_report_already_triggered():
    """Test that early report trigger is False when already triggered"""
    from research import should_trigger_early_report, tracker

    # Reset tracker with already triggered state
    tracker.iteration_count = 90
    tracker.recursion_limit = 100
    tracker.report_triggered = True  # Already triggered

    # Remove any existing report
    if os.path.exists(FINAL_REPORT_PATH):
        os.remove(FINAL_REPORT_PATH)

    # Should not trigger again
    assert not should_trigger_early_report(), "Should not trigger if already triggered"


def test_should_trigger_early_report_with_existing_report():
    """Test that early report trigger is False when report already exists"""
    from research import should_trigger_early_report, tracker

    # Reset tracker above threshold
    tracker.iteration_count = 90
    tracker.recursion_limit = 100
    tracker.report_triggered = False

    # Create existing report
    os.makedirs(RESEARCH_FOLDER, exist_ok=True)
    with open(FINAL_REPORT_PATH, "w") as f:
        f.write("# Existing Report")

    # Should not trigger when report exists
    assert not should_trigger_early_report(), (
        "Should not trigger when report already exists"
    )

    # Cleanup
    os.remove(FINAL_REPORT_PATH)


def test_create_finalize_instruction_content():
    """Test that finalize instruction contains required elements"""
    from research import create_finalize_instruction

    instruction = create_finalize_instruction()

    # Verify essential content
    assert "URGENT" in instruction, "Should contain URGENT marker"
    assert "final_report.md" in instruction, "Should reference final_report.md"
    assert "write_file" in instruction, "Should mention write_file tool"
    assert "STOP" in instruction, "Should instruct to stop activities"
    assert (
        "research-agent" in instruction.lower() or "sub-agent" in instruction.lower()
    ), "Should mention sub-agents to stop"


def test_tracker_iteration_fields_exist():
    """Test that AgentTracker has all required iteration tracking fields"""
    from research import AgentTracker

    tracker = AgentTracker()

    # Verify fields exist with correct defaults
    assert hasattr(tracker, "iteration_count"), "Should have iteration_count"
    assert hasattr(tracker, "recursion_limit"), "Should have recursion_limit"
    assert hasattr(tracker, "report_triggered"), "Should have report_triggered"

    assert tracker.iteration_count == 0, "iteration_count should default to 0"
    assert tracker.recursion_limit == 100, "recursion_limit should default to 100"
    assert tracker.report_triggered is False, "report_triggered should default to False"


def test_detect_language_dutch():
    """Test Dutch language detection"""
    from research import detect_language

    dutch_text = (
        "De organisatie heeft een nieuwe strategie voor het implementeren van AI"
    )
    assert detect_language(dutch_text) == "nl"


def test_detect_language_english():
    """Test English language detection"""
    from research import detect_language

    english_text = "The organization has a new strategy for implementing AI"
    assert detect_language(english_text) == "en"


def test_detect_language_mixed_defaults_english():
    """Test that mixed/unclear text defaults to English"""
    from research import detect_language

    mixed_text = "AI agents transform business processes"
    assert detect_language(mixed_text) == "en"


def test_refine_emergency_report_with_llm_returns_structured():
    """Test that LLM refinement returns structured output when mocked"""
    from research import refine_emergency_report_with_llm

    mock_response_text = """## Management Samenvatting
Dit is een test samenvatting.

## Bevindingen
Test bevindingen.

## Conclusie
Test conclusie."""

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_response = Mock()
        mock_response.content = [Mock(text=mock_response_text)]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        result = refine_emergency_report_with_llm(
            "Test vraag", "Dit zijn ruwe bevindingen met voldoende content " * 50
        )

        assert result is not None
        assert "Management Samenvatting" in result


def test_refine_emergency_report_with_llm_returns_none_for_short_content():
    """Test that LLM refinement returns None for insufficient content"""
    from research import refine_emergency_report_with_llm

    result = refine_emergency_report_with_llm("Test vraag", "Short")
    assert result is None


def test_refine_emergency_report_with_llm_handles_exception():
    """Test that LLM refinement handles exceptions gracefully"""
    from research import refine_emergency_report_with_llm

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

        result = refine_emergency_report_with_llm(
            "Test vraag", "Voldoende content voor de test " * 50
        )

        assert result is None


def test_search_status_display_add_search():
    """Test that SearchStatusDisplay correctly tracks searches"""
    from research import SearchStatusDisplay

    display = SearchStatusDisplay(max_history=3)

    # Add searches
    display.add_search(1, "test query 1", 10, "TestProvider", False)
    display.add_search(2, "test query 2", 5, "TestProvider", True)

    assert len(display.recent_searches) == 2
    assert display.recent_searches[0]["num"] == 1
    assert display.recent_searches[1]["cached"] is True


def test_search_status_display_max_history():
    """Test that SearchStatusDisplay respects max_history limit"""
    from research import SearchStatusDisplay

    display = SearchStatusDisplay(max_history=2)

    # Add more than max_history searches
    display.add_search(1, "query 1", 10, "Provider", False)
    display.add_search(2, "query 2", 10, "Provider", False)
    display.add_search(3, "query 3", 10, "Provider", False)

    # Should only keep last 2
    assert len(display.recent_searches) == 2
    assert display.recent_searches[0]["num"] == 2
    assert display.recent_searches[1]["num"] == 3


if __name__ == "__main__":
    print("Running report guarantee tests...")
    pytest.main([__file__, "-v"])
