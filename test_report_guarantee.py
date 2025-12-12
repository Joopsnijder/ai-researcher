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


if __name__ == "__main__":
    print("Running report guarantee tests...")
    pytest.main([__file__, "-v"])
