"""
Unit tests for deterministic report generation guarantee

These tests verify that final_report.md is ALWAYS created after agent execution,
regardless of whether the agent successfully creates it or not.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock


def test_ensure_report_exists_with_no_report():
    """Test that ensure_report_exists creates report when missing"""
    from research import ensure_report_exists

    # Remove report if exists
    if os.path.exists('final_report.md'):
        os.remove('final_report.md')

    # Mock result with some research content
    mock_result = {
        "messages": [
            Mock(content="This is research finding 1 about Python programming language."),
            Mock(content="This is research finding 2 about data structures.")
        ]
    }

    # Call ensure_report_exists
    ensure_report_exists("What is Python?", mock_result, partial=False)

    # Verify report was created
    assert os.path.exists('final_report.md'), "Report should be created"

    # Verify content
    with open('final_report.md', 'r') as f:
        content = f.read()
        assert "What is Python?" in content
        assert "Research Report" in content
        assert "research finding" in content.lower()

    # Cleanup
    os.remove('final_report.md')


def test_ensure_report_exists_with_existing_report():
    """Test that ensure_report_exists doesn't overwrite existing report"""
    from research import ensure_report_exists

    # Create existing report
    original_content = "# Original Report\n\nThis is the original content."
    with open('final_report.md', 'w') as f:
        f.write(original_content)

    # Call ensure_report_exists
    mock_result = {"messages": []}
    ensure_report_exists("Test question", mock_result, partial=False)

    # Verify original report is unchanged
    with open('final_report.md', 'r') as f:
        content = f.read()
        assert content == original_content, "Existing report should not be overwritten"

    # Cleanup
    os.remove('final_report.md')


def test_ensure_report_exists_with_no_research():
    """Test emergency report when no research content available"""
    from research import ensure_report_exists

    # Remove report if exists
    if os.path.exists('final_report.md'):
        os.remove('final_report.md')

    # Call with None result (no research)
    ensure_report_exists("Test question", None, partial=True)

    # Verify report was created
    assert os.path.exists('final_report.md'), "Report should be created even without research"

    # Verify it indicates no content
    with open('final_report.md', 'r') as f:
        content = f.read()
        assert "Partial Research Report" in content
        assert "Test question" in content

    # Cleanup
    os.remove('final_report.md')


def test_extract_research_from_messages():
    """Test extraction of research content from messages"""
    from research import extract_research_from_messages

    messages = [
        Mock(content="You are a research agent"),  # Should be skipped (system message)
        Mock(content="Short"),  # Should be skipped (too short)
        Mock(content="This is a valid research finding about quantum computing. " * 5),  # Should be included
        Mock(content="Successfully wrote file"),  # Should be skipped (tool confirmation)
        Mock(content="Another valid research finding about machine learning algorithms. " * 5),  # Should be included
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
    from research import run_research, search_tool, HybridSearchTool
    import research

    # Remove any existing report
    if os.path.exists('final_report.md'):
        os.remove('final_report.md')

    # Initialize search tool
    research.search_tool = HybridSearchTool(provider="multi-search")

    # Mock agent to return immediately without creating report
    with patch('research.agent.stream', return_value=[]):
        with patch('research.agent.invoke', return_value={"messages": [
            Mock(content="Mock research finding about the test question. " * 10)
        ]}):
            # Run research with mock
            result = run_research("Test question", recursion_limit=50)

    # CRITICAL: Verify report exists (this is the guarantee!)
    assert os.path.exists('final_report.md'), "Report MUST exist after run_research, regardless of agent behavior"

    # Cleanup
    if os.path.exists('final_report.md'):
        os.remove('final_report.md')


if __name__ == "__main__":
    print("Running report guarantee tests...")
    pytest.main([__file__, "-v"])
