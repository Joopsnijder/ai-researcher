"""Quick test script for research agent with low recursion limit"""

import os
import sys

# Set minimal recursion limit for quick test
os.environ['TEST_MODE'] = '1'

# Import after setting env
from research import run_research, search_tool, HybridSearchTool

# Initialize search tool with multi-search (free tier)
search_tool = HybridSearchTool(provider="multi-search")

# Simple test question
test_question = "What is Python?"

print("=" * 60)
print("QUICK TEST - Low Recursion Limit (50)")
print("=" * 60)
print(f"Question: {test_question}")
print("Recursion limit: 50")
print("=" * 60)
print()

# Run with very low recursion limit to test report generation
run_research(test_question, recursion_limit=50)

print("\n" + "=" * 60)
print("Test complete - check if final_report.md was created")
print("=" * 60)
