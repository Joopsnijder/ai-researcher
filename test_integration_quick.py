"""
Quick integration test to verify report generation guarantee

This test uses a trivial question with low recursion limit to quickly
validate that final_report.md is ALWAYS created.
"""

import os
import sys

# Remove any existing report
if os.path.exists('final_report.md'):
    os.remove('final_report.md')
    print("✓ Cleaned up existing final_report.md")

# Import after cleanup
from research import run_research, search_tool, HybridSearchTool

# Initialize search tool
search_tool = HybridSearchTool(provider="multi-search")

print("=" * 60)
print("INTEGRATION TEST: Report Generation Guarantee")
print("=" * 60)
print("Question: What is 2+2?")
print("Recursion limit: 30 (very low - agent may not complete properly)")
print("Expected: final_report.md MUST exist regardless")
print("=" * 60)
print()

try:
    result = run_research("What is 2+2?", recursion_limit=30)
    print("\n" + "=" * 60)
    print("Agent completed")
except Exception as e:
    print(f"\n" + "=" * 60)
    print(f"Agent crashed with: {type(e).__name__}: {e}")

# CRITICAL TEST: Check if report exists
print("=" * 60)
if os.path.exists('final_report.md'):
    print("✅ SUCCESS: final_report.md exists!")
    print("\nReport preview:")
    with open('final_report.md', 'r') as f:
        content = f.read()
        preview = content[:300] + "\n..." if len(content) > 300 else content
        print(preview)
else:
    print("❌ FAILURE: final_report.md does NOT exist!")
    print("The guarantee is broken!")
    sys.exit(1)

print("\n" + "=" * 60)
print("Test PASSED - Report generation guarantee works!")
print("=" * 60)
