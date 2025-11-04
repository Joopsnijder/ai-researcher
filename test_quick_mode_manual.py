"""
Manual test for quick research mode

Tests the new quick research functionality with a simple question
"""

import os
import research
from research import run_quick_research, HybridSearchTool

# Clean up
if os.path.exists('final_report.md'):
    os.remove('final_report.md')
    print("✓ Cleaned up existing final_report.md")

# Initialize search tool (must set on module level)
research.search_tool = HybridSearchTool(provider="multi-search")

print("=" * 60)
print("MANUAL TEST: Quick Research Mode")
print("=" * 60)
print("Question: What is quantum computing?")
print("Expected: Complete in 1-3 minutes, create final_report.md")
print("=" * 60)
print()

try:
    result = run_quick_research("What is quantum computing?", max_searches=3)

    print("\n" + "=" * 60)
    print("Test completed successfully!")

    if os.path.exists('final_report.md'):
        print("✅ final_report.md exists")
        with open('final_report.md', 'r') as f:
            content = f.read()
            word_count = len(content.split())
            print(f"📄 Report size: {len(content)} chars, ~{word_count} words")
    else:
        print("❌ final_report.md NOT created!")

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
