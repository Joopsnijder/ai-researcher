#!/usr/bin/env python3
"""
Test script om de prompt wijzigingen te valideren.

Tests:
1. Prompt bevat de juiste instructies
2. Bestaand rapport scannen op problemen
"""

import re
import sys


def test_prompts():
    """Controleer dat de prompts de juiste instructies bevatten."""
    from research import research_instructions, sub_research_prompt

    print("\n=== PROMPT TESTS ===\n")
    errors = []

    # Test 1: Rapport structuur template
    if "MANDATORY REPORT STRUCTURE" not in research_instructions:
        errors.append("❌ Ontbreekt: MANDATORY REPORT STRUCTURE")
    else:
        print("✓ MANDATORY REPORT STRUCTURE gevonden")

    if "Management Samenvatting" not in research_instructions:
        errors.append("❌ Ontbreekt: Management Samenvatting sectie")
    else:
        print("✓ Management Samenvatting gevonden")

    if "Onderzoeksvraag:" not in research_instructions:
        errors.append("❌ Ontbreekt: Metadata header (Onderzoeksvraag)")
    else:
        print("✓ Metadata header gevonden")

    # Test 2: Schrijfstijl instructies
    if "WRITING STYLE - PROSE" not in research_instructions:
        errors.append("❌ Ontbreekt: WRITING STYLE - PROSE instructies")
    else:
        print("✓ WRITING STYLE - PROSE gevonden")

    if "BAD EXAMPLE" not in research_instructions:
        errors.append("❌ Ontbreekt: BAD EXAMPLE voor bullets")
    else:
        print("✓ BAD/GOOD voorbeelden gevonden")

    # Test 3: Verboden agent statements
    if "FORBIDDEN AGENT STATEMENTS" not in research_instructions:
        errors.append("❌ Ontbreekt: FORBIDDEN AGENT STATEMENTS")
    else:
        print("✓ FORBIDDEN AGENT STATEMENTS gevonden")

    if "Now I'll compile" in research_instructions or "Now I will" not in research_instructions:
        # Check dat het in de verboden lijst staat
        if '"Now I\'ll compile..."' not in research_instructions:
            errors.append("❌ Ontbreekt: 'Now I'll compile' in verboden lijst")
        else:
            print("✓ Agent statements in verboden lijst")

    # Test 4: Sub-research prompt
    if "RAW FINDINGS" not in sub_research_prompt.upper() and "raw research findings" not in sub_research_prompt:
        errors.append("❌ Sub-research prompt: ontbreekt 'raw findings' instructie")
    else:
        print("✓ Sub-research prompt: raw findings instructie aanwezig")

    if "# ONDERZOEKSRAPPORT" in sub_research_prompt:
        # Dit moet in de BAD example staan
        if "DO NOT DO THIS" in sub_research_prompt:
            print("✓ Sub-research prompt: BAD example met headers")
        else:
            errors.append("❌ Sub-research prompt: headers verbieden niet duidelijk")

    print()
    if errors:
        print("=== FOUTEN GEVONDEN ===")
        for e in errors:
            print(e)
        return False
    else:
        print("=== ALLE PROMPT TESTS GESLAAGD ===")
        return True


def scan_report(filepath: str):
    """Scan een rapport op problemen."""
    print(f"\n=== RAPPORT SCAN: {filepath} ===\n")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Bestand niet gevonden: {filepath}")
        return False

    issues = []
    warnings = []

    # Check 1: Metadata header
    if "**Onderzoeksvraag:**" not in content:
        issues.append("❌ Ontbreekt: Metadata header (Onderzoeksvraag)")
    else:
        print("✓ Metadata header aanwezig")

    if "**Type:** AI-gegenereerd" not in content:
        warnings.append("⚠️  Ontbreekt: Type indicator")

    # Check 2: Management Samenvatting
    if "## Management Samenvatting" not in content:
        issues.append("❌ Ontbreekt: Management Samenvatting sectie")
    else:
        print("✓ Management Samenvatting aanwezig")

    # Check 3: Agent statements
    agent_patterns = [
        r"Now I'll compile",
        r"Now I will",
        r"Let me analyze",
        r"Let me summarize",
        r"I will now",
        r"I'm going to",
        r"Based on my research",
        r"In my analysis",
        r"I found that",
        r"I discovered",
    ]
    found_agent_statements = []
    for pattern in agent_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            found_agent_statements.extend(matches)

    if found_agent_statements:
        issues.append(f"❌ Agent statements gevonden: {found_agent_statements[:3]}...")
    else:
        print("✓ Geen agent statements")

    # Check 4: Bullet overuse
    bullet_lines = len(re.findall(r"^[-•*]\s", content, re.MULTILINE))
    total_lines = len(content.split("\n"))
    bullet_ratio = bullet_lines / max(total_lines, 1)

    if bullet_ratio > 0.3:
        warnings.append(f"⚠️  Veel bullets: {bullet_ratio:.0%} van regels zijn bullets")
    else:
        print(f"✓ Bullet ratio OK: {bullet_ratio:.0%}")

    # Check 5: Unicode bullets
    unicode_bullets = content.count("•")
    if unicode_bullets > 0:
        issues.append(f"❌ Unicode bullets gevonden: {unicode_bullets}x '•'")
    else:
        print("✓ Geen unicode bullets")

    # Check 6: Bronnen sectie
    if "## Bronnen" not in content and "## Sources" not in content:
        warnings.append("⚠️  Ontbreekt: Bronnen sectie")
    else:
        print("✓ Bronnen sectie aanwezig")

    # Statistieken
    word_count = len(content.split())
    print(f"\nStatistieken:")
    print(f"  - Woorden: {word_count:,}")
    print(f"  - Karakters: {len(content):,}")
    print(f"  - Regels: {total_lines}")

    # Resultaat
    print()
    if issues:
        print("=== PROBLEMEN ===")
        for i in issues:
            print(i)
    if warnings:
        print("=== WAARSCHUWINGEN ===")
        for w in warnings:
            print(w)
    if not issues and not warnings:
        print("=== RAPPORT ZIET ER GOED UIT ===")

    return len(issues) == 0


if __name__ == "__main__":
    # Run prompt tests
    prompts_ok = test_prompts()

    # Scan rapport als argument gegeven
    if len(sys.argv) > 1:
        report_ok = scan_report(sys.argv[1])
    else:
        print("\nTip: python test_prompt_changes.py <rapport.md> om een rapport te scannen")
        report_ok = True

    # Exit code
    sys.exit(0 if prompts_ok and report_ok else 1)
