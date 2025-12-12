#!/usr/bin/env python3
"""Refine an existing research report into a narrative format."""

import time
import dotenv

dotenv.load_dotenv()

from anthropic import Anthropic  # noqa: E402


def main():
    # Read the existing report
    input_file = "research/wat-verandert-er-aan-leiderschap-bij-de-inzet-van-1.md"
    with open(input_file, "r") as f:
        raw_content = f.read()

    question = "Wat verandert er aan leiderschap bij de inzet van AI-agents in bedrijfsprocessen?"

    client = Anthropic()

    # Verbeterde prompt voor narratief rapport
    refinement_prompt = f"""Je bent een professionele research editor. Je hebt ruwe research bevindingen uit een onderzoeksproces.

TAAK: Schrijf een professioneel, goed leesbaar onderzoeksrapport in vloeiend Nederlands.

KRITIEKE STIJLVEREISTEN:
- Schrijf in LOPENDE TEKST, geen bullet points
- Elke sectie moet 3-5 alinea's bevatten met vloeiende zinnen
- Gebruik een academische, professionele schrijfstijl
- Verbind ideeën met overgangszinnen tussen alinea's
- Minimaal 4000 woorden voor een grondig rapport
- Citeer bronnen inline met [nummer] notatie
- Gebruik markdown links voor bronnen: [nummer] [Titel](URL)
- BELANGRIJK: Gebruik Nederlandse titel-casing voor koppen (alleen eerste woord met hoofdletter)
  Correct: "De veranderende rol van leiderschap"
  Fout: "De Veranderende Rol van Leiderschap"

STRUCTUUR:
1. Management samenvatting (250 woorden, lopende tekst)
2. Inleiding (context en relevantie van de vraag)
3. De veranderende rol van leiderschap (uitgebreide analyse)
4. Nieuwe competenties voor leiders (met voorbeelden)
5. Uitdagingen en weerstand (psychologische en organisatorische aspecten)
6. Praktische implementatie (concrete aanpak)
7. Case studies (IBM, BBVA, Netflix, etc.)
8. Conclusie en aanbevelingen
9. Bronnen (alle [nummer] referenties met URLs als markdown links)

ORIGINELE VRAAG:
{question}

RUWE BEVINDINGEN:
{raw_content[:100000]}

Schrijf nu het volledige rapport in vloeiende, goed leesbare Nederlandse tekst. GEEN bullet points in de hoofdtekst."""

    print("Generating narrative report with Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        messages=[{"role": "user", "content": refinement_prompt}],
    )

    refined = response.content[0].text

    # Create the new report
    new_report = f"""# Leiderschap in het tijdperk van AI-agents

| | |
|---|---|
| **Onderzoeksvraag** | {question} |
| **Type** | AI-gegenereerd onderzoeksrapport |
| **Datum** | {time.strftime("%Y-%m-%d")} |

{refined}

---
*Dit rapport is automatisch gestructureerd uit verzamelde research bevindingen.*
*Gegenereerd op: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""

    # Save to new file
    output_file = "research/leiderschap-ai-agents-narratief.md"
    with open(output_file, "w") as f:
        f.write(new_report)

    print(f"\n✓ Narratief rapport opgeslagen als: {output_file}")
    print(f"  Lengte: {len(new_report)} karakters ({len(new_report.split())} woorden)")


if __name__ == "__main__":
    main()
