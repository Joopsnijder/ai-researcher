"""
Unit tests for the templates module.

Tests template loading, validation, and prompt generation.
"""

import pytest
import os
from pathlib import Path


class TestTemplateLoader:
    """Tests for template loading functionality."""

    def test_list_templates_returns_list(self):
        """Test that list_templates returns a non-empty list."""
        from ai_researcher.templates import list_templates

        templates = list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_list_templates_contains_default(self):
        """Test that default template is available."""
        from ai_researcher.templates import list_templates

        templates = list_templates()
        assert "default" in templates

    def test_list_templates_contains_swot(self):
        """Test that swot template is available."""
        from ai_researcher.templates import list_templates

        templates = list_templates()
        assert "swot" in templates

    def test_list_templates_contains_comparison(self):
        """Test that comparison template is available."""
        from ai_researcher.templates import list_templates

        templates = list_templates()
        assert "comparison" in templates

    def test_list_templates_contains_market(self):
        """Test that market template is available."""
        from ai_researcher.templates import list_templates

        templates = list_templates()
        assert "market" in templates


class TestLoadTemplate:
    """Tests for load_template function."""

    def test_load_default_template(self):
        """Test loading the default template."""
        from ai_researcher.templates import load_template

        template = load_template("default")
        assert template is not None
        assert template.get("name") == "default"

    def test_load_swot_template(self):
        """Test loading the swot template."""
        from ai_researcher.templates import load_template

        template = load_template("swot")
        assert template is not None
        assert template.get("name") == "swot"
        assert "sections" in template
        assert len(template["sections"]) > 0

    def test_load_comparison_template(self):
        """Test loading the comparison template."""
        from ai_researcher.templates import load_template

        template = load_template("comparison")
        assert template is not None
        assert template.get("name") == "comparison"
        assert "sections" in template

    def test_load_market_template(self):
        """Test loading the market template."""
        from ai_researcher.templates import load_template

        template = load_template("market")
        assert template is not None
        assert template.get("name") == "market"
        assert "sections" in template

    def test_load_nonexistent_template_raises_error(self):
        """Test that loading a non-existent template raises TemplateNotFoundError."""
        from ai_researcher.templates import load_template, TemplateNotFoundError

        with pytest.raises(TemplateNotFoundError):
            load_template("nonexistent_template_xyz")

    def test_template_has_description(self):
        """Test that templates have descriptions."""
        from ai_researcher.templates import load_template

        for name in ["swot", "comparison", "market"]:
            template = load_template(name)
            assert "description" in template
            assert len(template["description"]) > 0


class TestTemplateInfo:
    """Tests for get_template_info function."""

    def test_get_template_info_returns_list(self):
        """Test that get_template_info returns a list of dicts."""
        from ai_researcher.templates.loader import get_template_info

        info = get_template_info()
        assert isinstance(info, list)
        assert len(info) > 0

    def test_template_info_has_name_and_description(self):
        """Test that each template info has name and description."""
        from ai_researcher.templates.loader import get_template_info

        info = get_template_info()
        for t in info:
            assert "name" in t
            assert "description" in t


class TestGetTemplatePrompt:
    """Tests for get_template_prompt function."""

    def test_default_template_returns_empty_prompt(self):
        """Test that default template returns empty prompt (no sections)."""
        from ai_researcher.templates import load_template, get_template_prompt

        template = load_template("default")
        prompt = get_template_prompt(template)
        # Default template has no mandatory sections
        assert prompt == ""

    def test_swot_template_generates_sections(self):
        """Test that SWOT template generates section prompts."""
        from ai_researcher.templates import load_template, get_template_prompt

        template = load_template("swot")
        prompt = get_template_prompt(template)

        # Should contain SWOT sections
        assert "Strength" in prompt or "Sterktes" in prompt
        assert "Weakness" in prompt or "Zwaktes" in prompt
        assert "Opportunit" in prompt or "Kansen" in prompt
        assert "Threat" in prompt or "Bedreigingen" in prompt

    def test_template_prompt_english(self):
        """Test template prompt in English."""
        from ai_researcher.templates import load_template, get_template_prompt

        template = load_template("swot")
        prompt = get_template_prompt(template, language="en")

        assert "MANDATORY REPORT STRUCTURE" in prompt
        assert "Section" in prompt or "Focus" in prompt

    def test_template_prompt_dutch(self):
        """Test template prompt in Dutch."""
        from ai_researcher.templates import load_template, get_template_prompt

        template = load_template("swot")
        prompt = get_template_prompt(template, language="nl")

        assert "VERPLICHTE RAPPORT STRUCTUUR" in prompt

    def test_comparison_template_generates_sections(self):
        """Test that comparison template generates correct sections."""
        from ai_researcher.templates import load_template, get_template_prompt

        template = load_template("comparison")
        prompt = get_template_prompt(template)

        # Should contain comparison-related sections
        assert "Comparison" in prompt or "Vergelijk" in prompt or "Overview" in prompt

    def test_market_template_generates_sections(self):
        """Test that market template generates correct sections."""
        from ai_researcher.templates import load_template, get_template_prompt

        template = load_template("market")
        prompt = get_template_prompt(template)

        # Should contain market analysis sections
        assert "Market" in prompt or "Markt" in prompt


class TestTemplateStructure:
    """Tests for template YAML structure."""

    def test_swot_has_five_sections(self):
        """Test that SWOT template has 5 sections."""
        from ai_researcher.templates import load_template

        template = load_template("swot")
        sections = template.get("sections", [])
        assert len(sections) == 5

    def test_comparison_has_five_sections(self):
        """Test that comparison template has 5 sections."""
        from ai_researcher.templates import load_template

        template = load_template("comparison")
        sections = template.get("sections", [])
        assert len(sections) == 5

    def test_market_has_six_sections(self):
        """Test that market template has 6 sections."""
        from ai_researcher.templates import load_template

        template = load_template("market")
        sections = template.get("sections", [])
        assert len(sections) == 6

    def test_sections_have_title_and_prompt(self):
        """Test that template sections have title and prompt fields."""
        from ai_researcher.templates import load_template

        for name in ["swot", "comparison", "market"]:
            template = load_template(name)
            for section in template.get("sections", []):
                assert "title" in section
                assert "prompt" in section


class TestGetDefaultTemplate:
    """Tests for get_default_template function."""

    def test_get_default_template_returns_dict(self):
        """Test that get_default_template returns a dictionary."""
        from ai_researcher.templates.loader import get_default_template

        template = get_default_template()
        assert isinstance(template, dict)
        assert "name" in template

    def test_get_default_template_name_is_default(self):
        """Test that get_default_template returns 'default' template."""
        from ai_researcher.templates.loader import get_default_template

        template = get_default_template()
        assert template.get("name") == "default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
