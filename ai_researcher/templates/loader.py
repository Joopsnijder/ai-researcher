"""Template loader for research report templates."""

from pathlib import Path

import yaml


class TemplateNotFoundError(Exception):
    """Raised when a template cannot be found."""

    pass


# Directory containing template YAML files
TEMPLATES_DIR = Path(__file__).parent


def load_template(name: str) -> dict:
    """
    Load a template by name.

    Args:
        name: Template name (without .yaml extension)

    Returns:
        Dictionary with template configuration

    Raises:
        TemplateNotFoundError: If template doesn't exist
    """
    template_path = TEMPLATES_DIR / f"{name}.yaml"

    if not template_path.exists():
        available = list_templates()
        raise TemplateNotFoundError(
            f"Template '{name}' not found. Available templates: {', '.join(available)}"
        )

    with open(template_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    # Validate required fields
    if not template.get("name"):
        template["name"] = name
    if not template.get("sections"):
        template["sections"] = []

    return template


def list_templates() -> list[str]:
    """
    List all available template names.

    Returns:
        List of template names (without .yaml extension)
    """
    templates = []
    for file in TEMPLATES_DIR.glob("*.yaml"):
        templates.append(file.stem)
    return sorted(templates)


def get_template_info() -> list[dict]:
    """
    Get information about all available templates.

    Returns:
        List of dicts with 'name' and 'description' for each template
    """
    info = []
    for name in list_templates():
        try:
            template = load_template(name)
            info.append(
                {
                    "name": template.get("name", name),
                    "description": template.get("description", "No description"),
                }
            )
        except Exception:
            info.append({"name": name, "description": "Error loading template"})
    return info


def get_template_prompt(template: dict, language: str = "en") -> str:
    """
    Generate prompt section from template.

    Args:
        template: Template dictionary (from load_template)
        language: Language code ('en' or 'nl')

    Returns:
        String with formatted section instructions for the prompt
    """
    sections = template.get("sections", [])
    if not sections:
        return ""

    # Language-specific labels
    if language == "nl":
        header = "VERPLICHTE RAPPORT STRUCTUUR (volg dit template exact):"
        section_label = "Sectie"
        focus_label = "Focus"
    else:
        header = "MANDATORY REPORT STRUCTURE (follow this template exactly):"
        section_label = "Section"
        focus_label = "Focus"

    lines = [header, ""]

    for i, section in enumerate(sections, 1):
        title = section.get("title", f"{section_label} {i}")
        prompt = section.get("prompt", "")

        lines.append(f"## {title}")
        if prompt:
            lines.append(f"{focus_label}: {prompt}")
        lines.append("")

    # Add template-specific notes if present
    notes = template.get("notes")
    if notes:
        lines.append(notes)
        lines.append("")

    return "\n".join(lines)


def get_default_template() -> dict:
    """
    Get the default template.

    Returns:
        Default template dictionary, or minimal template if not found
    """
    try:
        return load_template("default")
    except TemplateNotFoundError:
        # Return minimal template if default.yaml doesn't exist
        return {
            "name": "default",
            "description": "Standard research report",
            "sections": [],
        }
