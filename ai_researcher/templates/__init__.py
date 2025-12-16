"""Research report templates module."""

from .loader import (
    load_template,
    list_templates,
    get_template_prompt,
    get_template_info,
    TemplateNotFoundError,
)

__all__ = [
    "load_template",
    "list_templates",
    "get_template_prompt",
    "get_template_info",
    "TemplateNotFoundError",
]
