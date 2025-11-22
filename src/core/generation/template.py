"""Template Engine for Jinja2-based C++ code generation."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import filters


class TemplateEngine:
    """
    Manages Jinja2 templates for C++ code generation.
    """

    def __init__(self, template_dir: str | None = None):
        """
        Initialize template engine.

        Args:
            template_dir: Directory containing Jinja2 templates.
                         If None, uses default templates/ directory.
        """
        path_dir: Path
        if template_dir is None:
            # Default to src/templates/ directory
            src_root = Path(__file__).parent.parent.parent
            path_dir = src_root / "templates"
        else:
            path_dir = Path(template_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(path_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self._register_filters()

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters for C++ code generation."""
        self.env.filters["cpp_type"] = filters.cpp_type
        self.env.filters["arithmetic_op"] = filters.arithmetic_op
        self.env.filters["format_args"] = filters.format_args
        self.env.filters["cpp_lib"] = filters.cpp_lib

    def render_cpp_code(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render C++ code from template.

        Args:
            template_name: Name of the template file
            context: Dictionary of template variables

        Returns:
            Rendered C++ code as string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_cmake(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render CMakeLists.txt from template.

        Args:
            template_name: Name of the template file
            context: Dictionary of template variables

        Returns:
            Rendered CMake content as string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_image_header(self) -> str:
        """Render image header file (dependency-free)"""
        template = self.env.get_template("headers/image.h.j2")
        return template.render()

    def render_validator_header(self) -> str:
        """Render validator header file"""
        template = self.env.get_template("headers/validator.h.j2")
        return template.render()
