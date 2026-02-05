"""Fault template models for simulation."""

from .module_library import ModuleSpec, MODULE_LIBRARY, module_spec_by_label, module_specs_by_system, module_templates
from .templates import TemplateResult, apply_template, select_template

__all__ = [
    "TemplateResult",
    "apply_template",
    "select_template",
    "ModuleSpec",
    "MODULE_LIBRARY",
    "module_spec_by_label",
    "module_specs_by_system",
    "module_templates",
]
