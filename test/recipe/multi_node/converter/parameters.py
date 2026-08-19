"""Resolve Recipe parameter defaults and render scenario scripts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import replace

import yaml

from .model import (
    ConversionError,
    ParameterValue,
    ScenarioSource,
    ScriptSource,
)

_PARAMETER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLACEHOLDER = re.compile(r"{{\s*([^{}]+?)\s*}}")


def _validate_name(value: object, field: str) -> str:
    """Require a name that can appear unambiguously in ``{{name}}``."""
    if not isinstance(value, str) or _PARAMETER_NAME.fullmatch(value) is None:
        raise ConversionError(
            f"{field} must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return value


def _validate_value(value: object, field: str) -> ParameterValue:
    """Accept only deterministic scalar values suitable for direct rendering."""
    if not isinstance(value, (str, int, float, bool)):
        raise ConversionError(f"{field} must be a non-null scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ConversionError(f"{field} must be finite")
    return value


def _decode_override(value: str, index: int) -> tuple[str, ParameterValue]:
    """Decode one ``--set name=value`` argument as a YAML scalar."""
    if not isinstance(value, str) or "=" not in value:
        raise ConversionError(
            f"Override {index} must use name=value syntax, got {value!r}"
        )
    raw_name, raw_value = value.split("=", 1)
    name = _validate_name(raw_name, f"Override {index} name")
    try:
        decoded = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConversionError(
            f"Override {index} has an invalid YAML scalar: {error}"
        ) from error
    return name, _validate_value(decoded, f"Override {name!r}")


def _referenced_parameters(source: ScenarioSource) -> set[str]:
    """Return parameter names referenced by scenario scripts."""
    referenced: set[str] = set()
    for script in source.scripts.values():
        for match in _PLACEHOLDER.finditer(script.content):
            token = match.group(1).strip()
            if not _is_script_reference(token) and _PARAMETER_NAME.fullmatch(token):
                referenced.add(token)
    return referenced


def load_parameters(
    source: ScenarioSource, overrides: list[str]
) -> dict[str, ParameterValue]:
    """Resolve used Recipe defaults, applying ordered ``--set`` values last.

    Unused frontend defaults do not affect conversion. Explicit overrides are
    retained even when unused so rendering can report likely misspellings.
    """
    referenced = _referenced_parameters(source)
    values = {
        name: _validate_value(value, f"Parameter default {name!r}")
        for name, value in source.parameter_defaults.items()
        if name in referenced
    }
    for index, item in enumerate(overrides):
        name, value = _decode_override(item, index)
        values[name] = value
    return values


def _render_value(value: ParameterValue) -> str:
    """Render a scalar consistently across YAML and command-line sources."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_script_reference(token: str) -> bool:
    """Identify page-level script inclusions, which this phase must preserve."""
    return token.startswith("script:")


def render_scenario(
    source: ScenarioSource, values: Mapping[str, object]
) -> ScenarioSource:
    """Render parameter placeholders in scripts and return a new scenario.

    Shell variables such as ``$HOST`` and ``${PORT}`` are ordinary text. Page
    inclusions such as ``{{script:api-0}}`` are also retained for their owning
    renderer. Every other double-brace placeholder must resolve here.
    """
    parameters: dict[str, ParameterValue] = {}
    for raw_name, raw_value in values.items():
        name = _validate_name(raw_name, f"Parameter name {raw_name!r}")
        parameters[name] = _validate_value(raw_value, f"Parameter {name!r}")

    missing: dict[str, set[str]] = {}
    residual: dict[str, set[str]] = {}
    used: set[str] = set()
    rendered_scripts: dict[str, ScriptSource] = {}

    for script_name, script in source.scripts.items():
        def substitute(match: re.Match[str]) -> str:
            """Resolve one value placeholder while collecting diagnostics."""
            token = match.group(1).strip()
            if _is_script_reference(token):
                return match.group(0)
            if _PARAMETER_NAME.fullmatch(token) is None:
                residual.setdefault(script_name, set()).add(match.group(0))
                return match.group(0)
            if token not in parameters:
                missing.setdefault(token, set()).add(script_name)
                return match.group(0)
            used.add(token)
            return _render_value(parameters[token])

        content = _PLACEHOLDER.sub(substitute, script.content)
        for match in _PLACEHOLDER.finditer(content):
            token = match.group(1).strip()
            if _is_script_reference(token) or token in missing:
                continue
            residual.setdefault(script_name, set()).add(match.group(0))
        rendered_scripts[script_name] = replace(script, content=content)

    errors: list[str] = []
    if missing:
        details = "; ".join(
            f"{name} (scripts: {', '.join(sorted(script_names))})"
            for name, script_names in sorted(missing.items())
        )
        errors.append(f"missing parameters: {details}")
    if residual:
        details = "; ".join(
            f"{name}: {', '.join(sorted(placeholders))}"
            for name, placeholders in sorted(residual.items())
        )
        errors.append(f"unresolved placeholders: {details}")
    unused = sorted(set(parameters) - used)
    if unused:
        errors.append(f"unused parameters: {', '.join(unused)}")
    if errors:
        raise ConversionError("Cannot render scenario scripts; " + "; ".join(errors))

    return replace(source, scripts=rendered_scripts)
