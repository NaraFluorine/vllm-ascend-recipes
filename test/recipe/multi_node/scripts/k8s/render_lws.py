#!/usr/bin/env python3
"""Render the simple LeaderWorkerSet manifest template with strict tokens."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import yaml


TOKEN = re.compile(r"{{\s*([a-z][a-z0-9_]*)\s*}}")


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Render an LWS template only when tokens and values match exactly.

    The renderer intentionally supports string substitution rather than a
    general template language. The final YAML shape is parsed and checked before
    it can be handed to kubectl.
    """
    placeholders = set(TOKEN.findall(template))
    provided = set(values)
    missing = placeholders - provided
    unexpected = provided - placeholders
    if missing:
        raise ValueError(f"missing LWS template values: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(
            f"unexpected LWS template values: {', '.join(sorted(unexpected))}"
        )
    if not all(isinstance(value, str) for value in values.values()):
        raise ValueError("LWS template values must all be strings")

    rendered = TOKEN.sub(lambda match: values[match.group(1)], template)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("unresolved or malformed LWS template token")

    documents = list(yaml.safe_load_all(rendered))
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError("rendered LWS template must contain one YAML mapping")
    manifest = documents[0]
    if manifest.get("apiVersion") != "leaderworkerset.x-k8s.io/v1":
        raise ValueError("rendered manifest has an unexpected apiVersion")
    if manifest.get("kind") != "LeaderWorkerSet":
        raise ValueError("rendered manifest is not a LeaderWorkerSet")
    return rendered


def parse_args() -> argparse.Namespace:
    """Parse paths for the template, strict JSON values, and output manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Render and atomically hand off one validated manifest file."""
    args = parse_args()
    raw_values = json.loads(args.values.read_text(encoding="utf-8"))
    if not isinstance(raw_values, dict) or not all(
        isinstance(name, str) for name in raw_values
    ):
        raise ValueError("LWS values file must contain a JSON object")
    rendered = render_template(
        args.template.read_text(encoding="utf-8"), raw_values
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
