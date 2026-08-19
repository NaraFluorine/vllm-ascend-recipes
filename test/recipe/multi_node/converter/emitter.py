"""Safely emit and verify an executable multi-node plan bundle."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from .model import BundleSpec


class EmitError(ValueError):
    """The bundle cannot be written safely or does not match generated output."""


_KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MULTI_NODE_ROOT = Path(__file__).resolve().parents[1]
_GENERATED_ROOT = _MULTI_NODE_ROOT / ".generated"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _field(value: object, name: str, default: Any = None) -> Any:
    """Read one planner field from a dataclass or mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _plain(value: Any) -> Any:
    """Convert frozen planner dataclasses into YAML-safe built-in values."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _node_document(node: object) -> dict[str, Any]:
    """Serialize one logical node without leaking absent optional fields."""
    result = {
        "id": _field(node, "id"),
        "role": _field(node, "role"),
        "launch": _field(node, "launch"),
    }
    readiness = _field(node, "readiness")
    if readiness is not None:
        result["readiness"] = _plain(readiness)
    return result


def _stage_document(stage: object) -> dict[str, Any]:
    """Serialize stage and step specs in the order chosen by the planner."""
    steps = []
    for step in _field(stage, "steps", []):
        item = {
            "id": _field(step, "id"),
            "script": _field(step, "script"),
            "timeout_seconds": _field(step, "timeout_seconds"),
        }
        inputs = _field(step, "inputs", {})
        if inputs:
            item["inputs"] = _plain(inputs)
        steps.append(item)
    return {
        "id": _field(stage, "id"),
        "failure_category": _field(stage, "failure_category"),
        "steps": steps,
    }


def _source_recipe(bundle: BundleSpec) -> str:
    """Normalize source provenance to one repository-relative POSIX path."""
    source = Path(bundle.source_recipe)
    if source.is_absolute():
        try:
            source = source.resolve().relative_to(_REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise EmitError(
                f"source Recipe must be inside {_REPOSITORY_ROOT}: {source}"
            ) from error
    if ".." in source.parts:
        raise EmitError(f"source Recipe must not escape the repository: {source}")
    return source.as_posix()


def plan_document(bundle: BundleSpec) -> dict[str, Any]:
    """Build the stable v1 YAML representation of a planned bundle."""
    document: dict[str, Any] = {
        "api_version": "multi-node/v1",
        "kind": "MultiNodePlan",
        "metadata": {
            "name": bundle.name,
            "source_recipe": _source_recipe(bundle),
            "test_id": bundle.test_id,
            "digests": {
                "recipe": bundle.recipe_digest,
                "parameters": bundle.parameter_digest,
            },
        },
        "model": {
            "id": bundle.model_id,
            "cache_path": bundle.model_id,
            "served_name": bundle.served_name,
        },
        "resources": {"npu_per_node": bundle.npu_per_node},
        "nodes": [_node_document(node) for node in bundle.nodes],
    }
    if bundle.gateway is not None:
        document["gateway"] = _plain(bundle.gateway)
    document["stages"] = [_stage_document(stage) for stage in bundle.stages]
    return document


def _safe_output(output: Path) -> Path:
    """Restrict generated bundles to the ignored repository-owned output tree."""
    output = output.expanduser()
    if output.is_symlink():
        raise EmitError(f"output must not be a symlink: {output}")
    output = output.resolve(strict=False)
    generated_root = _GENERATED_ROOT.resolve()
    try:
        relative = output.relative_to(generated_root)
    except ValueError as error:
        raise EmitError(
            f"output must be inside {generated_root}: {output}"
        ) from error
    if not relative.parts:
        raise EmitError("output cannot be the generated root")
    if not _KEBAB_CASE.fullmatch(output.name):
        raise EmitError(f"plan name must be kebab-case: {output.name}")
    return output


def _safe_relative_file(name: str) -> Path:
    """Reject absolute paths, traversal, and converter-owned file collisions."""
    posix = PurePosixPath(name)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise EmitError(f"invalid generated file path: {name}")
    if posix.parts[0] in {".", ""} or name in {"plan.yaml", "README.md"}:
        raise EmitError(f"reserved generated file path: {name}")
    return Path(*posix.parts)


def _readme(bundle: BundleSpec) -> str:
    """Return concise, deterministic provenance for generated files."""
    return (
        f"# {bundle.name}\n\n"
        "This directory is generated by the multi-node Recipe converter. "
        "Do not edit it manually.\n\n"
        f"- Source Recipe: `{_source_recipe(bundle)}`\n"
        f"- Test ID: `{bundle.test_id}`\n"
        f"- Recipe digest: `{bundle.recipe_digest}`\n"
        f"- Parameter digest: `{bundle.parameter_digest}`\n"
    )


def _load_generated_plan(path: Path) -> None:
    """Round-trip the generated plan through the runtime's real loader."""
    try:
        from test.recipe.multi_node.scripts.plan import load_plan
    except ModuleNotFoundError:
        from scripts.plan import load_plan

    load_plan(path)


def _write_candidate(bundle: BundleSpec, directory: Path) -> None:
    """Write, syntax-check, and runtime-decode a complete candidate bundle."""
    document = plan_document(bundle)
    (directory / "plan.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (directory / "README.md").write_text(_readme(bundle), encoding="utf-8")

    for name, content in bundle.files.items():
        relative = _safe_relative_file(str(name))
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(content, str):
            raise EmitError(f"generated file must contain text: {name}")
        target.write_text(content, encoding="utf-8")
        if target.suffix == ".sh":
            target.chmod(0o755)

    shell_scripts = sorted(directory.rglob("*.sh"))
    for script in shell_scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise EmitError(f"bash syntax check failed for {script}: {detail}")
    _load_generated_plan(directory / "plan.yaml")


def emit_bundle(bundle: BundleSpec, output: Path) -> None:
    """Generate and atomically replace one ignored runtime bundle."""
    output = _safe_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    )
    try:
        _write_candidate(bundle, candidate)
        backup: Path | None = None
        if output.exists():
            if not output.is_dir() or output.is_symlink():
                raise EmitError(f"output must be a real directory: {output}")
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.old.", dir=output.parent)
            )
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(candidate, output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


__all__ = ["EmitError", "emit_bundle", "plan_document"]
