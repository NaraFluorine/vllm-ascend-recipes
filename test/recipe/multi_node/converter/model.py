"""Typed source and output models shared by the multi-node converter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias


ParameterValue: TypeAlias = str | int | float | bool


class ConversionError(ValueError):
    """Report an invalid converter input with a user-facing explanation."""


@dataclass(frozen=True)
class ScriptSource:
    """One named script embedded in a Recipe scenario."""

    language: str
    content: str


@dataclass(frozen=True)
class ScenarioSource:
    """The complete scenario-level input consumed by the converter."""

    recipe_path: Path
    test_id: str
    npu: str
    deployment: str
    case: str
    npu_per_node: int
    aisbench: tuple[str, ...]
    parameter_defaults: Mapping[str, ParameterValue]
    scripts: Mapping[str, ScriptSource]


@dataclass(frozen=True)
class ReadinessSpec:
    """HTTP endpoints polled while a node service starts."""

    port_start: int
    count: int = 1
    health_path: str = "/health"


@dataclass(frozen=True)
class NodeSpec:
    """One logical node and its generated launch script."""

    id: str
    role: str
    launch: str
    readiness: ReadinessSpec | None


@dataclass(frozen=True)
class GatewaySpec:
    """An optional gateway in front of the generated node services."""

    launch: str
    port: int
    health_path: str = "/healthcheck"


@dataclass(frozen=True)
class StepSpec:
    """One executable step in a generated test stage."""

    id: str
    script: str
    timeout_seconds: int
    inputs: Mapping[str, object]


@dataclass(frozen=True)
class StageSpec:
    """An ordered set of steps sharing a failure category."""

    id: str
    failure_category: str
    steps: tuple[StepSpec, ...]


@dataclass(frozen=True)
class BundleSpec:
    """The complete deterministic output prepared for plan emission."""

    name: str
    source_recipe: str
    test_id: str
    recipe_digest: str
    parameter_digest: str
    model_id: str
    served_name: str
    npu_per_node: int
    nodes: tuple[NodeSpec, ...]
    gateway: GatewaySpec | None
    stages: tuple[StageSpec, ...]
    files: Mapping[str, str]
