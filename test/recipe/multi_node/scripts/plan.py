#!/usr/bin/env python3
"""Load the executable Multi-node framework v1 intermediate plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PLAN_API_VERSION = "multi-node/v1"
PLAN_KIND = "MultiNodePlan"
MAX_PORT = 65535


class PlanError(ValueError):
    """The plan or local hosts file cannot be executed."""


@dataclass(frozen=True)
class Model:
    """Model identity and cache information required by every node."""
    id: str
    cache_path: str
    served_name: str


@dataclass(frozen=True)
class Resources:
    """Per-node accelerator requirements used by infrastructure adapters."""
    npu_per_node: int


@dataclass(frozen=True)
class Readiness:
    """One or more consecutive HTTP endpoints exposed by a node launcher."""
    port_start: int
    count: int = 1
    health_path: str = "/health"


@dataclass(frozen=True)
class Node:
    """A logical node and its executable launch script."""
    id: str
    index: int
    role: str
    launch: str
    readiness: Readiness | None


@dataclass(frozen=True)
class Gateway:
    """An optional leader-side gateway placed in front of node services."""
    launch: str
    port: int
    health_path: str = "/healthcheck"


@dataclass(frozen=True)
class ScriptStep:
    """One executable stage step with generic converter-owned inputs."""
    id: str
    script: str
    timeout_seconds: int
    inputs: dict[str, Any]


@dataclass(frozen=True)
class Stage:
    """An ordered collection of steps sharing a failure category."""
    id: str
    failure_category: str
    steps: list[ScriptStep]


@dataclass(frozen=True)
class Plan:
    """The complete executable intermediate representation for one run."""
    path: Path
    name: str
    model: Model
    resources: Resources
    nodes: list[Node]
    gateway: Gateway | None
    stages: list[Stage]

    @property
    def directory(self) -> Path:
        """Return the directory used to resolve every plan-relative script."""
        return self.path.parent

    @property
    def leader(self) -> Node:
        """Return the first node, which owns coordination and stage execution."""
        return self.nodes[0]

    def node(self, node_id: str) -> Node:
        """Resolve a logical node by id."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise PlanError(f"Unknown node: {node_id}")


@dataclass(frozen=True)
class Host:
    """Runtime address and optional communication interface for one node."""
    address: str
    interface: str | None = None


def _mapping(
    value: Any, field: str, required: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Require a mapping and optionally verify the presence of named fields."""
    if not isinstance(value, dict):
        raise PlanError(f"{field} must be a mapping")
    missing = [key for key in required if key not in value]
    if missing:
        raise PlanError(f"{field} is missing fields: {', '.join(missing)}")
    return value


def _text(value: Any, field: str) -> str:
    """Require a non-empty string while preserving a precise field name."""
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{field} must be a non-empty string")
    return value


def _list(value: Any, field: str, *, non_empty: bool = False) -> list[Any]:
    """Require a list and optionally reject an empty protocol collection."""
    if not isinstance(value, list):
        raise PlanError(f"{field} must be a list")
    if non_empty and not value:
        raise PlanError(f"{field} must not be empty")
    return value


def _positive_integer(value: Any, field: str) -> int:
    """Require a positive integer without accepting booleans or floats."""
    if type(value) is not int or value <= 0:
        raise PlanError(f"{field} must be a positive integer")
    return value


def _port(value: Any, field: str) -> int:
    """Require one valid TCP port."""
    port = _positive_integer(value, field)
    if port > MAX_PORT:
        raise PlanError(f"{field} must be at most {MAX_PORT}")
    return port


def _health_path(value: Any, field: str) -> str:
    """Require an absolute HTTP path used by readiness probes."""
    path = _text(value, field)
    if not path.startswith("/"):
        raise PlanError(f"{field} must start with /")
    return path


def _script_path(value: Any, field: str, plan_directory: Path) -> str:
    """Validate one existing plan-relative script without allowing escape."""
    text = _text(value, field)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise PlanError(f"{field} must be a plan-relative path")
    root = plan_directory.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PlanError(f"{field} must stay inside the plan directory") from error
    if not candidate.is_file():
        raise PlanError(f"{field} does not exist: {text}")
    return relative.as_posix()


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping and normalize file and parser failures."""
    if not path.is_file():
        raise PlanError(f"File not found: {path}")
    try:
        return _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")), str(path)
        )
    except OSError as error:
        raise PlanError(f"Cannot read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise PlanError(f"Invalid YAML in {path}: {error}") from error


def _decode_readiness(value: Any, field: str) -> Readiness | None:
    """Decode optional node readiness without deriving topology semantics."""
    if value is None:
        return None
    raw = _mapping(value, field, ("port_start",))
    port_start = _port(raw["port_start"], f"{field}.port_start")
    count = _positive_integer(raw.get("count", 1), f"{field}.count")
    if port_start + count - 1 > MAX_PORT:
        raise PlanError(f"{field} port range exceeds {MAX_PORT}")
    return Readiness(
        port_start=port_start,
        count=count,
        health_path=_health_path(
            raw.get("health_path", "/health"), f"{field}.health_path"
        ),
    )


def _decode_step(
    value: Any, field: str, plan_directory: Path
) -> ScriptStep:
    """Decode a converter-produced executable step."""
    raw = _mapping(value, field, ("id", "script"))
    step_id = _text(raw["id"], f"{field}.id")
    return ScriptStep(
        id=step_id,
        script=_script_path(raw["script"], f"{field}.script", plan_directory),
        timeout_seconds=_positive_integer(
            raw.get("timeout_seconds", 300), f"{field}.timeout_seconds"
        ),
        inputs=_mapping(raw.get("inputs", {}), f"{field}.inputs"),
    )


def _decode_stage(value: Any, index: int, plan_directory: Path) -> Stage:
    """Decode a stage while preserving converter-defined step order."""
    field = f"stages[{index}]"
    raw = _mapping(value, field, ("id", "failure_category", "steps"))
    stage_id = _text(raw["id"], f"{field}.id")
    steps = [
        _decode_step(step, f"{field}.steps[{step_index}]", plan_directory)
        for step_index, step in enumerate(
            _list(raw["steps"], f"{field}.steps", non_empty=True)
        )
    ]
    step_ids = [step.id for step in steps]
    if len(step_ids) != len(set(step_ids)):
        raise PlanError(f"{field}.steps must have unique ids")
    return Stage(
        id=stage_id,
        failure_category=_text(
            raw["failure_category"], f"{field}.failure_category"
        ),
        steps=steps,
    )


def load_plan(path: Path) -> Plan:
    """Validate and decode one executable intermediate plan.

    Converter-owned topology semantics stay outside the Runtime. This boundary
    still verifies the protocol envelope, primitive types, endpoint ranges, and
    executable paths so stale or damaged bundles fail with actionable errors.
    """
    path = path.resolve()
    plan_directory = path.parent
    raw = _mapping(
        _read_yaml(path),
        "plan",
        (
            "api_version",
            "kind",
            "metadata",
            "model",
            "resources",
            "nodes",
            "stages",
        ),
    )
    if raw["api_version"] != PLAN_API_VERSION:
        raise PlanError(f"plan api_version must be {PLAN_API_VERSION}")
    if raw["kind"] != PLAN_KIND:
        raise PlanError(f"plan kind must be {PLAN_KIND}")

    metadata = _mapping(raw["metadata"], "metadata", ("name",))
    model = _mapping(
        raw["model"], "model", ("id", "cache_path", "served_name")
    )
    resources = _mapping(raw["resources"], "resources", ("npu_per_node",))

    nodes: list[Node] = []
    for index, value in enumerate(_list(raw["nodes"], "nodes", non_empty=True)):
        field = f"nodes[{index}]"
        node = _mapping(value, field, ("id", "role", "launch"))
        nodes.append(
            Node(
                id=_text(node["id"], f"{field}.id"),
                index=index,
                role=_text(node["role"], f"{field}.role"),
                launch=_script_path(
                    node["launch"], f"{field}.launch", plan_directory
                ),
                readiness=_decode_readiness(
                    node.get("readiness"), f"{field}.readiness"
                ),
            )
        )
    node_ids = [node.id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise PlanError("nodes must have unique ids")

    gateway_raw = raw.get("gateway")
    if gateway_raw is None:
        gateway = None
    else:
        gateway_mapping = _mapping(gateway_raw, "gateway", ("launch", "port"))
        gateway = Gateway(
            launch=_script_path(
                gateway_mapping["launch"], "gateway.launch", plan_directory
            ),
            port=_port(gateway_mapping["port"], "gateway.port"),
            health_path=_health_path(
                gateway_mapping.get("health_path", "/healthcheck"),
                "gateway.health_path",
            ),
        )
    if gateway is None and nodes[0].readiness is None:
        raise PlanError("leader readiness is required when gateway is absent")

    stages = [
        _decode_stage(stage, index, plan_directory)
        for index, stage in enumerate(_list(raw["stages"], "stages"))
    ]
    stage_ids = [stage.id for stage in stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise PlanError("stages must have unique ids")

    return Plan(
        path=path,
        name=_text(metadata["name"], "metadata.name"),
        model=Model(
            id=_text(model["id"], "model.id"),
            cache_path=_text(model["cache_path"], "model.cache_path"),
            served_name=_text(model["served_name"], "model.served_name"),
        ),
        resources=Resources(
            npu_per_node=_positive_integer(
                resources["npu_per_node"], "resources.npu_per_node"
            )
        ),
        nodes=nodes,
        gateway=gateway,
        stages=stages,
    )


def load_hosts(path: Path, plan: Plan) -> dict[str, Host]:
    """Load runtime hosts and require an exact match with logical plan nodes."""
    raw = _mapping(_read_yaml(path.resolve()), "hosts file", ("version", "hosts"))
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise PlanError("hosts version must be 1")
    hosts_raw = _mapping(raw["hosts"], "hosts")
    expected = {node.id for node in plan.nodes}
    actual = set(hosts_raw)
    if actual != expected:
        raise PlanError(
            "hosts keys must match plan nodes; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(str(key) for key in actual - expected)}"
        )

    hosts = {}
    for node_id, value in hosts_raw.items():
        field = f"hosts.{node_id}"
        host = _mapping(value, field, ("address",))
        interface = host.get("interface")
        hosts[node_id] = Host(
            address=_text(host["address"], f"{field}.address"),
            interface=(
                _text(interface, f"{field}.interface")
                if interface is not None
                else None
            ),
        )
    return hosts


def format_topology_summary(
    plan: Plan, hosts: dict[str, Host] | None = None
) -> str:
    """Print only information useful before starting a local or CI run."""
    lines = [
        f"Plan: {plan.name} ({len(plan.nodes)} nodes, "
        f"{plan.resources.npu_per_node} NPUs/node)",
        f"Model: {plan.model.id} (served as {plan.model.served_name})",
    ]
    for node in plan.nodes:
        host = ""
        if hosts:
            item = hosts[node.id]
            host = f" @{item.address}%{item.interface or 'auto'}"
        ready = ""
        if node.readiness:
            ready = f" ports={node.readiness.port_start}"
            if node.readiness.count > 1:
                ready += f"-{node.readiness.port_start + node.readiness.count - 1}"
        lines.append(f"{node.id}{host}: {node.role}, {node.launch}{ready}")
    if plan.gateway:
        lines.append(f"Gateway: {plan.gateway.launch} port={plan.gateway.port}")
    endpoint_port = (
        plan.gateway.port if plan.gateway else plan.leader.readiness.port_start
    )
    if hosts:
        lines.append(
            f"Endpoint: http://{hosts[plan.leader.id].address}:{endpoint_port}"
        )
    lines.append(
        "Stages: "
        + ", ".join(f"{stage.id}={len(stage.steps)}" for stage in plan.stages)
    )
    return "\n".join(lines)
