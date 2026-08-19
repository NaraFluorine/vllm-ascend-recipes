#!/usr/bin/env python3
"""Immutable Multi-node framework outcomes and small JSON serialization helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


RESULT_SCHEMA_VERSION = "multi-node-result/v2"
NODE_RESULT_SCHEMA_VERSION = "multi-node-node-result/v2"
STOP_SIGNAL_SCHEMA_VERSION = "multi-node-stop-signal/v1"
EXECUTION_STATUSES = frozenset({"passed", "failed", "cancelled", "aborted"})
RUN_STATUSES = frozenset({"passed", "failed", "cancelled"})
STOP_KINDS = frozenset({"completed", "failed", "cancelled"})


def _non_empty_string(value: object, field: str) -> str:
    """Validate a required string used by a serialized protocol object."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    """Validate an object-shaped protocol field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


@dataclass(frozen=True)
class RunFailure:
    """A stable failure category and human-readable diagnostic message."""
    category: str
    message: str

    def __post_init__(self) -> None:
        """Reject incomplete failures at construction time."""
        _non_empty_string(self.category, "failure.category")
        _non_empty_string(self.message, "failure.message")

    def to_dict(self) -> dict[str, str]:
        """Serialize the failure without adding transport-specific fields."""
        return {"category": self.category, "message": self.message}

    @classmethod
    def from_dict(cls, value: object) -> RunFailure:
        """Decode and validate a failure object."""
        raw = _mapping(value, "failure")
        return cls(
            category=_non_empty_string(raw.get("category"), "failure.category"),
            message=_non_empty_string(raw.get("message"), "failure.message"),
        )


def _failure_or_none(value: object) -> RunFailure | None:
    """Decode an optional failure field."""
    return None if value is None else RunFailure.from_dict(value)


def _cleanup_failures(errors: Iterable[RunFailure]) -> tuple[RunFailure, ...]:
    """Freeze cleanup errors and enforce their dedicated category."""
    cleanup = tuple(errors)
    if any(error.category != "cleanup_failed" for error in cleanup):
        raise ValueError("cleanup_errors must use category cleanup_failed")
    return cleanup


@dataclass(frozen=True)
class StopSignal:
    """An early run-wide signal that tells every node to begin cleanup."""

    kind: str
    origin_node_id: str
    failure: RunFailure | None = None

    def __post_init__(self) -> None:
        """Enforce consistency between stop kind and failure metadata."""
        if self.kind not in STOP_KINDS:
            raise ValueError(f"unknown stop kind: {self.kind}")
        _non_empty_string(self.origin_node_id, "origin_node_id")
        if self.kind == "completed" and self.failure is not None:
            raise ValueError("completed stop signal must not contain a failure")
        if self.kind in {"failed", "cancelled"} and self.failure is None:
            raise ValueError(f"{self.kind} stop signal requires a failure")
        if self.kind == "cancelled" and self.failure is not None:
            if self.failure.category != "cancelled":
                raise ValueError("cancelled stop signal requires cancelled failure")
        if self.kind == "failed" and self.failure is not None:
            if self.failure.category in {"cancelled", "cleanup_failed"}:
                raise ValueError("failed stop signal requires an execution failure")

    def to_dict(self) -> dict[str, Any]:
        """Serialize a versioned stop signal for coordinator transport."""
        return {
            "schema_version": STOP_SIGNAL_SCHEMA_VERSION,
            "kind": self.kind,
            "origin_node_id": self.origin_node_id,
            "failure": self.failure.to_dict() if self.failure else None,
        }

    @classmethod
    def from_dict(cls, value: object) -> StopSignal:
        """Decode a stop signal and reject unknown schema versions."""
        raw = _mapping(value, "stop signal")
        if raw.get("schema_version") != STOP_SIGNAL_SCHEMA_VERSION:
            raise ValueError("unknown stop signal schema_version")
        return cls(
            kind=_non_empty_string(raw.get("kind"), "stop signal.kind"),
            origin_node_id=_non_empty_string(
                raw.get("origin_node_id"), "stop signal.origin_node_id"
            ),
            failure=_failure_or_none(raw.get("failure")),
        )


@dataclass(frozen=True)
class NodeOutcome:
    """One node's immutable outcome, constructed only after local cleanup."""

    node_id: str
    execution_status: str
    failure: RunFailure | None = None
    cleanup_errors: tuple[RunFailure, ...] = ()

    def __post_init__(self) -> None:
        """Enforce local execution and cleanup outcome invariants."""
        _non_empty_string(self.node_id, "node_id")
        if self.execution_status not in EXECUTION_STATUSES:
            raise ValueError(f"unknown execution status: {self.execution_status}")
        object.__setattr__(self, "cleanup_errors", _cleanup_failures(self.cleanup_errors))

        if self.execution_status in {"failed", "cancelled"}:
            if self.failure is None:
                raise ValueError(
                    f"{self.execution_status} execution requires a failure"
                )
        elif self.failure is not None:
            raise ValueError(
                f"{self.execution_status} execution must not contain a local failure"
            )

        if self.execution_status == "cancelled" and self.failure is not None:
            if self.failure.category != "cancelled":
                raise ValueError("cancelled execution requires cancelled failure")
        if self.execution_status == "failed" and self.failure is not None:
            if self.failure.category in {"cancelled", "cleanup_failed"}:
                raise ValueError("failed execution requires an execution failure")

    @property
    def status(self) -> str:
        """Return the final local status without discarding execution status."""
        return "failed" if self.cleanup_errors else self.execution_status

    def to_dict(self) -> dict[str, Any]:
        """Serialize the final post-cleanup node outcome."""
        return {
            "schema_version": NODE_RESULT_SCHEMA_VERSION,
            "node_id": self.node_id,
            "status": self.status,
            "execution_status": self.execution_status,
            "failure": self.failure.to_dict() if self.failure else None,
            "cleanup_errors": [error.to_dict() for error in self.cleanup_errors],
        }

    @classmethod
    def from_dict(cls, value: object) -> NodeOutcome:
        """Decode a node outcome and verify its derived final status."""
        raw = _mapping(value, "node outcome")
        if raw.get("schema_version") != NODE_RESULT_SCHEMA_VERSION:
            raise ValueError("unknown node outcome schema_version")
        raw_cleanup = raw.get("cleanup_errors")
        if not isinstance(raw_cleanup, list):
            raise ValueError("node outcome.cleanup_errors must be a list")
        outcome = cls(
            node_id=_non_empty_string(raw.get("node_id"), "node outcome.node_id"),
            execution_status=_non_empty_string(
                raw.get("execution_status"), "node outcome.execution_status"
            ),
            failure=_failure_or_none(raw.get("failure")),
            cleanup_errors=tuple(RunFailure.from_dict(item) for item in raw_cleanup),
        )
        if raw.get("status") != outcome.status:
            raise ValueError("node outcome.status does not match its final outcome")
        return outcome


@dataclass(frozen=True)
class RunOutcome:
    """The leader's final outcome, built once from final node outcomes."""

    plan: str
    status: str
    nodes: Mapping[str, NodeOutcome]
    stages: Mapping[str, Any]
    failure: RunFailure | None = None
    failure_node_id: str | None = None
    missing_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the leader's aggregate against every reported node."""
        _non_empty_string(self.plan, "plan")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unknown run status: {self.status}")

        nodes = dict(self.nodes)
        for node_id, outcome in nodes.items():
            if node_id != outcome.node_id:
                raise ValueError(f"node outcome key does not match {outcome.node_id}")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "stages", dict(self.stages))

        missing = tuple(self.missing_nodes)
        if len(set(missing)) != len(missing) or any(not item for item in missing):
            raise ValueError("missing_nodes must contain unique non-empty node ids")
        if set(missing) & nodes.keys():
            raise ValueError("reported and missing nodes must not overlap")
        object.__setattr__(self, "missing_nodes", missing)

        if self.failure_node_id is not None:
            _non_empty_string(self.failure_node_id, "failure_node_id")
            if self.failure_node_id not in nodes:
                raise ValueError("failure_node_id must identify a reported node")
            if nodes[self.failure_node_id].execution_status == "aborted":
                raise ValueError("aborted node must not be the primary failure")

        if self.status == "passed":
            if self.failure is not None or self.failure_node_id is not None or missing:
                raise ValueError("passed run must not contain failure metadata")
            if any(outcome.status != "passed" for outcome in nodes.values()):
                raise ValueError("passed run requires every node to pass")
        else:
            if self.failure is None:
                raise ValueError(f"{self.status} run requires a failure")

        if self.status == "cancelled":
            if self.failure is None or self.failure.category != "cancelled":
                raise ValueError("cancelled run requires cancelled failure")
            if missing or any(outcome.status == "failed" for outcome in nodes.values()):
                raise ValueError("hard or missing node failure requires failed run")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the versioned aggregate result."""
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "plan": self.plan,
            "status": self.status,
            "failure": self.failure.to_dict() if self.failure else None,
            "failure_node_id": self.failure_node_id,
            "missing_nodes": list(self.missing_nodes),
            "nodes": {
                node_id: outcome.to_dict() for node_id, outcome in self.nodes.items()
            },
            "stages": dict(self.stages),
        }

    @classmethod
    def from_dict(cls, value: object) -> RunOutcome:
        """Decode and fully validate an aggregate result."""
        raw = _mapping(value, "run outcome")
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("unknown run outcome schema_version")
        raw_nodes = _mapping(raw.get("nodes"), "run outcome.nodes")
        raw_stages = _mapping(raw.get("stages"), "run outcome.stages")
        raw_missing = raw.get("missing_nodes")
        if not isinstance(raw_missing, list):
            raise ValueError("run outcome.missing_nodes must be a list")
        failure_node_id = raw.get("failure_node_id")
        if failure_node_id is not None and not isinstance(failure_node_id, str):
            raise ValueError("run outcome.failure_node_id must be a string or null")
        return cls(
            plan=_non_empty_string(raw.get("plan"), "run outcome.plan"),
            status=_non_empty_string(raw.get("status"), "run outcome.status"),
            failure=_failure_or_none(raw.get("failure")),
            failure_node_id=failure_node_id,
            missing_nodes=tuple(
                _non_empty_string(item, "run outcome.missing_nodes item")
                for item in raw_missing
            ),
            nodes={
                _non_empty_string(node_id, "run outcome node id"): NodeOutcome.from_dict(
                    outcome
                )
                for node_id, outcome in raw_nodes.items()
            },
            stages=dict(raw_stages),
        )


def build_node_result(outcome: NodeOutcome) -> dict[str, Any]:
    """Serialize an already-final NodeOutcome without deriving another status."""
    return outcome.to_dict()


def build_final_result(outcome: RunOutcome) -> dict[str, Any]:
    """Serialize an already-final RunOutcome without re-aggregating node state."""
    return outcome.to_dict()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object while rejecting scalar or list documents."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
