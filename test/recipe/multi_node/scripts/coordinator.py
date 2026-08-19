#!/usr/bin/env python3
"""HTTP coordination for Multi-node framework nodes."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

from scripts.result import NodeOutcome, RunOutcome, StopSignal


RUNNING = "running"
STOPPING = "stopping"


class CoordinatorError(RuntimeError):
    """A control-plane error with a stable machine-readable category."""

    def __init__(self, message: str, *, code: str = "coordinator_error") -> None:
        """Create an error suitable for local handling or an HTTP response."""
        super().__init__(message)
        self.code = code


class ObservedGlobalStop(RuntimeError):
    """Control flow for a stop initiated by this node's peer."""

    def __init__(self, signal: StopSignal) -> None:
        """Preserve the peer-originated signal that interrupted local work."""
        self.signal = signal
        message = signal.failure.message if signal.failure else signal.kind
        super().__init__(f"{signal.origin_node_id}: {message}")


class RunState:
    """Thread-safe transport state; final node outcomes are immutable."""

    def __init__(self, node_ids: list[str]) -> None:
        """Initialize pending readiness and empty immutable outcome storage."""
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique and non-empty")
        self.node_ids = tuple(node_ids)
        self.readiness = {node_id: "pending" for node_id in node_ids}
        self.stop_signal: StopSignal | None = None
        self.outcomes: dict[str, NodeOutcome] = {}
        self.final_outcome: RunOutcome | None = None
        self.condition = threading.Condition()

    def mark_ready(self, node_id: str) -> None:
        """Mark one known node ready, idempotently, before a stop is issued."""
        with self.condition:
            status = self._readiness(node_id)
            if status == "ready":
                return
            if self.stop_signal is not None:
                raise CoordinatorError(f"node {node_id} cannot become ready after stop")
            self.readiness[node_id] = "ready"
            self.condition.notify_all()

    def request_stop(self, signal: StopSignal) -> None:
        """Publish the first run-wide completion, failure, or cancellation."""
        with self.condition:
            self._readiness(signal.origin_node_id)
            if self.stop_signal is not None:
                if self.stop_signal == signal:
                    return
                raise CoordinatorError("run already has a different stop signal")
            if self.final_outcome is not None:
                raise CoordinatorError("run is already finalized")
            if signal.kind == "completed" and any(
                status != "ready" for status in self.readiness.values()
            ):
                raise CoordinatorError("execution cannot complete before all nodes are ready")
            self.stop_signal = signal
            self.condition.notify_all()

    def report_outcome(self, outcome: NodeOutcome) -> None:
        """Store one immutable post-cleanup node outcome."""
        with self.condition:
            self._readiness(outcome.node_id)
            previous = self.outcomes.get(outcome.node_id)
            if previous is not None:
                if previous == outcome:
                    return
                raise CoordinatorError(
                    f"node {outcome.node_id} already reported a different outcome"
                )
            if self.stop_signal is None:
                raise CoordinatorError("node outcome cannot be reported before stop")
            if self.final_outcome is not None:
                raise CoordinatorError("run is already finalized")
            self._validate_origin_outcome(outcome)
            self.outcomes[outcome.node_id] = outcome
            self.condition.notify_all()

    def finalize(self, outcome: RunOutcome) -> None:
        """Freeze the leader aggregate after validating reported/missing nodes."""
        with self.condition:
            if self.final_outcome is not None:
                if self.final_outcome == outcome:
                    return
                raise CoordinatorError("run already has a different final outcome")
            if self.stop_signal is None:
                raise CoordinatorError("run cannot finalize before stop")
            if dict(outcome.nodes) != self.outcomes:
                raise CoordinatorError("final outcome does not match reported node outcomes")
            missing = set(self.node_ids) - self.outcomes.keys()
            if set(outcome.missing_nodes) != missing:
                raise CoordinatorError("final outcome does not identify missing nodes")
            self.final_outcome = outcome
            self.condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        """Return a transport-safe snapshot for polling clients."""
        with self.condition:
            if self.final_outcome is not None:
                status = self.final_outcome.status
            elif self.stop_signal is not None:
                status = STOPPING
            else:
                status = RUNNING
            return {
                "status": status,
                "readiness": dict(sorted(self.readiness.items())),
                "stop_signal": (
                    self.stop_signal.to_dict() if self.stop_signal else None
                ),
                "outcomes": {
                    node_id: outcome.to_dict()
                    for node_id, outcome in sorted(self.outcomes.items())
                },
                "final_outcome": (
                    self.final_outcome.to_dict() if self.final_outcome else None
                ),
            }

    def _readiness(self, node_id: str) -> str:
        """Resolve a known node or normalize the error for HTTP callers."""
        try:
            return self.readiness[node_id]
        except KeyError as error:
            raise CoordinatorError(f"unknown node: {node_id}") from error

    def _validate_origin_outcome(self, outcome: NodeOutcome) -> None:
        """Require the stop origin's final facts to match its earlier signal."""
        signal = self.stop_signal
        if signal is None or signal.origin_node_id != outcome.node_id:
            return
        expected_status = {
            "completed": "passed",
            "failed": "failed",
            "cancelled": "cancelled",
        }[signal.kind]
        if outcome.execution_status != expected_status:
            raise CoordinatorError(
                f"stop origin {outcome.node_id} must report {expected_status} execution"
            )
        if signal.failure != outcome.failure:
            raise CoordinatorError("stop origin outcome does not match its stop failure")


def _handler(state: RunState) -> type[BaseHTTPRequestHandler]:
    """Bind a RunState instance into a minimal HTTP request handler class."""

    class Handler(BaseHTTPRequestHandler):
        """Expose state polling and the three node mutation endpoints."""

        def do_GET(self) -> None:  # noqa: N802
            """Serve the read-only state endpoint."""
            if self.path == "/state":
                self._send(200, state.snapshot())
            else:
                self._send(404, {"error": "endpoint not found"})

        def do_POST(self) -> None:  # noqa: N802
            """Dispatch ready, stop, and final-outcome node requests."""
            parts = self.path.strip("/").split("/")
            try:
                if len(parts) == 3 and parts[0] == "nodes":
                    node_id, action = parts[1:]
                    if action == "ready":
                        self._body_object(allow_empty=True)
                        state.mark_ready(node_id)
                    elif action == "stop":
                        signal = StopSignal.from_dict(self._body_object())
                        if signal.origin_node_id != node_id:
                            raise CoordinatorError(
                                "stop signal origin does not match request node"
                            )
                        state.request_stop(signal)
                    elif action == "outcome":
                        outcome = NodeOutcome.from_dict(self._body_object())
                        if outcome.node_id != node_id:
                            raise CoordinatorError(
                                "node outcome id does not match request node"
                            )
                        state.report_outcome(outcome)
                    else:
                        self._send(404, {"error": "endpoint not found"})
                        return
                else:
                    self._send(404, {"error": "endpoint not found"})
                    return
            except (CoordinatorError, ValueError) as error:
                self._send(400, {"error": str(error)})
                return
            self._send(200, state.snapshot())

        def _body_object(self, *, allow_empty: bool = False) -> Mapping[str, Any]:
            """Decode a JSON object body with optional empty-body support."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0 and allow_empty:
                    return {}
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError
                return value
            except (TypeError, ValueError) as error:
                raise CoordinatorError("invalid request body") from error

        def _send(self, status: int, value: object) -> None:
            """Write one compact JSON response with an exact content length."""
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress BaseHTTPRequestHandler access logs in service output."""
            pass

    return Handler


class _CoordinatorHTTPServer(ThreadingHTTPServer):
    """Wait for request threads so close cannot race the last outcome report."""

    daemon_threads = False
    block_on_close = True


class LeaderCoordinator:
    """Own the leader-local HTTP server and blocking state wait operations."""

    def __init__(
        self, node_ids: list[str], port: int, *, host: str = "0.0.0.0"
    ) -> None:
        """Create, but do not yet start, the coordinator server thread."""
        self.state = RunState(node_ids)
        self.server = _CoordinatorHTTPServer((host, port), _handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.started = False
        self.closed = False

    @property
    def port(self) -> int:
        """Return the bound port, including an OS-selected ephemeral port."""
        return int(self.server.server_address[1])

    def start(self) -> None:
        """Start serving once; repeated calls are harmless."""
        if self.started:
            return
        self.thread.start()
        self.started = True

    def close(self) -> None:
        """Stop serving and wait for all in-flight response threads."""
        if self.closed:
            return
        if self.started:
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()
        self.closed = True

    def wait_ready(
        self,
        timeout: int,
        check_processes: Callable[[], None],
        *,
        progress_callback: Callable[[dict[str, str]], None] | None = None,
        progress_interval_seconds: float = 30,
    ) -> None:
        """Wait for every node while continuing to supervise local processes."""

        def report_progress() -> None:
            """Copy mutable readiness before exposing it to diagnostics."""
            if progress_callback is not None:
                progress_callback(dict(self.state.readiness))

        self._wait(
            lambda: all(status == "ready" for status in self.state.readiness.values()),
            timeout,
            "nodes to become ready",
            check_processes,
            stop_on_signal=True,
            progress_callback=(
                report_progress
                if progress_callback is not None and progress_interval_seconds > 0
                else None
            ),
            progress_interval_seconds=progress_interval_seconds,
        )

    def wait_outcomes(self, timeout: int) -> dict[str, NodeOutcome]:
        """Wait for every reachable node to report its post-cleanup outcome."""
        self._wait(
            lambda: self.state.outcomes.keys() == self.state.readiness.keys(),
            timeout,
            "nodes to report final outcomes",
            lambda: None,
        )
        return dict(self.state.outcomes)

    def raise_if_stopped(self) -> None:
        """Interrupt leader work as soon as any node has requested a stop."""
        with self.state.condition:
            if self.state.stop_signal is not None:
                raise ObservedGlobalStop(self.state.stop_signal)

    def _wait(
        self,
        complete: Callable[[], bool],
        timeout: int,
        description: str,
        check_processes: Callable[[], None],
        *,
        stop_on_signal: bool = False,
        progress_callback: Callable[[], None] | None = None,
        progress_interval_seconds: float = 30,
    ) -> None:
        """Wait on the state condition with timeout and runtime supervision."""
        deadline = time.monotonic() + timeout
        next_progress = time.monotonic()
        with self.state.condition:
            while not complete():
                check_processes()
                if stop_on_signal and self.state.stop_signal is not None:
                    raise ObservedGlobalStop(self.state.stop_signal)
                now = time.monotonic()
                if progress_callback is not None and now >= next_progress:
                    progress_callback()
                    next_progress = now + progress_interval_seconds
                remaining = deadline - now
                if remaining <= 0:
                    raise CoordinatorError(f"timed out waiting for {description}")
                wait_seconds = min(1, remaining)
                if progress_callback is not None:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.001, next_progress - now),
                    )
                self.state.condition.wait(wait_seconds)


class CoordinatorClient:
    """A no-proxy HTTP client used by worker nodes and the leader runner."""

    def __init__(
        self, host: str, port: int, *, request_timeout: float = 5.0
    ) -> None:
        """Configure a direct client with a per-request timeout ceiling."""
        self.base_url = f"http://{host}:{port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.request_timeout = request_timeout

    def mark_ready(self, node_id: str, timeout: int) -> None:
        """Publish local readiness."""
        self._request(f"/nodes/{node_id}/ready", {}, timeout)

    def request_stop(self, signal: StopSignal, timeout: int = 5) -> None:
        """Publish a run-wide stop signal on behalf of its origin node."""
        self._request(
            f"/nodes/{signal.origin_node_id}/stop", signal.to_dict(), timeout
        )

    def report_outcome(self, outcome: NodeOutcome, timeout: int = 5) -> None:
        """Publish a node's immutable post-cleanup outcome."""
        self._request(
            f"/nodes/{outcome.node_id}/outcome", outcome.to_dict(), timeout
        )

    def wait_available(
        self, timeout: int, check_processes: Callable[[], None]
    ) -> None:
        """Retry startup connection refusal until the shared deadline expires."""
        deadline = time.monotonic() + timeout
        while True:
            check_processes()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CoordinatorError(
                    "timed out waiting for the coordinator",
                    code="coordinator_unreachable",
                )
            try:
                self._request("/state", None, remaining)
                return
            except CoordinatorError as error:
                if error.code != "coordinator_unreachable":
                    raise
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def wait_stop(
        self, timeout: int, check_processes: Callable[[], None]
    ) -> StopSignal:
        """Poll state until the leader or a peer publishes a stop signal."""
        deadline = time.monotonic() + timeout
        while True:
            check_processes()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CoordinatorError("timed out waiting for the run stop signal")
            state = self._request("/state", None, remaining)
            raw_signal = state.get("stop_signal")
            if raw_signal is not None:
                try:
                    return StopSignal.from_dict(raw_signal)
                except ValueError as error:
                    raise CoordinatorError(
                        "coordinator returned an invalid stop signal"
                    ) from error
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def _request(
        self, path: str, value: object | None, timeout: float
    ) -> dict[str, Any]:
        """Issue one direct JSON request and normalize transport failures."""
        body = None if value is None else json.dumps(value).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"} if body else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with self.opener.open(
                request, timeout=min(timeout, self.request_timeout)
            ) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("error")
            except (AttributeError, ValueError):
                message = None
            error.close()
            raise CoordinatorError(
                str(message or f"coordinator returned HTTP {error.code}")
            ) from error
        except OSError as error:
            raise CoordinatorError(
                f"cannot reach coordinator at {self.base_url}",
                code="coordinator_unreachable",
            ) from error
        except ValueError as error:
            raise CoordinatorError("coordinator returned invalid JSON") from error
        if not isinstance(result, dict):
            raise CoordinatorError("coordinator returned invalid JSON")
        return result
