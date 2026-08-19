#!/usr/bin/env python3
"""Execute one node from a multi-node framework intermediate plan."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[4]
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from scripts.coordinator import (  # noqa: E402
    CoordinatorClient,
    CoordinatorError,
    LeaderCoordinator,
    ObservedGlobalStop,
)
from scripts.plan import (  # noqa: E402
    Host,
    Node,
    Plan,
    PlanError,
    Stage,
    format_topology_summary,
    load_hosts,
    load_plan,
)
from scripts.process import (  # noqa: E402
    CancellationRequested,
    ManagedProcess,
    ManagedProcessExited,
    check_processes,
    signal_cancellation_event,
    start_process,
    stop_processes,
    wait_for_process,
)
from scripts.result import (  # noqa: E402
    NodeOutcome,
    RunFailure,
    RunOutcome,
    StopSignal,
    build_final_result,
    build_node_result,
    read_json,
    write_json_atomic,
)


DEFAULT_VLLM_ASCEND_ROOT = Path("/vllm-workspace/vllm-ascend")
MODEL_CACHE_ROOT = Path("/root/.cache/modelscope/hub/models")
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
MAX_RESULT_SUMMARY_CHARACTERS = 4096


class RunnerError(RuntimeError):
    """The top-level error raised when a node cannot execute its plan."""


class StageFailure(RuntimeError):
    """Carry one structured stage failure through the node lifecycle."""

    def __init__(self, failure: RunFailure) -> None:
        """Preserve the public failure category while remaining an exception."""
        self.failure = failure
        super().__init__(failure.message)


def parse_args() -> argparse.Namespace:
    """Parse the internal command-line interface used by ``run.sh``.

    ``--validate-only`` decodes and prints a plan without hosts, services, or
    hardware access. Normal users should invoke the shell entrypoint instead.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--hosts", type=Path)
    parser.add_argument("--node-id")
    parser.add_argument("--vllm-ascend-root", type=Path)
    parser.add_argument("--control-port", type=int, default=29599)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--run-timeout-seconds", type=int, default=7200)
    parser.add_argument("--progress-interval-seconds", type=float, default=30)
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/multi-node"))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def interface_addresses() -> dict[str, str]:
    """Return Linux interface-to-IPv4 mappings for communication setup.

    Minimal runtime images may omit iproute2, so an ioctl fallback avoids
    modifying the image solely to discover the local interface.
    """
    addresses: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    else:
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 4:
                addresses[fields[1]] = fields[3].split("/", 1)[0]
        if addresses:
            return addresses

    # Minimal runtime images may not contain iproute2. SIOCGIFADDR keeps local
    # and hostNetwork execution usable without adding another image dependency.
    try:
        import fcntl
        import struct

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            for _, interface in socket.if_nameindex():
                try:
                    request = struct.pack("256s", interface[:15].encode())
                    response = fcntl.ioctl(probe.fileno(), 0x8915, request)
                except OSError:
                    continue
                addresses[interface] = socket.inet_ntoa(response[20:24])
    except (ImportError, OSError):
        pass
    return addresses


def select_interface(host: Host) -> str:
    """Select the interface used by HCCL, Gloo, and tensor parallel traffic.

    An explicit hosts entry wins. Otherwise the address must map to a local
    interface; failing closed prevents collectives from silently using the
    wrong network.
    """
    if host.interface:
        return host.interface
    for interface, address in interface_addresses().items():
        if address == host.address:
            return interface
    raise RunnerError("cannot detect the local interface; set it in hosts.yaml")


def resolve_vllm_ascend_root(requested: Path | None) -> Path:
    """Resolve the vLLM Ascend source root exposed to plan-local scripts.

    The runner does not require the directory itself; scripts that consume an
    upstream launcher or proxy own that concrete dependency check.
    """
    root = requested or Path(
        os.environ.get("VLLM_ASCEND_ROOT", str(DEFAULT_VLLM_ASCEND_ROOT))
    )
    # The runner exposes this runtime contract but does not require every plan to
    # consume the upstream source tree. A plan that uses it fails at its own script.
    return root.expanduser().resolve()


def base_environment(
    plan: Plan,
    node: Node,
    hosts: dict[str, Host],
    interface: str,
    model_path: str,
    vllm_ascend_root: Path,
    control_port: int,
    plan_artifact_directory: Path,
    node_artifact_directory: Path,
) -> dict[str, str]:
    """Build the common environment for services, gateways, and stage steps.

    Static plan data and runtime hosts are combined once so business scripts do
    not re-parse protocol files or independently derive topology.
    """
    local_ip = hosts[node.id].address
    leader_ip = hosts[plan.leader.id].address
    environment = os.environ.copy()
    environment.update(
        {
            "MULTI_NODE_PLAN_DIR": str(plan.directory),
            "MULTI_NODE_REPOSITORY_ROOT": str(ROOT),
            "MULTI_NODE_NODE_ID": node.id,
            "MULTI_NODE_NODE_INDEX": str(node.index),
            "MULTI_NODE_NODE_ROLE": node.role,
            "MULTI_NODE_LOCAL_IP": local_ip,
            "MULTI_NODE_LOCAL_INTERFACE": interface,
            "MULTI_NODE_LEADER_IP": leader_ip,
            "MULTI_NODE_CONTROL_PORT": str(control_port),
            "MULTI_NODE_MODEL_ID": plan.model.id,
            "MULTI_NODE_MODEL_PATH": model_path,
            "MULTI_NODE_SERVED_MODEL_NAME": plan.model.served_name,
            "MULTI_NODE_VLLM_ASCEND_ROOT": str(vllm_ascend_root),
            "MULTI_NODE_ARTIFACT_ROOT": str(plan_artifact_directory),
            "MULTI_NODE_NODE_ARTIFACT_DIR": str(node_artifact_directory),
            "HCCL_IF_IP": local_ip,
            "HCCL_SOCKET_IFNAME": interface,
            "GLOO_SOCKET_IFNAME": interface,
            "TP_SOCKET_IFNAME": interface,
        }
    )
    if node.readiness:
        environment["MULTI_NODE_SERVICE_PORT_START"] = str(node.readiness.port_start)
        environment["MULTI_NODE_SERVICE_COUNT"] = str(node.readiness.count)
    if plan.gateway:
        environment["MULTI_NODE_GATEWAY_PORT"] = str(plan.gateway.port)
    for plan_node in plan.nodes:
        environment[f"MULTI_NODE_NODE_{plan_node.index}_IP"] = hosts[
            plan_node.id
        ].address

    no_proxy = environment.get("NO_PROXY", environment.get("no_proxy", "")).split(
        ","
    )
    no_proxy.extend(host.address for host in hosts.values())
    environment["NO_PROXY"] = ",".join(
        dict.fromkeys(item for item in no_proxy if item)
    )
    environment["no_proxy"] = environment["NO_PROXY"]
    return environment


def wait_http_ready(
    url: str,
    timeout: int,
    check_runtime: Callable[[], None],
    *,
    progress_label: str | None = None,
    progress_interval_seconds: float = 30,
) -> None:
    """Poll one HTTP health endpoint until success, timeout, or runtime failure.

    Requests bypass user proxies because endpoints are on the node network.
    Runtime supervision runs before each attempt so a dead process interrupts
    the wait immediately.
    """
    started = time.monotonic()
    deadline = time.monotonic() + timeout
    next_progress = started + progress_interval_seconds
    while time.monotonic() < deadline:
        check_runtime()
        try:
            with DIRECT_OPENER.open(url, timeout=2) as response:
                if response.status < 400:
                    return
        except (OSError, urllib.error.URLError):
            pass
        now = time.monotonic()
        if (
            progress_label is not None
            and progress_interval_seconds > 0
            and now >= next_progress
        ):
            print(
                f"{progress_label} status=waiting "
                f"elapsed={math.floor(now - started)}s"
            )
            next_progress = now + progress_interval_seconds
        time.sleep(min(1, max(0, deadline - now)))
    raise TimeoutError(f"timed out waiting for {url}")


def wait_node_ready(
    node: Node,
    host: Host,
    timeout: int,
    check_runtime: Callable[[], None],
    progress_interval_seconds: float,
) -> None:
    """Check every consecutive endpoint declared by a node's readiness block.

    Nodes without an HTTP endpoint, such as a headless DP member, are ready once
    their launcher remains alive.
    """
    if node.readiness is None:
        check_runtime()
        print(
            f"[startup] node={node.id} has no HTTP rank endpoints; "
            "launcher status=running"
        )
        return
    deadline = time.monotonic() + timeout
    for offset in range(node.readiness.count):
        remaining = max(1, int(deadline - time.monotonic()))
        label = f"[startup] node={node.id} rank={offset}"
        url = (
            f"http://{host.address}:{node.readiness.port_start + offset}"
            f"{node.readiness.health_path}"
        )
        print(f"{label} status=polling")
        wait_http_ready(
            url,
            remaining,
            check_runtime,
            progress_label=label,
            progress_interval_seconds=progress_interval_seconds,
        )
        print(
            f"{label} status=ready local_ready={offset + 1}/"
            f"{node.readiness.count}"
        )
    print(
        f"[startup] node={node.id} all local ranks ready="
        f"{node.readiness.count}/{node.readiness.count}"
    )


def remaining_timeout(deadline: float, label: str) -> int:
    """Return the whole seconds left in a shared phase deadline."""
    remaining = math.ceil(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError(f"{label} timed out")
    return remaining


def run_stage(
    stage: Stage,
    plan: Plan,
    environment: dict[str, str],
    artifact_directory: Path,
    managed_processes: list[ManagedProcess],
    check_runtime: Callable[[], None],
    cancellation,
    execution_deadline: float,
    progress_interval_seconds: float,
) -> dict[str, object]:
    """Run a stage sequentially and validate the shared step result contract.

    Every step owns an input file, log, result, and artifact directory. The
    runner understands only exit status and ``status=passed``; tool-specific
    metrics remain the responsibility of the step adapter.
    """
    results: dict[str, object] = {}
    for step in stage.steps:
        stage_directory = artifact_directory / stage.id
        step_directory = stage_directory / step.id
        step_directory.mkdir(parents=True, exist_ok=True)
        result_path = step_directory / "result.json"
        input_path = step_directory / "input.json"
        write_json_atomic(input_path, step.inputs)
        step_environment = environment.copy()
        step_environment.update(
            {
                "MULTI_NODE_STEP_ARTIFACT_DIR": str(step_directory),
                "MULTI_NODE_STEP_INPUT_FILE": str(input_path),
                "MULTI_NODE_STEP_RESULT_FILE": str(result_path),
            }
        )
        script = plan.directory / step.script
        log_path = stage_directory / f"{step.id}.log"
        step_label = f"{stage.id}/{step.id}"
        step_started = time.monotonic()
        print(f"[stage] {step_label} status=started log={log_path}")
        item = start_process(
            f"{stage.id} {step.id}",
            ["bash", script.name],
            cwd=script.parent,
            environment=step_environment,
            log_path=log_path,
            stage=stage.id,
        )
        managed_processes.append(item)
        try:
            timeout_seconds = min(
                step.timeout_seconds,
                remaining_timeout(execution_deadline, "execution"),
            )
        except TimeoutError as error:
            raise StageFailure(
                RunFailure(
                    category=stage.failure_category,
                    message=f"execution timed out before {stage.id} {step.id}",
                )
            ) from error
        try:
            return_code = wait_for_process(
                item,
                timeout_seconds,
                check_runtime=check_runtime,
                cancellation=cancellation,
                progress_callback=lambda elapsed, label=step_label: print(
                    f"[stage] {label} status=running "
                    f"elapsed={math.floor(elapsed)}s service=running"
                ),
                progress_interval_seconds=progress_interval_seconds,
            )
        except subprocess.TimeoutExpired as error:
            print(
                f"[stage] {step_label} status=timed_out "
                f"elapsed={math.floor(time.monotonic() - step_started)}s"
            )
            raise StageFailure(
                RunFailure(
                    category=stage.failure_category,
                    message=(
                        f"{stage.id} {step.id} timed out after "
                        f"{timeout_seconds}s; see "
                        f"{log_path.relative_to(artifact_directory.parent)}"
                    ),
                )
            ) from error
        if return_code != 0:
            print(
                f"[stage] {step_label} status=failed exit_code={return_code} "
                f"elapsed={math.floor(time.monotonic() - step_started)}s "
                f"log={log_path}"
            )
            raise StageFailure(
                RunFailure(
                    category=stage.failure_category,
                    message=(
                        f"{stage.id} {step.id} exited with {return_code}; "
                        f"see {log_path}"
                    ),
                )
            )

        result: dict[str, object] = {"status": "passed"}
        if result_path.exists():
            try:
                result = read_json(result_path)
            except (OSError, ValueError) as error:
                raise StageFailure(
                    RunFailure(
                        category=stage.failure_category,
                        message=f"invalid step result {result_path}: {error}",
                    )
                ) from error
            if result.get("status") != "passed":
                print(
                    f"[stage] {step_label} result="
                    f"{format_result_summary(result)}"
                )
                raise StageFailure(
                    RunFailure(
                        category=stage.failure_category,
                        message=(
                            f"{stage.id} {step.id} reported status "
                            f"{result.get('status')!r}"
                        ),
                    )
                )
        else:
            raise StageFailure(
                RunFailure(
                    category=stage.failure_category,
                    message=f"{stage.id} {step.id} did not write {result_path}",
                )
            )
        print(
            f"[stage] {step_label} status=completed exit_code=0 "
            f"elapsed={math.floor(time.monotonic() - step_started)}s"
        )
        print(f"[stage] {step_label} result={format_result_summary(result)}")
        results[step.id] = result
    return results


def format_result_summary(result: dict[str, object]) -> str:
    """Return a single bounded JSON line suitable for GitHub's live log."""
    summary = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(summary) <= MAX_RESULT_SUMMARY_CHARACTERS:
        return summary
    return summary[: MAX_RESULT_SUMMARY_CHARACTERS - 3] + "..."


def aggregate_run_outcome(
    *,
    plan: Plan,
    outcomes: dict[str, NodeOutcome],
    stages: dict[str, object],
    stop_signal: StopSignal,
    collection_failure: RunFailure | None = None,
) -> RunOutcome:
    """Aggregate post-cleanup node facts into the single final run outcome.

    Primary attribution prefers the original execution failure, then node and
    cleanup failures, missing outcomes, cancellation, and finally success.
    Observer nodes marked ``aborted`` can never become the primary failure.
    """
    ordered = {
        item.id: outcomes[item.id] for item in plan.nodes if item.id in outcomes
    }
    missing = tuple(item.id for item in plan.nodes if item.id not in outcomes)

    if stop_signal.kind == "failed":
        assert stop_signal.failure is not None
        origin = ordered.get(stop_signal.origin_node_id)
        return RunOutcome(
            plan=plan.name,
            status="failed",
            nodes=ordered,
            stages=stages,
            failure=stop_signal.failure,
            failure_node_id=(
                stop_signal.origin_node_id
                if origin is not None and origin.execution_status != "aborted"
                else None
            ),
            missing_nodes=missing,
        )

    for node_id, outcome in ordered.items():
        if outcome.execution_status == "failed":
            assert outcome.failure is not None
            return RunOutcome(
                plan=plan.name,
                status="failed",
                nodes=ordered,
                stages=stages,
                failure=outcome.failure,
                failure_node_id=node_id,
                missing_nodes=missing,
            )

    for node_id, outcome in ordered.items():
        if outcome.cleanup_errors:
            return RunOutcome(
                plan=plan.name,
                status="failed",
                nodes=ordered,
                stages=stages,
                failure=outcome.cleanup_errors[0],
                failure_node_id=node_id,
                missing_nodes=missing,
            )

    if collection_failure is not None or missing:
        failure = collection_failure or RunFailure(
            category="coordinator_unreachable",
            message=f"missing final outcomes: {', '.join(missing)}",
        )
        return RunOutcome(
            plan=plan.name,
            status="failed",
            nodes=ordered,
            stages=stages,
            failure=failure,
            missing_nodes=missing,
        )

    if stop_signal.kind == "cancelled":
        assert stop_signal.failure is not None
        origin = ordered.get(stop_signal.origin_node_id)
        return RunOutcome(
            plan=plan.name,
            status="cancelled",
            nodes=ordered,
            stages=stages,
            failure=stop_signal.failure,
            failure_node_id=(
                stop_signal.origin_node_id
                if origin is not None and origin.execution_status == "cancelled"
                else None
            ),
        )

    if all(outcome.status == "passed" for outcome in ordered.values()):
        return RunOutcome(
            plan=plan.name,
            status="passed",
            nodes=ordered,
            stages=stages,
        )

    return RunOutcome(
        plan=plan.name,
        status="failed",
        nodes=ordered,
        stages=stages,
        failure=RunFailure(
            category="internal_error",
            message="node outcomes are inconsistent with the completed stop signal",
        ),
    )


def run_node(
    plan: Plan,
    hosts: dict[str, Host],
    node: Node,
    args: argparse.Namespace,
) -> None:
    """Execute the complete lifecycle for one logical node.

    The fixed sequence is: establish coordination, launch the local service,
    report readiness, optionally start the leader gateway, execute leader-only
    stages, publish a stop, clean process groups, report the immutable node
    outcome, and let the leader aggregate the run.

    Business topology is never inferred here. P/D, DP/TP, ranks, KV connectors,
    and gateway backends must already be explicit in plan-local scripts.
    """
    host = hosts[node.id]
    interface = select_interface(host)
    model_path = str(MODEL_CACHE_ROOT / plan.model.cache_path)
    plan_artifact_directory = (args.artifact_root / plan.name).resolve()
    artifact_directory = plan_artifact_directory / node.id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    environment = base_environment(
        plan,
        node,
        hosts,
        interface,
        model_path,
        resolve_vllm_ascend_root(args.vllm_ascend_root),
        args.control_port,
        plan_artifact_directory,
        artifact_directory,
    )
    endpoint_port = (
        plan.gateway.port if plan.gateway else plan.leader.readiness.port_start
    )
    endpoint_host = hosts[plan.leader.id].address
    environment.update(
        {
            "MULTI_NODE_ENDPOINT_HOST": endpoint_host,
            "MULTI_NODE_ENDPOINT_PORT": str(endpoint_port),
            "MULTI_NODE_ENDPOINT": f"http://{endpoint_host}:{endpoint_port}",
        }
    )

    coordinator: LeaderCoordinator | None = None
    client = CoordinatorClient(hosts[plan.leader.id].address, args.control_port)
    managed_processes: list[ManagedProcess] = []
    runtime_processes: list[ManagedProcess] = []
    execution_status: str | None = None
    execution_failure: RunFailure | None = None
    service_ready = False
    stage_results: dict[str, object] = {}
    node_outcome: NodeOutcome | None = None
    final_outcome: RunOutcome | None = None

    try:
        with signal_cancellation_event() as cancellation:
            try:
                startup_deadline = time.monotonic() + args.startup_timeout_seconds
                # Establish the control plane first. Node 0 owns the HTTP
                # coordinator; every other node waits for the same endpoint.
                def check_cancellation() -> None:
                    """Convert the signal event into ordinary control flow."""
                    if cancellation.is_set():
                        raise CancellationRequested("cancellation requested")

                if node.id == plan.leader.id:
                    coordinator = LeaderCoordinator(
                        [item.id for item in plan.nodes], args.control_port
                    )
                    coordinator.start()
                else:
                    print("waiting for the leader coordinator")
                    client.wait_available(
                        remaining_timeout(startup_deadline, "startup"),
                        check_cancellation,
                    )

                # The plan-local launcher owns vLLM-specific process topology.
                # The runner deliberately supervises it as one process group.
                launch_script = plan.directory / node.launch
                print(
                    "starting service launcher; "
                    f"log: {artifact_directory / 'service.log'}"
                )
                service_process = start_process(
                    "service launcher",
                    ["bash", launch_script.name],
                    cwd=launch_script.parent,
                    environment=environment,
                    log_path=artifact_directory / "service.log",
                    stage="service",
                )
                managed_processes.append(service_process)
                runtime_processes.append(service_process)

                def check_local_runtime() -> None:
                    """Supervise cancellation and all node-local processes."""
                    check_cancellation()
                    check_processes(runtime_processes)

                # Local readiness is checked here; cross-node readiness belongs
                # to the coordinator so every node observes one shared state.
                try:
                    wait_node_ready(
                        node,
                        host,
                        remaining_timeout(startup_deadline, "startup"),
                        check_local_runtime,
                        args.progress_interval_seconds,
                    )
                except TimeoutError as error:
                    raise StageFailure(
                        RunFailure(
                            category="startup_timeout",
                            message=f"{error}; see {node.id}/service.log",
                        )
                    ) from error
                service_ready = True

                if coordinator is not None:
                    # Only the leader waits for the full group and executes the
                    # gateway and validation stages.
                    coordinator.state.mark_ready(node.id)
                    print(
                        "local service ready; waiting for the other nodes"
                    )

                    def report_cluster_readiness(
                        readiness: dict[str, str],
                    ) -> None:
                        """Print the leader's compact view of the node barrier."""
                        ready = sum(
                            status == "ready" for status in readiness.values()
                        )
                        pending = ",".join(
                            node_id
                            for node_id, status in sorted(readiness.items())
                            if status != "ready"
                        )
                        suffix = f" pending={pending}" if pending else ""
                        print(
                            f"[startup] cluster nodes ready={ready}/"
                            f"{len(readiness)}{suffix}"
                        )

                    coordinator.wait_ready(
                        remaining_timeout(startup_deadline, "startup"),
                        check_local_runtime,
                        progress_callback=report_cluster_readiness,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                    report_cluster_readiness(coordinator.state.readiness)

                    if plan.gateway:
                        # A gateway is an ordinary plan-owned process. The runner
                        # knows only how to start, probe, and stop it.
                        gateway_script = plan.directory / plan.gateway.launch
                        print(
                            "starting gateway; "
                            f"log: {artifact_directory / 'gateway.log'}"
                        )
                        gateway_process = start_process(
                            "gateway",
                            ["bash", gateway_script.name],
                            cwd=gateway_script.parent,
                            environment=environment,
                            log_path=artifact_directory / "gateway.log",
                            stage="gateway",
                        )
                        managed_processes.append(gateway_process)
                        runtime_processes.append(gateway_process)

                    def check_leader_runtime() -> None:
                        """Supervise services, gateway, cancellation, and peers."""
                        check_local_runtime()
                        coordinator.raise_if_stopped()

                    if plan.gateway:
                        try:
                            wait_http_ready(
                                environment["MULTI_NODE_ENDPOINT"]
                                + plan.gateway.health_path,
                                remaining_timeout(startup_deadline, "startup"),
                                check_leader_runtime,
                                progress_label="[startup] gateway",
                                progress_interval_seconds=(
                                    args.progress_interval_seconds
                                ),
                            )
                            print("[startup] gateway status=ready")
                        except TimeoutError as error:
                            raise StageFailure(
                                RunFailure(
                                    category="gateway_failed",
                                    message=f"{error}; see {node.id}/gateway.log",
                                )
                            ) from error

                    # Stage and step order is converter-owned and preserved.
                    # Any step failure becomes the first global stop signal.
                    execution_deadline = (
                        time.monotonic() + args.run_timeout_seconds
                    )
                    for stage in plan.stages:
                        stage_results[stage.id] = run_stage(
                            stage,
                            plan,
                            environment,
                            artifact_directory,
                            managed_processes,
                            check_leader_runtime,
                            cancellation,
                            execution_deadline,
                            args.progress_interval_seconds,
                        )
                    check_leader_runtime()
                    coordinator.state.request_stop(
                        StopSignal(kind="completed", origin_node_id=node.id)
                    )
                    execution_status = "passed"
                else:
                    # Workers report readiness and then wait. They never execute
                    # the leader-owned gateway or validation stages.
                    client.mark_ready(
                        node.id, remaining_timeout(startup_deadline, "startup")
                    )
                    print(
                        "local service ready; waiting for execution to stop"
                    )
                    worker_wait_timeout = max(
                        1, math.ceil(startup_deadline - time.monotonic())
                    ) + args.run_timeout_seconds
                    stop_signal = client.wait_stop(
                        worker_wait_timeout, check_local_runtime
                    )
                    execution_status = (
                        "passed" if stop_signal.kind == "completed" else "aborted"
                    )
            except ObservedGlobalStop as error:
                execution_status = (
                    "passed" if error.signal.kind == "completed" else "aborted"
                )
            except StageFailure as error:
                execution_status = "failed"
                execution_failure = error.failure
            except CancellationRequested as error:
                execution_status = "cancelled"
                execution_failure = RunFailure(
                    category="cancelled", message=str(error)
                )
            except ManagedProcessExited as error:
                if error.item.stage == "gateway":
                    category = "gateway_failed"
                elif not service_ready:
                    category = "launch_failed"
                else:
                    category = "node_failed"
                execution_status = "failed"
                execution_failure = RunFailure(category=category, message=str(error))
            except CoordinatorError as error:
                category = (
                    "coordinator_unreachable"
                    if error.code == "coordinator_unreachable"
                    else "node_failed"
                )
                execution_status = "failed"
                execution_failure = RunFailure(category=category, message=str(error))
            except (OSError, RunnerError) as error:
                execution_status = "failed"
                execution_failure = RunFailure(
                    category="launch_failed", message=str(error)
                )
            except Exception as error:
                execution_status = "failed"
                execution_failure = RunFailure(
                    category="internal_error",
                    message=f"{type(error).__name__}: {error}",
                )

            if execution_status is None:
                execution_status = "failed"
                execution_failure = RunFailure(
                    category="internal_error", message="execution ended without an outcome"
                )

            # A stop signal converges nodes quickly but is not the final result;
            # aggregation waits for post-cleanup outcomes.
            if execution_status in {"failed", "cancelled"}:
                assert execution_failure is not None
                signal = StopSignal(
                    kind=execution_status,
                    origin_node_id=node.id,
                    failure=execution_failure,
                )
                try:
                    if coordinator is not None:
                        coordinator.state.request_stop(signal)
                    else:
                        client.request_stop(signal)
                except CoordinatorError as error:
                    print(f"warning: could not publish stop signal: {error}")

            # Every managed process group follows the same TERM/wait/KILL path.
            # Cleanup errors are retained without replacing the primary failure.
            cleanup_errors = tuple(
                RunFailure(category="cleanup_failed", message=message)
                for message in stop_processes(managed_processes)
            )
            # NodeOutcome is constructed only after cleanup facts are final.
            node_outcome = NodeOutcome(
                node_id=node.id,
                execution_status=execution_status,
                failure=execution_failure,
                cleanup_errors=cleanup_errors,
            )
            write_json_atomic(
                artifact_directory / "node-result.json",
                build_node_result(node_outcome),
            )

            # Report final node facts. A reporting failure becomes a local
            # outcome because the leader cannot safely assume delivery.
            try:
                if coordinator is not None:
                    coordinator.state.report_outcome(node_outcome)
                else:
                    client.report_outcome(node_outcome)
            except CoordinatorError as error:
                if coordinator is not None:
                    raise
                report_failure = RunFailure(
                    category="coordinator_unreachable",
                    message=f"could not report final node outcome: {error}",
                )
                node_outcome = NodeOutcome(
                    node_id=node.id,
                    execution_status="failed",
                    failure=report_failure,
                    cleanup_errors=cleanup_errors,
                )
                write_json_atomic(
                    artifact_directory / "node-result.json",
                    build_node_result(node_outcome),
                )

            if coordinator is not None:
                # The leader alone writes the aggregate result after collecting
                # all reachable node outcomes.
                collection_failure = None
                try:
                    outcomes = coordinator.wait_outcomes(60)
                except CoordinatorError as error:
                    outcomes = dict(coordinator.state.outcomes)
                    collection_failure = RunFailure(
                        category="coordinator_unreachable", message=str(error)
                    )
                stop_signal = coordinator.state.stop_signal
                if stop_signal is None:
                    raise RunnerError("coordinator stopped without a stop signal")
                final_outcome = aggregate_run_outcome(
                    plan=plan,
                    outcomes=outcomes,
                    stages=stage_results,
                    stop_signal=stop_signal,
                    collection_failure=collection_failure,
                )
                write_json_atomic(
                    plan_artifact_directory / "result.json",
                    build_final_result(final_outcome),
                )
                coordinator.state.finalize(final_outcome)
    finally:
        if coordinator is not None:
            coordinator.close()

    assert node_outcome is not None
    effective_status = final_outcome.status if final_outcome else node_outcome.status
    if effective_status != "passed":
        failure = (
            final_outcome.failure
            if final_outcome is not None
            else node_outcome.failure
            or (
                node_outcome.cleanup_errors[0]
                if node_outcome.cleanup_errors
                else None
            )
        )
        raise RunnerError(f"{effective_status}: {failure.message if failure else node.id}")
    print("plan completed")


def main() -> int:
    """Load the plan and run either validation-only or one node lifecycle."""
    args = parse_args()
    try:
        plan = load_plan(args.plan)
        hosts = load_hosts(args.hosts, plan) if args.hosts else None
        if args.validate_only:
            print(format_topology_summary(plan, hosts))
            return 0
        if hosts is None:
            raise RunnerError("--hosts is required unless --validate-only is used")
        if not args.node_id:
            raise RunnerError("--node-id is required")
        if args.progress_interval_seconds < 0:
            raise RunnerError("--progress-interval-seconds must be non-negative")
        node = plan.node(args.node_id)
        run_node(plan, hosts, node, args)
        return 0
    except (OSError, PlanError, CoordinatorError, RunnerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
