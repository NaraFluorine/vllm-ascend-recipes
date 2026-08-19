from __future__ import annotations

import errno
import json
import sys
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/recipe/multi_node"))

from scripts.coordinator import (  # noqa: E402
    CoordinatorClient,
    CoordinatorError,
    LeaderCoordinator,
    RunState,
)
from scripts.result import (  # noqa: E402
    NodeOutcome,
    RunFailure,
    RunOutcome,
    StopSignal,
)


class FakeResponse:
    def __init__(self, value: object) -> None:
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class SequenceOpener:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def open(self, request: object, timeout: float) -> FakeResponse:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome


def refused() -> urllib.error.URLError:
    return urllib.error.URLError(
        ConnectionRefusedError(errno.ECONNREFUSED, "connection refused")
    )


def failure(message: str = "service stopped") -> RunFailure:
    return RunFailure(category="node_failed", message=message)


class RunStateTests(unittest.TestCase):
    def test_ready_and_identical_stop_requests_are_idempotent(self) -> None:
        state = RunState(["node0", "node1"])
        state.mark_ready("node0")
        state.mark_ready("node0")
        signal = StopSignal(kind="failed", origin_node_id="node0", failure=failure())

        state.request_stop(signal)
        state.request_stop(signal)

        snapshot = state.snapshot()
        self.assertEqual(snapshot["readiness"]["node0"], "ready")
        self.assertEqual(snapshot["stop_signal"], signal.to_dict())

    def test_first_stop_signal_is_preserved(self) -> None:
        state = RunState(["node0", "node1"])
        first = StopSignal(kind="failed", origin_node_id="node0", failure=failure())
        second = StopSignal(
            kind="cancelled",
            origin_node_id="node1",
            failure=RunFailure(category="cancelled", message="SIGTERM"),
        )

        state.request_stop(first)
        with self.assertRaisesRegex(CoordinatorError, "different stop signal"):
            state.request_stop(second)

        self.assertEqual(state.stop_signal, first)

    def test_observed_remote_failure_is_not_a_second_local_failure(self) -> None:
        state = RunState(["node0", "node1"])
        stop = StopSignal(kind="failed", origin_node_id="node0", failure=failure())
        state.request_stop(stop)
        state.report_outcome(
            NodeOutcome(
                node_id="node0", execution_status="failed", failure=stop.failure
            )
        )
        state.report_outcome(
            NodeOutcome(node_id="node1", execution_status="aborted")
        )

        snapshot = state.snapshot()
        self.assertEqual(
            snapshot["outcomes"]["node0"]["execution_status"], "failed"
        )
        self.assertEqual(
            snapshot["outcomes"]["node1"]["execution_status"], "aborted"
        )
        self.assertIsNone(snapshot["outcomes"]["node1"]["failure"])

    def test_outcome_report_is_immutable_and_idempotent(self) -> None:
        state = RunState(["node0"])
        state.mark_ready("node0")
        state.request_stop(StopSignal(kind="completed", origin_node_id="node0"))
        outcome = NodeOutcome(node_id="node0", execution_status="passed")

        state.report_outcome(outcome)
        state.report_outcome(outcome)
        with self.assertRaisesRegex(CoordinatorError, "different outcome"):
            state.report_outcome(
                NodeOutcome(
                    node_id="node0",
                    execution_status="passed",
                    cleanup_errors=(
                        RunFailure(category="cleanup_failed", message="group survived"),
                    ),
                )
            )

        self.assertEqual(state.outcomes, {"node0": outcome})

    def test_cleanup_failure_is_part_of_final_outcome(self) -> None:
        state = RunState(["node0", "node1"])
        for node_id in state.node_ids:
            state.mark_ready(node_id)
        state.request_stop(StopSignal(kind="completed", origin_node_id="node0"))
        leader = NodeOutcome(node_id="node0", execution_status="passed")
        cleanup = RunFailure(category="cleanup_failed", message="group survived")
        worker = NodeOutcome(
            node_id="node1",
            execution_status="passed",
            cleanup_errors=(cleanup,),
        )
        state.report_outcome(leader)
        state.report_outcome(worker)
        final = RunOutcome(
            plan="fixture",
            status="failed",
            nodes={"node0": leader, "node1": worker},
            stages={},
            failure=cleanup,
            failure_node_id="node1",
        )

        state.finalize(final)
        state.finalize(final)

        snapshot = state.snapshot()
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["final_outcome"], final.to_dict())

    def test_finalize_can_explicitly_record_missing_outcomes(self) -> None:
        state = RunState(["node0", "node1"])
        local_failure = failure()
        state.request_stop(
            StopSignal(
                kind="failed", origin_node_id="node0", failure=local_failure
            )
        )
        node0 = NodeOutcome(
            node_id="node0", execution_status="failed", failure=local_failure
        )
        state.report_outcome(node0)
        final = RunOutcome(
            plan="fixture",
            status="failed",
            nodes={"node0": node0},
            stages={},
            failure=local_failure,
            failure_node_id="node0",
            missing_nodes=("node1",),
        )

        state.finalize(final)

        self.assertEqual(state.final_outcome, final)

    def test_unknown_node_is_rejected(self) -> None:
        with self.assertRaisesRegex(CoordinatorError, "unknown node"):
            RunState(["node0"]).mark_ready("node9")


class CoordinatorHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = LeaderCoordinator(
            ["node0", "node1"], 0, host="127.0.0.1"
        )
        self.coordinator.start()
        self.addCleanup(self.coordinator.close)
        self.client = CoordinatorClient("127.0.0.1", self.coordinator.port)

    def test_ready_stop_and_outcome_round_trip(self) -> None:
        for node_id in ("node0", "node1"):
            self.client.mark_ready(node_id, 1)
        self.coordinator.wait_ready(1, lambda: None)

        signal = StopSignal(kind="completed", origin_node_id="node0")
        self.client.request_stop(signal)
        self.client.request_stop(signal)
        self.assertEqual(self.client.wait_stop(1, lambda: None), signal)

        outcomes = {
            node_id: NodeOutcome(node_id=node_id, execution_status="passed")
            for node_id in ("node0", "node1")
        }
        for outcome in outcomes.values():
            self.client.report_outcome(outcome)
            self.client.report_outcome(outcome)
        self.assertEqual(self.coordinator.wait_outcomes(1), outcomes)

        final = RunOutcome(
            plan="fixture",
            status="passed",
            nodes=outcomes,
            stages={"checks": {"health": {"status": "passed"}}},
        )
        self.coordinator.state.finalize(final)
        self.assertEqual(self.coordinator.state.final_outcome, final)

    def test_invalid_request_returns_a_simple_error(self) -> None:
        with self.assertRaisesRegex(CoordinatorError, "unknown node"):
            self.client.mark_ready("node9", 1)

    def test_ready_wait_reports_periodic_node_snapshots(self) -> None:
        snapshots: list[dict[str, str]] = []
        self.client.mark_ready("node0", 1)

        def mark_worker_ready() -> None:
            time.sleep(0.05)
            self.client.mark_ready("node1", 1)

        worker = threading.Thread(target=mark_worker_ready)
        worker.start()
        self.coordinator.wait_ready(
            1,
            lambda: None,
            progress_callback=snapshots.append,
            progress_interval_seconds=0.01,
        )
        worker.join()

        self.assertTrue(snapshots)
        self.assertTrue(
            any(snapshot["node1"] == "pending" for snapshot in snapshots)
        )


class CoordinatorStartupTests(unittest.TestCase):
    def test_startup_wait_retries_connection_refusal_until_available(self) -> None:
        client = CoordinatorClient("coordinator.invalid", 1, request_timeout=0.01)
        opener = SequenceOpener(
            [refused(), refused(), FakeResponse({"status": "running"})]
        )
        client.opener = opener  # type: ignore[assignment]

        with patch("scripts.coordinator.time.sleep"):
            client.wait_available(3, lambda: None)

        self.assertEqual(opener.calls, 3)

    def test_normal_operations_do_not_apply_generic_retries(self) -> None:
        client = CoordinatorClient("coordinator.invalid", 1, request_timeout=0.01)
        opener = SequenceOpener([refused(), FakeResponse({})])
        client.opener = opener  # type: ignore[assignment]

        with self.assertRaises(CoordinatorError) as raised:
            client.mark_ready("node0", 1)

        self.assertEqual(raised.exception.code, "coordinator_unreachable")
        self.assertEqual(opener.calls, 1)


if __name__ == "__main__":
    unittest.main()
