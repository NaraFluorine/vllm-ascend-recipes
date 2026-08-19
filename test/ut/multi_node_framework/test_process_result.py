from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/recipe/multi_node"))

from scripts.process import (  # noqa: E402
    ManagedProcess,
    ManagedProcessExited,
    check_processes,
    process_group_exists,
    signal_cancellation_event,
    start_process,
    stop_processes,
    tail_log,
    wait_for_process,
)
from scripts.result import (  # noqa: E402
    NodeOutcome,
    RunFailure,
    RunOutcome,
    StopSignal,
    build_final_result,
    build_node_result,
    write_json_atomic,
)


class ProcessTests(unittest.TestCase):
    def test_sigterm_handler_only_sets_the_cancellation_event(self) -> None:
        previous = signal.getsignal(signal.SIGTERM)
        with signal_cancellation_event() as cancellation:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(cancellation.wait(1))
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_wait_for_process_returns_normal_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = start_process(
                "successful child",
                ["bash", "-c", "exit 0"],
                cwd=root,
                environment=os.environ,
                log_path=root / "child.log",
            )
            try:
                self.assertEqual(wait_for_process(child, 2), 0)
            finally:
                stop_processes([child], grace_period_seconds=0.1)

    def test_check_processes_reports_unexpected_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = start_process(
                "failed child",
                ["bash", "-c", "echo failed; exit 7"],
                cwd=root,
                environment=os.environ,
                log_path=root / "child.log",
            )
            try:
                child.process.wait(timeout=2)
                with self.assertRaises(ManagedProcessExited) as raised:
                    check_processes([child])
                self.assertEqual(raised.exception.return_code, 7)
                self.assertIn("failed", str(raised.exception))
                self.assertNotIn("last log lines", str(raised.exception))
            finally:
                stop_processes([child], grace_period_seconds=0.1)

    def test_wait_for_process_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = start_process(
                "slow child",
                ["bash", "-c", "sleep 30"],
                cwd=root,
                environment=os.environ,
                log_path=root / "child.log",
            )
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    wait_for_process(child, 0.01, poll_interval_seconds=0.01)
            finally:
                stop_processes([child], grace_period_seconds=0.1)

    def test_wait_for_process_reports_progress_without_streaming_the_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress: list[float] = []
            child = start_process(
                "progress child",
                ["bash", "-c", "echo private-output; sleep 0.08"],
                cwd=root,
                environment=os.environ,
                log_path=root / "child.log",
            )
            try:
                return_code = wait_for_process(
                    child,
                    2,
                    poll_interval_seconds=0.005,
                    progress_callback=progress.append,
                    progress_interval_seconds=0.01,
                )
            finally:
                stop_processes([child], grace_period_seconds=0.1)

            self.assertEqual(return_code, 0)
            self.assertGreaterEqual(len(progress), 1)
            self.assertIn(
                "private-output",
                (root / "child.log").read_text(encoding="utf-8"),
            )

    def test_log_tail_replaces_invalid_utf8_and_cleanup_kills_the_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "child.log"
            child = start_process(
                "stubborn child",
                ["bash", "-c", "printf '\\377tail\\n'; trap '' TERM; sleep 30"],
                cwd=root,
                environment=os.environ,
                log_path=log_path,
            )
            deadline = time.monotonic() + 2
            while log_path.stat().st_size == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            errors = stop_processes(
                [child], grace_period_seconds=0.1, kill_timeout_seconds=1
            )

            self.assertEqual(errors, [])
            self.assertFalse(process_group_exists(child.process_group))
            self.assertIn("tail", tail_log(log_path))
            self.assertIn("�", tail_log(log_path))
            self.assertEqual(stop_processes([child], grace_period_seconds=0), [])

    def test_cleanup_returns_signal_diagnostics(self) -> None:
        fake_process = Mock()
        fake_process.pid = os.getpgrp() + 100_000
        fake_process.poll.return_value = None
        item = ManagedProcess(
            name="unmanageable child",
            process=fake_process,
            log_path=Path("/tmp/unmanageable-child.log"),
            log_file=io.BytesIO(),
        )

        with (
            patch("scripts.process.process_group_exists", return_value=True),
            patch("scripts.process.os.killpg", side_effect=OSError("denied")),
        ):
            errors = stop_processes(
                [item], grace_period_seconds=0, kill_timeout_seconds=0
            )

        self.assertTrue(any("could not SIGTERM" in error for error in errors))
        self.assertTrue(any("could not SIGKILL" in error for error in errors))
        self.assertTrue(any("still exists" in error for error in errors))


class ResultTests(unittest.TestCase):
    def test_stop_signal_round_trip(self) -> None:
        signal = StopSignal(
            kind="failed",
            origin_node_id="node1",
            failure=RunFailure(category="node_failed", message="service exited"),
        )

        self.assertEqual(StopSignal.from_dict(signal.to_dict()), signal)

    def test_node_outcome_preserves_execution_and_cleanup_status(self) -> None:
        cleanup = RunFailure(category="cleanup_failed", message="group survived")
        outcome = NodeOutcome(
            node_id="node1",
            execution_status="passed",
            cleanup_errors=(cleanup,),
        )

        serialized = build_node_result(outcome)

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(serialized["execution_status"], "passed")
        self.assertEqual(serialized["status"], "failed")
        self.assertIsNone(serialized["failure"])
        self.assertEqual(NodeOutcome.from_dict(serialized), outcome)

    def test_aborted_node_has_no_local_failure(self) -> None:
        outcome = NodeOutcome(node_id="node1", execution_status="aborted")

        self.assertEqual(outcome.status, "aborted")
        self.assertIsNone(outcome.failure)
        with self.assertRaisesRegex(ValueError, "must not contain a local failure"):
            NodeOutcome(
                node_id="node1",
                execution_status="aborted",
                failure=RunFailure(category="node_failed", message="remote failure"),
            )

    def test_run_outcome_uses_generic_stages_and_exact_node_outcomes(self) -> None:
        nodes = {
            "node0": NodeOutcome(node_id="node0", execution_status="passed"),
            "node1": NodeOutcome(node_id="node1", execution_status="passed"),
        }
        outcome = RunOutcome(
            plan="fixture",
            status="passed",
            nodes=nodes,
            stages={
                "checks": {"health": {"status": "passed"}},
                "accuracy": {"gsm8k": {"status": "passed"}},
            },
        )

        serialized = build_final_result(outcome)

        self.assertEqual(set(serialized["stages"]), {"checks", "accuracy"})
        self.assertEqual(serialized["nodes"]["node1"], nodes["node1"].to_dict())
        self.assertEqual(RunOutcome.from_dict(serialized), outcome)

    def test_cleanup_failure_can_be_the_run_primary_failure(self) -> None:
        cleanup = RunFailure(category="cleanup_failed", message="group survived")
        node = NodeOutcome(
            node_id="node1",
            execution_status="passed",
            cleanup_errors=(cleanup,),
        )

        outcome = RunOutcome(
            plan="fixture",
            status="failed",
            nodes={"node1": node},
            stages={},
            failure=cleanup,
            failure_node_id="node1",
        )

        self.assertEqual(outcome.failure, cleanup)
        self.assertEqual(outcome.nodes["node1"].failure, None)

    def test_cancelled_run_can_contain_aborted_observers(self) -> None:
        cancellation = RunFailure(category="cancelled", message="SIGTERM")
        nodes = {
            "node0": NodeOutcome(
                node_id="node0",
                execution_status="cancelled",
                failure=cancellation,
            ),
            "node1": NodeOutcome(node_id="node1", execution_status="aborted"),
        }

        outcome = RunOutcome(
            plan="fixture",
            status="cancelled",
            nodes=nodes,
            stages={},
            failure=cancellation,
            failure_node_id="node0",
        )

        self.assertEqual(outcome.status, "cancelled")
        self.assertEqual(outcome.nodes["node1"].execution_status, "aborted")

    def test_aborted_node_cannot_be_the_primary_failure(self) -> None:
        failure = RunFailure(category="node_failed", message="lost node")
        with self.assertRaisesRegex(ValueError, "aborted node"):
            RunOutcome(
                plan="fixture",
                status="failed",
                nodes={
                    "node1": NodeOutcome(
                        node_id="node1", execution_status="aborted"
                    )
                },
                stages={},
                failure=failure,
                failure_node_id="node1",
            )

    def test_missing_outcomes_are_explicit_run_level_failure(self) -> None:
        failure = RunFailure(
            category="coordinator_unreachable", message="node1 did not report"
        )
        outcome = RunOutcome(
            plan="fixture",
            status="failed",
            nodes={
                "node0": NodeOutcome(node_id="node0", execution_status="passed")
            },
            stages={},
            failure=failure,
            missing_nodes=("node1",),
        )

        self.assertIsNone(outcome.failure_node_id)
        self.assertEqual(outcome.missing_nodes, ("node1",))

    def test_atomic_json_replaces_an_existing_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_atomic(path, {"status": "running"})
            write_json_atomic(path, {"status": "passed"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"status": "passed"}
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
