#!/usr/bin/env python3
"""Small process-lifecycle helpers for the Multi-node framework runner."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence


DEFAULT_LOG_TAIL_LINES = 50
DEFAULT_LOG_TAIL_BYTES = 16 * 1024


@dataclass
class ManagedProcess:
    """A child started in its own session and process group."""

    name: str
    process: subprocess.Popen[bytes]
    log_path: Path
    log_file: BinaryIO
    stage: str | None = None

    @property
    def process_group(self) -> int:
        """Return the stable process-group id created for this command."""
        # start_new_session=True makes the child PID its stable process-group ID.
        return self.process.pid


class ManagedProcessExited(RuntimeError):
    """A supervised process exited before its owner expected it to."""

    def __init__(self, item: ManagedProcess, return_code: int) -> None:
        """Build an actionable error pointing to the complete saved log."""
        self.item = item
        self.return_code = return_code
        super().__init__(
            f"{item.name} exited with {return_code}; see {item.log_path}"
        )


class CancellationRequested(RuntimeError):
    """SIGINT or SIGTERM requested an orderly runner shutdown."""


def start_process(
    name: str,
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    stage: str | None = None,
) -> ManagedProcess:
    """Start a logged command in a new session and process group."""
    if not command:
        raise ValueError("command must not be empty")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise

    return ManagedProcess(
        name=name,
        process=process,
        log_path=log_path,
        log_file=log_file,
        stage=stage,
    )


def tail_log(
    path: Path,
    *,
    max_lines: int = DEFAULT_LOG_TAIL_LINES,
    max_bytes: int = DEFAULT_LOG_TAIL_BYTES,
) -> str:
    """Return a bounded, replacement-decoded tail while preserving the full log."""
    try:
        with path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            data = log_file.read(max_bytes)
    except OSError:
        return ""

    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def poll_processes(
    processes: Sequence[ManagedProcess],
) -> tuple[ManagedProcess, int] | None:
    """Return the first exited managed process, if any."""
    for item in processes:
        return_code = item.process.poll()
        if return_code is not None:
            return item, return_code
    return None


def check_processes(processes: Sequence[ManagedProcess]) -> None:
    """Raise with a bounded log tail when a supervised process has exited."""
    exited = poll_processes(processes)
    if exited is not None:
        raise ManagedProcessExited(*exited)


def wait_for_process(
    item: ManagedProcess,
    timeout_seconds: float,
    *,
    check_runtime: Callable[[], None] | None = None,
    cancellation: threading.Event | None = None,
    poll_interval_seconds: float = 0.5,
    progress_callback: Callable[[float], None] | None = None,
    progress_interval_seconds: float = 30,
) -> int:
    """Wait while supervising runtime state and reporting bounded progress."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    next_progress = started + progress_interval_seconds
    while True:
        if cancellation is not None and cancellation.is_set():
            raise CancellationRequested("cancellation requested")

        return_code = item.process.poll()
        if return_code is not None:
            return return_code

        if check_runtime is not None:
            check_runtime()

        return_code = item.process.poll()
        if return_code is not None:
            return return_code

        now = time.monotonic()
        if (
            progress_callback is not None
            and progress_interval_seconds > 0
            and now >= next_progress
        ):
            progress_callback(now - started)
            next_progress = now + progress_interval_seconds

        remaining = deadline - now
        if remaining <= 0:
            raise subprocess.TimeoutExpired(item.name, timeout_seconds)

        delay = min(poll_interval_seconds, remaining)
        if cancellation is None:
            time.sleep(delay)
        else:
            cancellation.wait(delay)


@contextmanager
def signal_cancellation_event(
    handled_signals: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
) -> Iterator[threading.Event]:
    """Set an event from minimal signal handlers and restore old handlers."""
    cancellation = threading.Event()
    previous_handlers: dict[int, signal.Handlers] = {}

    def request_cancellation(_signum: int, _frame: FrameType | None) -> None:
        """Record cancellation without performing unsafe work in a handler."""
        cancellation.set()

    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_cancellation)

    try:
        yield cancellation
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def process_group_exists(process_group: int) -> bool:
    """Return whether a process group still has at least one member."""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_groups(
    processes: Sequence[ManagedProcess], deadline: float
) -> list[ManagedProcess]:
    """Wait until process groups disappear and return any remaining groups."""
    while True:
        alive: list[ManagedProcess] = []
        for item in processes:
            item.process.poll()
            if process_group_exists(item.process_group):
                alive.append(item)
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))


def stop_processes(
    processes: Sequence[ManagedProcess],
    *,
    grace_period_seconds: float = 10,
    kill_timeout_seconds: float = 5,
) -> list[str]:
    """TERM/KILL process groups in reverse order, close logs, and verify cleanup.

    Cleanup diagnostics are returned instead of raised so a runner can retain its
    original execution failure as ``primary_failure``.
    """
    cleanup_errors: list[str] = []
    reversed_processes = list(reversed(processes))
    signal_targets: list[ManagedProcess] = []
    seen_groups: set[int] = set()
    current_group = os.getpgrp()

    for item in reversed_processes:
        if item.process_group in seen_groups:
            continue
        seen_groups.add(item.process_group)
        if item.process_group == current_group:
            cleanup_errors.append(
                f"refusing to signal runner process group {current_group} "
                f"for {item.name}"
            )
            continue
        signal_targets.append(item)

    try:
        for item in signal_targets:
            if not process_group_exists(item.process_group):
                continue
            try:
                os.killpg(item.process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                cleanup_errors.append(
                    f"could not SIGTERM {item.name} process group "
                    f"{item.process_group}: {error}"
                )

        alive = _wait_for_process_groups(
            signal_targets, time.monotonic() + grace_period_seconds
        )
        for item in alive:
            try:
                os.killpg(item.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                cleanup_errors.append(
                    f"could not SIGKILL {item.name} process group "
                    f"{item.process_group}: {error}"
                )

        _wait_for_process_groups(
            alive, time.monotonic() + kill_timeout_seconds
        )
    finally:
        for item in reversed_processes:
            try:
                item.log_file.close()
            except Exception as error:
                cleanup_errors.append(f"could not close {item.log_path}: {error}")

    for item in signal_targets:
        item.process.poll()
        if process_group_exists(item.process_group):
            cleanup_errors.append(
                f"{item.name} process group {item.process_group} still exists "
                "after SIGKILL"
            )

    return cleanup_errors
