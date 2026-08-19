"""Static analyzers for the shell fragments embedded in Recipe scenarios.

The converter deliberately does not execute Recipe shell.  This module only
recognizes the small command contracts used by multi-node scenarios and turns
them into typed values that the planner can validate.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, Sequence


class ShellAnalysisError(ValueError):
    """Report a shell fragment that is outside the converter contract."""


@dataclass(frozen=True)
class VllmServeCommand:
    """The statically visible parts of one ``vllm serve`` command."""

    model_id: str
    options: Mapping[str, str | None]
    command: str

    def require(self, option: str) -> str:
        """Return an option value or fail with a useful conversion error."""
        value = self.options.get(option)
        if value is None:
            raise ShellAnalysisError(f"vllm serve requires {option}")
        return value


@dataclass(frozen=True)
class ExternalDpCommand:
    """Arguments supplied to vLLM Ascend's external-online-DP launcher."""

    launcher: str
    dp_size: int
    tp_size: int
    dp_size_local: int
    dp_rank_start: int
    dp_address: str
    dp_rpc_port: int
    vllm_start_port: int


@dataclass(frozen=True)
class GatewayCommand:
    """Static endpoint lists supplied to the P/D load-balancing proxy."""

    launcher: str
    host: str
    port: int
    prefiller_hosts: tuple[str, ...]
    prefiller_ports: tuple[int, ...]
    decoder_hosts: tuple[str, ...]
    decoder_ports: tuple[int, ...]


@dataclass(frozen=True)
class CompletionCheck:
    """Endpoint and served model asserted by a completion smoke test."""

    endpoint: str
    served_name: str


def logical_commands(script: str) -> tuple[str, ...]:
    """Fold backslash continuations without evaluating any shell syntax."""
    commands: list[str] = []
    pending = ""
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or (line.startswith("#") and not pending):
            continue
        continued = line.endswith("\\")
        part = line[:-1].rstrip() if continued else line
        pending = f"{pending} {part}".strip()
        if not continued:
            commands.append(pending)
            pending = ""
    if pending:
        raise ShellAnalysisError("shell fragment ends with an unfinished continuation")
    return tuple(commands)


def _tokenize(command: str) -> list[str]:
    """Tokenize one folded command without expansion or execution."""
    try:
        return shlex.split(command, posix=True)
    except ValueError as exc:
        raise ShellAnalysisError(f"cannot parse shell command: {exc}") from exc


def _find_command(script: str, marker: Sequence[str]) -> tuple[str, list[str]]:
    """Locate the first logical command containing an exact token marker."""
    for command in logical_commands(script):
        tokens = _tokenize(command)
        for index in range(len(tokens) - len(marker) + 1):
            if tokens[index : index + len(marker)] == list(marker):
                return command, tokens[index:]
    raise ShellAnalysisError(f"missing {' '.join(marker)} command")


def _options(tokens: Sequence[str], *, start: int) -> dict[str, str | None]:
    """Decode single-value and boolean GNU-style options."""
    parsed: dict[str, str | None] = {}
    index = start
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ShellAnalysisError(f"unexpected positional argument {token!r}")
        if "=" in token:
            key, value = token.split("=", 1)
            parsed[key] = value
            index += 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            parsed[token] = tokens[index + 1]
            index += 2
        else:
            parsed[token] = None
            index += 1
    return parsed


def _integer(value: str | None, label: str) -> int:
    """Decode a visible non-negative integer option."""
    if value is None:
        raise ShellAnalysisError(f"{label} requires a value")
    try:
        result = int(value)
    except ValueError as exc:
        raise ShellAnalysisError(f"{label} must be an integer, got {value!r}") from exc
    if result < 0:
        raise ShellAnalysisError(f"{label} must not be negative")
    return result


def parse_vllm_serve(script: str) -> VllmServeCommand:
    """Parse the single supported ``vllm serve MODEL [OPTIONS]`` command."""
    matches: list[tuple[str, list[str]]] = []
    for command in logical_commands(script):
        tokens = _tokenize(command)
        for index in range(len(tokens) - 1):
            if tokens[index : index + 2] == ["vllm", "serve"]:
                matches.append((command, tokens[index:]))
                break
    if len(matches) != 1:
        raise ShellAnalysisError(
            f"expected exactly one vllm serve command, found {len(matches)}"
        )
    command, tokens = matches[0]
    if len(tokens) < 3 or tokens[2].startswith("--"):
        raise ShellAnalysisError("vllm serve requires a literal model argument")
    model_id = tokens[2]
    if not model_id or model_id.startswith("$") or "{{" in model_id:
        raise ShellAnalysisError("vllm serve model must be a literal relative model id")
    model_path = PurePosixPath(model_id)
    if model_path.is_absolute() or ".." in model_path.parts:
        raise ShellAnalysisError(
            "vllm serve model must not be absolute or contain parent traversal"
        )
    return VllmServeCommand(model_id, _options(tokens, start=3), command)


def parse_external_dp(script: str) -> ExternalDpCommand:
    """Parse an ``external_online_dp/launch_online_dp.py`` invocation."""
    command, tokens = _find_command(script, ("python",))
    del command
    if len(tokens) < 2 or not tokens[1].endswith(
        "/examples/external_online_dp/launch_online_dp.py"
    ):
        raise ShellAnalysisError(
            "launch script must call examples/external_online_dp/launch_online_dp.py"
        )
    options = _options(tokens, start=2)

    def require(name: str) -> str:
        """Return one required external-DP option value."""
        value = options.get(name)
        if value is None:
            raise ShellAnalysisError(f"external-DP launcher requires {name}")
        return value

    return ExternalDpCommand(
        launcher=tokens[1],
        dp_size=_integer(require("--dp-size"), "--dp-size"),
        tp_size=_integer(require("--tp-size"), "--tp-size"),
        dp_size_local=_integer(require("--dp-size-local"), "--dp-size-local"),
        dp_rank_start=_integer(require("--dp-rank-start"), "--dp-rank-start"),
        dp_address=require("--dp-address"),
        dp_rpc_port=_integer(require("--dp-rpc-port"), "--dp-rpc-port"),
        vllm_start_port=_integer(require("--vllm-start-port"), "--vllm-start-port"),
    )


def _option_group(tokens: Sequence[str], name: str) -> tuple[str, ...]:
    """Read one gateway option followed by one or more values."""
    try:
        start = tokens.index(name) + 1
    except ValueError as exc:
        raise ShellAnalysisError(f"gateway requires {name}") from exc
    end = start
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    if start == end:
        raise ShellAnalysisError(f"gateway {name} requires at least one value")
    return tuple(tokens[start:end])


def parse_gateway(script: str) -> GatewayCommand:
    """Parse the supported vLLM Ascend P/D proxy command."""
    _, tokens = _find_command(script, ("python",))
    if len(tokens) < 2 or not tokens[1].endswith(
        "/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
    ):
        raise ShellAnalysisError(
            "gateway script must call the vLLM Ascend P/D proxy example"
        )
    host = _option_group(tokens, "--host")
    port = _option_group(tokens, "--port")
    if len(host) != 1 or len(port) != 1:
        raise ShellAnalysisError("gateway host and port must each have one value")
    prefill_ports = tuple(
        _integer(value, "--prefiller-ports")
        for value in _option_group(tokens, "--prefiller-ports")
    )
    decode_ports = tuple(
        _integer(value, "--decoder-ports")
        for value in _option_group(tokens, "--decoder-ports")
    )
    return GatewayCommand(
        launcher=tokens[1],
        host=host[0],
        port=_integer(port[0], "--port"),
        prefiller_hosts=_option_group(tokens, "--prefiller-hosts"),
        prefiller_ports=prefill_ports,
        decoder_hosts=_option_group(tokens, "--decoder-hosts"),
        decoder_ports=decode_ports,
    )


_URL_RE = re.compile(r"https?://[^\s\"']+/v1/completions")
_DATA_RE = re.compile(r"(?:-d|--data(?:-raw)?)\s+([\"'])(.+?)\1", re.DOTALL)


def parse_completion_check(script: str) -> CompletionCheck:
    """Validate the minimal curl-and-JSON completion check contract."""
    if "curl " not in script or "--fail" not in script:
        raise ShellAnalysisError("service-check must use curl --fail")
    url_match = _URL_RE.search(script)
    if url_match is None:
        raise ShellAnalysisError("service-check must call /v1/completions")
    data_match = _DATA_RE.search(script)
    if data_match is None:
        raise ShellAnalysisError("service-check must send a JSON request body")
    try:
        payload = json.loads(data_match.group(2))
    except json.JSONDecodeError as exc:
        raise ShellAnalysisError("service-check request body must be literal JSON") from exc
    served_name = payload.get("model")
    if not isinstance(served_name, str) or not served_name:
        raise ShellAnalysisError("service-check request body requires a model")
    if 'json.load(sys.stdin)["choices"]' not in script and "['choices']" not in script:
        raise ShellAnalysisError("service-check must assert a non-empty choices response")
    return CompletionCheck(url_match.group(0), served_name)


def export_assignments(script: str) -> tuple[str, ...]:
    """Return literal export statements that are safe to retain in generated scripts."""
    exports: list[str] = []
    for command in logical_commands(script):
        if command.startswith("export ") and "ASCEND_RT_VISIBLE_DEVICES=" not in command:
            if "$(" in command or "`" in command:
                raise ShellAnalysisError(
                    "export assignments cannot contain command substitution"
                )
            exports.append(command)
    return tuple(exports)
