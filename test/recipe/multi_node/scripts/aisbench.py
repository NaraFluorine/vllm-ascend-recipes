#!/usr/bin/env python3
"""Render AISBench inputs and translate artifacts to the step-result contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[4]
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from scripts.result import write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the single runtime entrypoint used by evaluation steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _required_text(value: object, field: str) -> str:
    """Return a required non-empty text field or raise a config error."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"aisbench.{field} must be a non-empty string")
    return value


def _positive_integer(value: object, field: str) -> int:
    """Return a strictly positive integer field from step inputs."""
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"aisbench.{field} must be a positive integer")
    return value


def load_run_config(path: Path) -> dict[str, object]:
    """Load and strictly validate the AISBench portion of a step input file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid AISBench input JSON: {path}") from error
    if not isinstance(raw, dict) or set(raw) != {"aisbench"}:
        raise RuntimeError("step inputs must contain only an aisbench mapping")
    config = raw["aisbench"]
    if not isinstance(config, dict):
        raise RuntimeError("step inputs.aisbench must be a mapping")

    case_type = _required_text(config.get("case_type"), "case_type")
    if case_type not in {"accuracy", "performance"}:
        raise RuntimeError("aisbench.case_type must be accuracy or performance")
    _required_text(config.get("dataset_path"), "dataset_path")
    dataset_conf = _required_text(config.get("dataset_conf"), "dataset_conf")
    request_conf = _required_text(config.get("request_conf"), "request_conf")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", dataset_conf):
        raise RuntimeError("aisbench.dataset_conf must be group/config_name")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", request_conf):
        raise RuntimeError("aisbench.request_conf must be a config name")
    for field in ("num_prompts", "max_out_len", "batch_size"):
        _positive_integer(config.get(field), field)
    for field in ("request_rate", "temperature"):
        if field in config and not isinstance(config[field], (int, float)):
            raise RuntimeError(f"aisbench.{field} must be numeric")
    if "dump_eval_details" in config and type(config["dump_eval_details"]) is not bool:
        raise RuntimeError("aisbench.dump_eval_details must be boolean")
    return config


def download_dataset(dataset: str) -> Path:
    """Resolve a local dataset path or download a ModelScope dataset snapshot."""
    local = Path(dataset).expanduser()
    if local.exists():
        return local.resolve()
    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "modelscope is required to download AISBench data"
        ) from error

    cache_directory = os.environ.get(
        "MODELSCOPE_CACHE", "/root/.cache/modelscope/hub"
    )
    downloaded = snapshot_download(
        model_id=dataset,
        repo_type="dataset",
        cache_dir=cache_directory,
    )
    return Path(downloaded).resolve()


def _replace_config_value(content: str, field: str, value: object) -> str:
    """Replace exactly one top-level assignment in an AISBench Python config."""
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(field)}\s*=\s*)(?P<value>[^,\n]+)(?P<suffix>,.*)$",
        re.MULTILINE,
    )
    replacement = repr(value) if isinstance(value, str) else str(value)
    content, count = pattern.subn(
        lambda match: f"{match.group('prefix')}{replacement}{match.group('suffix')}",
        content,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"AISBench template field not found: {field}")
    return content


def prepare_configs(
    config: Mapping[str, object],
    source: Path,
    output: Path,
    dataset_directory: Path,
) -> tuple[str, str]:
    """Copy pinned AISBench templates and fill runtime-specific values.

    Templates remain immutable in the shared AISBench installation. Each step
    receives a private rendered copy under its artifact directory.
    """
    dataset_conf = str(config["dataset_conf"])
    dataset_group, dataset_name = dataset_conf.split("/", 1)
    request_name = str(config["request_conf"])
    model_source = (
        source
        / "ais_bench/benchmark/configs/models/vllm_api"
        / f"{request_name}.py"
    )
    dataset_source = (
        source
        / "ais_bench/benchmark/configs/datasets"
        / dataset_group
        / f"{dataset_name}.py"
    )
    for path in (model_source, dataset_source):
        if not path.is_file():
            raise RuntimeError(f"AISBench config template not found: {path}")

    model_content = model_source.read_text(encoding="utf-8")
    model_values: dict[str, object] = {
        "path": os.environ["MULTI_NODE_MODEL_PATH"],
        "model": os.environ["MULTI_NODE_SERVED_MODEL_NAME"],
        "host_ip": os.environ["MULTI_NODE_ENDPOINT_HOST"],
        "host_port": int(os.environ["MULTI_NODE_ENDPOINT_PORT"]),
        "max_out_len": int(config["max_out_len"]),
        "batch_size": int(config["batch_size"]),
        "request_rate": config.get("request_rate", 0),
        "temperature": config.get("temperature", 0.01),
    }
    for field, value in model_values.items():
        model_content = _replace_config_value(model_content, field, value)

    dataset_content = _replace_config_value(
        dataset_source.read_text(encoding="utf-8"),
        "path",
        str(dataset_directory),
    )
    model_output = output / "models" / "vllm_api" / f"{request_name}.py"
    dataset_output = output / "datasets" / dataset_group / f"{dataset_name}.py"
    model_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(model_content, encoding="utf-8")
    dataset_output.write_text(dataset_content, encoding="utf-8")
    return request_name, dataset_name


def aisbench_command(
    config: Mapping[str, object],
    executable: str,
    config_directory: Path,
    request_name: str,
    dataset_name: str,
) -> list[str]:
    """Build the AISBench CLI command from validated intermediate inputs."""
    command = [
        executable,
        "--config-dir",
        str(config_directory),
        "--models",
        request_name,
        "--datasets",
        dataset_name,
        "--num-prompts",
        str(config["num_prompts"]),
    ]
    if config["case_type"] == "performance":
        command.extend(
            [
                "--mode",
                "perf",
                "--summarizer",
                str(config.get("summarizer", "default_perf")),
            ]
        )
    else:
        command.extend(["--mode", "all"])
        if config.get("dump_eval_details", False):
            command.append("--dump-eval-details")
    return command


def run_aisbench(config_file: Path) -> None:
    """Execute one AISBench step and publish the framework result contract."""
    config = load_run_config(config_file)
    artifact_directory = Path(os.environ["MULTI_NODE_STEP_ARTIFACT_DIR"]).resolve()
    result_file = Path(os.environ["MULTI_NODE_STEP_RESULT_FILE"]).resolve()
    source = Path(os.environ["MULTI_NODE_AISBENCH_SOURCE"]).resolve()
    executable = os.environ["MULTI_NODE_AISBENCH_BIN"]
    dataset_directory = download_dataset(str(config["dataset_path"]))
    config_directory = artifact_directory / "aisbench-config"
    request_name, dataset_name = prepare_configs(
        config, source, config_directory, dataset_directory
    )

    command = aisbench_command(
        config,
        executable,
        config_directory,
        request_name,
        dataset_name,
    )
    subprocess.run(command, cwd=artifact_directory, check=True)

    if config["case_type"] == "accuracy":
        result, failure_threshold = build_accuracy_result(
            artifact_directory,
            baseline=config.get("baseline"),
            allowed_drop=float(config.get("allowed_drop", 0.0)),
        )
        write_json_atomic(
            result_file,
            result,
        )
        if result["status"] == "failed":
            raise RuntimeError(
                f"accuracy {result['metrics']['accuracy']} is below "
                f"{failure_threshold}"
            )
    else:
        metrics, sources = performance_metrics(artifact_directory)
        write_json_atomic(
            result_file,
            {
                "status": "passed",
                "type": "performance",
                "metrics": metrics,
                "artifacts": relative_artifacts(sources, artifact_directory),
            },
        )


def _latest_run(directory: Path) -> Path:
    """Return the newest AISBench run directory from the fixed output layout."""
    output_root = directory / "outputs/default"
    runs = (
        [path for path in output_root.iterdir() if path.is_dir()]
        if output_root.is_dir()
        else []
    )
    if not runs:
        raise RuntimeError(f"AISBench run directory not found under {output_root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _latest_file(directory: Path, pattern: str, label: str) -> Path:
    """Return the newest matching artifact or fail with a focused message."""
    files = list(directory.glob(pattern))
    if not files:
        raise RuntimeError(f"AISBench {label} not found under {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def _number_with_unit(value: object, unit: str, field: str) -> float:
    """Parse a numeric AISBench field while enforcing its expected unit."""
    if not isinstance(value, str) or not value.endswith(unit):
        raise RuntimeError(f"invalid AISBench {field}: {value!r}")
    try:
        return float(value.removesuffix(unit))
    except ValueError as error:
        raise RuntimeError(f"invalid AISBench {field}: {value!r}") from error


def accuracy_score(directory: Path) -> tuple[float, Path]:
    """Read the fixed AISBench 3.1 summary CSV contract."""
    summary_directory = _latest_run(directory) / "summary"
    path = _latest_file(summary_directory, "summary_*.csv", "summary CSV")
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        prefix = ["dataset", "version", "metric", "mode"]
        if fields[:4] != prefix:
            raise RuntimeError(f"invalid AISBench accuracy columns in {path}")
        score_fields = fields[4:]
        if score_fields and score_fields[0] == "total_count":
            score_fields = score_fields[1:]
        if len(score_fields) != 1:
            raise RuntimeError(f"expected one AISBench model column in {path}")
        rows = [row for row in reader if row["metric"] == "accuracy"]

    if len(rows) != 1:
        raise RuntimeError(f"expected one AISBench accuracy row in {path}")
    value = rows[0][score_fields[0]]
    match = (
        re.fullmatch(
            r"(-?(?:\d+(?:\.\d*)?|\.\d+))(?: \(\d+/\d+\))?", value
        )
        if value is not None
        else None
    )
    if match is None:
        raise RuntimeError(f"invalid AISBench accuracy score in {path}: {value!r}")
    return float(match.group(1)), path


def performance_metrics(directory: Path) -> tuple[dict[str, float], list[Path]]:
    """Read the fixed AISBench 3.1 default_perf JSON/CSV pair."""
    performance_root = _latest_run(directory) / "performances"
    json_files = list(performance_root.glob("*/*.json"))
    if len(json_files) != 1:
        raise RuntimeError(
            f"expected one AISBench performance JSON under {performance_root}"
        )
    json_path = json_files[0]
    csv_path = json_path.with_suffix(".csv")
    if not csv_path.is_file():
        raise RuntimeError(f"AISBench performance CSV not found: {csv_path}")

    try:
        value = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid AISBench performance JSON: {json_path}") from error
    try:
        request_rate = value["Request Throughput"]["total"]
        output_rate = value["Output Token Throughput"]["total"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"invalid AISBench performance JSON: {json_path}") from error

    expected_columns = [
        "Performance Parameters",
        "Stage",
        "Average",
        "Min",
        "Max",
        "Median",
        "P75",
        "P90",
        "P99",
        "N",
    ]
    with csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != expected_columns:
            raise RuntimeError(f"invalid AISBench performance columns in {csv_path}")
        rows = {
            row["Performance Parameters"]: row
            for row in reader
            if row["Stage"] == "total"
        }

    try:
        e2el = rows["E2EL"]["Average"]
        ttft = rows["TTFT"]["Average"]
        tpot = rows["TPOT"]["Average"]
    except KeyError as error:
        raise RuntimeError(f"invalid AISBench performance CSV: {csv_path}") from error

    metrics = {
        "request_per_second": _number_with_unit(
            request_rate, " req/s", "Request Throughput.total"
        ),
        "output_token_per_second": _number_with_unit(
            output_rate, " token/s", "Output Token Throughput.total"
        ),
        "e2e_latency_ms": _number_with_unit(e2el, " ms", "E2EL.Average"),
        "ttft_ms": _number_with_unit(ttft, " ms", "TTFT.Average"),
        "tpot_ms": _number_with_unit(tpot, " ms", "TPOT.Average"),
    }
    return metrics, [json_path, csv_path]


def build_accuracy_result(
    directory: Path,
    *,
    baseline: object = None,
    allowed_drop: float = 0.0,
) -> tuple[dict[str, object], float | None]:
    """Translate an accuracy summary and apply an optional baseline gate.

    The returned threshold is populated only when the gate fails, allowing the
    caller to persist the structured result before raising a stage error.
    """
    score, source_file = accuracy_score(directory)
    metrics: dict[str, float] = {"accuracy": score}
    status = "passed"
    threshold = None
    if baseline is not None:
        numeric_baseline = float(baseline)
        metrics.update(
            baseline=numeric_baseline,
            allowed_drop=allowed_drop,
        )
        threshold = numeric_baseline - allowed_drop
        if score < threshold:
            status = "failed"
        else:
            threshold = None
    return (
        {
            "status": status,
            "type": "accuracy",
            "mode": "gate" if baseline is not None else "smoke",
            "metrics": metrics,
            "artifacts": relative_artifacts([source_file], directory),
        },
        threshold,
    )


def relative_artifacts(paths: Iterable[Path], root: Path) -> list[str]:
    """Convert artifact paths to portable paths relative to a step root."""
    return [path.relative_to(root).as_posix() for path in paths]


def main() -> int:
    """Run AISBench and convert expected operational errors to exit status 1."""
    args = parse_args()
    try:
        run_aisbench(args.config)
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
