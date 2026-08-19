from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/recipe/multi_node"))

from scripts.aisbench import (  # noqa: E402
    accuracy_score,
    aisbench_command,
    build_accuracy_result,
    load_run_config,
    performance_metrics,
    prepare_configs,
)


class AisbenchResultTests(unittest.TestCase):
    def test_pinned_templates_are_copied_and_filled_in_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            model_template = (
                source
                / "ais_bench/benchmark/configs/models/vllm_api/general.py"
            )
            dataset_template = (
                source
                / "ais_bench/benchmark/configs/datasets/gsm8k/sample.py"
            )
            model_template.parent.mkdir(parents=True)
            dataset_template.parent.mkdir(parents=True)
            model_template.write_text(
                "models = [dict(\n"
                "    path='',\n    model='',\n    request_rate=0,\n"
                "    host_ip='localhost',\n    host_port=8080,\n"
                "    max_out_len=512,\n    batch_size=1,\n"
                "    generation_kwargs=dict(\n        temperature=0.01,\n    ),\n)]\n",
                encoding="utf-8",
            )
            dataset_template.write_text(
                "datasets = [dict(\n    path='ais_bench/datasets/gsm8k',\n)]\n",
                encoding="utf-8",
            )
            config = {
                "case_type": "accuracy",
                "dataset_path": "vllm-ascend/gsm8k-lite",
                "dataset_conf": "gsm8k/sample",
                "request_conf": "general",
                "num_prompts": 1,
                "max_out_len": 16,
                "batch_size": 2,
                "request_rate": 3,
                "temperature": 0.2,
            }
            environment = {
                "MULTI_NODE_MODEL_PATH": "/models/fake",
                "MULTI_NODE_SERVED_MODEL_NAME": "fake",
                "MULTI_NODE_ENDPOINT_HOST": "10.0.0.8",
                "MULTI_NODE_ENDPOINT_PORT": "38085",
            }

            with mock.patch.dict(os.environ, environment, clear=False):
                request, dataset = prepare_configs(
                    config,
                    source,
                    root / "artifact/config",
                    root / "dataset-cache/gsm8k-lite",
                )

            self.assertEqual((request, dataset), ("general", "sample"))
            model = (
                root / "artifact/config/models/vllm_api/general.py"
            ).read_text()
            dataset_config = (
                root / "artifact/config/datasets/gsm8k/sample.py"
            ).read_text()
            self.assertIn("path='/models/fake'", model)
            self.assertIn("model='fake'", model)
            self.assertIn("host_ip='10.0.0.8'", model)
            self.assertIn("host_port=38085", model)
            self.assertIn("max_out_len=16", model)
            self.assertIn("batch_size=2", model)
            self.assertIn("request_rate=3", model)
            self.assertIn("temperature=0.2", model)
            self.assertIn(str(root / "dataset-cache/gsm8k-lite"), dataset_config)

    def test_step_input_is_strict_and_keeps_num_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            value = {
                "aisbench": {
                    "case_type": "accuracy",
                    "dataset_path": "vllm-ascend/gsm8k-lite",
                    "dataset_conf": "gsm8k/sample",
                    "request_conf": "general",
                    "num_prompts": 3,
                    "max_out_len": 16,
                    "batch_size": 1,
                }
            }
            path.write_text(json.dumps(value), encoding="utf-8")

            config = load_run_config(path)

            self.assertEqual(config["num_prompts"], 3)

    def test_command_uses_plan_prompt_limit_and_case_mode(self) -> None:
        command = aisbench_command(
            {
                "case_type": "performance",
                "num_prompts": 7,
                "summarizer": "default_perf",
            },
            "/cache/bin/ais_bench",
            Path("/artifact/config"),
            "vllm_api_stream_chat",
            "gsm8k_gen_0_shot_cot_str_perf",
        )

        self.assertIn("--num-prompts", command)
        self.assertEqual(command[command.index("--num-prompts") + 1], "7")
        self.assertIn("perf", command)
        self.assertIn("default_perf", command)

    def test_accuracy_summary_is_translated_and_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            summary = (
                artifact
                / "outputs/default/20260809_142027/summary/summary_20260809.csv"
            )
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "dataset,version,metric,mode,total_count,multi-node-vllm\n"
                "gsm8k,1,accuracy,gen,8,82.00 (7/8)\n",
                encoding="utf-8",
            )
            score, source = accuracy_score(artifact)
            self.assertEqual(score, 82.0)
            self.assertEqual(source, summary)

            value, threshold = build_accuracy_result(
                artifact, baseline=85, allowed_drop=2
            )

            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["mode"], "gate")
            self.assertEqual(value["metrics"]["accuracy"], 82.0)
            self.assertEqual(threshold, 83.0)

    def test_performance_json_and_csv_are_translated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            performance = (
                artifact
                / "outputs/default/20260809_142027/performances/multi-node-vllm"
            )
            performance.mkdir(parents=True)
            (performance / "gsm8k.json").write_text(
                json.dumps(
                    {
                        "Request Throughput": {"total": "0.2665 req/s"},
                        "Output Token Throughput": {"total": "8.529 token/s"},
                    }
                ),
                encoding="utf-8",
            )
            (performance / "gsm8k.csv").write_text(
                "Performance Parameters,Stage,Average,Min,Max,Median,P75,P90,P99,N\n"
                "E2EL,total,7503.7 ms,1,2,3,4,5,6,2\n"
                "TTFT,total,100.5 ms,1,2,3,4,5,6,2\n"
                "TPOT,total,20.25 ms,1,2,3,4,5,6,2\n",
                encoding="utf-8",
            )

            metrics, sources = performance_metrics(artifact)

            self.assertEqual(metrics["request_per_second"], 0.2665)
            self.assertEqual(metrics["output_token_per_second"], 8.529)
            self.assertEqual(metrics["e2e_latency_ms"], 7503.7)
            self.assertEqual(metrics["ttft_ms"], 100.5)
            self.assertEqual(metrics["tpot_ms"], 20.25)
            self.assertEqual(
                {path.name for path in sources},
                {"gsm8k.json", "gsm8k.csv"},
            )

    def test_performance_parser_requires_the_current_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            (artifact / "performance.json").write_text(
                json.dumps({"Request Throughput": "1 req/s"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "run directory not found"):
                performance_metrics(artifact)


if __name__ == "__main__":
    unittest.main()
