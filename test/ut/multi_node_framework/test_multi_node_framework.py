from __future__ import annotations

import io
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/recipe/multi_node"))

from converter.cli import main as convert_main  # noqa: E402
from scripts.plan import load_hosts, load_plan  # noqa: E402
from scripts.result import NodeOutcome, RunFailure, StopSignal  # noqa: E402
from scripts.runner import (  # noqa: E402
    aggregate_run_outcome,
    format_result_summary,
    wait_http_ready,
)


EXAMPLE = ROOT / "test/recipe/multi_node/.generated/template-pd/pd-2n2c"
GENERIC_DP_EXAMPLE = (
    ROOT / "test/recipe/multi_node/.generated/template2-non-pd/dp-2n2c"
)
RECIPE_TEMPLATE_CASES = (
    (
        "DeepSeek/template_pd.yaml",
        "pd-2n2c",
        "pd",
        "1p1d",
        {
            "prefill-0-template",
            "decode-0-template",
            "prefill-0-launch",
            "decode-0-launch",
            "gateway-0",
            "service-check",
        },
    ),
    (
        "Qwen/template2_non_pd.yaml",
        "dp-2n2c",
        "non-pd",
        "2-node",
        {"api-0", "headless-0", "service-check"},
    ),
)


def setUpModule() -> None:
    """Generate both ignored runtime fixtures before this module reads them."""
    cases = (
        ("models/en/DeepSeek/template_pd.yaml", "pd-2n2c"),
        ("models/en/Qwen/template2_non_pd.yaml", "dp-2n2c"),
    )
    for recipe, test_id in cases:
        status = convert_main(["--recipe", str(ROOT / recipe), "--test-id", test_id])
        if status:
            raise RuntimeError(f"converter failed for {recipe}#{test_id}")


def tearDownModule() -> None:
    """Remove only the two fixtures created by ``setUpModule``."""
    for directory in (EXAMPLE, GENERIC_DP_EXAMPLE):
        shutil.rmtree(directory, ignore_errors=True)
    for directory in (
        EXAMPLE.parent,
        GENERIC_DP_EXAMPLE.parent,
        EXAMPLE.parents[1],
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


class RunnerProgressTests(unittest.TestCase):
    def test_http_readiness_prints_periodic_waiting_heartbeat(self) -> None:
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        output = io.StringIO()

        with (
            patch(
                "scripts.runner.DIRECT_OPENER.open",
                side_effect=[urllib.error.URLError("not ready"), response],
            ),
            patch(
                "scripts.runner.time.monotonic",
                side_effect=[0, 0, 0, 31, 31],
            ),
            patch("scripts.runner.time.sleep"),
            redirect_stdout(output),
        ):
            wait_http_ready(
                "http://127.0.0.1:8000/health",
                100,
                lambda: None,
                progress_label="[startup] node=node0 rank=0",
                progress_interval_seconds=30,
            )

        self.assertIn("status=waiting elapsed=31s", output.getvalue())


class PlanTests(unittest.TestCase):
    def test_examples_have_bilingual_recipe_templates(self) -> None:
        for (
            recipe_path,
            test_id,
            deployment,
            case,
            expected_scripts,
        ) in RECIPE_TEMPLATE_CASES:
            documents_by_language = {}
            for language in ("en", "zh"):
                recipe = yaml.safe_load(
                    (ROOT / "models" / language / recipe_path).read_text(
                        encoding="utf-8"
                    )
                )
                scenario = next(
                    scenario
                    for scenario in recipe["scenarios"]
                    if scenario.get("test_id") == test_id
                )
                test_ids = [
                    item["test_id"]
                    for item in recipe["scenarios"]
                    if "test_id" in item
                ]
                self.assertEqual(len(test_ids), len(set(test_ids)))
                self.assertEqual(scenario["deployment"], deployment)
                self.assertEqual(scenario["case"], case)
                self.assertEqual(set(scenario["scripts"]), expected_scripts)
                self.assertEqual(scenario["npu_per_node"], 2)
                self.assertEqual(
                    scenario["aisbench"], ["accuracy", "performance"]
                )
                documents_by_language[language] = (recipe, scenario)

            recipe_en, scenario_en = documents_by_language["en"]
            recipe_zh, scenario_zh = documents_by_language["zh"]
            for field in (
                "test_id",
                "npu",
                "precision",
                "deployment",
                "case",
                "npu_per_node",
                "aisbench",
                "scripts",
            ):
                self.assertEqual(
                    scenario_en[field],
                    scenario_zh[field],
                    f"{recipe_path} differs between en/zh at {field}",
                )

            def effective_defaults(recipe, scenario):
                definitions = dict(recipe.get("config_params", {}))
                definitions.update(scenario.get("config_params", {}))
                return {
                    name: definition.get("default")
                    for name, definition in definitions.items()
                }

            self.assertEqual(
                effective_defaults(recipe_en, scenario_en),
                effective_defaults(recipe_zh, scenario_zh),
                f"{recipe_path} has different en/zh parameter defaults",
            )

        qwen_recipe = yaml.safe_load(
            (ROOT / "models/en/Qwen/template2_non_pd.yaml").read_text(
                encoding="utf-8"
            )
        )
        qwen_scenario = next(
            scenario
            for scenario in qwen_recipe["scenarios"]
            if scenario.get("test_id") == "dp-2n2c"
        )
        self.assertIn(
            "--data-parallel-size 4",
            qwen_scenario["scripts"]["api-0"]["content"],
        )
        self.assertIn(
            "--headless",
            qwen_scenario["scripts"]["headless-0"]["content"],
        )
        qwen_scripts = "\n".join(
            script["content"] for script in qwen_scenario["scripts"].values()
        )
        self.assertIn("vllm serve Qwen/Qwen3-30B-A3B", qwen_scripts)
        self.assertEqual(
            {
                name: definition["default"]
                for name, definition in qwen_recipe["config_params"].items()
            },
            {
                "max_model_len": 4096,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.9,
            },
        )

        deepseek_recipe = yaml.safe_load(
            (ROOT / "models/en/DeepSeek/template_pd.yaml").read_text(
                encoding="utf-8"
            )
        )
        deepseek_scenario = next(
            scenario
            for scenario in deepseek_recipe["scenarios"]
            if scenario.get("test_id") == "pd-2n2c"
        )
        prefill = deepseek_scenario["scripts"]["prefill-0-template"]["content"]
        decode = deepseek_scenario["scripts"]["decode-0-template"]["content"]
        gateway = deepseek_scenario["scripts"]["gateway-0"]["content"]
        service_check = deepseek_scenario["scripts"]["service-check"]["content"]
        self.assertIn('"kv_port":"30000"', prefill)
        self.assertIn('"kv_port":"30200"', decode)
        self.assertIn("--served-model-name deepseek-v2-lite", prefill)
        self.assertIn("--port 38085", gateway)
        self.assertIn("--prefiller-ports 7100 7101", gateway)
        self.assertIn("--decoder-ports 7100 7101", gateway)
        self.assertIn("curl --fail --silent --show-error", service_check)
        self.assertIn(":38085/v1/completions", service_check)
        self.assertIn('["choices"]', service_check)
        self.assertIn("vllm serve vllm-ascend/DeepSeek-V2-Lite-W8A8", prefill)
        self.assertIn(
            "/vllm-workspace/vllm-ascend/examples/external_online_dp/launch_online_dp.py",
            deepseek_scenario["scripts"]["prefill-0-launch"]["content"],
        )

    def test_example_has_two_independent_two_instance_nodes(self) -> None:
        plan = load_plan(EXAMPLE / "plan.yaml")

        self.assertEqual(plan.name, "template-pd-pd-2n2c")
        self.assertEqual(plan.leader.id, "node0")
        self.assertEqual([node.role for node in plan.nodes], ["prefill", "decode"])
        self.assertEqual([node.index for node in plan.nodes], [0, 1])
        self.assertEqual([node.readiness.count for node in plan.nodes], [2, 2])
        self.assertEqual(
            [node.launch for node in plan.nodes],
            ["nodes/node0/run.sh", "nodes/node1/run.sh"],
        )
        self.assertEqual(plan.gateway.port, 38085)
        for node in plan.nodes:
            template = (EXAMPLE / node.launch).parent / "run_dp_template.sh"
            template_text = template.read_text(encoding="utf-8")
            self.assertIn(
                'rank_log_directory="$MULTI_NODE_NODE_ARTIFACT_DIR/servers"',
                template_text,
            )
            self.assertIn(
                'rank_log="$rank_log_directory/rank-$4.log"', template_text
            )
            self.assertIn('} > "$rank_log" 2>&1', template_text)
        self.assertEqual(
            [stage.id for stage in plan.stages],
            ["completion", "accuracy", "performance"],
        )

    def test_generic_dp_example_has_one_four_rank_group(self) -> None:
        plan = load_plan(GENERIC_DP_EXAMPLE / "plan.yaml")
        api_run = (GENERIC_DP_EXAMPLE / plan.nodes[0].launch).read_text(
            encoding="utf-8"
        )
        headless_run = (GENERIC_DP_EXAMPLE / plan.nodes[1].launch).read_text(
            encoding="utf-8"
        )

        self.assertEqual(plan.name, "template2-non-pd-dp-2n2c")
        self.assertEqual([node.id for node in plan.nodes], ["node0", "node1"])
        self.assertEqual([node.role for node in plan.nodes], ["api", "headless"])
        self.assertEqual(plan.nodes[0].readiness.count, 1)
        self.assertIsNone(plan.nodes[1].readiness)
        self.assertIsNone(plan.gateway)
        self.assertIn("--data-parallel-size 4", api_run)
        self.assertIn("--data-parallel-size-local 2", api_run)
        self.assertIn("--headless", headless_run)
        self.assertIn("--data-parallel-start-rank 2", headless_run)
        self.assertIn(
            '--data-parallel-address "$MULTI_NODE_NODE_0_IP"', headless_run
        )

    def test_examples_declare_aisbench_inputs_without_vendored_configs(self) -> None:
        for example in (
            EXAMPLE,
            GENERIC_DP_EXAMPLE,
        ):
            plan = load_plan(example / "plan.yaml")
            accuracy = plan.stages[1].steps[0]
            performance = plan.stages[2].steps[0]
            run_script = (example / "evaluations/run_aisbench.sh").read_text()

            self.assertEqual(accuracy.script, "evaluations/run_aisbench.sh")
            self.assertEqual(performance.script, "evaluations/run_aisbench.sh")
            self.assertEqual(
                accuracy.inputs["aisbench"]["request_conf"],
                "vllm_api_general_chat",
            )
            self.assertEqual(
                performance.inputs["aisbench"]["request_conf"],
                "vllm_api_stream_chat",
            )
            self.assertIn("MULTI_NODE_STEP_INPUT_FILE", run_script)
            self.assertFalse(any((example / "aisbench").rglob("*.py")))

    def test_small_plans_use_modelscope_datasets_and_prompt_limit(self) -> None:
        for example in (EXAMPLE, GENERIC_DP_EXAMPLE):
            plan = load_plan(example / "plan.yaml")
            accuracy = plan.stages[1].steps[0].inputs["aisbench"]
            performance = plan.stages[2].steps[0].inputs["aisbench"]

            self.assertEqual(accuracy["dataset_path"], "vllm-ascend/gsm8k-lite")
            self.assertEqual(
                performance["dataset_path"],
                "vllm-ascend/GSM8K-in3500-bs400",
            )
            self.assertEqual(accuracy["num_prompts"], 1)
            self.assertEqual(performance["num_prompts"], 1)
            self.assertFalse(any((example / "aisbench").rglob("*.jsonl")))

    def test_result_summary_is_single_line_and_bounded(self) -> None:
        summary = format_result_summary(
            {"status": "passed", "metrics": {"detail": "x" * 5000}}
        )

        self.assertNotIn("\n", summary)
        self.assertLessEqual(len(summary), 4096)
        self.assertTrue(summary.endswith("..."))

    def test_hosts_must_match_plan_nodes(self) -> None:
        plan = load_plan(EXAMPLE / "plan.yaml")
        with tempfile.TemporaryDirectory() as directory:
            hosts_path = Path(directory) / "hosts.yaml"
            hosts_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "hosts": {
                            "node0": {"address": "127.0.0.1", "interface": "lo"},
                            "node1": {"address": "127.0.0.2"},
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            hosts = load_hosts(hosts_path, plan)

        self.assertEqual(set(hosts), {"node0", "node1"})
        self.assertEqual(hosts["node0"].interface, "lo")

    def test_node_template_maps_launcher_index_to_selected_card(self) -> None:
        template = EXAMPLE / "nodes/node0/run_dp_template.sh"
        with tempfile.TemporaryDirectory() as directory:
            fake_bin = Path(directory)
            artifact_directory = fake_bin / "artifacts"
            fake_vllm = fake_bin / "vllm"
            fake_vllm.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$ASCEND_RT_VISIBLE_DEVICES"\n',
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "MULTI_NODE_VISIBLE_DEVICES": "4,5,6,7",
                    "MULTI_NODE_MODEL_PATH": "/models/fake",
                    "MULTI_NODE_SERVED_MODEL_NAME": "fake",
                    "MULTI_NODE_NODE_ARTIFACT_DIR": str(artifact_directory),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    str(template),
                    "1",
                    "7101",
                    "2",
                    "1",
                    "127.0.0.1",
                    "12321",
                    "1",
                ],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            )
            rank_log = (
                artifact_directory / "servers/rank-1.log"
            ).read_text(encoding="utf-8")

        self.assertEqual(result.stdout, "")
        self.assertIn("external DP rank=1 device=5 port=7101", rank_log)
        self.assertEqual(rank_log.splitlines()[-1], "5")

class LocalRunnerTests(unittest.TestCase):
    def test_two_nodes_run_every_check_and_evaluation_declared_by_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory)
            control_port = free_port()
            prefill_port = free_port()
            decode_port = free_port()
            gateway_port = free_port()
            self._write_fake_runtime(plan_dir)
            self._write_fake_plan(plan_dir, prefill_port, decode_port, gateway_port)

            artifact_root = plan_dir / "artifacts"
            command = ["bash", str(ROOT / "test/recipe/multi_node/scripts/run.sh")]
            common_environment = os.environ.copy()
            common_environment.update(
                {
                    "PATH": f"{Path(sys.executable).parent}:{common_environment['PATH']}",
                    "MULTI_NODE_PLAN": str(plan_dir / "plan.yaml"),
                    "MULTI_NODE_CLUSTER_IPS": "127.0.0.1,127.0.0.1",
                    "MULTI_NODE_INTERFACE": "lo",
                    "VLLM_ASCEND_ROOT": str(plan_dir / "vllm-ascend"),
                    "MULTI_NODE_CONTROL_PORT": str(control_port),
                    "MULTI_NODE_STARTUP_TIMEOUT_SECONDS": "20",
                    "MULTI_NODE_RUN_TIMEOUT_SECONDS": "20",
                    "MULTI_NODE_ARTIFACT_ROOT": str(artifact_root),
                }
            )
            leader_environment = common_environment | {
                "MULTI_NODE_NODE_INDEX": "0"
            }
            worker_environment = common_environment | {
                "MULTI_NODE_NODE_INDEX": "1"
            }
            leader = subprocess.Popen(
                command,
                env=leader_environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            worker = subprocess.Popen(
                command,
                env=worker_environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            leader_output, _ = leader.communicate(timeout=30)
            worker_output, _ = worker.communicate(timeout=30)

            self.assertEqual(
                leader.returncode,
                0,
                f"{leader_output}\nworker output:\n{worker_output}",
            )
            self.assertEqual(worker.returncode, 0, worker_output)
            self.assertIn("local service ready", leader_output)
            self.assertIn(
                "[startup] node=node0 rank=0 status=ready", leader_output
            )
            self.assertIn("[startup] cluster nodes ready=2/2", leader_output)
            self.assertIn("starting gateway", leader_output)
            self.assertIn("[startup] gateway status=ready", leader_output)
            self.assertIn(
                '[stage] accuracy/accuracy result={"metrics":{"accuracy":1.0}',
                leader_output,
            )
            self.assertIn("plan completed", leader_output)
            self.assertIn("plan completed", worker_output)
            leader_artifacts = artifact_root / "local-runner-test" / "node0"
            self.assertTrue((leader_artifacts / "completion/health.log").is_file())
            self.assertEqual(
                (leader_artifacts / "accuracy/accuracy/result.txt").read_text(
                    encoding="utf-8"
                ),
                f"127.0.0.1:{gateway_port}\n",
            )
            final_result = json.loads(
                (artifact_root / "local-runner-test/result.json").read_text(
                    encoding="utf-8"
                )
            )
            leader_result = json.loads(
                (leader_artifacts / "node-result.json").read_text(encoding="utf-8")
            )
            worker_result = json.loads(
                (
                    artifact_root
                    / "local-runner-test/node1/node-result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(final_result["status"], "passed")
            self.assertEqual(
                {
                    node_id: result["status"]
                    for node_id, result in final_result["nodes"].items()
                },
                {"node0": "passed", "node1": "passed"},
            )
            self.assertEqual(
                final_result["stages"]["accuracy"]["accuracy"]["metrics"][
                    "accuracy"
                ],
                1.0,
            )
            self.assertEqual(
                final_result["stages"]["performance"]["performance"][
                    "metrics"
                ]["request_per_second"],
                2.0,
            )
            self.assertEqual(
                final_result["stages"]["custom-stage"]["custom"]["metrics"][
                    "value"
                ],
                1,
            )
            self.assertEqual(
                json.loads(
                    (
                        leader_artifacts / "custom-stage/custom/input.json"
                    ).read_text(encoding="utf-8")
                ),
                {"fixture": {"value": 7}},
            )
            self.assertEqual(leader_result["status"], "passed")
            self.assertEqual(worker_result["status"], "passed")
            self.assertEqual(
                set(leader_result),
                {
                    "schema_version",
                    "node_id",
                    "status",
                    "execution_status",
                    "failure",
                    "cleanup_errors",
                },
            )
            self.assertEqual(set(worker_result), set(leader_result))

    def test_remote_service_failure_interrupts_a_supervised_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory)
            control_port = free_port()
            prefill_port = free_port()
            decode_port = free_port()
            gateway_port = free_port()
            self._write_fake_runtime(plan_dir)
            self._write_fake_plan(plan_dir, prefill_port, decode_port, gateway_port)
            (plan_dir / "checks/health.sh").write_text(
                "sleep 20\n", encoding="utf-8"
            )
            (plan_dir / "nodes/node1/run.sh").write_text(
                'python3 "$MULTI_NODE_PLAN_DIR/fake_service.py" '
                '"$MULTI_NODE_LOCAL_IP" "$MULTI_NODE_SERVICE_PORT_START" &\n'
                "service_pid=$!\n"
                "sleep 3\n"
                'kill "$service_pid"\n'
                'wait "$service_pid"\n',
                encoding="utf-8",
            )
            hosts_path = plan_dir / "hosts.yaml"
            hosts_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "hosts": {
                            "node0": {"address": "127.0.0.1", "interface": "lo"},
                            "node1": {"address": "127.0.0.1", "interface": "lo"},
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            artifact_root = plan_dir / "artifacts"
            command = [
                sys.executable,
                str(ROOT / "test/recipe/multi_node/scripts/runner.py"),
                "--plan",
                str(plan_dir / "plan.yaml"),
                "--hosts",
                str(hosts_path),
                "--control-port",
                str(control_port),
                "--startup-timeout-seconds",
                "15",
                "--run-timeout-seconds",
                "15",
                "--artifact-root",
                str(artifact_root),
            ]
            started = time.monotonic()
            leader = subprocess.Popen(
                [*command, "--node-id", "node0"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            worker = subprocess.Popen(
                [*command, "--node-id", "node1"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            leader_output, _ = leader.communicate(timeout=25)
            worker_output, _ = worker.communicate(timeout=25)

            self.assertLess(time.monotonic() - started, 15)
            self.assertNotEqual(leader.returncode, 0, leader_output)
            self.assertNotEqual(worker.returncode, 0, worker_output)
            final_result = json.loads(
                (artifact_root / "local-runner-test/result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(final_result["status"], "failed")
            self.assertEqual(final_result["failure"]["category"], "node_failed")
            self.assertIn("node1", final_result["failure"]["message"])
            self.assertEqual(final_result["failure_node_id"], "node1")
            self.assertEqual(
                final_result["nodes"]["node0"]["execution_status"], "aborted"
            )
            self.assertIsNone(final_result["nodes"]["node0"]["failure"])
            self.assertEqual(
                final_result["nodes"]["node1"]["execution_status"], "failed"
            )

    def test_leader_stage_failure_aborts_worker_without_local_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory)
            control_port = free_port()
            prefill_port = free_port()
            decode_port = free_port()
            gateway_port = free_port()
            self._write_fake_runtime(plan_dir)
            self._write_fake_plan(plan_dir, prefill_port, decode_port, gateway_port)
            (plan_dir / "evaluations/accuracy.sh").write_text(
                "echo intentional evaluation failure >&2\nexit 9\n",
                encoding="utf-8",
            )

            artifact_root = plan_dir / "artifacts"
            leader, worker, leader_output, worker_output = self._run_direct_nodes(
                plan_dir, control_port, artifact_root
            )

            self.assertNotEqual(leader.returncode, 0, leader_output)
            self.assertNotEqual(worker.returncode, 0, worker_output)
            final_result = json.loads(
                (artifact_root / "local-runner-test/result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(final_result["failure_node_id"], "node0")
            self.assertEqual(
                final_result["nodes"]["node0"]["execution_status"], "failed"
            )
            self.assertEqual(
                final_result["nodes"]["node1"]["execution_status"], "aborted"
            )
            self.assertIsNone(final_result["nodes"]["node1"]["failure"])

    def test_cleanup_failure_from_worker_makes_the_aggregate_fail(self) -> None:
        plan = load_plan(EXAMPLE / "plan.yaml")
        cleanup = RunFailure(
            category="cleanup_failed", message="worker process group survived"
        )
        outcomes = {
            "node0": NodeOutcome(node_id="node0", execution_status="passed"),
            "node1": NodeOutcome(
                node_id="node1",
                execution_status="passed",
                cleanup_errors=(cleanup,),
            ),
        }

        result = aggregate_run_outcome(
            plan=plan,
            outcomes=outcomes,
            stages={},
            stop_signal=StopSignal(kind="completed", origin_node_id="node0"),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_node_id, "node1")
        self.assertEqual(result.failure, cleanup)
        self.assertEqual(result.nodes["node1"].execution_status, "passed")
        self.assertEqual(result.nodes["node1"].status, "failed")

    def test_first_failed_stop_remains_primary_when_another_node_also_fails(self) -> None:
        plan = load_plan(EXAMPLE / "plan.yaml")
        leader_failure = RunFailure(category="node_failed", message="leader failed later")
        worker_failure = RunFailure(category="node_failed", message="worker failed first")
        outcomes = {
            "node0": NodeOutcome(
                node_id="node0", execution_status="failed", failure=leader_failure
            ),
            "node1": NodeOutcome(
                node_id="node1", execution_status="failed", failure=worker_failure
            ),
        }

        result = aggregate_run_outcome(
            plan=plan,
            outcomes=outcomes,
            stages={},
            stop_signal=StopSignal(
                kind="failed", origin_node_id="node1", failure=worker_failure
            ),
        )

        self.assertEqual(result.failure_node_id, "node1")
        self.assertEqual(result.failure, worker_failure)

    def test_leader_sigterm_cancels_run_after_every_node_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory)
            control_port = free_port()
            prefill_port = free_port()
            decode_port = free_port()
            gateway_port = free_port()
            self._write_fake_runtime(plan_dir)
            self._write_fake_plan(plan_dir, prefill_port, decode_port, gateway_port)
            (plan_dir / "checks/health.sh").write_text(
                "sleep 30\nprintf '%s\\n' '{\"status\":\"passed\"}' "
                '> "$MULTI_NODE_STEP_RESULT_FILE"\n',
                encoding="utf-8",
            )

            artifact_root = plan_dir / "artifacts"
            command = ["bash", str(ROOT / "test/recipe/multi_node/scripts/run.sh")]
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{Path(sys.executable).parent}:{environment['PATH']}",
                    "MULTI_NODE_PLAN": str(plan_dir / "plan.yaml"),
                    "MULTI_NODE_CLUSTER_IPS": "127.0.0.1,127.0.0.1",
                    "MULTI_NODE_INTERFACE": "lo",
                    "VLLM_ASCEND_ROOT": str(plan_dir / "vllm-ascend"),
                    "MULTI_NODE_CONTROL_PORT": str(control_port),
                    "MULTI_NODE_STARTUP_TIMEOUT_SECONDS": "20",
                    "MULTI_NODE_RUN_TIMEOUT_SECONDS": "20",
                    "MULTI_NODE_ARTIFACT_ROOT": str(artifact_root),
                }
            )
            leader = subprocess.Popen(
                command,
                env=environment | {"MULTI_NODE_NODE_INDEX": "0"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            worker = subprocess.Popen(
                command,
                env=environment | {"MULTI_NODE_NODE_INDEX": "1"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            check_log = artifact_root / "local-runner-test/node0/completion/health.log"
            deadline = time.monotonic() + 15
            while not check_log.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(check_log.exists(), "leader never started the long check")
            leader.send_signal(signal.SIGTERM)
            leader_output, _ = leader.communicate(timeout=25)
            worker_output, _ = worker.communicate(timeout=25)

            self.assertNotEqual(leader.returncode, 0, leader_output)
            self.assertNotEqual(worker.returncode, 0, worker_output)
            final_result = json.loads(
                (artifact_root / "local-runner-test/result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(final_result["status"], "cancelled")
            self.assertEqual(
                final_result["nodes"]["node0"]["execution_status"], "cancelled"
            )
            self.assertEqual(
                final_result["nodes"]["node1"]["execution_status"], "aborted"
            )
            self.assertEqual(final_result["missing_nodes"], [])

    @staticmethod
    def _run_direct_nodes(
        plan_dir: Path, control_port: int, artifact_root: Path
    ) -> tuple[
        subprocess.Popen[str], subprocess.Popen[str], str, str
    ]:
        hosts_path = plan_dir / "hosts.yaml"
        hosts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "hosts": {
                        "node0": {"address": "127.0.0.1", "interface": "lo"},
                        "node1": {"address": "127.0.0.1", "interface": "lo"},
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(ROOT / "test/recipe/multi_node/scripts/runner.py"),
            "--plan",
            str(plan_dir / "plan.yaml"),
            "--hosts",
            str(hosts_path),
            "--control-port",
            str(control_port),
            "--startup-timeout-seconds",
            "15",
            "--run-timeout-seconds",
            "15",
            "--artifact-root",
            str(artifact_root),
        ]
        leader = subprocess.Popen(
            [*command, "--node-id", "node0"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        worker = subprocess.Popen(
            [*command, "--node-id", "node1"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        leader_output, _ = leader.communicate(timeout=25)
        worker_output, _ = worker.communicate(timeout=25)
        return leader, worker, leader_output, worker_output

    @staticmethod
    def _write_fake_runtime(plan_dir: Path) -> None:
        required = (
            "examples/external_online_dp/launch_online_dp.py",
        )
        for relative_path in required:
            path = plan_dir / "vllm-ascend" / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fake runtime tool\n", encoding="utf-8")

    @staticmethod
    def _write_fake_plan(
        plan_dir: Path,
        prefill_port: int,
        decode_port: int,
        gateway_port: int,
    ) -> None:
        for directory in (
            "nodes/node0",
            "nodes/node1",
            "gateway",
            "checks",
            "evaluations",
        ):
            (plan_dir / directory).mkdir(parents=True)

        (plan_dir / "fake_service.py").write_text(
            """from http.server import BaseHTTPRequestHandler, HTTPServer
import sys

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass

HTTPServer((sys.argv[1], int(sys.argv[2])), Handler).serve_forever()
""",
            encoding="utf-8",
        )
        service_script = (
            'exec python3 "$MULTI_NODE_PLAN_DIR/fake_service.py" '
            '"$MULTI_NODE_LOCAL_IP" "$MULTI_NODE_SERVICE_PORT_START"\n'
        )
        (plan_dir / "nodes/node0/run.sh").write_text(service_script, encoding="utf-8")
        (plan_dir / "nodes/node1/run.sh").write_text(service_script, encoding="utf-8")
        (plan_dir / "gateway/run.sh").write_text(
            'exec python3 "$MULTI_NODE_PLAN_DIR/fake_service.py" '
            '"$MULTI_NODE_LOCAL_IP" "$MULTI_NODE_GATEWAY_PORT"\n',
            encoding="utf-8",
        )
        (plan_dir / "checks/health.sh").write_text(
            "python3 -c 'import os, urllib.request; "
            'urllib.request.urlopen(os.environ["MULTI_NODE_ENDPOINT"] + "/healthcheck")\'\n'
            "printf '%s\\n' '{\"status\":\"passed\"}' "
            '> "$MULTI_NODE_STEP_RESULT_FILE"\n',
            encoding="utf-8",
        )
        (plan_dir / "evaluations/accuracy.sh").write_text(
            'echo "$MULTI_NODE_ENDPOINT_HOST:$MULTI_NODE_ENDPOINT_PORT" '
            '> "$MULTI_NODE_STEP_ARTIFACT_DIR/result.txt"\n'
            "printf '%s\\n' '{\"status\": \"passed\", \"type\": \"accuracy\", "
            "\"metrics\": {\"accuracy\": 1.0}}' > \"$MULTI_NODE_STEP_RESULT_FILE\"\n",
            encoding="utf-8",
        )
        (plan_dir / "evaluations/performance.sh").write_text(
            "printf '%s\\n' '{\"status\": \"passed\", "
            "\"type\": \"performance\", \"metrics\": "
            "{\"request_per_second\": 2.0}}' > \"$MULTI_NODE_STEP_RESULT_FILE\"\n",
            encoding="utf-8",
        )
        (plan_dir / "evaluations/custom.sh").write_text(
            "printf '%s\\n' '{\"status\": \"passed\", "
            '"metrics": {"value": 1}}\' > "$MULTI_NODE_STEP_RESULT_FILE"\n',
            encoding="utf-8",
        )

        plan_data = {
            "api_version": "multi-node/v1",
            "kind": "MultiNodePlan",
            "metadata": {"name": "local-runner-test"},
            "model": {
                "id": "fake/model",
                "cache_path": "fake/model",
                "served_name": "fake",
            },
            "resources": {"npu_per_node": 1},
            "nodes": [
                {
                    "id": "node0",
                    "role": "prefill",
                    "launch": "nodes/node0/run.sh",
                    "readiness": {"port_start": prefill_port},
                },
                {
                    "id": "node1",
                    "role": "decode",
                    "launch": "nodes/node1/run.sh",
                    "readiness": {"port_start": decode_port},
                },
            ],
            "gateway": {"launch": "gateway/run.sh", "port": gateway_port},
            "stages": [
                {
                    "id": "completion",
                    "failure_category": "check_failed",
                    "steps": [
                        {
                            "id": "health",
                            "script": "checks/health.sh",
                            "timeout_seconds": 5,
                        }
                    ],
                },
                {
                    "id": "accuracy",
                    "failure_category": "evaluation_failed",
                    "steps": [
                        {
                            "id": "accuracy",
                            "script": "evaluations/accuracy.sh",
                            "timeout_seconds": 5,
                        }
                    ],
                },
                {
                    "id": "performance",
                    "failure_category": "evaluation_failed",
                    "steps": [
                        {
                            "id": "performance",
                            "script": "evaluations/performance.sh",
                            "timeout_seconds": 5,
                        }
                    ],
                },
                {
                    "id": "custom-stage",
                    "failure_category": "custom_failure",
                    "steps": [
                        {
                            "id": "custom",
                            "script": "evaluations/custom.sh",
                            "timeout_seconds": 5,
                            "inputs": {"fixture": {"value": 7}},
                        }
                    ],
                },
            ],
        }
        (plan_dir / "plan.yaml").write_text(
            yaml.safe_dump(plan_data, sort_keys=False), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
