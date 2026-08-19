from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/recipe/multi_node"))

from scripts.plan import (  # noqa: E402
    PlanError,
    format_topology_summary,
    load_hosts,
    load_plan,
)


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.plan_directory = Path(self.temporary_directory.name)
        for relative_path in (
            "nodes/node0/run.sh",
            "nodes/node1/run.sh",
            "gateway/run.sh",
            "checks/completion.sh",
            "evaluations/accuracy.sh",
            "evaluations/performance.sh",
        ):
            path = self.plan_directory / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

        self.plan_data: dict[str, Any] = {
            "api_version": "multi-node/v1",
            "kind": "MultiNodePlan",
            "metadata": {"name": "framework-plan"},
            "model": {
                "id": "example/model",
                "cache_path": "example/model",
                "served_name": "example",
            },
            "resources": {"npu_per_node": 2},
            "nodes": [
                {
                    "id": "node0",
                    "role": "prefill",
                    "launch": "nodes/node0/run.sh",
                    "readiness": {
                        "port_start": 7100,
                        "count": 2,
                        "health_path": "/health",
                    },
                },
                {
                    "id": "node1",
                    "role": "decode",
                    "launch": "nodes/node1/run.sh",
                    "readiness": {"port_start": 7200},
                },
            ],
            "gateway": {
                "launch": "gateway/run.sh",
                "port": 38085,
                "health_path": "/healthcheck",
            },
            "stages": [
                {
                    "id": "completion",
                    "failure_category": "check_failed",
                    "steps": [
                        {
                            "id": "completion",
                            "script": "checks/completion.sh",
                            "timeout_seconds": 300,
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
                            "timeout_seconds": 600,
                            "inputs": {
                                "aisbench": {"num_prompts": 4},
                            },
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
                            "timeout_seconds": 900,
                        }
                    ],
                },
            ],
        }
        self.hosts_data = {
            "version": 1,
            "hosts": {
                "node0": {"address": "192.0.2.10", "interface": "eth0"},
                "node1": {"address": "192.0.2.11"},
            },
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_yaml(self, name: str, data: Any) -> Path:
        path = self.plan_directory / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    def write_plan(self, data: dict[str, Any] | None = None) -> Path:
        return self.write_yaml(
            "plan.yaml", self.plan_data if data is None else data
        )

    def write_hosts(self, data: dict[str, Any] | None = None) -> Path:
        return self.write_yaml(
            "hosts.yaml", self.hosts_data if data is None else data
        )

    def test_loads_executable_fields_and_ignores_generator_metadata(self) -> None:
        data = copy.deepcopy(self.plan_data)
        data["generated_by"] = "recipe compiler"
        data["metadata"]["source_recipe"] = "recipes/example.yaml"

        plan = load_plan(self.write_plan(data))

        self.assertEqual(plan.name, "framework-plan")
        self.assertEqual(plan.model.cache_path, "example/model")
        self.assertEqual(plan.resources.npu_per_node, 2)
        self.assertEqual([node.id for node in plan.nodes], ["node0", "node1"])
        self.assertEqual(plan.nodes[0].readiness.count, 2)
        self.assertEqual(plan.nodes[1].readiness.count, 1)
        self.assertEqual(plan.gateway.port, 38085)
        self.assertEqual(
            [stage.id for stage in plan.stages],
            ["completion", "accuracy", "performance"],
        )
        self.assertEqual(plan.stages[0].failure_category, "check_failed")
        self.assertEqual(plan.stages[1].steps[0].timeout_seconds, 600)
        self.assertEqual(
            plan.stages[1].steps[0].inputs,
            {"aisbench": {"num_prompts": 4}},
        )
        self.assertEqual(plan.stages[0].steps[0].inputs, {})

    def test_rejects_unknown_plan_protocol(self) -> None:
        invalid_values = (
            ("api_version", "multi-node/v2", "api_version must be multi-node/v1"),
            ("kind", "RecipePlan", "kind must be MultiNodePlan"),
        )
        for field, value, message in invalid_values:
            with self.subTest(field=field):
                data = copy.deepcopy(self.plan_data)
                data[field] = value
                with self.assertRaisesRegex(PlanError, message):
                    load_plan(self.write_plan(data))

    def test_rejects_invalid_node_and_endpoint_contracts(self) -> None:
        cases = []

        empty_nodes = copy.deepcopy(self.plan_data)
        empty_nodes["nodes"] = []
        cases.append((empty_nodes, "nodes must not be empty"))

        duplicate_nodes = copy.deepcopy(self.plan_data)
        duplicate_nodes["nodes"][1]["id"] = "node0"
        cases.append((duplicate_nodes, "nodes must have unique ids"))

        invalid_npu_count = copy.deepcopy(self.plan_data)
        invalid_npu_count["resources"]["npu_per_node"] = 2.0
        cases.append((invalid_npu_count, "npu_per_node must be a positive integer"))

        invalid_port_range = copy.deepcopy(self.plan_data)
        invalid_port_range["nodes"][0]["readiness"] = {
            "port_start": 65535,
            "count": 2,
        }
        cases.append((invalid_port_range, "port range exceeds 65535"))

        missing_direct_readiness = copy.deepcopy(self.plan_data)
        del missing_direct_readiness["gateway"]
        missing_direct_readiness["nodes"][0]["readiness"] = None
        cases.append(
            (
                missing_direct_readiness,
                "leader readiness is required when gateway is absent",
            )
        )

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PlanError, message):
                    load_plan(self.write_plan(data))

    def test_rejects_scripts_outside_or_missing_from_plan(self) -> None:
        cases = []

        escaping = copy.deepcopy(self.plan_data)
        escaping["nodes"][0]["launch"] = "../run.sh"
        cases.append((escaping, "must be a plan-relative path"))

        missing = copy.deepcopy(self.plan_data)
        missing["stages"][0]["steps"][0]["script"] = "checks/missing.sh"
        cases.append((missing, "does not exist"))

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PlanError, message):
                    load_plan(self.write_plan(data))

    def test_gateway_and_leader_readiness_define_the_runtime_endpoint(self) -> None:
        direct = copy.deepcopy(self.plan_data)
        del direct["gateway"]
        plan = load_plan(self.write_plan(direct))
        hosts = load_hosts(self.write_hosts(), plan)
        self.assertIn(
            "Endpoint: http://192.0.2.10:7100",
            format_topology_summary(plan, hosts),
        )

    def test_hosts_exactly_match_nodes_and_have_required_text(self) -> None:
        plan = load_plan(self.write_plan())
        hosts = load_hosts(self.write_hosts(), plan)
        self.assertEqual(hosts["node0"].interface, "eth0")
        self.assertIsNone(hosts["node1"].interface)
        self.assertEqual(plan.node("node1").index, 1)
        with self.assertRaisesRegex(PlanError, "Unknown node: node2"):
            plan.node("node2")

        missing = copy.deepcopy(self.hosts_data)
        del missing["hosts"]["node1"]
        with self.assertRaisesRegex(PlanError, r"missing=\['node1'\]"):
            load_hosts(self.write_hosts(missing), plan)

        invalid = copy.deepcopy(self.hosts_data)
        invalid["hosts"]["node0"]["address"] = ""
        with self.assertRaisesRegex(PlanError, r"hosts\.node0\.address"):
            load_hosts(self.write_hosts(invalid), plan)

        invalid_version = copy.deepcopy(self.hosts_data)
        invalid_version["version"] = 1.0
        with self.assertRaisesRegex(PlanError, "hosts version must be 1"):
            load_hosts(self.write_hosts(invalid_version), plan)

    def test_topology_summary_is_compact(self) -> None:
        plan = load_plan(self.write_plan())
        summary = format_topology_summary(plan, load_hosts(self.write_hosts(), plan))

        self.assertIn("Plan: framework-plan (2 nodes, 2 NPUs/node)", summary)
        self.assertIn(
            "node0 @192.0.2.10%eth0: prefill, nodes/node0/run.sh "
            "ports=7100-7101",
            summary,
        )
        self.assertIn("Gateway: gateway/run.sh port=38085", summary)
        self.assertIn("Stages: completion=1, accuracy=1, performance=1", summary)


if __name__ == "__main__":
    unittest.main()
