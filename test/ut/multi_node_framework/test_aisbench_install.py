from __future__ import annotations

import hashlib
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "test/recipe/multi_node/scripts/install_aisbench.sh"
CONSTRAINTS = ROOT / "test/recipe/multi_node/scripts/aisbench-constraints.txt"


class AisbenchInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache_root = self.root / "cache"
        self.environment_file = self.root / "aisbench.env"
        self.constraints = self.root / "constraints.txt"
        self.fake_bin = self.root / "bin"
        self.pip_arguments = self.root / "pip-arguments"
        self.environment_identity = "controller=fixture-cpu;runtime=fixture-npu"
        self.package_version = "3.1.fixture20260817"
        self.constraints.write_text("\n", encoding="utf-8")
        self.fake_bin.mkdir()
        flock = self.fake_bin / "flock"
        flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        flock.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @property
    def python_version(self) -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def environment(
        self,
        *,
        identity: str | None = None,
        python: Path | None = None,
        overrides: dict[str, str] | None = None,
    ) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "AIS_BENCH_CACHE_ROOT": str(self.cache_root),
                "AIS_BENCH_CACHE_SCHEMA": "fixture-v2",
                "AIS_BENCH_ENVIRONMENT_IDENTITY": (
                    identity if identity is not None else self.environment_identity
                ),
                "AIS_BENCH_CONSTRAINTS": str(self.constraints),
                "AIS_BENCH_PACKAGE_VERSION": self.package_version,
                "AIS_BENCH_PYTHON": str(python or sys.executable),
                "FAKE_PIP_ARGS_FILE": str(self.pip_arguments),
                "PIP_NO_INDEX": "1",
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
            }
        )
        environment.update(overrides or {})
        return environment

    def run_installer(
        self,
        *,
        identity: str | None = None,
        python: Path | None = None,
        overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(INSTALLER), "--env-file", str(self.environment_file)],
            env=self.environment(
                identity=identity,
                python=python,
                overrides=overrides,
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def expected_key(self, identity: str | None = None) -> str:
        constraints_hash = hashlib.sha256(
            self.constraints.read_bytes()
        ).hexdigest()[:16]
        environment_hash = hashlib.sha256(
            (identity or self.environment_identity).encode()
        ).hexdigest()[:16]
        return (
            f"fixture-v2-ais_bench_benchmark-{self.package_version}-"
            f"env{environment_hash}-"
            f"py{self.python_version}-{platform.machine()}-{constraints_hash}"
        )

    def source_directory(self, cache_directory: Path) -> Path:
        return (
            cache_directory
            / "venv/lib"
            / f"python{self.python_version}"
            / "site-packages"
        )

    def populate_valid_cache(self, identity: str | None = None) -> Path:
        cache_directory = self.cache_root / self.expected_key(identity)
        source = self.source_directory(cache_directory)
        command = cache_directory / "venv/bin/ais_bench"
        (source / "ais_bench/benchmark/configs/datasets").mkdir(parents=True)
        (source / "ais_bench/benchmark/configs/models/vllm_api").mkdir(
            parents=True
        )
        command.parent.mkdir(parents=True)
        command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        command.chmod(0o755)
        (cache_directory / "READY").write_text(
            self.expected_key(identity) + "\n", encoding="utf-8"
        )
        return cache_directory

    def fake_python_with_cold_install(self) -> Path:
        fake_python = self.root / "fake-python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" || "${1:-}" == "-" ]]; then
    exec __REAL_PYTHON__ "$@"
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
    venv_directory=${!#}
    mkdir -p "$venv_directory/bin"
    cat > "$venv_directory/bin/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
bin_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
venv_directory=$(dirname -- "$bin_directory")
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
    printf '%s\n' "$@" > "$FAKE_PIP_ARGS_FILE"
    source_directory="$venv_directory/lib/python__PYTHON_VERSION__/site-packages"
    mkdir -p "$source_directory/ais_bench/benchmark/configs/datasets/fixture"
    mkdir -p "$source_directory/ais_bench/benchmark/configs/models/vllm_api"
    touch "$source_directory/ais_bench/benchmark/configs/datasets/fixture/dataset.py"
    touch "$source_directory/ais_bench/benchmark/configs/models/vllm_api/model.py"
    cat > "$bin_directory/ais_bench" <<'ENTRYPOINT'
#!/usr/bin/env python3
raise SystemExit(0)
ENTRYPOINT
    chmod +x "$bin_directory/ais_bench"
    exit 0
fi
if [[ "${1:-}" == "$bin_directory/_ais_bench_entrypoint.py" && "${2:-}" == "-h" ]]; then
    exit 0
fi
echo "unexpected fake venv Python invocation: $*" >&2
exit 1
SH
    chmod +x "$venv_directory/bin/python"
    exit 0
fi
echo "unexpected fake Python invocation: $*" >&2
exit 1
"""
            .replace("__REAL_PYTHON__", shlex.quote(sys.executable))
            .replace("__PYTHON_VERSION__", self.python_version),
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        return fake_python

    def output_environment(self) -> dict[str, str]:
        return dict(
            line.split("=", 1)
            for line in self.environment_file.read_text(encoding="utf-8").splitlines()
        )

    def test_installer_is_executable_and_uses_the_constraints_file(self) -> None:
        self.assertTrue(os.access(INSTALLER, os.X_OK))
        self.assertIn("--constraint", INSTALLER.read_text(encoding="utf-8"))
        self.assertEqual(
            CONSTRAINTS.read_text(encoding="utf-8").splitlines()[-1],
            "opencv-python-headless==4.11.0.86",
        )

    def test_cache_key_includes_every_runtime_input(self) -> None:
        expected_key = self.expected_key()
        self.populate_valid_cache()

        result = self.run_installer()

        self.assertEqual(result.returncode, 0, result.stdout)
        output = self.output_environment()
        self.assertEqual(
            set(output),
            {
                "MULTI_NODE_AISBENCH_BIN",
                "MULTI_NODE_AISBENCH_CACHE_KEY",
                "MULTI_NODE_AISBENCH_SOURCE",
            },
        )
        self.assertEqual(output["MULTI_NODE_AISBENCH_CACHE_KEY"], expected_key)
        self.assertEqual(
            Path(output["MULTI_NODE_AISBENCH_BIN"]),
            self.cache_root / expected_key / "venv/bin/ais_bench",
        )
        self.assertEqual(
            Path(output["MULTI_NODE_AISBENCH_SOURCE"]),
            self.source_directory(self.cache_root / expected_key),
        )

    def test_valid_cache_is_reused_without_installing_again(self) -> None:
        self.populate_valid_cache()
        first = self.run_installer()
        self.assertEqual(first.returncode, 0, first.stdout)
        marker = Path(self.output_environment()["MULTI_NODE_AISBENCH_BIN"])
        first_mtime = marker.stat().st_mtime_ns

        second = self.run_installer()

        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertIn("Reusing AISBench cache", second.stdout)
        self.assertEqual(marker.stat().st_mtime_ns, first_mtime)

    def test_environment_identity_partitions_the_shared_cache(self) -> None:
        other_identity = "controller=fixture-cpu-v2;runtime=fixture-npu"
        first_cache = self.populate_valid_cache()
        second_cache = self.populate_valid_cache(other_identity)

        first = self.run_installer()
        first_key = self.output_environment()["MULTI_NODE_AISBENCH_CACHE_KEY"]
        second = self.run_installer(identity=other_identity)
        second_key = self.output_environment()["MULTI_NODE_AISBENCH_CACHE_KEY"]

        self.assertEqual(first.returncode, 0, first.stdout)
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertNotEqual(first_key, second_key)
        self.assertEqual(first_cache.name, first_key)
        self.assertEqual(second_cache.name, second_key)

    def test_environment_identity_is_required(self) -> None:
        environment = self.environment()
        environment.pop("AIS_BENCH_ENVIRONMENT_IDENTITY")

        result = subprocess.run(
            ["bash", str(INSTALLER)],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AIS_BENCH_ENVIRONMENT_IDENTITY is required", result.stdout)

    def test_cold_cache_is_installed_verified_and_published(self) -> None:
        result = self.run_installer(python=self.fake_python_with_cold_install())

        self.assertEqual(result.returncode, 0, result.stdout)
        output = self.output_environment()
        cache_directory = self.cache_root / output["MULTI_NODE_AISBENCH_CACHE_KEY"]
        command = cache_directory / "venv/bin/ais_bench"
        self.assertEqual(output["MULTI_NODE_AISBENCH_CACHE_KEY"], self.expected_key())
        self.assertEqual(
            Path(output["MULTI_NODE_AISBENCH_SOURCE"]),
            self.source_directory(cache_directory),
        )
        self.assertTrue(command.is_file())
        self.assertTrue(
            (cache_directory / "venv/bin/_ais_bench_entrypoint.py").is_file()
        )
        self.assertEqual(
            (cache_directory / "READY").read_text().strip(), cache_directory.name
        )
        self.assertEqual(list(self.cache_root.glob(".*.tmp.*")), [])
        pip_arguments = self.pip_arguments.read_text(encoding="utf-8").splitlines()
        self.assertIn(
            f"ais_bench_benchmark=={self.package_version}", pip_arguments
        )
        self.assertEqual(
            pip_arguments[pip_arguments.index("--retries") + 1], "3"
        )

    def test_invalid_package_version_is_rejected(self) -> None:
        result = self.run_installer(
            overrides={"AIS_BENCH_PACKAGE_VERSION": "3.1; echo unsafe"}
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contains unsupported characters", result.stdout)

    def test_installation_is_verified_before_atomic_publication(self) -> None:
        text = INSTALLER.read_text(encoding="utf-8")

        self.assertIn("mktemp -d", text)
        self.assertIn("flock 9", text)
        self.assertIn("_ais_bench_entrypoint.py", text)
        self.assertNotIn('venv/bin/ais_bench.py"', text)
        self.assertIn('"$staging_directory/venv/bin/ais_bench" -h', text)
        self.assertIn('mv "$staging_directory" "$cache_directory"', text)
        self.assertLess(
            text.index('"$staging_directory/venv/bin/ais_bench" -h'),
            text.index('mv "$staging_directory" "$cache_directory"'),
        )

    def test_package_install_is_pinned_and_retried(self) -> None:
        text = INSTALLER.read_text(encoding="utf-8")

        self.assertIn(
            "AIS_BENCH_PACKAGE_VERSION=${AIS_BENCH_PACKAGE_VERSION:-3.1.20260630}",
            text,
        )
        self.assertIn(
            '"${AIS_BENCH_PACKAGE_NAME}==${AIS_BENCH_PACKAGE_VERSION}"', text
        )
        self.assertIn("AIS_BENCH_PIP_RETRIES=${AIS_BENCH_PIP_RETRIES:-3}", text)


if __name__ == "__main__":
    unittest.main()
