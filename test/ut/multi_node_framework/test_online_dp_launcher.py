from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "test/recipe/multi_node/scripts/run_online_dp.py"


class OnlineDpLauncherTests(unittest.TestCase):
    def run_launcher(
        self, worker_exit_code: int, *, expose_processes: bool = True
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            launcher = Path(temporary_directory) / "launch_online_dp.py"
            processes_line = "processes = []" if expose_processes else "workers = []"
            append_line = (
                "processes.append(process)"
                if expose_processes
                else "workers.append(process)"
            )
            launcher.write_text(
                textwrap.dedent(
                    f"""
                    import multiprocessing
                    import sys

                    def worker(exit_code):
                        raise SystemExit(exit_code)

                    {processes_line}
                    if __name__ == "__main__":
                        process = multiprocessing.Process(
                            target=worker, args=(int(sys.argv[1]),)
                        )
                        {append_line}
                        process.start()
                        process.join()
                    """
                ),
                encoding="utf-8",
            )
            return subprocess.run(
                [sys.executable, str(WRAPPER), str(launcher), str(worker_exit_code)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def test_successful_worker_keeps_zero_exit_code(self) -> None:
        result = self.run_launcher(0)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_failed_worker_makes_launcher_fail(self) -> None:
        result = self.run_launcher(7)
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertIn("external-DP worker 0 exited with 7", result.stdout)

    def test_upstream_processes_contract_change_fails_closed(self) -> None:
        result = self.run_launcher(7, expose_processes=False)

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("external-DP launcher contract changed", result.stdout)

if __name__ == "__main__":
    unittest.main()
