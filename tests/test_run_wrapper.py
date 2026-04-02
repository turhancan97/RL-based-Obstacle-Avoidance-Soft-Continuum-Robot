from __future__ import annotations

import subprocess
import sys


def test_run_py_help_smoke():
    result = subprocess.run([sys.executable, "run.py", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "Legacy continuum RL command wrapper" in result.stdout
