"""
Sandboxed Python experiment execution with timeout.
Extracted from AI-Scientist-v2 interpreter.py.
Runs experiment scripts in a separate process with stdout/stderr capture.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentResult:
    """Result of running an experiment script."""
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    exec_time: float = 0.0
    timed_out: bool = False
    script_path: str = ""


def run_experiment(
    script_path: str,
    working_dir: Optional[str] = None,
    timeout: int = 3600,
    env_vars: Optional[dict] = None,
) -> ExperimentResult:
    """Run a Python experiment script in a subprocess with timeout.

    Args:
        script_path: Path to the Python script to run.
        working_dir: Working directory for execution. Defaults to script's directory.
        timeout: Maximum execution time in seconds (default: 1 hour).
        env_vars: Additional environment variables to set.

    Returns:
        ExperimentResult with stdout, stderr, return code, and timing.
    """
    script_path = str(Path(script_path).resolve())
    if working_dir is None:
        working_dir = str(Path(script_path).parent)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    result = ExperimentResult(script_path=script_path)
    start_time = time.time()

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
        )
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.return_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        result.stdout = e.stdout or ""
        result.stderr = (e.stderr or "") + f"\n[TIMEOUT after {timeout}s]"
        result.return_code = -1
        result.timed_out = True
    except Exception as e:
        result.stderr = str(e)
        result.return_code = -1

    result.exec_time = time.time() - start_time
    return result


def run_experiment_background(
    script_path: str,
    log_path: str,
    working_dir: Optional[str] = None,
    env_vars: Optional[dict] = None,
) -> int:
    """Launch an experiment script in the background.

    Args:
        script_path: Path to the Python script.
        log_path: Path to write stdout+stderr log.
        working_dir: Working directory. Defaults to script's directory.
        env_vars: Additional environment variables.

    Returns:
        PID of the background process.
    """
    script_path = str(Path(script_path).resolve())
    if working_dir is None:
        working_dir = str(Path(script_path).parent)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=working_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    return proc.pid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiment_runner.py <script.py> [timeout_seconds]")
        sys.exit(1)

    script = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 3600

    print(f"Running {script} with timeout={timeout}s...")
    result = run_experiment(script, timeout=timeout)
    print(f"Return code: {result.return_code}")
    print(f"Execution time: {result.exec_time:.1f}s")
    if result.timed_out:
        print("TIMED OUT")
    if result.stdout:
        print(f"--- stdout ---\n{result.stdout[-2000:]}")
    if result.stderr:
        print(f"--- stderr ---\n{result.stderr[-2000:]}")
