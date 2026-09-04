"""Detached pipeline runs for the Streamlit UI.

young_pipeline.sh is launched in its own process session with its output
going to a log file on disk, not to a pipe back into Streamlit. The
reduction therefore keeps running on the machine when the browser tab, the
SSH tunnel, or the Streamlit server itself goes away, and the page can
reattach to it (live or finished) the next time it loads.

Everything lives in <repo>/.pipeline_run/:
    state.json     pid, start time, log path
    pipeline.log   combined stdout+stderr of young_pipeline.sh
    config.yaml    snapshot of the config the run was started with
    exit_code      written by the wrapper shell when young_pipeline.sh exits
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RUN_DIRNAME = ".pipeline_run"
LOG_TAIL_BYTES = 512 * 1024

# tqdm-style progress lines ("  3%|▎    | 1/32 [...]"). Consecutive ones are
# collapsed so a progress bar occupies one line, as it would in a terminal.
_TQDM_LINE = re.compile(r"\d+%\|")

# Popen handles for runs started by this Streamlit process. Polling them
# reaps the child when it exits, so a finished run never lingers as a zombie
# that os.kill(pid, 0) would still report as alive.
_PROCS: dict[int, subprocess.Popen] = {}


@dataclass
class RunInfo:
    pid: int
    started_at: datetime
    log_path: Path
    exit_code: int | None
    finished_at: datetime | None

    @property
    def running(self) -> bool:
        return self.exit_code is None and _pid_alive(self.pid)

    @property
    def crashed(self) -> bool:
        """Process is gone but never recorded an exit code (killed, OOM, ...)."""
        return self.exit_code is None and not _pid_alive(self.pid)


def run_dir(repo_root: Path) -> Path:
    return Path(repo_root) / RUN_DIRNAME


def _pid_alive(pid: int) -> bool:
    proc = _PROCS.get(pid)
    if proc is not None:
        return proc.poll() is None
    # No handle (Streamlit restarted, or the module was reloaded). If the
    # process is still our child, reap it here so a finished run does not sit
    # around as a zombie that os.kill(pid, 0) would report as alive.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_run(repo_root: Path) -> RunInfo | None:
    d = run_dir(repo_root)
    state_file = d / "state.json"
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text())
        pid = int(state["pid"])
        started_at = datetime.fromisoformat(state["started_at"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None

    exit_code = None
    finished_at = None
    exit_file = d / "exit_code"
    if exit_file.exists():
        try:
            exit_code = int(exit_file.read_text().strip())
        except ValueError:
            exit_code = -1
        finished_at = datetime.fromtimestamp(exit_file.stat().st_mtime)

    return RunInfo(
        pid=pid,
        started_at=started_at,
        log_path=d / "pipeline.log",
        exit_code=exit_code,
        finished_at=finished_at,
    )


def start_run(repo_root: Path, script_path: Path, config_path: Path) -> RunInfo:
    """Launch young_pipeline.sh detached and record it. Returns its RunInfo."""
    repo_root = Path(repo_root)
    d = run_dir(repo_root)
    d.mkdir(exist_ok=True)
    for name in ("state.json", "exit_code", "pipeline.log"):
        try:
            (d / name).unlink()
        except FileNotFoundError:
            pass
    shutil.copy(config_path, d / "config.yaml")

    exit_file = d / "exit_code"
    log_path = d / "pipeline.log"
    # The wrapper records the script's exit status once it finishes. Writing
    # to a temp name and renaming keeps a half-written file from ever being
    # read as a completed run.
    wrapper = (
        f"bash {shlex.quote(str(script_path))}; rc=$?; "
        f"echo $rc > {shlex.quote(str(exit_file) + '.tmp')} && "
        f"mv {shlex.quote(str(exit_file) + '.tmp')} {shlex.quote(str(exit_file))}"
    )

    env = dict(os.environ)
    # Stage scripts print to a file now rather than a pipe/terminal, so make
    # Python flush every line or the log would only update in large chunks.
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            ["bash", "-c", wrapper],
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # own session: survives SIGHUP and Streamlit exiting
        )
    _PROCS[proc.pid] = proc

    started_at = datetime.now()
    (d / "state.json").write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "started_at": started_at.isoformat(timespec="seconds"),
                "log": str(log_path),
            },
            indent=2,
        )
    )
    return RunInfo(
        pid=proc.pid,
        started_at=started_at,
        log_path=log_path,
        exit_code=None,
        finished_at=None,
    )


def stop_run(repo_root: Path, grace_seconds: float = 10.0) -> None:
    """Terminate the run's whole process group and record it as stopped."""
    run = load_run(repo_root)
    if run is None or not run.running:
        return

    # start_new_session made the wrapper the leader of its own process group,
    # so signalling the group reaches young_pipeline.sh, the stage scripts
    # and all their multiprocessing workers.
    for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(run.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and _pid_alive(run.pid):
            time.sleep(0.2)
        if not _pid_alive(run.pid):
            break

    exit_file = run_dir(repo_root) / "exit_code"
    if not exit_file.exists():
        exit_file.write_text("130\n")  # conventional "interrupted" status
    with open(run.log_path, "ab") as log:
        log.write(b"\n[Pipeline stopped from the interface]\n")


def read_log_tail(log_path: Path, max_lines: int) -> list[str]:
    """Return the last max_lines of the log with tqdm progress lines collapsed."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - LOG_TAIL_BYTES))
            chunk = f.read()
    except FileNotFoundError:
        return []

    text = chunk.decode("utf-8", errors="replace")
    if size > LOG_TAIL_BYTES:
        # Drop the partial first line of the window.
        text = text.split("\n", 1)[-1]

    lines: list[str] = []
    # tqdm redraws with carriage returns, so split on those as well as
    # newlines; the collapse below then folds the redraws onto one line.
    for raw in re.split(r"\r\n|\r|\n", text):
        if lines and _TQDM_LINE.search(raw) and (
            _TQDM_LINE.search(lines[-1]) or lines[-1] == ""
        ):
            # Either a redraw of the previous progress line, or the empty
            # segment left by the "\n\r" that precedes the first redraw.
            lines[-1] = raw
        else:
            lines.append(raw)
    if lines and lines[-1] == "":
        lines.pop()
    return lines[-max_lines:]
