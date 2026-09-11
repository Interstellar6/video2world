"""Conservative same-host process identity and owned-process cleanup."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def process_identity(pid: int) -> dict:
    identity = {"pid": pid, "hostname": socket.gethostname()}
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        identity["proc_start_ticks"] = stat.read_text().rsplit(") ", 1)[1].split()[19]
    else:
        result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            identity["process_started_at"] = result.stdout.strip()
    return identity


def process_state(identity: dict) -> str:
    if not isinstance(identity, dict) or identity.get("hostname") != socket.gethostname():
        return "unknown"
    pid = identity.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except (PermissionError, OSError):
        return "unknown"
    try:
        current = process_identity(pid)
    except (OSError, IndexError):
        return "unknown"
    for field in ("proc_start_ticks", "process_started_at"):
        if identity.get(field) is not None and current.get(field) is not None:
            return "alive" if str(identity[field]) == str(current[field]) else "exited"
    return "unknown"


def provider_process_state(identity: dict, *, leader_state: str | None = None) -> str:
    state = process_state(identity) if leader_state is None else leader_state
    if state != "exited":
        return state
    try:
        os.kill(identity["pid"], 0)
        # PID reuse implies the original process group no longer reserves its leader ID.
        return "exited"
    except ProcessLookupError:
        group = identity.get("process_group_id", identity["pid"])
        if not isinstance(group, int) or isinstance(group, bool) or group <= 0:
            return "unknown"
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return "exited"
        except (PermissionError, OSError):
            return "unknown"
        return "alive"
    except (PermissionError, OSError):
        return "unknown"


def terminate_started_process(process: subprocess.Popen, identity: dict, *, term_timeout: float = 2,
                              kill_timeout: float = 2) -> dict:
    """Only call with the Popen object just created with start_new_session=True."""
    group = process.pid
    audit = {"pid": process.pid, "process_group_id": group, "signals": [], "errors": [],
             "confirmed_exited": False, "term_timeout_seconds": term_timeout, "kill_timeout_seconds": kill_timeout}

    def state():
        if process.returncode is not None:
            try:
                os.kill(process.pid, 0)
                return "exited"
            except ProcessLookupError:
                pass
            except OSError:
                return "unknown"
        return provider_process_state(identity)

    def send(sig):
        try:
            os.killpg(group, sig)
            audit["signals"].append(signal.Signals(sig).name)
        except ProcessLookupError:
            pass
        except OSError as error:
            audit["errors"].append(f"{signal.Signals(sig).name}: {error}")

    def wait_group(timeout):
        deadline = time.monotonic() + timeout
        while True:
            process.poll()
            if process.returncode is not None and state() == "exited":
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(.05, max(0, deadline - time.monotonic())))

    # Before wait/poll reaps the child, its PID cannot be reused, even if identity capture failed.
    if process.returncode is None or state() == "alive":
        send(signal.SIGTERM)
    if wait_group(term_timeout):
        audit["confirmed_exited"] = True
    else:
        if process.returncode is None or state() == "alive":
            send(signal.SIGKILL)
        audit["confirmed_exited"] = wait_group(kill_timeout)
    audit["returncode"] = process.returncode
    audit["final_process_state"] = state()
    return audit
