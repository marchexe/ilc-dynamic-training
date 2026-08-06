#!/usr/bin/env python3
"""Process-tree cleanup for run_proxy_audit.py.

Phase 1 (the initial audit implementation) verified live that killing only
the orchestrator process (`kill $PID`) left orphaned local `ssh` clients and
remote Weaver processes running on iutgpu01: `run_tiered_evaluation`
(training.pbt.execution.backend, shared PBT code, not modified here)
dispatches each checkpoint's evaluation via `subprocess.Popen(...,
start_new_session=True)`, which detaches every worker into its own session
and process group -- exactly so a crash in one worker doesn't take down the
others, but it also means the OS does not propagate a signal sent only to
the parent down to these children automatically.

This module tracks and terminates only the *direct OS-level children of the
current process* -- never a pattern-matched "kill anything that looks like
weaver/ssh" sweep, which could hit another user's or another run's job.
Since `run_tiered_evaluation` spawns its workers directly in this process
(not via an intermediate wrapper process), every worker it ever starts is,
for the lifetime of this process, a direct child of `os.getpid()` -- so
querying "direct children of my own PID" at the moment a shutdown is
requested is both complete (nothing it could have spawned is missed) and
exactly scoped (nothing owned by anyone else can appear in that list).
"""

import os
import signal
import subprocess
import time


class AuditShutdownRequested(Exception):
    """Raised from a signal handler (SIGTERM, SIGINT, or the SIGALRM-based
    timeout) so orchestration code can catch it at a safe point in
    run_proxy_audit.py's main loop -- the same mechanism Python's own
    KeyboardInterrupt already uses for SIGINT, extended to SIGTERM and to
    an explicit timeout."""

    def __init__(self, reason, signal_name=None):
        super().__init__(f"audit shutdown requested: reason={reason} signal={signal_name}")
        self.reason = reason
        self.signal_name = signal_name


def direct_child_pids(parent_pid):
    """PIDs of direct children of parent_pid, via `pgrep -P` -- the
    kernel's own parent-child bookkeeping, not string-matching on command
    lines. Returns [] both when there genuinely are no children and when
    pgrep itself is unavailable/errors (best-effort; never raises)."""
    try:
        result = subprocess.run(["pgrep", "-P", str(parent_pid)], capture_output=True, text=True, check=False, timeout=10)
    except Exception:
        return []
    if result.returncode not in (0, 1):  # 1 = "no processes matched", not an error
        return []
    return [int(pid) for pid in result.stdout.split() if pid.strip()]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal -- shouldn't happen for our own children


def _still_running(pid):
    """Non-blocking liveness check for a PID that is (or was) a genuine OS
    child of the calling process. Uses os.waitpid(pid, WNOHANG) rather
    than os.kill(pid, 0): after SIGTERM/SIGKILL a child becomes a zombie
    until its parent reaps it, and a zombie still answers a signal-0 probe
    successfully -- os.kill(pid, 0) alone cannot tell "still running" from
    "dead but not yet reaped", which would make every termination here
    look like it silently failed. waitpid both answers correctly AND reaps
    the zombie in the same call, so this doubles as the reap step.
    Returns False (not running) if the PID was never our child or was
    already reaped by something else (ChildProcessError / errno ECHILD)."""
    try:
        reaped_pid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return reaped_pid == 0


def terminate_child_processes(parent_pid, grace_period_seconds=10.0, poll_interval_seconds=0.2):
    """SIGTERM every direct child's process group (each is its own group
    leader, a consequence of start_new_session=True), wait up to
    grace_period_seconds for them to exit on their own, then SIGKILL only
    whichever ones are still alive. Returns (terminated_pids, failures);
    failures is a list of {"pid", "stage", "error"} dicts, empty on a
    clean run. Idempotent and safe to call with no children present.

    Only meaningful when parent_pid == the CALLING process's own pid (or
    another process this one can reap via waitpid): `_still_running` reaps
    as it checks, which only works for genuine children of the caller.
    """
    pids = direct_child_pids(parent_pid)
    failures = []

    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception as error:
            failures.append({"pid": pid, "stage": "sigterm", "error": str(error)})

    deadline = time.monotonic() + grace_period_seconds
    remaining = {pid for pid in pids if _still_running(pid)}
    while remaining and time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        remaining = {pid for pid in remaining if _still_running(pid)}

    for pid in remaining:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception as error:
            failures.append({"pid": pid, "stage": "sigkill", "error": str(error)})

    if remaining:
        time.sleep(poll_interval_seconds)
    still_alive = [pid for pid in pids if _still_running(pid)]
    for pid in still_alive:
        failures.append({"pid": pid, "stage": "verify", "error": "still alive after SIGKILL"})

    terminated = [pid for pid in pids if pid not in still_alive]
    return terminated, failures


def remote_processes_matching(host, match_substring, timeout_seconds=15):
    """Best-effort: PIDs on `host` whose command line contains
    match_substring. Used only to VERIFY/log post-cleanup remote state and
    as a narrowly-scoped fallback kill target -- callers must pass a
    substring unique to this run (its experiment_dir path, which embeds
    run_id), never a generic pattern like "weaver" alone that could match
    another run's or another user's job. Returns (pids_or_None,
    error_or_None); never raises."""
    try:
        result = subprocess.run(
            ["ssh", host, "pgrep", "-f", match_substring],
            capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
    except Exception as error:
        return None, str(error)
    if result.returncode not in (0, 1):
        return None, (result.stderr.strip() or f"ssh/pgrep exited {result.returncode}")
    pids = [int(pid) for pid in result.stdout.split() if pid.strip()]
    return pids, None


def kill_remote_processes(host, pids, timeout_seconds=15):
    """Fallback only: directly kill specific, already-identified remote
    PIDs (from remote_processes_matching, i.e. already scoped to this
    run). Not used as a first resort -- closing the local ssh client
    (terminate_child_processes) is the primary mechanism and was verified
    live to cleanly take the remote Weaver process down with it."""
    if not pids:
        return True, None
    try:
        result = subprocess.run(
            ["ssh", host, "kill", "-9", *[str(pid) for pid in pids]],
            capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
    except Exception as error:
        return False, str(error)
    return result.returncode == 0, (result.stderr.strip() or None)


def install_shutdown_handlers():
    """Registers SIGTERM/SIGINT handlers that raise AuditShutdownRequested
    instead of SIGTERM's default (immediate termination, no cleanup) or
    (for SIGINT) letting a bare KeyboardInterrupt propagate without the
    signal name recorded."""

    def handler(signum, _frame):
        raise AuditShutdownRequested("signal", signal_name=signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def install_timeout(timeout_seconds):
    """Unix SIGALRM-based overall audit timeout: fires once, timeout_seconds
    after this call, raising AuditShutdownRequested(reason="timeout") via
    the same signal mechanism as install_shutdown_handlers. No-op if
    timeout_seconds is falsy. Call cancel_timeout() once the guarded work
    finishes so a stray alarm can't fire later."""
    if not timeout_seconds:
        return

    def handler(_signum, _frame):
        raise AuditShutdownRequested("timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout_seconds))


def cancel_timeout():
    signal.alarm(0)
