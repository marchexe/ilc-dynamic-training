import os
import signal
import subprocess
import sys
import time
import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from research.process_cleanup import (
    AuditShutdownRequested,
    cancel_timeout,
    direct_child_pids,
    install_shutdown_handlers,
    install_timeout,
    kill_remote_processes,
    pid_alive,
    remote_processes_matching,
    terminate_child_processes,
)


def spawn_sleeper(seconds=30, ignore_sigterm=False):
    """A real, detached (start_new_session=True) child process -- the same
    shape run_tiered_evaluation's ssh/weaver workers take -- used as a
    generic stand-in so these tests don't need a real GPU/SSH host."""
    if ignore_sigterm:
        cmd = [sys.executable, "-c", f"import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep({seconds})"]
    else:
        cmd = ["sleep", str(seconds)]
    return subprocess.Popen(cmd, start_new_session=True)


class DirectChildPidsTest(unittest.TestCase):
    def test_finds_a_real_spawned_child(self):
        proc = spawn_sleeper(10)
        try:
            time.sleep(0.3)
            self.assertIn(proc.pid, direct_child_pids(os.getpid()))
        finally:
            proc.kill()
            proc.wait()

    def test_empty_for_a_process_with_no_children_of_its_own(self):
        proc = spawn_sleeper(5)
        try:
            time.sleep(0.3)
            self.assertEqual(direct_child_pids(proc.pid), [])
        finally:
            proc.kill()
            proc.wait()


class TerminateChildProcessesTest(unittest.TestCase):
    def test_sigterm_cleanup_of_a_responsive_child(self):
        proc = spawn_sleeper(30)
        time.sleep(0.3)
        self.assertTrue(pid_alive(proc.pid))

        terminated, failures = terminate_child_processes(os.getpid(), grace_period_seconds=3.0)

        self.assertIn(proc.pid, terminated)
        self.assertEqual(failures, [])
        self.assertFalse(pid_alive(proc.pid))
        proc.wait(timeout=2)

    def test_sigkill_fallback_for_a_child_that_ignores_sigterm(self):
        proc = spawn_sleeper(30, ignore_sigterm=True)
        time.sleep(0.3)
        self.assertTrue(pid_alive(proc.pid))

        terminated, failures = terminate_child_processes(os.getpid(), grace_period_seconds=1.0, poll_interval_seconds=0.1)

        self.assertIn(proc.pid, terminated)
        self.assertEqual(failures, [])
        self.assertFalse(pid_alive(proc.pid))
        proc.wait(timeout=2)

    def test_no_orphaned_children_remain_after_cleanup(self):
        procs = [spawn_sleeper(30) for _ in range(3)]
        time.sleep(0.3)
        terminate_child_processes(os.getpid(), grace_period_seconds=3.0)
        remaining = direct_child_pids(os.getpid())
        for proc in procs:
            self.assertNotIn(proc.pid, remaining)
            proc.wait(timeout=2)

    def test_unrelated_process_is_not_touched(self):
        """Cleanup scoped to a parent_pid must never touch a process that
        exists but is not that parent's child -- confirms this is real
        OS-level parent-child matching, not a broad pattern-based sweep
        that could hit another run's or another user's process."""
        unrelated = spawn_sleeper(10)
        try:
            time.sleep(0.3)
            fake_parent_pid = 999999
            terminated, _ = terminate_child_processes(fake_parent_pid, grace_period_seconds=0.5)
            self.assertEqual(terminated, [])
            self.assertTrue(pid_alive(unrelated.pid))
        finally:
            unrelated.kill()
            unrelated.wait()

    def test_no_children_is_a_clean_noop(self):
        terminated, failures = terminate_child_processes(os.getpid(), grace_period_seconds=0.2)
        self.assertEqual(terminated, [])
        self.assertEqual(failures, [])


class SignalHandlerTest(unittest.TestCase):
    def setUp(self):
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)
        self._orig_sigint = signal.getsignal(signal.SIGINT)

    def tearDown(self):
        signal.signal(signal.SIGTERM, self._orig_sigterm)
        signal.signal(signal.SIGINT, self._orig_sigint)

    def test_sigterm_raises_audit_shutdown_requested(self):
        install_shutdown_handlers()
        with self.assertRaises(AuditShutdownRequested) as context:
            os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(context.exception.reason, "signal")
        self.assertEqual(context.exception.signal_name, "SIGTERM")

    def test_sigint_raises_audit_shutdown_requested(self):
        install_shutdown_handlers()
        with self.assertRaises(AuditShutdownRequested) as context:
            os.kill(os.getpid(), signal.SIGINT)
        self.assertEqual(context.exception.reason, "signal")
        self.assertEqual(context.exception.signal_name, "SIGINT")


class TimeoutHandlingTest(unittest.TestCase):
    def tearDown(self):
        cancel_timeout()

    def test_timeout_raises_audit_shutdown_requested(self):
        install_timeout(1)
        with self.assertRaises(AuditShutdownRequested) as context:
            time.sleep(3)
        self.assertEqual(context.exception.reason, "timeout")

    def test_cancel_timeout_prevents_a_late_alarm(self):
        install_timeout(1)
        cancel_timeout()
        time.sleep(1.5)  # must NOT raise: alarm was cancelled before it could fire

    def test_falsy_timeout_is_a_noop(self):
        install_timeout(None)
        install_timeout(0)
        time.sleep(0.1)  # must NOT raise: no alarm was ever armed


class RemoteVerificationTest(unittest.TestCase):
    def test_unreachable_host_returns_error_not_exception(self):
        pids, error = remote_processes_matching("nonexistent-test-host-xyz.invalid", "some-run-id", timeout_seconds=5)
        self.assertIsNone(pids)
        self.assertIsNotNone(error)

    def test_kill_remote_processes_with_no_pids_is_a_noop_success(self):
        ok, error = kill_remote_processes("nonexistent-test-host-xyz.invalid", [])
        self.assertTrue(ok)
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
