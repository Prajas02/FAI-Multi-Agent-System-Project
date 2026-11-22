"""Unit tests for src/analytics/execution.py (design Sections 7.3, 10 and 14).

The module-level ``adapter_*`` functions are fake adapters. Worker processes
import them by reference (``"<this module>:adapter_name"``), so they must stay
at module level. Temporary modules (outside the repository) are used for
import-time behaviour; they are removed after each test.

This file deliberately imports only the standard library so that the spawned
worker processes (which import this module) start quickly.
"""

import importlib
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest import mock

from src.analytics import execution
from src.analytics.execution import (
    AdapterResolutionError,
    ExecutionError,
    InProcessBackend,
    StageOutcome,
    Stopwatch,
    StopwatchError,
    SubprocessBackend,
    resolve_adapter,
)

ADAPTER_MODULE = __name__
TREE_KILL_SUPPORTED = os.name != "nt" or shutil.which("taskkill") is not None
TREE_KILL_SKIP_REASON = "process-tree termination needs POSIX process groups or the Windows 'taskkill' tool"


def adapter_ref(name):
    return "%s:%s" % (ADAPTER_MODULE, name)


# ---------------------------------------------------------------------------
# Fake adapters (module level, importable by worker processes)
# ---------------------------------------------------------------------------

def adapter_echo(scenario, seed, time_limit):
    return {
        "scenario": scenario,
        "seed": seed,
        "time_limit": time_limit,
        "worker_pid": os.getpid(),
        "nested": {"values": [1, 2.5, (3, 4)], "text": "café"},
        "runtime_seconds": 987654.0,        # a "solver-reported" runtime that must be ignored
    }


def adapter_raises(scenario, seed, time_limit):
    raise ValueError("boom\nsecond line")


def adapter_sleep(scenario, seed, time_limit):
    time.sleep(scenario["sleep"])
    return "slept"


def adapter_crash(scenario, seed, time_limit):
    os._exit(3)


def adapter_large(scenario, seed, time_limit):
    return b"x" * (8 * 1024 * 1024)        # larger than any pipe buffer


def adapter_unpicklable_result(scenario, seed, time_limit):
    return lambda: None


def _write(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def adapter_process_tree(scenario, seed, time_limit):
    """Start a long-lived child process, record both pids, then act per ``scenario['mode']``."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _write(scenario["worker_pid_file"], str(os.getpid()))
    _write(scenario["child_pid_file"], str(child.pid))
    mode = scenario["mode"]
    if mode == "hang":
        time.sleep(300)
    if mode == "raise":
        raise RuntimeError("after starting a child")
    if mode == "crash":
        os._exit(9)
    return "returned"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class TempModules:
    """Create importable temporary modules outside the repository."""

    def __init__(self, **sources):
        self.sources = {"%s_%s" % (name, uuid.uuid4().hex[:8]): text for name, text in sources.items()}
        self.names = dict(zip(sources, self.sources))
        self.directory = None

    def __enter__(self):
        self.directory = tempfile.mkdtemp(prefix="exec_test_modules_")
        for name, text in self.sources.items():
            with open(os.path.join(self.directory, name + ".py"), "w") as handle:
                handle.write(text)
        sys.path.insert(0, self.directory)
        importlib.invalidate_caches()
        return self

    def __exit__(self, *exc_info):
        try:
            sys.path.remove(self.directory)
        except ValueError:
            pass
        for name in self.sources:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        shutil.rmtree(self.directory, ignore_errors=True)
        return False

    def ref(self, key, attribute):
        return "%s:%s" % (self.names[key], attribute)


def process_exists(pid):
    if os.name == "nt":
        output = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60,
        ).stdout
        return ('"%d"' % pid) in output
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_quietly(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
        else:
            os.kill(pid, 9)
    except Exception:
        pass


def read_pid(path):
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with open(path) as handle:
                text = handle.read().strip()
            if text:
                return int(text)
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError("pid file was not written: %s" % path)


class FakeClock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.ticks.pop(0)


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------

class TestStopwatch(unittest.TestCase):
    def test_start_stop_and_elapsed_with_injected_clock(self):
        clock = FakeClock(10.0, 12.5)
        watch = Stopwatch(clock=clock)
        self.assertFalse(watch.running)
        watch.start()
        self.assertTrue(watch.running)
        self.assertEqual(watch.stop(), 2.5)
        self.assertFalse(watch.running)
        self.assertEqual(watch.elapsed, 2.5)
        self.assertEqual(clock.calls, 2)

    def test_elapsed_while_running_is_live_and_frozen_after_stop(self):
        clock = FakeClock(0.0, 1.0, 3.0, 9.0)
        watch = Stopwatch(clock=clock)
        watch.start()
        self.assertEqual(watch.elapsed, 1.0)
        self.assertEqual(watch.elapsed, 3.0)
        self.assertEqual(watch.stop(), 9.0)
        self.assertEqual(watch.elapsed, 9.0)     # frozen: the clock is not read again
        self.assertEqual(clock.calls, 4)

    def test_default_clock_is_perf_counter(self):
        with mock.patch("time.perf_counter", side_effect=[100.0, 103.5]) as perf_counter:
            watch = Stopwatch()
            watch.start()
            self.assertEqual(watch.stop(), 3.5)
        self.assertEqual(perf_counter.call_count, 2)

    def test_wall_clock_is_never_used(self):
        with mock.patch("time.time", side_effect=AssertionError("wall clock used")), \
                mock.patch("time.monotonic", side_effect=AssertionError("monotonic used")):
            watch = Stopwatch()
            watch.start()
            self.assertGreaterEqual(watch.stop(), 0.0)

    def test_real_clock_measures_a_sleep(self):
        watch = Stopwatch()
        watch.start()
        time.sleep(0.05)
        elapsed = watch.stop()
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 2.0)

    def test_stop_before_start_is_an_error(self):
        with self.assertRaises(StopwatchError):
            Stopwatch().stop()

    def test_elapsed_before_start_is_an_error(self):
        with self.assertRaises(StopwatchError):
            Stopwatch().elapsed

    def test_start_twice_is_an_error(self):
        watch = Stopwatch(clock=FakeClock(0.0))
        watch.start()
        with self.assertRaises(StopwatchError):
            watch.start()

    def test_stop_twice_is_an_error(self):
        watch = Stopwatch(clock=FakeClock(0.0, 1.0))
        watch.start()
        watch.stop()
        with self.assertRaises(StopwatchError):
            watch.stop()

    def test_restart_after_stop_measures_from_zero(self):
        watch = Stopwatch(clock=FakeClock(0.0, 2.0, 10.0, 11.0))
        watch.start()
        watch.stop()
        watch.start()
        self.assertEqual(watch.stop(), 1.0)

    def test_reset(self):
        watch = Stopwatch(clock=FakeClock(0.0, 2.0))
        watch.start()
        watch.stop()
        watch.reset()
        with self.assertRaises(StopwatchError):
            watch.elapsed

    def test_stopwatch_error_is_a_runtime_error(self):
        self.assertTrue(issubclass(StopwatchError, RuntimeError))


# ---------------------------------------------------------------------------
# StageOutcome
# ---------------------------------------------------------------------------

class TestStageOutcome(unittest.TestCase):
    def test_fields(self):
        self.assertEqual(
            StageOutcome._fields,
            ("started", "completed", "elapsed_seconds", "error", "timed_out", "payload"),
        )


# ---------------------------------------------------------------------------
# In-process backend
# ---------------------------------------------------------------------------

class TestInProcessBackend(unittest.TestCase):
    def setUp(self):
        self.backend = InProcessBackend()

    def test_successful_adapter(self):
        adapter = mock.Mock(return_value={"assignments": {1: [1]}})
        outcome = self.backend.run(adapter, "scenario", 7, 12.5)
        self.assertIsInstance(outcome, StageOutcome)
        self.assertTrue(outcome.started)
        self.assertTrue(outcome.completed)
        self.assertFalse(outcome.timed_out)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.payload, {"assignments": {1: [1]}})
        self.assertIsInstance(outcome.elapsed_seconds, float)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.0)

    def test_adapter_receives_scenario_seed_and_time_limit(self):
        scenario = object()
        adapter = mock.Mock(return_value=None)
        self.backend.run(adapter, scenario, 1593276916, 300.0)
        adapter.assert_called_once()
        args, kwargs = adapter.call_args
        self.assertEqual(kwargs, {})
        self.assertIs(args[0], scenario)            # passed through, not copied
        self.assertEqual(args[1], 1593276916)
        self.assertEqual(args[2], 300.0)

    def test_integer_time_limit_is_passed_unchanged(self):
        adapter = mock.Mock(return_value=None)
        self.backend.run(adapter, None, 1, 300)
        self.assertEqual(adapter.call_args[0][2], 300)

    def test_adapter_exception_becomes_a_failed_outcome(self):
        outcome = self.backend.run(adapter_raises, None, 1, 5.0)
        self.assertTrue(outcome.started)
        self.assertFalse(outcome.completed)
        self.assertFalse(outcome.timed_out)
        self.assertIsNone(outcome.payload)
        self.assertEqual(outcome.error, "ValueError: boom second line")   # single line, type and message

    def test_elapsed_is_recorded_for_failed_calls(self):
        def slow_failure(scenario, seed, time_limit):
            time.sleep(0.05)
            raise RuntimeError("late")

        outcome = self.backend.run(slow_failure, None, 1, 5.0)
        self.assertFalse(outcome.completed)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.04)

    def test_elapsed_measures_only_the_adapter_call(self):
        with mock.patch("time.perf_counter", side_effect=[50.0, 52.25]) as perf_counter:
            outcome = self.backend.run(mock.Mock(return_value="x"), None, 1, 5.0)
        self.assertEqual(outcome.elapsed_seconds, 2.25)
        self.assertEqual(perf_counter.call_count, 2)          # one start, one stop

    def test_wall_clock_is_not_used(self):
        with mock.patch("time.time", side_effect=AssertionError("wall clock used")):
            outcome = self.backend.run(mock.Mock(return_value="x"), None, 1, 5.0)
        self.assertTrue(outcome.completed)

    def test_no_hard_timeout_is_enforced(self):
        # The adapter overruns the limit it was given; the backend must not stop it.
        outcome = self.backend.run(adapter_sleep, {"sleep": 0.25}, 1, 0.05)
        self.assertTrue(outcome.completed)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.payload, "slept")
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.2)
        self.assertIn("No hard timeout", InProcessBackend.__doc__)

    def test_solver_reported_runtime_is_ignored(self):
        class Result:
            runtime_seconds = 123456.0
            solver_name = "liar"

        outcome = self.backend.run(mock.Mock(return_value=Result()), None, 1, 5.0)
        self.assertLess(outcome.elapsed_seconds, 5.0)
        self.assertEqual(outcome.payload.runtime_seconds, 123456.0)     # payload untouched
        echoed = self.backend.run(adapter_echo, None, 1, 5.0)
        self.assertLess(echoed.elapsed_seconds, 5.0)
        self.assertEqual(echoed.payload["runtime_seconds"], 987654.0)

    def test_error_text_is_bounded(self):
        def long_failure(scenario, seed, time_limit):
            raise ValueError("x" * 5000)

        outcome = self.backend.run(long_failure, None, 1, 5.0)
        self.assertLessEqual(len(outcome.error), execution.MAX_ERROR_LENGTH)
        self.assertTrue(outcome.error.endswith("..."))
        self.assertTrue(outcome.error.startswith("ValueError: xxx"))

    def test_exception_without_message(self):
        def silent_failure(scenario, seed, time_limit):
            raise KeyError()

        self.assertEqual(self.backend.run(silent_failure, None, 1, 5.0).error, "KeyError")

    def test_keyboard_interrupt_and_system_exit_propagate(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__):
                with self.assertRaises(exception):
                    self.backend.run(mock.Mock(side_effect=exception()), None, 1, 5.0)

    def test_not_callable_adapter_is_a_harness_error(self):
        with self.assertRaises(ExecutionError):
            self.backend.run("not callable", None, 1, 5.0)

    def test_invalid_seed_and_time_limit(self):
        adapter = mock.Mock()
        for seed in ("1", 1.5, None, True):
            with self.assertRaises(TypeError):
                self.backend.run(adapter, None, seed, 5.0)
        for limit in (0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.backend.run(adapter, None, 1, limit)
        for limit in ("5", None, True):
            with self.assertRaises(TypeError):
                self.backend.run(adapter, None, 1, limit)
        adapter.assert_not_called()


# ---------------------------------------------------------------------------
# Adapter resolution
# ---------------------------------------------------------------------------

class TestResolveAdapter(unittest.TestCase):
    def test_valid_module_and_attribute(self):
        adapter = resolve_adapter("ALG", {"ALG": "math:sqrt"})
        self.assertTrue(callable(adapter))
        self.assertEqual(adapter(9.0), 3.0)

    def test_resolves_an_adapter_from_this_module(self):
        adapter = resolve_adapter("ECHO", {"ECHO": adapter_ref("adapter_echo")})
        self.assertIs(adapter, adapter_echo)

    def test_resolves_from_a_temporary_module(self):
        with TempModules(good="def adapter(scenario, seed, time_limit):\n    return 'ok'\n") as modules:
            adapter = resolve_adapter("X", {"X": modules.ref("good", "adapter")})
            self.assertEqual(adapter(None, 1, 1.0), "ok")

    def test_missing_algorithm_id(self):
        with self.assertRaisesRegex(AdapterResolutionError, "no adapter is registered.*'MILP'"):
            resolve_adapter("MILP", {"GA": "math:sqrt"})
        with self.assertRaises(AdapterResolutionError):
            resolve_adapter("MILP", {})
        with self.assertRaises(AdapterResolutionError):
            resolve_adapter(None, {"GA": "math:sqrt"})
        with self.assertRaises(AdapterResolutionError):
            resolve_adapter(["unhashable"], {"GA": "math:sqrt"})

    def test_algorithm_id_is_matched_exactly(self):
        with self.assertRaises(AdapterResolutionError):
            resolve_adapter("milp", {"MILP": "math:sqrt"})

    def test_registry_must_be_a_mapping(self):
        for registry in (None, ["math:sqrt"], "math:sqrt"):
            with self.assertRaises(AdapterResolutionError):
                resolve_adapter("X", registry)

    def test_malformed_references(self):
        for reference in ("math", ":sqrt", "math:", "math:sqrt:extra", "ma th:sqrt", "math:s qrt",
                          "1math:sqrt", "math..sqrt:f", "math:1sqrt", "", ":", "math.:sqrt",
                          "math:sqrt.real", "../etc:passwd"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AdapterResolutionError, "malformed"):
                    resolve_adapter("X", {"X": reference})
        for reference in (None, 5, ["math:sqrt"], b"math:sqrt"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AdapterResolutionError, "must be a string"):
                    resolve_adapter("X", {"X": reference})

    def test_missing_module_keeps_the_original_exception_as_cause(self):
        with self.assertRaises(AdapterResolutionError) as caught:
            resolve_adapter("X", {"X": "no_such_module_for_tests_xyz:fn"})
        self.assertIn("no_such_module_for_tests_xyz", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_missing_attribute_keeps_the_original_exception_as_cause(self):
        with self.assertRaises(AdapterResolutionError) as caught:
            resolve_adapter("X", {"X": "math:no_such_attribute"})
        self.assertIn("no_such_attribute", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, AttributeError)

    def test_non_callable_attribute(self):
        with self.assertRaisesRegex(AdapterResolutionError, "non-callable"):
            resolve_adapter("X", {"X": "math:pi"})

    def test_module_that_fails_while_importing(self):
        with TempModules(bad="raise RuntimeError('import-time failure')\n") as modules:
            with self.assertRaises(AdapterResolutionError) as caught:
                resolve_adapter("X", {"X": modules.ref("bad", "adapter")})
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertIn("import-time failure", str(caught.exception))

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(AdapterResolutionError, ExecutionError))
        self.assertTrue(issubclass(ExecutionError, RuntimeError))

    def test_there_is_no_built_in_registry(self):
        for name in ("REGISTRY", "ALGORITHM_ADAPTERS", "ADAPTERS"):
            self.assertFalse(hasattr(execution, name))
        self.assertNotIn("milp", " ".join(vars(execution)).lower().replace("milp ", ""))


# ---------------------------------------------------------------------------
# Subprocess backend
# ---------------------------------------------------------------------------

class SubprocessTestCase(unittest.TestCase):
    def setUp(self):
        self.backend = SubprocessBackend(startup_timeout=60.0)
        self.dir = tempfile.mkdtemp(prefix="exec_test_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(self.assert_no_active_children)

    def assert_no_active_children(self):
        deadline = time.time() + 10
        while multiprocessing.active_children() and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(multiprocessing.active_children(), [])

    def pid_files(self, mode):
        return {
            "mode": mode,
            "worker_pid_file": os.path.join(self.dir, "worker.pid"),
            "child_pid_file": os.path.join(self.dir, "child.pid"),
        }

    def assert_tree_is_gone(self, scenario):
        pids = [read_pid(scenario["worker_pid_file"]), read_pid(scenario["child_pid_file"])]
        for pid in pids:
            self.addCleanup(kill_quietly, pid)
        deadline = time.time() + 20
        while time.time() < deadline and any(process_exists(pid) for pid in pids):
            time.sleep(0.1)
        for pid in pids:
            self.assertFalse(process_exists(pid), "process %d is still running" % pid)


class TestSubprocessSuccessAndFailures(SubprocessTestCase):
    def test_successful_adapter_and_payload_transfer(self):
        scenario = {"name": "S1", "robots": [1, 2, 3], "origin": (0.5, 1.5)}
        outcome = self.backend.run(adapter_ref("adapter_echo"), scenario, 4211836502, 30.0)
        self.assertIsInstance(outcome, StageOutcome)
        self.assertTrue(outcome.started)
        self.assertTrue(outcome.completed)
        self.assertFalse(outcome.timed_out)
        self.assertIsNone(outcome.error)
        payload = outcome.payload
        self.assertEqual(payload["scenario"], scenario)
        self.assertEqual(payload["seed"], 4211836502)
        self.assertEqual(payload["time_limit"], 30.0)
        self.assertEqual(payload["nested"], {"values": [1, 2.5, (3, 4)], "text": "café"})
        self.assertNotEqual(payload["worker_pid"], os.getpid())       # really another process
        self.assertIsInstance(outcome.elapsed_seconds, float)

    def test_elapsed_is_worker_measured_and_ignores_reported_runtime(self):
        started = time.perf_counter()
        outcome = self.backend.run(adapter_ref("adapter_sleep"), {"sleep": 0.3}, 1, 30.0)
        wall = time.perf_counter() - started
        self.assertTrue(outcome.completed)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.25)
        self.assertLess(outcome.elapsed_seconds, wall - 0.02)   # excludes process start-up and transfer
        echoed = self.backend.run(adapter_ref("adapter_echo"), None, 1, 30.0)
        self.assertLess(echoed.elapsed_seconds, 30.0)           # not the reported 987654.0
        self.assertEqual(echoed.payload["runtime_seconds"], 987654.0)

    def test_large_payload_does_not_deadlock(self):
        outcome = self.backend.run(adapter_ref("adapter_large"), None, 1, 60.0)
        self.assertTrue(outcome.completed)
        self.assertEqual(len(outcome.payload), 8 * 1024 * 1024)
        self.assertEqual(outcome.payload[:3] + outcome.payload[-3:], b"xxxxxx")

    def test_adapter_exception(self):
        outcome = self.backend.run(adapter_ref("adapter_raises"), None, 1, 30.0)
        self.assertTrue(outcome.started)
        self.assertFalse(outcome.completed)
        self.assertFalse(outcome.timed_out)
        self.assertIsNone(outcome.payload)
        self.assertEqual(outcome.error, "ValueError: boom second line")
        self.assertIsInstance(outcome.elapsed_seconds, float)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.0)

    def test_adapter_exception_keeps_worker_measured_elapsed(self):
        with TempModules(slowfail=(
            "import time\n"
            "def adapter(scenario, seed, time_limit):\n"
            "    time.sleep(0.3)\n"
            "    raise RuntimeError('late')\n"
        )) as modules:
            outcome = self.backend.run(modules.ref("slowfail", "adapter"), None, 1, 30.0)
        self.assertEqual(outcome.error, "RuntimeError: late")
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.25)

    def test_worker_crash_after_the_adapter_started(self):
        outcome = self.backend.run(adapter_ref("adapter_crash"), None, 1, 30.0)
        self.assertTrue(outcome.started)
        self.assertFalse(outcome.completed)
        self.assertFalse(outcome.timed_out)
        self.assertIsNone(outcome.payload)
        self.assertTrue(outcome.error.startswith("WorkerCrash:"), outcome.error)
        self.assertIn("exit code 3", outcome.error)
        self.assertIsInstance(outcome.elapsed_seconds, float)       # parent-measured

    def test_crash_and_adapter_exception_are_distinguishable(self):
        crash = self.backend.run(adapter_ref("adapter_crash"), None, 1, 30.0)
        failure = self.backend.run(adapter_ref("adapter_raises"), None, 1, 30.0)
        self.assertTrue(crash.error.startswith("WorkerCrash:"))
        self.assertFalse(failure.error.startswith("WorkerCrash:"))
        self.assertNotIn("TimeoutError", crash.error + failure.error)

    def test_worker_sys_exit_is_reported_as_a_crash_not_an_exception(self):
        with TempModules(exiting=(
            "import sys\n"
            "def adapter(scenario, seed, time_limit):\n"
            "    sys.exit(5)\n"
        )) as modules:
            outcome = self.backend.run(modules.ref("exiting", "adapter"), None, 1, 30.0)
        self.assertTrue(outcome.error.startswith("WorkerCrash:"))
        self.assertIn("exit code 5", outcome.error)

    def test_uses_the_spawn_start_method_explicitly(self):
        real_get_context = multiprocessing.get_context
        with mock.patch("src.analytics.execution.multiprocessing.get_context", side_effect=real_get_context) as spy, \
                mock.patch("multiprocessing.set_start_method", side_effect=AssertionError("global start method changed")):
            outcome = self.backend.run(adapter_ref("adapter_echo"), None, 1, 30.0)
        self.assertTrue(outcome.completed)
        spy.assert_called_with("spawn")


class TestSubprocessTimeout(SubprocessTestCase):
    def test_adapter_timeout(self):
        started = time.perf_counter()
        outcome = self.backend.run(adapter_ref("adapter_sleep"), {"sleep": 60}, 1, 0.5)
        wall = time.perf_counter() - started
        self.assertTrue(outcome.started)
        self.assertFalse(outcome.completed)
        self.assertTrue(outcome.timed_out)
        self.assertIsNone(outcome.payload)
        self.assertIn("TimeoutError", outcome.error)
        self.assertIn("0.5", outcome.error)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.45)      # parent-measured from ready
        self.assertLess(outcome.elapsed_seconds, 5.0)
        self.assertLess(wall, 30.0)

    def test_timeout_clock_starts_after_ready_not_at_process_creation(self):
        # The module needs 1.5 s to import, which is longer than the 0.8 s limit.
        with TempModules(slowimport=(
            "import time\n"
            "time.sleep(1.5)\n"
            "def adapter(scenario, seed, time_limit):\n"
            "    time.sleep(0.2)\n"
            "    return 'done'\n"
        )) as modules:
            started = time.perf_counter()
            outcome = self.backend.run(modules.ref("slowimport", "adapter"), None, 1, 0.8)
            wall = time.perf_counter() - started
        self.assertTrue(outcome.completed, outcome)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.payload, "done")
        self.assertGreater(wall, 1.5)                               # start-up exceeded the limit ...
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.15)
        self.assertLess(outcome.elapsed_seconds, 0.8)               # ... but only the call was timed

    def test_timeout_still_applies_after_a_slow_import(self):
        with TempModules(slowhang=(
            "import time\n"
            "time.sleep(1.0)\n"
            "def adapter(scenario, seed, time_limit):\n"
            "    time.sleep(60)\n"
        )) as modules:
            outcome = self.backend.run(modules.ref("slowhang", "adapter"), None, 1, 0.5)
        self.assertTrue(outcome.timed_out)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0.45)
        self.assertLess(outcome.elapsed_seconds, 0.5 + 3.0)         # the 1 s import is not included

    def test_slow_startup_beyond_the_startup_budget_is_a_harness_error(self):
        with TempModules(neverready="import time\ntime.sleep(20)\n") as modules:
            backend = SubprocessBackend(startup_timeout=0.8)
            started = time.perf_counter()
            with self.assertRaisesRegex(ExecutionError, "did not signal ready"):
                backend.run(modules.ref("neverready", "adapter"), None, 1, 30.0)
            self.assertLess(time.perf_counter() - started, 15.0)

    def test_invalid_startup_timeout(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SubprocessBackend(startup_timeout=value)
        for value in ("1", None, True):
            with self.assertRaises(TypeError):
                SubprocessBackend(startup_timeout=value)


class TestSubprocessInfrastructureErrors(SubprocessTestCase):
    def test_unpicklable_scenario_is_a_harness_error_not_an_adapter_failure(self):
        with self.assertRaises(ExecutionError) as caught:
            self.backend.run(adapter_ref("adapter_echo"), {"lock": threading.Lock()}, 1, 30.0)
        self.assertIn("could not start", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)

    def test_unpicklable_adapter_output_is_a_harness_error(self):
        with self.assertRaisesRegex(ExecutionError, "could not be transferred"):
            self.backend.run(adapter_ref("adapter_unpicklable_result"), None, 1, 30.0)

    def test_worker_that_dies_before_ready_is_a_harness_error(self):
        with TempModules(diesonimport="import os\nos._exit(7)\n") as modules:
            with self.assertRaises(ExecutionError) as caught:
                self.backend.run(modules.ref("diesonimport", "adapter"), None, 1, 30.0)
        self.assertIn("before it was ready", str(caught.exception))
        self.assertIn("exit code 7", str(caught.exception))
        self.assertNotIsInstance(caught.exception, AdapterResolutionError)

    def test_adapter_that_cannot_be_imported_in_the_worker(self):
        with self.assertRaisesRegex(AdapterResolutionError, "no_such_module_for_tests_xyz"):
            self.backend.run("no_such_module_for_tests_xyz:adapter", None, 1, 30.0)

    def test_missing_attribute_in_the_worker(self):
        with TempModules(noattr="VALUE = 1\n") as modules:
            with self.assertRaisesRegex(AdapterResolutionError, "no attribute 'adapter'"):
                self.backend.run(modules.ref("noattr", "adapter"), None, 1, 30.0)

    def test_non_callable_attribute_in_the_worker(self):
        with TempModules(notcallable="adapter = 42\n") as modules:
            with self.assertRaisesRegex(AdapterResolutionError, "non-callable"):
                self.backend.run(modules.ref("notcallable", "adapter"), None, 1, 30.0)

    def test_malformed_reference_is_rejected_before_any_process_is_started(self):
        with mock.patch("src.analytics.execution.multiprocessing.get_context",
                        side_effect=AssertionError("a process was prepared")):
            for reference in ("nocolon", "a:b:c", "", None, 5):
                with self.subTest(reference=reference):
                    with self.assertRaises(AdapterResolutionError):
                        self.backend.run(reference, None, 1, 30.0)

    def test_invalid_seed_and_time_limit_are_rejected_before_any_process_is_started(self):
        with mock.patch("src.analytics.execution.multiprocessing.get_context",
                        side_effect=AssertionError("a process was prepared")):
            with self.assertRaises(TypeError):
                self.backend.run(adapter_ref("adapter_echo"), None, "1", 30.0)
            with self.assertRaises(ValueError):
                self.backend.run(adapter_ref("adapter_echo"), None, 1, 0)
            with self.assertRaises(TypeError):
                self.backend.run(adapter_ref("adapter_echo"), None, 1, "30")

    def test_process_start_failure_is_a_clear_harness_error(self):
        real_context = multiprocessing.get_context("spawn")
        connections = []

        def recording_pipe(duplex=True):
            pair = real_context.Pipe(duplex=duplex)
            connections.extend(pair)
            return pair

        class FailingProcess:
            """A process whose start() fails; any attempt to manage it is a bug."""
            pid = None

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise OSError("cannot fork")

            def __getattr__(self, name):
                raise AssertionError("unstarted process was used: %s" % name)

        fake_context = mock.Mock()
        fake_context.Pipe = recording_pipe
        fake_context.Process = FailingProcess
        with mock.patch("src.analytics.execution.multiprocessing.get_context", return_value=fake_context), \
                mock.patch("src.analytics.execution._terminate_process_tree") as terminate, \
                mock.patch("src.analytics.execution._reap") as reap:
            with self.assertRaises(ExecutionError) as caught:
                self.backend.run(adapter_ref("adapter_echo"), None, 1, 30.0)
        # the original start error is preserved, with its cause
        self.assertIn("could not start the worker process", str(caught.exception))
        self.assertIn("cannot fork", str(caught.exception))
        self.assertNotIn("terminated", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(str(caught.exception.__cause__), "cannot fork")
        # an unstarted process is neither terminated nor reaped
        terminate.assert_not_called()
        reap.assert_not_called()
        # both pipe ends are closed
        self.assertEqual(len(connections), 2)
        self.assertTrue(all(connection.closed for connection in connections))

    def test_start_failure_after_a_real_pickling_error_is_not_a_termination_error(self):
        with mock.patch("src.analytics.execution._terminate_process_tree") as terminate, \
                mock.patch("src.analytics.execution._reap") as reap:
            with self.assertRaises(ExecutionError) as caught:
                self.backend.run(adapter_ref("adapter_echo"), {"lock": threading.Lock()}, 1, 30.0)
        self.assertIn("could not start the worker process", str(caught.exception))
        self.assertNotIn("could not be terminated", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        terminate.assert_not_called()
        reap.assert_not_called()

    def test_pipe_creation_failure_is_a_clear_harness_error(self):
        fake_context = mock.Mock()
        fake_context.Pipe.side_effect = OSError("too many open files")
        with mock.patch("src.analytics.execution.multiprocessing.get_context", return_value=fake_context):
            with self.assertRaisesRegex(ExecutionError, "could not prepare the worker process"):
                self.backend.run(adapter_ref("adapter_echo"), None, 1, 30.0)

    def test_failed_runs_are_not_retried(self):
        marker = os.path.join(self.dir, "runs.txt")
        with TempModules(counting=(
            "import os\n"
            "def adapter(scenario, seed, time_limit):\n"
            "    with open(scenario, 'a') as handle:\n"
            "        handle.write('x')\n"
            "    raise RuntimeError('always fails')\n"
        )) as modules:
            outcome = self.backend.run(modules.ref("counting", "adapter"), marker, 1, 30.0)
        self.assertFalse(outcome.completed)
        with open(marker) as handle:
            self.assertEqual(handle.read(), "x")                    # the adapter ran exactly once


class TestCleanupHappensExactlyOnce(SubprocessTestCase):
    """Process-tree termination is centralized in run() and happens once per started worker."""

    def record(self, function):
        events = []
        real_terminate = execution._terminate_process_tree
        real_reap = execution._reap

        def terminate(process):
            events.append(("terminate", process))
            return real_terminate(process)

        def reap(process):
            events.append(("reap", process))
            return real_reap(process)

        with mock.patch("src.analytics.execution._terminate_process_tree", terminate), \
                mock.patch("src.analytics.execution._reap", reap):
            try:
                return events, function(), None
            except BaseException as exc:
                return events, None, exc

    def assert_cleaned_up_once(self, events):
        self.assertEqual([name for name, _ in events], ["terminate", "reap"], events)
        self.assertIs(events[0][1], events[1][1])        # the same worker, terminated before it is reaped

    def test_success(self):
        events, outcome, error = self.record(
            lambda: self.backend.run(adapter_ref("adapter_echo"), None, 1, 30.0))
        self.assertIsNone(error)
        self.assertTrue(outcome.completed)
        self.assert_cleaned_up_once(events)

    def test_adapter_exception(self):
        events, outcome, error = self.record(
            lambda: self.backend.run(adapter_ref("adapter_raises"), None, 1, 30.0))
        self.assertIsNone(error)
        self.assertFalse(outcome.completed)
        self.assert_cleaned_up_once(events)

    def test_timeout(self):
        events, outcome, error = self.record(
            lambda: self.backend.run(adapter_ref("adapter_sleep"), {"sleep": 60}, 1, 0.5))
        self.assertIsNone(error)
        self.assertTrue(outcome.timed_out)
        self.assert_cleaned_up_once(events)

    def test_worker_crash_after_ready(self):
        events, outcome, error = self.record(
            lambda: self.backend.run(adapter_ref("adapter_crash"), None, 1, 30.0))
        self.assertIsNone(error)
        self.assertTrue(outcome.error.startswith("WorkerCrash:"))
        self.assertIn("exit code 3", outcome.error)       # exit code read after the single cleanup
        self.assert_cleaned_up_once(events)

    def test_worker_dies_before_ready(self):
        with TempModules(dies="import os\nos._exit(7)\n") as modules:
            events, outcome, error = self.record(
                lambda: self.backend.run(modules.ref("dies", "adapter"), None, 1, 30.0))
        self.assertIsInstance(error, ExecutionError)
        self.assertIn("exit code 7", str(error))
        self.assert_cleaned_up_once(events)

    def test_adapter_cannot_be_loaded_in_the_worker(self):
        events, outcome, error = self.record(
            lambda: self.backend.run("no_such_module_for_tests_xyz:adapter", None, 1, 30.0))
        self.assertIsInstance(error, AdapterResolutionError)
        self.assert_cleaned_up_once(events)

    def test_startup_timeout(self):
        with TempModules(slow="import time\ntime.sleep(20)\n") as modules:
            events, outcome, error = self.record(
                lambda: SubprocessBackend(startup_timeout=0.8).run(modules.ref("slow", "adapter"), None, 1, 30.0))
        self.assertIsInstance(error, ExecutionError)
        self.assertIn("did not signal ready", str(error))
        self.assert_cleaned_up_once(events)

    def test_unpicklable_adapter_output(self):
        events, outcome, error = self.record(
            lambda: self.backend.run(adapter_ref("adapter_unpicklable_result"), None, 1, 30.0))
        self.assertIsInstance(error, ExecutionError)
        self.assert_cleaned_up_once(events)

    def test_keyboard_interrupt(self):
        real_wait = execution._wait
        calls = []

        def interrupting_wait(connections, timeout):
            calls.append(1)
            if len(calls) >= 2:                  # the first call returns "ready"; interrupt the timed phase
                raise KeyboardInterrupt()
            return real_wait(connections, timeout)

        with mock.patch("src.analytics.execution._wait", interrupting_wait):
            events, outcome, error = self.record(
                lambda: self.backend.run(adapter_ref("adapter_sleep"), {"sleep": 60}, 1, 120.0))
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assert_cleaned_up_once(events)

    def test_supervise_never_terminates_joins_or_polls_the_worker(self):
        # A process object that only has a sentinel: touching anything else raises AttributeError.
        parent_end, worker_end = multiprocessing.Pipe(duplex=True)
        sentinel_end, _unused = multiprocessing.Pipe(duplex=False)
        self.addCleanup(parent_end.close)
        self.addCleanup(sentinel_end.close)
        self.addCleanup(_unused.close)
        process = mock.Mock(spec=["sentinel"])
        process.sentinel = sentinel_end
        worker_end.send(("ready",))
        worker_end.close()                       # the "worker" disappears without a final message
        with mock.patch("src.analytics.execution._terminate_process_tree") as terminate:
            with self.assertRaises(execution._WorkerGone) as caught:
                SubprocessBackend()._supervise(process, parent_end, 5.0)
        self.assertTrue(caught.exception.ready)
        self.assertIsNotNone(caught.exception.elapsed)
        terminate.assert_not_called()

    def test_supervise_reports_a_worker_gone_before_ready_without_touching_the_process(self):
        parent_end, worker_end = multiprocessing.Pipe(duplex=True)
        sentinel_end, _unused = multiprocessing.Pipe(duplex=False)
        self.addCleanup(parent_end.close)
        self.addCleanup(sentinel_end.close)
        self.addCleanup(_unused.close)
        process = mock.Mock(spec=["sentinel"])
        process.sentinel = sentinel_end
        worker_end.close()
        with self.assertRaises(execution._WorkerGone) as caught:
            SubprocessBackend()._supervise(process, parent_end, 5.0)
        self.assertFalse(caught.exception.ready)


@unittest.skipUnless(TREE_KILL_SUPPORTED, TREE_KILL_SKIP_REASON)
class TestSubprocessCleanup(SubprocessTestCase):
    def test_worker_and_child_are_terminated_after_a_timeout(self):
        scenario = self.pid_files("hang")
        outcome = self.backend.run(adapter_ref("adapter_process_tree"), scenario, 1, 1.5)
        self.assertTrue(outcome.timed_out)
        self.assertNotEqual(read_pid(scenario["worker_pid_file"]), os.getpid())
        self.assert_tree_is_gone(scenario)

    def test_worker_and_child_are_terminated_after_success(self):
        scenario = self.pid_files("return")
        outcome = self.backend.run(adapter_ref("adapter_process_tree"), scenario, 1, 30.0)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.payload, "returned")
        self.assert_tree_is_gone(scenario)

    def test_worker_and_child_are_terminated_after_an_adapter_exception(self):
        scenario = self.pid_files("raise")
        outcome = self.backend.run(adapter_ref("adapter_process_tree"), scenario, 1, 30.0)
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.error, "RuntimeError: after starting a child")
        self.assert_tree_is_gone(scenario)

    def test_worker_is_gone_after_a_crash(self):
        scenario = self.pid_files("crash")
        outcome = self.backend.run(adapter_ref("adapter_process_tree"), scenario, 1, 30.0)
        self.assertTrue(outcome.error.startswith("WorkerCrash:"))
        self.assertIn("exit code 9", outcome.error)
        worker_pid = read_pid(scenario["worker_pid_file"])
        self.addCleanup(kill_quietly, read_pid(scenario["child_pid_file"]))   # never leak the child
        deadline = time.time() + 20
        while time.time() < deadline and process_exists(worker_pid):
            time.sleep(0.1)
        self.assertFalse(process_exists(worker_pid))

    @unittest.skipIf(os.name == "nt", "Windows cannot find the children of a worker that has already died "
                                      "(taskkill /T needs the live parent); child cleanup after a crash "
                                      "is only guaranteed on POSIX process groups")
    def test_children_are_terminated_after_a_crash(self):
        scenario = self.pid_files("crash")
        outcome = self.backend.run(adapter_ref("adapter_process_tree"), scenario, 1, 30.0)
        self.assertTrue(outcome.error.startswith("WorkerCrash:"))
        self.assert_tree_is_gone(scenario)

    def test_worker_and_child_are_terminated_when_the_parent_is_interrupted(self):
        scenario = self.pid_files("hang")
        real_wait = execution._wait

        def interrupting_wait(connections, timeout):
            result = real_wait(connections, min(timeout, 0.05))
            if all(os.path.exists(scenario[key]) and os.path.getsize(scenario[key]) > 0
                   for key in ("worker_pid_file", "child_pid_file")):
                raise KeyboardInterrupt()
            return result

        with mock.patch("src.analytics.execution._wait", interrupting_wait):
            with self.assertRaises(KeyboardInterrupt):
                self.backend.run(adapter_ref("adapter_process_tree"), scenario, 1, 120.0)
        self.assert_tree_is_gone(scenario)


class TestSkippedCoverageIsExplicit(unittest.TestCase):
    def test_process_tree_tests_have_a_documented_skip_reason(self):
        self.assertIn("taskkill", TREE_KILL_SKIP_REASON)
        self.assertIn("POSIX", TREE_KILL_SKIP_REASON)


if __name__ == "__main__":
    unittest.main()
