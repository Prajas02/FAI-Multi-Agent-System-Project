"""Timers and execution backends for running one allocation stage.

Implements Sections 7.3 and 10 of docs/BENCHMARK_HARNESS_DESIGN.md:

- ``StageOutcome``: the structured result of one stage call.
- ``Stopwatch``: a small ``time.perf_counter`` timer (monotonic, never wall clock).
- ``InProcessBackend``: calls an adapter directly in this process. It measures
  only the adapter call and does **not** enforce a hard timeout; it relies on
  the adapter honouring the time limit it is given.
- ``SubprocessBackend``: runs an adapter, given as an import reference
  ``"module.path:attribute"``, in a ``spawn`` worker process and **enforces a
  hard timeout**, terminating the worker and its child processes.
- ``resolve_adapter``: imports an adapter from an injected registry.

The module only depends on the standard library. It does not contain any real
algorithm adapter, any reward functionality, evaluation, logging or
orchestration. Runtimes reported by solvers (``AllocationResult.runtime_seconds``
and similar) are never used: every duration is measured here.

Subprocess protocol (single duplex pipe, ``spawn`` start method):

1. The parent starts the worker with the adapter reference, scenario, seed and
   time limit (all must be picklable).
2. The worker establishes its own process group/session (POSIX), imports the
   adapter and then sends ``("ready",)``. Interpreter start-up and import time
   therefore never count against the algorithm's time limit.
3. The parent starts the timeout clock when it receives ``ready``.
4. The worker measures the adapter call with its own ``perf_counter`` and sends
   one final message: ``("result", payload, seconds)``,
   ``("adapter_error", text, seconds)`` or ``("payload_error", text, seconds)``.
   It then waits (it does not exit by itself).
5. After the final message, a timeout, a crash or an interruption, ``run()``
   terminates the worker's whole process tree **exactly once** (the supervising
   code never terminates or joins the worker itself) (POSIX: ``killpg``; Windows:
   ``taskkill /T /F``) and joins the process. The worker is deliberately still
   alive when the tree is terminated, so that on Windows the parent/child links
   needed by ``taskkill /T`` still exist.

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import importlib
import math
import multiprocessing
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from multiprocessing.connection import wait as _connection_wait
from numbers import Real
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

__all__ = [
    "AdapterResolutionError",
    "ExecutionError",
    "InProcessBackend",
    "StageOutcome",
    "Stopwatch",
    "StopwatchError",
    "SubprocessBackend",
    "resolve_adapter",
]

MAX_ERROR_LENGTH = 500

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ExecutionError(RuntimeError):
    """Harness/infrastructure failure (not an ordinary algorithm failure).

    Examples: the worker could not be started, the scenario could not be
    pickled, the worker died before it was ready, the adapter's output could not
    be transferred, or a worker could not be terminated.
    """


class AdapterResolutionError(ExecutionError):
    """An adapter reference or registry entry could not be resolved."""


class StopwatchError(RuntimeError):
    """A stopwatch operation was invalid (for example stopping before starting)."""


class StageOutcome(NamedTuple):
    """Result of one stage call (same meaning for both backends).

    Attributes:
        started: The adapter call actually began.
        completed: The adapter returned normally.
        elapsed_seconds: Measured duration of the adapter call. For a timeout it
            is the parent-measured time from the ``ready`` event until the
            timeout was detected; for a worker crash it is the parent-measured
            time from ``ready`` until the crash was detected.
        error: Concise description of the failure, or ``None`` on success.
            Adapter exceptions read ``"<ExceptionType>: <message>"``; a timeout
            starts with ``"TimeoutError:"``; a worker crash starts with
            ``"WorkerCrash:"``.
        timed_out: The call was stopped because it exceeded the time limit.
        payload: The adapter's return value on success, otherwise ``None``.
    """
    started: bool
    completed: bool
    elapsed_seconds: Optional[float]
    error: Optional[str]
    timed_out: bool
    payload: Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_exception(exc: BaseException) -> str:
    """One-line ``"Type: message"`` text, bounded in length."""
    message = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if len(text) > MAX_ERROR_LENGTH:
        text = text[: MAX_ERROR_LENGTH - 3] + "..."
    return text


def _check_run_arguments(seed: Any, time_limit: Any) -> float:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}.")
    if isinstance(time_limit, bool) or not isinstance(time_limit, Real):
        raise TypeError(f"time_limit must be a number of seconds, got {time_limit!r}.")
    limit = float(time_limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f"time_limit must be a finite number > 0, got {limit!r}.")
    return limit


def _parse_adapter_reference(reference: Any) -> Tuple[str, str]:
    """Split ``"module.path:attribute"``; raise ``AdapterResolutionError`` if malformed."""
    if not isinstance(reference, str):
        raise AdapterResolutionError(
            f"adapter reference must be a string 'module.path:attribute', got {reference!r}."
        )
    module_name, separator, attribute = reference.partition(":")
    parts = module_name.split(".")
    if (
        separator != ":"
        or not all(_NAME_RE.fullmatch(part) for part in parts)
        or _NAME_RE.fullmatch(attribute) is None
    ):
        raise AdapterResolutionError(
            f"malformed adapter reference {reference!r}; expected 'module.path:attribute'."
        )
    return module_name, attribute


def _load_adapter(reference: str) -> Callable[..., Any]:
    """Import and return the callable named by ``reference``."""
    module_name, attribute = _parse_adapter_reference(reference)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterResolutionError(
            f"cannot import module {module_name!r} for adapter {reference!r}: {_format_exception(exc)}"
        ) from exc
    except Exception as exc:  # the module raised while it was being imported
        raise AdapterResolutionError(
            f"importing module {module_name!r} for adapter {reference!r} raised "
            f"{_format_exception(exc)}"
        ) from exc
    try:
        adapter = getattr(module, attribute)
    except AttributeError as exc:
        raise AdapterResolutionError(
            f"module {module_name!r} has no attribute {attribute!r} (adapter {reference!r})."
        ) from exc
    if not callable(adapter):
        raise AdapterResolutionError(
            f"adapter {reference!r} resolved to a non-callable {type(adapter).__name__}."
        )
    return adapter


def resolve_adapter(algorithm_id: str, registry: "Mapping[str, str]") -> Callable[..., Any]:
    """Return the adapter callable registered for ``algorithm_id``.

    The registry is injected by the caller (there is no built-in registry):
    ``{"MILP": "some.module:adapter_function", ...}``.

    Raises:
        AdapterResolutionError: the registry is not a mapping, the id is not
            registered, the reference is malformed, the module cannot be
            imported (the original exception is kept as ``__cause__``), the
            attribute is missing, or it is not callable.
    """
    if not isinstance(registry, Mapping):
        raise AdapterResolutionError("registry must be a mapping of algorithm id -> 'module:attribute'.")
    try:
        known = algorithm_id in registry
    except TypeError:
        known = False
    if not known:
        raise AdapterResolutionError(
            f"no adapter is registered for algorithm {algorithm_id!r}; "
            f"registered ids: {sorted(str(key) for key in registry)}."
        )
    return _load_adapter(registry[algorithm_id])


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------

class Stopwatch:
    """Monotonic timer based on ``time.perf_counter`` (never the wall clock).

    The clock is resolved when the stopwatch is created, so tests can replace
    ``time.perf_counter`` or pass their own ``clock``. It never reads any
    runtime reported by a solver.

    * ``start()``: begin timing (error if already running; a stopped watch is
      restarted from zero).
    * ``stop()``: end timing and return the elapsed seconds (error if not
      running).
    * ``elapsed``: seconds so far while running, the final value once stopped
      (error if never started).
    """

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock = clock if clock is not None else time.perf_counter
        self._started_at: Optional[float] = None
        self._elapsed: Optional[float] = None

    @property
    def running(self) -> bool:
        return self._started_at is not None

    def start(self) -> None:
        if self._started_at is not None:
            raise StopwatchError("stopwatch is already running.")
        self._elapsed = None
        self._started_at = self._clock()

    def stop(self) -> float:
        if self._started_at is None:
            raise StopwatchError("stopwatch is not running (stop() before start(), or stopped twice).")
        self._elapsed = self._clock() - self._started_at
        self._started_at = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        if self._started_at is not None:
            return self._clock() - self._started_at
        if self._elapsed is None:
            raise StopwatchError("stopwatch was never started.")
        return self._elapsed

    def reset(self) -> None:
        """Discard any measurement and return to the never-started state."""
        self._started_at = None
        self._elapsed = None


# ---------------------------------------------------------------------------
# In-process backend
# ---------------------------------------------------------------------------

class InProcessBackend:
    """Call an adapter directly in the current process.

    Only the adapter call is timed. **No hard timeout is enforced**: the adapter
    receives ``time_limit`` and is trusted to honour it. Use
    :class:`SubprocessBackend` when a runaway solver must be stopped.
    """

    def run(self, adapter: Callable[..., Any], scenario: Any, seed: int, time_limit: float) -> StageOutcome:
        """Call ``adapter(scenario, seed, time_limit)`` and report the outcome.

        Ordinary exceptions become a failed outcome (with the elapsed time);
        ``KeyboardInterrupt`` and ``SystemExit`` propagate.

        Raises:
            ExecutionError: ``adapter`` is not callable (harness error).
            TypeError / ValueError: invalid ``seed`` or ``time_limit``.
        """
        if not callable(adapter):
            raise ExecutionError(f"adapter must be callable, got {type(adapter).__name__}.")
        _check_run_arguments(seed, time_limit)

        watch = Stopwatch()
        watch.start()
        try:
            payload = adapter(scenario, seed, time_limit)
        except Exception as exc:
            return StageOutcome(True, False, watch.stop(), _format_exception(exc), False, None)
        return StageOutcome(True, True, watch.stop(), None, False, payload)


# ---------------------------------------------------------------------------
# Subprocess backend
# ---------------------------------------------------------------------------

def _wait(connections: List[Any], timeout: float) -> List[Any]:
    """Wait for any connection/sentinel to become ready (replaceable in tests)."""
    return _connection_wait(connections, timeout)


def _has_data(connection: Any) -> bool:
    """``True`` if a message is waiting; a broken pipe counts as no data."""
    try:
        return bool(connection.poll(0))
    except (OSError, EOFError):
        return False


def _worker_main(connection: Any, adapter_reference: str, scenario: Any, seed: int, time_limit: float) -> None:
    """Worker process entry point (module level so that ``spawn`` can pickle it)."""
    try:
        if os.name != "nt":
            try:
                os.setsid()  # own session/process group: the parent can kill the whole group
            except OSError:
                pass
        adapter = _load_adapter(adapter_reference)
    except BaseException as exc:
        try:
            connection.send(("setup_error", _format_exception(exc)))
        except Exception:
            pass
        return

    connection.send(("ready",))

    watch = Stopwatch()
    watch.start()
    try:
        payload = adapter(scenario, seed, time_limit)
    except Exception as exc:
        seconds = watch.stop()
        connection.send(("adapter_error", _format_exception(exc), seconds))
    else:
        seconds = watch.stop()
        try:
            connection.send(("result", payload, seconds))
        except Exception as exc:  # the payload cannot be pickled / transferred
            connection.send(("payload_error", _format_exception(exc), seconds))

    # Stay alive until the parent has terminated the process tree (or closed the
    # pipe): the parent must be able to find this process's children.
    try:
        connection.recv()
    except (EOFError, OSError):
        pass


def _terminate_process_tree(process: Any) -> None:
    """Kill the worker and all its descendants. Safe to call repeatedly.

    Must be called **before** the process is joined or polled, so that a POSIX
    process-group id cannot have been reused by an unrelated process.
    """
    pid = process.pid
    if pid is None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(int(pid))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            process.kill()
        except Exception:
            pass
    else:
        try:
            os.killpg(int(pid), signal.SIGKILL)  # the worker is its own group leader
        except OSError:
            try:
                process.kill()
            except Exception:
                pass


def _reap(process: Any) -> Tuple[bool, Optional[int]]:
    """Join a **started** process; return ``(gone, exit_code)``.

    The exit code is read before the process object is closed. Must only be
    called for a process whose ``start()`` succeeded, and only after
    :func:`_terminate_process_tree`.
    """
    try:
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        if process.is_alive():
            return False, None
        exit_code = process.exitcode
        try:
            process.close()
        except ValueError:
            pass
        return True, exit_code
    except Exception:
        return False, None


def _close_quietly(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        pass


class _WorkerGone(Exception):
    """Internal signal from ``_supervise``: the worker disappeared without a final message.

    The exit code is only read by ``run()`` after the worker has been
    terminated and reaped, so that the process tree is terminated exactly once.
    """

    def __init__(self, ready: bool, elapsed: Optional[float]) -> None:
        super().__init__("worker process gone")
        self.ready = ready
        self.elapsed = elapsed


class SubprocessBackend:
    """Run an adapter in a ``spawn`` worker process with a hard timeout.

    Args:
        startup_timeout: Seconds allowed for the worker to start, import the
            adapter and signal ``ready``. This budget is separate from the
            algorithm's time limit.
    """

    def __init__(self, startup_timeout: float = 120.0) -> None:
        if isinstance(startup_timeout, bool) or not isinstance(startup_timeout, Real):
            raise TypeError("startup_timeout must be a number of seconds.")
        if not math.isfinite(float(startup_timeout)) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be a finite number > 0.")
        self.startup_timeout = float(startup_timeout)

    def run(self, adapter_ref: str, scenario: Any, seed: int, time_limit: float) -> StageOutcome:
        """Run ``adapter(scenario, seed, time_limit)`` in a worker process.

        Args:
            adapter_ref: ``"module.path:attribute"`` import reference.
            scenario: Passed to the adapter; must be picklable.
            seed: Passed to the adapter.
            time_limit: Seconds, enforced by the parent after the worker is ready.

        Returns:
            A ``StageOutcome``. Success, adapter exceptions, timeouts and worker
            crashes after the adapter started are all reported as outcomes.

        Raises:
            AdapterResolutionError: malformed reference, or the worker could not
                import the adapter.
            ExecutionError: infrastructure failure (worker could not be started,
                inputs could not be pickled, worker died or hung before it was
                ready, the adapter's output could not be transferred, or the
                worker could not be terminated).
            TypeError / ValueError: invalid ``seed`` or ``time_limit``.
            KeyboardInterrupt: propagated after the worker has been cleaned up.
        """
        _parse_adapter_reference(adapter_ref)
        limit = _check_run_arguments(seed, time_limit)

        try:
            context = multiprocessing.get_context("spawn")
            parent_connection, child_connection = context.Pipe(duplex=True)
        except Exception as exc:
            raise ExecutionError(f"could not prepare the worker process: {_format_exception(exc)}") from exc

        process = None
        started = False          # True only after process.start() succeeded
        outcome = None
        worker_gone = None
        exit_code = None
        leaked = False
        try:
            process = context.Process(
                target=_worker_main,
                args=(child_connection, adapter_ref, scenario, seed, limit),
                name="allocation-worker",
                daemon=False,  # a solver may need to start its own processes
            )
            try:
                process.start()
            except Exception as exc:
                raise ExecutionError(
                    f"could not start the worker process: {_format_exception(exc)}"
                ) from exc
            finally:
                _close_quietly(child_connection)  # the parent only keeps its own end
            started = True

            try:
                outcome = self._supervise(process, parent_connection, limit)
            except _WorkerGone as gone:
                worker_gone = gone
        finally:
            # Centralized cleanup. A worker that never started is neither
            # terminated nor reaped, so the original start error is preserved.
            # A started worker's process tree is terminated exactly once, here,
            # before the process is joined (a POSIX process-group id must not be
            # reused first) and before the pipe is closed (the worker is still
            # alive, so that its children can be found).
            try:
                if started:
                    _terminate_process_tree(process)
            finally:
                _close_quietly(parent_connection)
                _close_quietly(child_connection)
                if started:
                    reaped, exit_code = _reap(process)
                    leaked = not reaped

        if leaked:
            raise ExecutionError("the worker process could not be terminated.")
        if worker_gone is not None:
            if not worker_gone.ready:
                raise ExecutionError(
                    f"the worker process exited before it was ready (exit code {exit_code})."
                )
            return StageOutcome(
                True, False, worker_gone.elapsed,
                f"WorkerCrash: the worker process exited unexpectedly (exit code {exit_code}) "
                "without returning a result.",
                False, None,
            )
        return outcome

    # -- protocol ----------------------------------------------------------

    @staticmethod
    def _receive(connection: Any) -> Optional[tuple]:
        """Receive one message; ``None`` means the pipe is closed (worker gone)."""
        try:
            message = connection.recv()
        except (EOFError, ConnectionError):
            return None
        except OSError:
            return None
        except Exception as exc:  # the parent could not unpickle the message
            raise ExecutionError(
                f"could not decode a message from the worker: {_format_exception(exc)}"
            ) from exc
        if not isinstance(message, tuple) or not message or not isinstance(message[0], str):
            raise ExecutionError(f"unexpected message from the worker: {message!r}"[:MAX_ERROR_LENGTH])
        return message

    def _supervise(self, process: Any, connection: Any, time_limit: float) -> StageOutcome:
        """Follow the worker protocol and return the outcome.

        Never terminates, joins or polls the process: that is done once, by
        ``run()``. Raises ``_WorkerGone`` if the worker disappears without a
        final message.
        """
        # Phase 1: wait for "ready"; this does not count against the time limit.
        startup = Stopwatch()
        startup.start()
        while True:
            remaining = self.startup_timeout - startup.elapsed
            if remaining <= 0:
                raise ExecutionError(
                    f"the worker did not signal ready within {self.startup_timeout:g} s "
                    "(slow start-up or import)."
                )
            ready = _wait([connection, process.sentinel], remaining)
            if not ready:
                continue
            if connection in ready:
                message = self._receive(connection)
                if message is not None:
                    if message[0] == "ready":
                        break
                    if message[0] == "setup_error":
                        raise AdapterResolutionError(f"the worker could not load the adapter: {message[1]}")
                    raise ExecutionError(f"unexpected message before ready: {message[0]!r}.")
            elif _has_data(connection):
                continue
            # the worker is gone without having signalled ready (run() reports it)
            raise _WorkerGone(ready=False, elapsed=None)

        # Phase 2: the timeout clock starts now.
        watch = Stopwatch()
        watch.start()
        while True:
            remaining = time_limit - watch.elapsed
            if remaining <= 0:
                return StageOutcome(
                    True, False, watch.stop(),
                    f"TimeoutError: the adapter exceeded the time limit of {time_limit:g} s.",
                    True, None,
                )
            ready = _wait([connection, process.sentinel], remaining)
            if not ready:
                continue
            message = None
            if connection in ready:
                message = self._receive(connection)
            elif _has_data(connection):
                continue
            if message is None:
                # The worker is gone and sent no final message: a crash
                # (run() adds the exit code after the worker has been reaped).
                raise _WorkerGone(ready=True, elapsed=watch.stop())

            kind = message[0]
            if kind == "result" and len(message) == 3:
                return StageOutcome(True, True, float(message[2]), None, False, message[1])
            if kind == "adapter_error" and len(message) == 3:
                return StageOutcome(True, False, float(message[2]), str(message[1]), False, None)
            if kind == "payload_error" and len(message) == 3:
                raise ExecutionError(
                    "the adapter's output could not be transferred from the worker "
                    f"(not picklable?): {message[1]}"
                )
            raise ExecutionError(f"unexpected message from the worker: {kind!r}.")
