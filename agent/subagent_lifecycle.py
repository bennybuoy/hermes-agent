"""Public, plugin-safe lifecycle API for delegated Hermes subagents: immutable contracts, not ``AIAgent``
objects. Plugins obtain it via ``PluginContext.subagent_lifecycle``."""

from __future__ import annotations

import contextvars
import dataclasses
import enum
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
import contextlib
import weakref
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from agent.interrupt_compat import request_hard_interrupt

PUBLIC_CONTRACT_VERSION = 1
_MAX_GOAL_CHARS = 16_000
_MAX_CONTEXT_CHARS = 32_000
_MAX_METADATA_BYTES = 8_192
_MAX_RESULT_CHARS = 32_000
_TERMINAL_RETENTION_SECONDS = 3_600
# Herald-parity floor for the opt-in inactivity watchdog; lower values are rejected at launch.
_STALL_TIMEOUT_MIN_SECONDS = 30.0

# ── Two-phase stall monitor (ported from tools/async_delegation.py) ─────────
# A runner wedged before returning never reaches its terminal publication, so the record
# would stay RUNNING forever. No wall-clock timeout (heavy work must never be killed for
# taking long): one lazily-started monitor thread samples per-record PROGRESS via the
# structured ``(api_call_count, current_tool, last_activity_ts)`` token; a child frozen
# past the idle threshold (or the fixed in-tool ceiling) is hard-interrupted and given a
# FIXED grace window to unwind via the normal completion path; a record still unreturned
# after grace is force-finalized with a terminal FAILED/STALLED result. This is a
# liveness watchdog, not a progress detector: a stuck provider request with a functioning
# heartbeat stays "alive" to it; transport watchdogs and configured delegation timeouts
# remain independently effective.
_STALL_SWEEP_SECONDS = 30.0
_STALL_GRACE_SECONDS = 120.0
_STALL_IN_TOOL_SECONDS = 1200.0


class SubagentLifecycleError(ValueError):
    """A request cannot be safely accepted by the public lifecycle API."""


class SubagentState(str, enum.Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class SubagentLaunchRequest:
    goal: str
    context: Optional[str] = None
    role: str = "leaf"
    model: Optional[str] = None
    allowed_toolsets: Optional[tuple[str, ...]] = None
    blocked_tools: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    parent_session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    stall_timeout_seconds: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class SubagentHandle:
    contract_version: int
    subagent_id: str
    parent_session_id: Optional[str]
    correlation_id: Optional[str]
    created_at: float
    provider: Optional[str]
    model: Optional[str]
    role: str
    depth: int
    capability: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentHandle":
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("Malformed subagent handle.") from exc


@dataclasses.dataclass(frozen=True)
class SubagentStatus:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentTerminalState:
    handle: SubagentHandle
    state: SubagentState
    completed: bool
    timed_out: bool = False
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentCancelResult:
    accepted: bool
    already_terminal: bool = False
    unknown_handle: bool = False
    unsupported: bool = False
    state: SubagentState = SubagentState.UNKNOWN


@dataclasses.dataclass(frozen=True)
class SubagentResult:
    handle: SubagentHandle
    terminal_state: SubagentState
    ready: bool
    summary: Optional[str] = None
    structured_payload: Optional[Mapping[str, Any]] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_classification: Optional[str] = None
    error_message: Optional[str] = None
    usage_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    tool_execution_summary: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    # Additive, present only on stall finalizations (flat strings/numbers; kept immutable
    # so mutability never leaks into the frozen result). ``result_hash`` stays an opaque
    # versioned integrity token — do not test cross-version equality.
    stall_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    result_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentReconnectResult:
    connected: bool
    state: SubagentState
    diagnostic: Optional[str] = None


@dataclasses.dataclass
class _Record:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    agent: Any = None
    future: Optional[Future] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[SubagentResult] = None
    # Child-activity stall bookkeeping (see _STALL_* constants): last structured
    # progress sample, the monotonic instant it was first observed, whether the
    # child was inside a tool at sample time, and the fixed post-interrupt grace
    # deadline. All are read/written under _REGISTRY.lock.
    request_stall_timeout_seconds: Optional[float] = None
    last_progress_token: Optional[tuple] = None
    progress_started_at: Optional[float] = None
    in_tool: bool = False
    stall_grace_deadline: Optional[float] = None
    stall_quiet_seconds: Optional[float] = None
    stall_threshold_seconds: Optional[float] = None
    stall_phase: Optional[str] = None
    stall_interrupt_requested: bool = False
    # Set by cancel(): the monitor's force-finalization must honour cancellation precedence.
    stall_cancel_precedence: bool = False
    terminal_event: threading.Event = dataclasses.field(default_factory=threading.Event)


@dataclasses.dataclass
class _Registry:
    """Thread-safe terminal-retention registry; never returns live records."""

    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    records: dict[str, _Record] = dataclasses.field(default_factory=dict)
    correlations: dict[tuple[Optional[str], str], str] = dataclasses.field(default_factory=dict)
    # Stall-monitor lifecycle: start/exit synchronized under ``lock`` with a generation
    # counter so a racing launch either sees an alive monitor or starts one.
    monitor_thread: Optional[threading.Thread] = None
    monitor_owner_generation: int = 0
    monitor_generation: int = 0
    monitor_wake: threading.Event = dataclasses.field(default_factory=threading.Event)


_REGISTRY = _Registry()


def _ensure_stall_monitor() -> None:
    """Start the stall monitor if any record carries a stall policy.

    The start decision is synchronized under ``_REGISTRY.lock`` with a generation counter:
    a launch either sees the CURRENT generation's monitor alive or bumps the generation and
    starts a new thread — even a live-but-superseded thread never blocks re-arming, so a
    launch racing a monitor's final empty sweep can never strand its record.
    """
    with _REGISTRY.lock:
        if not any(record.request_stall_timeout_seconds is not None for record in _REGISTRY.records.values()):
            return
        thread = _REGISTRY.monitor_thread
        if thread is not None and thread.is_alive() and _REGISTRY.monitor_owner_generation == _REGISTRY.monitor_generation:
            return
        _REGISTRY.monitor_generation += 1
        generation = _REGISTRY.monitor_generation
        _REGISTRY.monitor_owner_generation = generation
        thread = threading.Thread(
            target=_stall_monitor_loop, args=(generation,), name="hermes-lifecycle-stall-monitor", daemon=True)
        _REGISTRY.monitor_thread = thread
    thread.start()


def _stall_monitor_loop(generation: int) -> None:
    """Sweep stall-capable records, then wait one sweep interval (30s) or a launch wake.

    Exits — under ``_REGISTRY.lock`` — when its generation was superseded by a newer monitor
    or a synchronized sweep finds nothing monitorable, clearing the slot it owns."""
    while True:
        SubagentLifecycleService._sweep()
        with _REGISTRY.lock:
            if _REGISTRY.monitor_owner_generation != generation:
                return  # superseded: a newer generation's thread owns the slot
            if not any(record.request_stall_timeout_seconds is not None and record.result is None
                       for record in _REGISTRY.records.values()):
                _REGISTRY.monitor_thread = None
                return
        if _REGISTRY.monitor_wake.wait(_STALL_SWEEP_SECONDS):
            _REGISTRY.monitor_wake.clear()


from agent.interrupt_compat import request_hard_interrupt as _request_hard_interrupt


def _stall_metadata(record: _Record, quiet_seconds: float, threshold: float) -> Mapping[str, Any]:
    """Structured stall metadata — additive, present only on stall finalizations. Returned
    frozen (read-only view) so mutability never leaks into the frozen result."""
    return MappingProxyType({
        "stalled_after_quiet_seconds": round(quiet_seconds, 2),
        "stall_threshold_seconds": threshold,
        "stall_phase": "in_tool" if record.in_tool else "idle",
        "stall_grace_seconds": _STALL_GRACE_SECONDS,
    })
from tools.daemon_pool import DaemonThreadPoolExecutor as _DaemonExecutor  # daemon: a wedged child never blocks exit
_EXECUTOR = _DaemonExecutor(max_workers=8, thread_name_prefix="hermes-lifecycle")
_SECRET = secrets.token_bytes(32)
_ACTIVE_PARENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar("hermes_subagent_lifecycle_parent", default=None)


@contextmanager
def bind_subagent_parent(parent_agent: Any):
    """Bind the host-owned parent for the current agent turn.

    Stored as a weakref: every asyncio Handle/Future scheduled from the turn
    (LSP reader loops, kernel pipes, ...) snapshots the Context, and those
    snapshots outlive the turn. A strong ref there pinned finished delegate
    children — each of which binds itself here for its own turn — in the
    parent process heap for the life of the background loop.
    """
    try:
        ref = weakref.ref(parent_agent)
    except TypeError:
        ref = lambda: parent_agent  # noqa: E731 — non-weakrefable test doubles
    token = _ACTIVE_PARENT_AGENT.set(ref)
    try:
        yield
    finally:
        _ACTIVE_PARENT_AGENT.reset(token)


def get_active_subagent_parent() -> Any:
    """Return the parent bound to this execution context, if any."""
    ref = _ACTIVE_PARENT_AGENT.get()
    return ref() if ref is not None else None


def _opt_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _sample_child_progress(child: Any, previous: Optional[tuple[tuple, bool]] = None) -> tuple[tuple, bool]:
    """Structured progress token for one child, mirroring delegate-tool's ``_batch_progress_token``:
    ``(api_call_count, current_tool, last_activity_ts)`` plus ``in_tool = bool(current_tool)``.
    An unreadable child keeps the previous sample — a child that cannot be read must never look
    healthy by accident."""
    try:
        summary = child.get_activity_summary()
        token = (summary.get("api_call_count", 0), summary.get("current_tool"), summary.get("last_activity_ts"))
        return token, bool(summary.get("current_tool"))
    except Exception:
        return previous if previous is not None else ((0, None, None), False)


def _session_id_of(agent: Any) -> Optional[str]:
    return str(getattr(agent, "session_id", "") or "") or None


def _clip(value: Any) -> Optional[str]:
    return str(value)[:_MAX_RESULT_CHARS] if value is not None else None


# Per-field shape check applied to a (possibly deserialized) handle before trusting it.
_HANDLE_FIELD_CHECKS: tuple[tuple[str, Callable[[Any], bool]], ...] = (
    ("contract_version", lambda v: type(v) is int and v == PUBLIC_CONTRACT_VERSION),
    ("subagent_id", lambda v: isinstance(v, str) and bool(v)),
    ("parent_session_id", _opt_str),
    ("correlation_id", _opt_str),
    ("created_at", lambda v: not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)),
    ("provider", _opt_str),
    ("model", _opt_str),
    ("role", lambda v: isinstance(v, str)),
    ("depth", lambda v: type(v) is int),
    ("capability", lambda v: isinstance(v, str)),
)

# Launch-request rejections in check order: (predicate, error). The type check leads so later predicates may
# dereference request fields.
_REQUEST_REJECTIONS: tuple[tuple[Callable[[Any], bool], str], ...] = (
    (lambda r: not isinstance(r, SubagentLaunchRequest) or not isinstance(r.goal, str) or not r.goal.strip() or len(r.goal) > _MAX_GOAL_CHARS,
     "goal must be a non-empty string of at most 16000 characters."),
    (lambda r: r.context is not None and (not isinstance(r.context, str) or len(r.context) > _MAX_CONTEXT_CHARS),
     "context must be a string of at most 32000 characters."),
    (lambda r: r.role not in {"leaf", "orchestrator"}, "role must be 'leaf' or 'orchestrator'."),
    (lambda r: r.timeout_seconds is not None, "Per-launch timeout is not supported; configure delegation timeout explicitly."),
    (lambda r: r.stall_timeout_seconds is not None
     and (isinstance(r.stall_timeout_seconds, bool) or not isinstance(r.stall_timeout_seconds, (int, float))
          or not math.isfinite(r.stall_timeout_seconds) or r.stall_timeout_seconds < _STALL_TIMEOUT_MIN_SECONDS),
     f"stall_timeout_seconds must be a finite number of at least {_STALL_TIMEOUT_MIN_SECONDS:g} seconds "
     "(None disables the stall monitor for this child)."),
    (lambda r: r.working_directory is not None,
     "working_directory is not supported because Hermes delegates use isolated task environments."),
    (lambda r: bool(r.blocked_tools),
     "Per-tool blocking is not supported; use allowed_toolsets. Hermes always blocks unsafe child tools."),
)


def _handle_is_well_formed(handle: Any) -> bool:
    return isinstance(handle, SubagentHandle) and all(check(getattr(handle, field)) for field, check in _HANDLE_FIELD_CHECKS)


class SubagentLifecycleService:
    """Stable public service behind :attr:`PluginContext.subagent_lifecycle`. Children run in-process only;
    completed results stay until process exit; ``reconnect`` reports that a serialized handle cannot
    reconnect after a restart instead of launching work again."""

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver

    def launch(self, request: SubagentLaunchRequest) -> SubagentHandle:
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError("No active Hermes parent session is available.")
        self._validate_request(request, parent)
        parent_session_id = _session_id_of(parent)
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError("parent_session_id does not match the active session.")
        correlation_key = (parent_session_id, request.correlation_id or "")
        with _REGISTRY.lock:
            self._cleanup_locked()
            if request.correlation_id and correlation_key in _REGISTRY.correlations:
                raise SubagentLifecycleError("Duplicate correlation_id for this parent session.")
        # Lazy: delegate construction stays internal, plugins never import private delegation helpers.
        from tools.delegate_tool import _build_child_preserving_parent_tools, DEFAULT_MAX_ITERATIONS
        child = _build_child_preserving_parent_tools(
            task_index=0, goal=request.goal, context=request.context,
            toolsets=list(request.allowed_toolsets) if request.allowed_toolsets else None,
            model=request.model, max_iterations=DEFAULT_MAX_ITERATIONS, task_count=1, parent_agent=parent, role=request.role,
        )
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        if not subagent_id:
            raise SubagentLifecycleError("Hermes failed to assign a child identity.")
        created = time.time()
        handle = SubagentHandle(
            PUBLIC_CONTRACT_VERSION, subagent_id, parent_session_id, request.correlation_id, created,
            getattr(child, "provider", None), getattr(child, "model", None), getattr(child, "_delegate_role", request.role),
            int(getattr(child, "_delegate_depth", 1) or 1), self._capability(subagent_id, parent_session_id, created),
        )
        record = _Record(handle, SubagentState.PENDING, created, agent=child,
                         request_stall_timeout_seconds=request.stall_timeout_seconds)
        with _REGISTRY.lock:
            _REGISTRY.records[subagent_id] = record
            if request.correlation_id:
                _REGISTRY.correlations[correlation_key] = subagent_id
        record.future = _EXECUTOR.submit(self._run, record, request.goal, parent)
        if request.stall_timeout_seconds is not None:
            _ensure_stall_monitor()
            _REGISTRY.monitor_wake.set()
        return handle

    @staticmethod
    def _stamp_progress_locked(record: _Record) -> None:
        """Initial/refreshed progress sample at the PENDING→RUNNING transition (monotonic clock)."""
        previous = None if record.last_progress_token is None else ((record.last_progress_token, record.in_tool))
        token, in_tool = _sample_child_progress(record.agent, previous=previous)
        record.last_progress_token, record.in_tool = token, in_tool
        record.progress_started_at = time.monotonic()

    @staticmethod
    def _sweep_locked() -> tuple[list, list, bool]:
        """Collect two-phase stall actions; caller holds ``_REGISTRY.lock``.

        Phase 1 candidates: records whose frozen sample stayed frozen past the effective
        threshold (idle = the request's ``stall_timeout_seconds``; in-tool = the FIXED
        ``_STALL_IN_TOOL_SECONDS``, regardless of the request). Each candidate is REVALIDATED
        here — the token is re-sampled under the lock, and a child that resumed since the
        observation is adopted as fresh instead of interrupted (stale-sample abort, mirroring
        the activity-tracking abort-claim pattern). Committing marks the record stalling with
        a FIXED grace deadline; the interrupt itself is requested by the caller OUTSIDE the
        lock. Phase 2 candidates: stalling records whose fixed grace expired — activity during
        grace never resets it. Queued (never-started) records are monitorable but not charged
        inactivity. Records without a stall policy are never monitor-engaged. Returns
        ``(interrupts, finalizes, any_monitorable)``.
        """
        now_mono = time.monotonic()
        to_interrupt: list[tuple[_Record, float, float]] = []  # (record, quiet_seconds, threshold)
        to_finalize: list[tuple[_Record, float, float]] = []
        any_monitorable = False
        for record in _REGISTRY.records.values():
            if record.request_stall_timeout_seconds is None:
                continue  # no policy on the request: never monitor-engaged
            if record.result is not None:
                continue  # already terminal
            any_monitorable = True
            if record.state is SubagentState.PENDING:
                continue  # queued behind the pool: monitorable, but inactivity is not charged
            if record.stall_grace_deadline is not None:
                # Phase 2: grace is fixed — a re-sample here could look like recovery the
                # instant the interruption itself triggers activity, so none is taken.
                if now_mono >= record.stall_grace_deadline:
                    to_finalize.append((record, record.stall_quiet_seconds or 0.0,
                                        record.stall_threshold_seconds or _STALL_IN_TOOL_SECONDS))
                continue
            quiet = max(0.0, now_mono - (record.progress_started_at or now_mono))
            threshold = _STALL_IN_TOOL_SECONDS if record.in_tool else record.request_stall_timeout_seconds
            if quiet < threshold:
                continue
            # Revalidate the frozen observation before committing the interrupt.
            token, in_tool = _sample_child_progress(record.agent, previous=(record.last_progress_token, record.in_tool))
            if token != record.last_progress_token:
                record.last_progress_token, record.in_tool = token, in_tool
                record.progress_started_at = now_mono
                continue  # activity resumed between observation and commit: abort
            record.stall_quiet_seconds, record.stall_threshold_seconds = quiet, threshold
            record.stall_phase = "in_tool" if in_tool else "idle"
            record.stall_grace_deadline = now_mono + _STALL_GRACE_SECONDS  # fixed from commit instant
            to_interrupt.append((record, quiet, threshold))
        return to_interrupt, to_finalize, any_monitorable

    @staticmethod
    def _sweep() -> int:
        """One two-phase monitor pass; returns the number of actions taken.

        Test/monitor seam: the monitor loop calls this every 30s; event-based tests drive it
        directly instead of sleeping. Hard interrupts are requested OUTSIDE the registry lock
        so a blocking interrupt cannot stall other records' deadlines."""
        with _REGISTRY.lock:
            to_interrupt, to_finalize, _ = SubagentLifecycleService._sweep_locked()
        actions = 0
        for record, quiet, threshold in to_interrupt:
            agent = record.agent
            if agent is None:
                continue
            with contextlib.suppress(Exception):
                _request_hard_interrupt(
                    agent,
                    f"Lifecycle stall monitor: no child activity for {quiet:.0f}s "
                    f"(threshold {threshold:.0f}s); fixed grace {_STALL_GRACE_SECONDS:.0f}s before force-finalization.",
                    tool_reason="subagent stall interrupt requested",
                )
            with _REGISTRY.lock:
                record.stall_interrupt_requested = True
            actions += 1
        for record, quiet, threshold in to_finalize:
            actions += SubagentLifecycleService._force_finalize_stalled(record, quiet, threshold)
        return actions

    @staticmethod
    def _force_finalize_stalled(record: _Record, quiet_seconds: float, threshold: float) -> int:
        """Terminal publication for a record whose fixed grace expired. Cancellation-vs-stall
        precedence is resolved here under the registry lock (via ``_publish_terminal``): a
        record whose runner returned CANCELLED first, or whose state is CANCEL_REQUESTED,
        keeps the cancellation classification — the stall does not overwrite it."""
        if record.state is SubagentState.CANCEL_REQUESTED:
            error_message = (
                f"Subagent {record.handle.subagent_id} was cancelled during the stall grace window "
                "and did not return; finalized as cancelled rather than stalled.")
            result = SubagentLifecycleService._with_result_hash(SubagentResult(
                record.handle, SubagentState.CANCELLED, True, summary=None,
                error_classification="CANCELLED", error_message=error_message,
                started_at=record.started_at, completed_at=time.time(),
                stall_metadata=_stall_metadata(record, quiet_seconds, threshold),
            ))
            return 1 if SubagentLifecycleService._publish_terminal(record, result) else 0
        error_message = (
            f"Subagent {record.handle.subagent_id} stalled: the child stopped making progress "
            "(no new API calls, tool transitions, or streamed tokens), did not respond to "
            "interruption within the grace window, and never returned. The worker may be wedged "
            "inside a model API call; transport watchdogs and configured delegation timeouts "
            "remain independently effective. Re-dispatch the task if it is still needed.")
        result = SubagentLifecycleService._with_result_hash(SubagentResult(
            record.handle, SubagentState.FAILED, True, summary=None,
            error_classification="STALLED", error_message=error_message,
            started_at=record.started_at, completed_at=time.time(),
            stall_metadata=_stall_metadata(record, quiet_seconds, threshold),
        ))
        return 1 if SubagentLifecycleService._publish_terminal(record, result) else 0

    @staticmethod
    def _publish_terminal(record: _Record, result: SubagentResult) -> bool:
        """Single-writer terminal publication: under ``_REGISTRY.lock``, the FIRST publication
        owns ``agent``/``result``/``state``/``completed_at``/``updated_at`` and sets the
        record's ``terminal_event``; later publications (a late runner return after
        force-finalization) change none of them. Never completes the executor-owned Future."""
        with _REGISTRY.lock:
            if record.result is not None:
                return False
            record.agent, record.result, record.state = None, result, result.terminal_state
            record.completed_at = record.updated_at = result.completed_at or time.time()
            record.terminal_event.set()
            return True

    @staticmethod
    def _with_result_hash(result: SubagentResult) -> SubagentResult:
        # Build the hash preimage manually: dataclasses.asdict deep-copies field values and
        # a frozen mapping view (stall_metadata) cannot be pickled/deep-copied. A plain-dict
        # copy keeps the preimage stable, printable, and serializable.
        payload: dict[str, Any] = {}
        for field in dataclasses.fields(result):
            value = getattr(result, field.name)
            if field.name == "result_hash":
                continue
            if field.name == "stall_metadata" and isinstance(value, MappingProxyType):
                value = dict(value)
            payload[field.name] = value
        return dataclasses.replace(
            result, result_hash=hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest())

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            return SubagentStatus(handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE")
        with _REGISTRY.lock:
            diagnostic = None
            if record.result is not None:
                pass  # terminal: phase-1 diagnostics cleared
            elif record.stall_grace_deadline is not None:
                # Phase-2/grace diagnostic; the metadata ``stall_phase`` keeps its
                # async-delegation meaning (idle|in_tool), never phase 1/2.
                remaining = max(0.0, record.stall_grace_deadline - time.monotonic())
                mode = record.stall_phase or ("in_tool" if record.in_tool else "idle")
                diagnostic = (
                    f"stall interrupt requested: quiet={record.stall_quiet_seconds or 0.0:.0f}s "
                    f"threshold={record.stall_threshold_seconds or 0.0:.0f}s mode={mode} "
                    f"grace={_STALL_GRACE_SECONDS:.0f}s (force-finalize in ~{remaining:.0f}s)")
            elif record.stall_interrupt_requested:
                mode = record.stall_phase or ("in_tool" if record.in_tool else "idle")
                diagnostic = (
                    f"stall interrupt requested: quiet={record.stall_quiet_seconds or 0.0:.0f}s "
                    f"threshold={record.stall_threshold_seconds or 0.0:.0f}s mode={mode} "
                    f"grace={_STALL_GRACE_SECONDS:.0f}s")
            elif record.request_stall_timeout_seconds is not None and record.state is not SubagentState.PENDING:
                diagnostic = (f"stall monitor armed: threshold={record.request_stall_timeout_seconds:.0f}s "
                              f"(idle), in-tool ceiling={_STALL_IN_TOOL_SECONDS:.0f}s")
            return SubagentStatus(record.handle, record.state, record.updated_at, diagnostic)

    def wait(self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None) -> SubagentTerminalState:
        record = self._record(handle)
        if record is None:
            return SubagentTerminalState(handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE")
        # Wait on the per-record terminal_event, NOT the runner Future: a record
        # force-finalized by the stall monitor has no completing Future to wait on, and a
        # blocked runner must never hold callers hostage. With a finite timeout the caller
        # gets the current state and ``timed_out=True`` — this limits the CALLER's wait,
        # never the child's.
        if not record.terminal_event.wait(timeout_seconds):
            with _REGISTRY.lock:
                return SubagentTerminalState(record.handle, record.state, False, True)
        with _REGISTRY.lock:
            return SubagentTerminalState(record.handle, record.state, record.result is not None)

    def cancel(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        record = self._record(handle)
        if record is None:
            return SubagentCancelResult(False, unknown_handle=True)
        with _REGISTRY.lock:
            if record.result is not None:
                return SubagentCancelResult(False, already_terminal=True, state=record.state)
            agent = record.agent
            record.state = SubagentState.CANCEL_REQUESTED
            record.updated_at = time.time()
            # Cancellation-vs-stall precedence is resolved at publication, under the same
            # lock: a record whose CANCEL_REQUESTED won the state must never be republished
            # as FAILED/STALLED by the stall monitor, and vice versa.
            record.stall_cancel_precedence = True
        accepted = False
        if agent is not None:
            with contextlib.suppress(Exception):
                accepted = request_hard_interrupt(
                    agent, f"Lifecycle cancellation requested: {reason[:500]}", tool_reason="subagent cancellation requested",
                )
        return SubagentCancelResult(bool(accepted), unsupported=not accepted, state=SubagentState.CANCEL_REQUESTED)

    def result(self, handle: SubagentHandle) -> SubagentResult:
        record = self._record(handle)
        if record is None:
            return SubagentResult(handle, SubagentState.UNKNOWN, False, error_classification="UNKNOWN_HANDLE")
        with _REGISTRY.lock:
            return record.result or SubagentResult(record.handle, record.state, False, error_classification="NOT_READY")

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            return SubagentReconnectResult(False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE")
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def _record(self, handle: SubagentHandle) -> Optional[_Record]:
        """Registry record for a well-formed, capability-verified handle owned by the active parent."""
        if not _handle_is_well_formed(handle):
            return None
        expected = self._capability(handle.subagent_id, handle.parent_session_id, handle.created_at)
        if not hmac.compare_digest(handle.capability, expected):
            return None
        if _session_id_of(self._parent_agent_resolver()) != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            return _REGISTRY.records.get(handle.subagent_id)

    @staticmethod
    def _cleanup_locked() -> None:
        """Retain terminal snapshots for a bounded period, never live work."""
        cutoff = time.time() - _TERMINAL_RETENTION_SECONDS
        expired = [
            sid for sid, record in _REGISTRY.records.items()
            if record.result is not None and record.completed_at is not None and record.completed_at < cutoff
        ]
        for subagent_id in expired:
            handle = _REGISTRY.records.pop(subagent_id).handle
            if handle.correlation_id:
                _REGISTRY.correlations.pop((handle.parent_session_id, handle.correlation_id), None)

    def _run(self, record: _Record, goal: str, parent: Any) -> None:
        with _REGISTRY.lock:
            if record.state is not SubagentState.CANCEL_REQUESTED:
                record.state = SubagentState.RUNNING
            record.started_at = record.updated_at = time.time()
            self._stamp_progress_locked(record)
        try:
            from tools.delegate_tool import _run_child_lifecycle
            raw = _run_child_lifecycle(0, goal, record.agent, parent)
            is_dict = isinstance(raw, dict)
            raw = raw if is_dict else {}
            status = str(raw.get("status", "error"))
            if status == "interrupted":
                state = SubagentState.CANCELLED if record.state == SubagentState.CANCEL_REQUESTED else SubagentState.INTERRUPTED
            else:
                state = SubagentState.SUCCEEDED if status == "completed" else SubagentState.FAILED
            fields: dict[str, Any] = dict(
                summary=_clip(raw.get("summary")), error_message=_clip(raw.get("error") or None),
                error_classification=None if state == SubagentState.SUCCEEDED else status.upper(),
                usage_metadata={"api_calls": raw.get("api_calls", 0)} if is_dict else {},
                tool_execution_summary={"duration_seconds": raw.get("duration_seconds", 0)} if is_dict else {},
            )
        except Exception as exc:
            state = SubagentState.FAILED
            fields = dict(error_classification=type(exc).__name__, error_message=_clip(exc))
        result = SubagentResult(record.handle, state, True, started_at=record.started_at, completed_at=time.time(), **fields)
        result = SubagentLifecycleService._with_result_hash(result)
        # Single-writer publication: a force-finalized record ignores this late return.
        SubagentLifecycleService._publish_terminal(record, result)

    @staticmethod
    def _capability(subagent_id: str, parent_session_id: Optional[str], created_at: float) -> str:
        value = f"{subagent_id}|{parent_session_id or ''}|{created_at:.6f}".encode()
        return hmac.new(_SECRET, value, hashlib.sha256).hexdigest()

    @staticmethod
    def _validate_request(request: SubagentLaunchRequest, parent: Any) -> None:
        for rejected, message in _REQUEST_REJECTIONS:
            if rejected(request):
                raise SubagentLifecycleError(message)
        try:
            metadata_bytes = len(json.dumps(dict(request.metadata), sort_keys=True).encode())
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("metadata must be JSON-serializable.") from exc
        if metadata_bytes > _MAX_METADATA_BYTES:
            raise SubagentLifecycleError("metadata exceeds 8192 bytes.")
        if not request.allowed_toolsets:
            return
        from toolsets import TOOLSETS
        unknown = set(request.allowed_toolsets) - set(TOOLSETS)
        if unknown:
            raise SubagentLifecycleError(f"Unknown toolsets: {', '.join(sorted(unknown))}.")
        enabled = getattr(parent, "enabled_toolsets", None)
        if enabled is not None and not set(request.allowed_toolsets).issubset(set(enabled)):
            raise SubagentLifecycleError("Requested toolsets would broaden parent permissions.")
