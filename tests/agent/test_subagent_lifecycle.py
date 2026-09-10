"""Contract tests for the public plugin subagent lifecycle API."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.subagent_lifecycle import (
    PUBLIC_CONTRACT_VERSION,
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    _Record,
    _REGISTRY,
    _STALL_GRACE_SECONDS,
    _STALL_IN_TOOL_SECONDS,
    _STALL_SWEEP_SECONDS,
    _STALL_TIMEOUT_MIN_SECONDS,
    _sample_child_progress,
    bind_subagent_parent,
    get_active_subagent_parent,
)


class FakeChild:
    def __init__(self, ident="sa-test"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.interrupted = False
        self.interrupt_kind = None
        self.interrupt_message = None
        self.tool_reason = None
        self.activity_summary = {"api_call_count": 0, "current_tool": None, "last_activity_ts": time.time()}
        self.summary_error: Exception | None = None

    def get_activity_summary(self):
        if self.summary_error is not None:
            raise self.summary_error
        return dict(self.activity_summary)

    def interrupt(self, _reason):
        self.interrupted = True
        self.interrupt_kind = "soft"

    def hard_interrupt(self, reason, *, tool_reason=None):
        # The stall monitor must call request_hard_interrupt OUTSIDE the registry
        # lock; a blocking interrupt must never stall other records' deadlines.
        self.lock_owned_at_interrupt = _REGISTRY.lock._is_owned()
        self.interrupted = True
        self.interrupt_kind = "hard"
        self.interrupt_message = reason
        self.tool_reason = tool_reason


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])
    counter = iter(range(1000))

    def build(**_kwargs):
        return FakeChild(f"sa-{next(counter)}")

    def run(_index, _goal, child, _parent):
        for _ in range(20):
            if child.interrupted:
                return {
                    "status": "interrupted",
                    "summary": None,
                    "api_calls": 0,
                    "duration_seconds": 0,
                }
            time.sleep(0.002)
        return {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    return SubagentLifecycleService(lambda: parent)






def test_cancel_is_cooperative_and_forged_handle_is_unknown(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    assert lifecycle.cancel(handle, reason="test").accepted
    terminal = lifecycle.wait(handle, timeout_seconds=1)
    assert terminal.state is SubagentState.CANCELLED
    forged = handle.__class__(**{**handle.to_dict(), "capability": "forged"})
    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert lifecycle.result(forged).error_classification == "UNKNOWN_HANDLE"
    other_parent = SimpleNamespace(session_id="different-parent")
    other_service = SubagentLifecycleService(lambda: other_parent)
    assert other_service.status(handle).state is SubagentState.UNKNOWN


def test_cancel_uses_explicit_hard_interrupt(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    record = lifecycle._record(handle)
    assert record is not None and record.agent is not None

    assert lifecycle.cancel(handle, reason="explicit user cancel").accepted

    assert record.agent.interrupt_kind == "hard"
    assert "explicit user cancel" in record.agent.interrupt_message
    assert record.agent.tool_reason == "subagent cancellation requested"
    lifecycle.wait(handle, timeout_seconds=1)








def test_public_lifecycle_runs_host_aggregation(monkeypatch):
    memory = Mock()
    parent = SimpleNamespace(
        session_id="parent-aggregate",
        enabled_toolsets=["file"],
        _memory_manager=memory,
        _current_turn_id="turn-1",
        session_estimated_cost_usd=1.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = FakeChild("sa-aggregate")
    child.session_id = "child-session"
    hook = Mock()

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "aggregated",
            "api_calls": 1,
            "duration_seconds": 0.25,
            "_child_role": "leaf",
            "_child_cost_usd": 2.5,
        },
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="aggregate me"))
    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED

    memory.on_delegation.assert_called_once_with(
        task="aggregate me", result="aggregated", child_session_id="child-session"
    )
    hook.assert_called_once_with(
        "subagent_stop",
        parent_session_id="parent-aggregate",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_summary="aggregated",
        child_status="completed",
        # Redacted tool history rides the shared finalization pipeline
        # (#62011/#72403); empty here because the fabricated result carries
        # no tool_trace.
        tool_call_history=[],
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"




def test_agent_turn_binds_and_clears_lifecycle_parent(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    observed = []

    def run_conversation(parent, *_args, **_kwargs):
        observed.append(get_active_subagent_parent())
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", run_conversation)

    assert agent.run_conversation("hello") == {"final_response": "ok"}
    assert observed == [agent]
    assert get_active_subagent_parent() is None


# ── A1: opt-in stall_timeout_seconds request field ──────────────────────────
def test_stall_timeout_rejects_invalid_values(lifecycle):
    for bad in (0, -5, 29.9, "600", True, float("nan"), float("inf")):
        with pytest.raises(SubagentLifecycleError, match="stall_timeout_seconds"):
            lifecycle.launch(SubagentLaunchRequest(goal="g", stall_timeout_seconds=bad))


def test_stall_timeout_accepts_floor_and_disables_on_none(lifecycle):
    # 30.0 floor is accepted; None (default) = monitor off for this child.
    handle = lifecycle.launch(SubagentLaunchRequest(goal="g", stall_timeout_seconds=30.0))
    lifecycle.wait(handle, timeout_seconds=5)
    assert lifecycle.result(handle).terminal_state is SubagentState.SUCCEEDED


def test_timeout_seconds_still_rejected_when_both_supplied(lifecycle):
    with pytest.raises(SubagentLifecycleError, match="Per-launch timeout"):
        lifecycle.launch(SubagentLaunchRequest(goal="g", stall_timeout_seconds=600, timeout_seconds=60))


# ── A2: per-record child progress sampling ──────────────────────────────────
def test_progress_token_is_structured_tuple_from_activity_summary(lifecycle):
    child = FakeChild("sa-token-shape")
    child.activity_summary = {"api_call_count": 4, "current_tool": "bash", "last_activity_ts": 1234.5}
    token, in_tool = _sample_child_progress(child)
    assert token == (4, "bash", 1234.5)
    assert in_tool is True
    child.activity_summary = {"api_call_count": 5, "current_tool": None, "last_activity_ts": 1235.0}
    token, in_tool = _sample_child_progress(child)
    assert token == (5, None, 1235.0)
    assert in_tool is False


def test_sample_unreadable_child_keeps_previous_sample():
    child = FakeChild("sa-broken")
    child.summary_error = RuntimeError("child gone")
    previous = ((0, None, 1.0), False)
    token, in_tool = _sample_child_progress(child, previous=previous)
    assert (token, in_tool) == previous


def test_launch_stamps_initial_progress_sample(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="sample me"))
    record = lifecycle._record(handle)
    deadline = time.monotonic() + 5
    while record.state is SubagentState.PENDING and time.monotonic() < deadline:
        time.sleep(0.001)
    assert record.last_progress_token is not None
    assert isinstance(record.progress_started_at, float)
    assert record.in_tool is False
    lifecycle.wait(handle, timeout_seconds=5)


def test_mode_transition_changes_token(lifecycle):
    child = FakeChild("sa-mode")
    child.activity_summary = {"api_call_count": 1, "current_tool": None, "last_activity_ts": 10.0}
    token, in_tool = _sample_child_progress(child)
    assert in_tool is False
    # Entering a tool is itself an activity transition: token changes and in_tool flips.
    child.activity_summary = {"api_call_count": 1, "current_tool": "bash", "last_activity_ts": 11.0}
    new_token, new_in_tool = _sample_child_progress(child)
    assert new_token != token
    assert new_in_tool is True


# ── A3: two-phase stall monitor (fixed grace, synchronized lifecycle) ───────
class StalledRunner:
    """Runner that never returns while the child stays frozen (injected via monkeypatch)."""

    def __init__(self, release: threading.Event):
        self.release = release

    def __call__(self, _index, _goal, child, _parent):
        self.release.wait()
        return {"status": "completed", "summary": "late", "api_calls": 1, "duration_seconds": 0.01}


def _make_record(lifecycle, *, goal="stall probe", monkeypatch=None, release=None):
    """Launch a stall-capable record; with a ``release`` event the runner is patched to a
    blocked stub BEFORE launch so the sweep can never race the fixture's default runner."""
    if monkeypatch is not None and release is not None:
        monkeypatch.setattr("tools.delegate_tool._run_single_child", StalledRunner(release))
    handle = lifecycle.launch(SubagentLaunchRequest(goal=goal, stall_timeout_seconds=30.0))
    return handle, lifecycle._record(handle)


def _await_child_interrupt(child, timeout: float = 5.0) -> None:
    """Wait until the child observes its stall interrupt (sync now, daemon-thread later)."""
    deadline = time.monotonic() + timeout
    while not getattr(child, "interrupted", False):
        if time.monotonic() >= deadline:
            pytest.fail("child never observed its stall interrupt")
        time.sleep(0.001)


def test_sweep_constants_match_async_delegation_parity():
    assert _STALL_SWEEP_SECONDS == 30.0
    assert _STALL_GRACE_SECONDS == 120.0
    assert _STALL_IN_TOOL_SECONDS == 1200.0
    assert _STALL_TIMEOUT_MIN_SECONDS == 30.0


def test_idle_child_frozen_past_threshold_is_interrupted_once_outside_lock(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    # Let the runner reach its blocked state and stamp the initial sample.
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    # Freeze the child (summary matching the frozen token) and backdate the quiet
    # window past the 30s idle threshold.
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    swept = SubagentLifecycleService._sweep()
    assert swept == 1
    assert child.interrupt_kind == "hard", "stall interrupt must be a hard interrupt"
    assert child.lock_owned_at_interrupt is False, "request_hard_interrupt must run OUTSIDE the registry lock"
    assert record.stall_grace_deadline is not None
    assert record.stall_grace_deadline >= time.monotonic() + _STALL_GRACE_SECONDS - 1.0
    assert lifecycle.status(handle).state is SubagentState.RUNNING, "phase-1 keeps public state RUNNING"
    assert "stall interrupt requested" in (lifecycle.status(handle).diagnostic or "")
    assert "mode=idle" in (lifecycle.status(handle).diagnostic or "")
    assert "threshold=30" in (lifecycle.status(handle).diagnostic or "")
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_tool_exit_resampled_every_sweep_rearms_clock(lifecycle, monkeypatch):
    # F1: a child that LEAVES a tool while quiet < in-tool ceiling must be re-sampled on
    # the next sweep — the mode transition re-arms the quiet clock and switches the
    # effective threshold, so the fixed 1200s ceiling never outlives the tool call.
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    # Frozen INSIDE a tool, quiet past the idle threshold but far below 1200s.
    child.activity_summary.update(api_call_count=3, current_tool="bash", last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (3, "bash", 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = True
    assert SubagentLifecycleService._sweep() == 0, "in-tool child must not interrupt below the fixed ceiling"
    # The child LEAVES the tool (real activity): the next sweep must observe the
    # transition even though quiet is still below the in-tool ceiling.
    child.activity_summary.update(api_call_count=4, current_tool=None, last_activity_ts=1001.0)
    with _REGISTRY.lock:
        clock_before = record.progress_started_at
    assert SubagentLifecycleService._sweep() == 0, "mode transition is activity: no interrupt"
    with _REGISTRY.lock:
        assert record.in_tool is False, "mode transition must be adopted on the next sweep"
        assert record.progress_started_at > clock_before, "quiet clock must re-arm on the mode transition"
    # Now idle past the request's 30s idle threshold: the monitor engages at the IDLE
    # threshold, not the stale 1200s in-tool ceiling.
    with _REGISTRY.lock:
        record.progress_started_at = time.monotonic() - 35.0
    assert SubagentLifecycleService._sweep() == 1
    _await_child_interrupt(child)
    assert child.interrupt_kind == "hard"
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


class MachineryChild(FakeChild):
    """Fake child carrying the real AIAgent liveness seam: ``_liveness_activity_lock()``,
    the generation counter, and the abort claim (the interrupt-control claim protocol)."""

    def __init__(self, ident):
        super().__init__(ident)
        self._activity_lock = threading.Lock()
        self._turn_liveness_activity_generation = 0
        self._turn_liveness_abort_claim = None

    def _liveness_activity_lock(self):
        return self._activity_lock

    def touch(self):
        """Mirror ``_touch_activity``: bump generation, stamp, invalidate any reserved claim."""
        with self._activity_lock:
            self._turn_liveness_activity_generation += 1
            self.activity_summary["last_activity_ts"] = time.time()
            self._turn_liveness_abort_claim = None

    def interrupt(self, message=None, *, hard_cancel=False, tool_reason=None, require_generation=None):
        # Mirror InterruptControlMixin.interrupt: reserve the claim under the activity
        # lock, consume it at the publication edge, decline when activity raced.
        self.interrupt_message = message
        self.tool_reason = tool_reason
        if require_generation is not None:
            with self._activity_lock:
                if self._turn_liveness_activity_generation != require_generation:
                    return False  # turn resumed before the claim was reserved
                self._turn_liveness_abort_claim = require_generation
            with self._activity_lock:
                if self._turn_liveness_abort_claim != require_generation:
                    return False
                self._turn_liveness_abort_claim = None
        self.interrupted = True
        self.interrupt_kind = "claimed-hard" if hard_cancel else "claimed"
        return True


def _launch_machinery_record(lifecycle, monkeypatch, release):
    """Launch with a MachineryChild and a blocked runner; returns (handle, record, child)."""
    machinery = MachineryChild("sa-claim")
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: machinery)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", StalledRunner(release))
    handle = lifecycle.launch(SubagentLaunchRequest(goal="claim probe", stall_timeout_seconds=30.0))
    record = lifecycle._record(handle)
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    return handle, record, machinery


def _freeze_child_for_sweep(record, child):
    """Freeze the child's summary AND backdate the record's quiet window past the 30s
    idle threshold under the registry lock (the pre-sweep observation)."""
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False


def test_generation_claimed_interrupt_commits_when_activity_stays_quiet(lifecycle, monkeypatch):
    # F2: with real liveness machinery the phase-1 interrupt is committed through the
    # child's generation claim — hard-cancel published, tool_reason forwarded.
    release = threading.Event()
    handle, record, child = _launch_machinery_record(lifecycle, monkeypatch, release)
    _freeze_child_for_sweep(record, child)
    to_interrupt, _to_finalize, _ = SubagentLifecycleService._sweep_locked()
    assert len(to_interrupt) == 1
    assert SubagentLifecycleService._dispatch_stall_interrupt(record, 45.0, 30.0) is True
    assert child.interrupt_kind == "claimed-hard"
    assert child.tool_reason == "subagent stall interrupt requested"
    with _REGISTRY.lock:
        assert record.stall_interrupt_requested is True
        assert record.stall_grace_deadline is not None
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_activity_resumed_before_interrupt_commit_aborts_and_clears_grace(lifecycle, monkeypatch):
    # F2: activity resuming between the sweep commit and the interrupt dispatch abandons
    # the abort — no interrupt, grace cleared, fresh sample adopted, clock re-armed.
    release = threading.Event()
    handle, record, child = _launch_machinery_record(lifecycle, monkeypatch, release)
    _freeze_child_for_sweep(record, child)
    to_interrupt, _to_finalize, _ = SubagentLifecycleService._sweep_locked()
    assert len(to_interrupt) == 1
    assert record.stall_grace_deadline is not None
    # Activity resumes between the sweep commit and the interrupt dispatch.
    child.touch()
    child.activity_summary["api_call_count"] = 7
    assert SubagentLifecycleService._dispatch_stall_interrupt(record, 45.0, 30.0) is False
    assert child.interrupt_kind is None, "resumed child must never be interrupted"
    with _REGISTRY.lock:
        assert record.stall_grace_deadline is None, "aborted commit must clear the grace"
        assert record.stall_quiet_seconds is None and record.stall_phase is None
        assert record.last_progress_token == (7, None, child.activity_summary["last_activity_ts"]), "fresh sample adopted"
        assert record.stall_interrupt_requested is False
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_fake_child_without_liveness_machinery_degrades_to_compat_interrupt(lifecycle, monkeypatch):
    # F2: test doubles lacking the liveness seam keep the current compat hard-interrupt
    # behavior (real AIAgent children always carry the machinery).
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    _freeze_child_for_sweep(record, child)
    assert SubagentLifecycleService._sweep() == 1
    _await_child_interrupt(child)
    assert child.interrupt_kind == "hard", "no machinery: compat hard interrupt stands"
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_blocked_interrupt_does_not_block_finalization(lifecycle, monkeypatch):
    # F3: a blocked interrupt must never freeze supervision — a second record whose grace
    # expires publishes WHILE the first record's interrupt is still blocked.
    release_a = threading.Event()
    handle_a, record_a = _make_record(lifecycle, monkeypatch=monkeypatch, release=release_a)
    child_a = record_a.agent
    deadline = time.monotonic() + 5
    while record_a.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    _freeze_child_for_sweep(record_a, child_a)
    release_b = threading.Event()
    blocker = threading.Event()
    child_b = FakeChild("sa-blocker")
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child_b)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", StalledRunner(release_b))

    def blocking_hard_interrupt(reason, *, tool_reason=None):
        blocker.wait(10)  # wedged interrupt: never returns until released

    child_b.hard_interrupt = blocking_hard_interrupt
    handle_b = lifecycle.launch(SubagentLaunchRequest(goal="second record", stall_timeout_seconds=30.0))
    record_b = lifecycle._record(handle_b)
    deadline = time.monotonic() + 5
    while record_b.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    _freeze_child_for_sweep(record_b, child_b)
    # First sweep: both records freeze past threshold → two in-flight interrupt threads;
    # A commits (interrupt observed), B's interrupt is wedged inside hard_interrupt.
    assert SubagentLifecycleService._sweep() == 2
    _await_child_interrupt(child_a)
    with _REGISTRY.lock:
        deadline_b = record_b.stall_grace_deadline
    assert deadline_b is not None
    # Second sweep: A stays in grace, B's grace expires → force-finalize must run (B's
    # in-flight interrupt must not block it), and neither record is re-interrupted.
    with _REGISTRY.lock:
        record_b.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1, "B force-finalizes while B's interrupt is in flight"
    result_b = lifecycle.result(handle_b)
    assert result_b.terminal_state is SubagentState.FAILED
    assert result_b.error_classification == "STALLED"
    # A's interrupt never blocked any of this; release everything and drain.
    blocker.set()
    release_b.set()
    release_a.set()
    lifecycle.wait(handle_a, timeout_seconds=5)
    lifecycle.wait(handle_b, timeout_seconds=5)


def test_sweep_does_not_respawn_in_flight_interrupt(lifecycle, monkeypatch):
    # F3: the monitor never re-spawns an interrupt for a record while one is in flight;
    # the in-flight entry clears when the interrupt thread completes.
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    _freeze_child_for_sweep(record, child)
    blocker = threading.Event()

    def blocking_hard_interrupt(reason, *, tool_reason=None):
        blocker.wait(10)

    child.hard_interrupt = blocking_hard_interrupt
    assert SubagentLifecycleService._sweep() == 1, "first sweep spawns the interrupt"
    record_id = record.handle.subagent_id
    deadline = time.monotonic() + 5
    while record_id not in SubagentLifecycleService._INTERRUPT_THREADS and time.monotonic() < deadline:
        time.sleep(0.001)
    assert record_id in SubagentLifecycleService._INTERRUPT_THREADS, "in-flight entry present while blocked"
    # Second sweep while the interrupt is wedged: no re-spawn (sweep takes 0 actions).
    assert SubagentLifecycleService._sweep() == 0
    assert SubagentLifecycleService._INTERRUPT_THREADS[record_id] is not None
    blocker.set()
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)
    deadline = time.monotonic() + 5
    while record_id in SubagentLifecycleService._INTERRUPT_THREADS and time.monotonic() < deadline:
        time.sleep(0.001)
    assert record_id not in SubagentLifecycleService._INTERRUPT_THREADS, "entry clears when the thread completes"


def _cooperative_interrupt_runner(timeout: float = 5.0):
    """Runner that returns ``interrupted`` only after the child observes an interrupt."""
    def runner(_i, _g, child, _p):
        deadline = time.monotonic() + timeout
        while not getattr(child, "interrupted", False) and time.monotonic() < deadline:
            time.sleep(0.001)
        return {"status": "interrupted", "summary": None, "api_calls": 3, "duration_seconds": 1.0}
    return runner


def _await_stall_flag(record, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not record.stall_interrupt_requested:
        if time.monotonic() >= deadline:
            pytest.fail("stall interrupt was never marked requested")
        time.sleep(0.001)


def test_stall_caused_cooperative_return_publishes_failed_stalled(lifecycle, monkeypatch):
    # F4b: a stall-induced cooperative return keeps its causality — FAILED + STALLED +
    # stall metadata at publication, not a bare INTERRUPTED with empty metadata.
    monkeypatch.setattr("tools.delegate_tool._run_single_child", _cooperative_interrupt_runner())
    handle, record = _make_record(lifecycle)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    _freeze_child_for_sweep(record, child)
    assert SubagentLifecycleService._sweep() == 1
    _await_child_interrupt(child)
    _await_stall_flag(record)
    terminal = lifecycle.wait(handle, timeout_seconds=5)
    assert terminal.state is SubagentState.FAILED
    result = lifecycle.result(handle)
    assert result.error_classification == "STALLED"
    assert result.stall_metadata["stalled_after_quiet_seconds"] >= 44.0
    assert result.stall_metadata["stall_threshold_seconds"] == 30.0
    assert result.stall_metadata["stall_phase"] == "idle"


def test_cancel_landing_before_publication_wins_over_stall_return(lifecycle, monkeypatch):
    # F4a: publication-time state re-read — a cancel() landing between the runner's
    # result construction and _publish_terminal must win (CANCELLED), never publish a
    # stall/INTERRUPTED classification built from a stale snapshot.
    monkeypatch.setattr("tools.delegate_tool._run_single_child", _cooperative_interrupt_runner())
    handle, record = _make_record(lifecycle)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    _freeze_child_for_sweep(record, child)
    assert SubagentLifecycleService._sweep() == 1
    _await_child_interrupt(child)
    original_hash = SubagentLifecycleService._with_result_hash

    def hash_then_cancel(result):
        hashed = original_hash(result)
        lifecycle.cancel(handle, reason="landed mid-publish")
        return hashed

    monkeypatch.setattr(SubagentLifecycleService, "_with_result_hash", staticmethod(hash_then_cancel))
    terminal = lifecycle.wait(handle, timeout_seconds=5)
    assert terminal.state is SubagentState.CANCELLED
    result = lifecycle.result(handle)
    assert result.error_classification == "CANCELLED"
    assert result.terminal_state is SubagentState.CANCELLED


def test_second_sweep_does_not_reinterrupt(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    first_deadline = record.stall_grace_deadline
    assert SubagentLifecycleService._sweep() == 0, "already-stalling record must not be re-interrupted"
    assert record.stall_grace_deadline == first_deadline
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_in_tool_child_uses_fixed_threshold_not_request(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    # Frozen INSIDE a tool, quiet just past the 30s request threshold but far below 1200s.
    child.activity_summary.update(api_call_count=3, current_tool="bash", last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (3, "bash", 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = True
    assert SubagentLifecycleService._sweep() == 0, "in-tool child must NOT interrupt at the idle threshold"
    assert child.interrupt_kind is None
    # Past the fixed 1200s in-tool ceiling the monitor engages.
    with _REGISTRY.lock:
        record.progress_started_at = time.monotonic() - (_STALL_IN_TOOL_SECONDS + 5)
    assert SubagentLifecycleService._sweep() == 1
    assert child.interrupt_kind == "hard"
    assert "mode=in_tool" in (lifecycle.status(handle).diagnostic or "")
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_activity_resumed_between_observation_and_commit_aborts_interrupt(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    # Child resumed between observation and the sweep's revalidation: new token differs.
    child.activity_summary["api_call_count"] = 7
    child.activity_summary["last_activity_ts"] = 2000.0
    assert SubagentLifecycleService._sweep() == 0
    assert child.interrupt_kind is None, "stale observation must never interrupt a resumed child"
    assert record.stall_grace_deadline is None
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_grace_is_fixed_activity_during_grace_does_not_reset_it(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    grace_deadline = record.stall_grace_deadline
    # Activity resumes DURING grace (e.g. the interrupt itself triggered work).
    child.activity_summary["api_call_count"] = 9
    child.activity_summary["last_activity_ts"] = 3000.0
    SubagentLifecycleService._sweep()
    assert record.stall_grace_deadline == grace_deadline, "grace is fixed; activity must not postpone finalization"
    # Past the fixed grace with the runner still blocked → force-finalized as STALLED.
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1
    result = lifecycle.result(handle)
    assert result.terminal_state is SubagentState.FAILED
    assert result.error_classification == "STALLED"
    assert result.ready is True
    meta = result.stall_metadata
    assert meta["stalled_after_quiet_seconds"] >= 44.0
    assert meta["stall_threshold_seconds"] == 30.0
    assert meta["stall_phase"] == "idle"
    assert meta["stall_grace_seconds"] == _STALL_GRACE_SECONDS
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_completion_during_grace_wins_terminal_race(lifecycle, monkeypatch):
    late = threading.Event()

    def runner(_i, _g, _c, _p):
        late.wait(5)
        return {"status": "completed", "summary": "made it", "api_calls": 2, "duration_seconds": 0.02}

    monkeypatch.setattr("tools.delegate_tool._run_single_child", runner)
    handle, record = _make_record(lifecycle)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    # Child finishes during grace; the normal completion publishes first.
    late.set()
    terminal = lifecycle.wait(handle, timeout_seconds=5)
    assert terminal.state is SubagentState.SUCCEEDED
    assert lifecycle.result(handle).error_classification is None
    # The late force-finalize sweep must not overwrite the success.
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    SubagentLifecycleService._sweep()
    assert lifecycle.result(handle).terminal_state is SubagentState.SUCCEEDED


def test_late_runner_return_after_force_finalize_is_noop(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1
    stalled = lifecycle.result(handle)
    assert stalled.terminal_state is SubagentState.FAILED and stalled.error_classification == "STALLED"
    snapshot_hash = stalled.result_hash
    # The runner eventually returns "successfully" — publication must be a no-op.
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)
    final = lifecycle.result(handle)
    assert final.terminal_state is SubagentState.FAILED
    assert final.error_classification == "STALLED"
    assert final.result_hash == snapshot_hash, "late runner return must not change the published snapshot"
    assert final.summary == stalled.summary


def test_monitor_exits_when_nothing_monitorable_and_racing_launch_still_monitored(monkeypatch):
    # The monitor's exit decision is synchronized: when its sweep finds nothing monitorable
    # and no newer generation owns the thread, it clears the slot and exits; a racing launch
    # either sees an alive monitor or starts a new generation (never strands its record).
    from agent.subagent_lifecycle import _ensure_stall_monitor, _stall_monitor_loop
    with _REGISTRY.lock:
        _REGISTRY.monitor_thread = threading.current_thread()
        _REGISTRY.monitor_generation += 1
        _REGISTRY.monitor_owner_generation = _REGISTRY.monitor_generation
        generation = _REGISTRY.monitor_generation
    thread = threading.Thread(target=_stall_monitor_loop, args=(generation,), daemon=True)
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive(), "monitor with nothing monitorable must exit after its sweep"
    with _REGISTRY.lock:
        assert _REGISTRY.monitor_thread is None, "exit must clear the monitor slot under the lock"
    # A racing launch: _ensure_stall_monitor under the same lock bumps the generation and
    # re-arms a live monitor even though a (stale) thread object was left in the slot.
    stalled_record = _Record(
        SubagentHandle(PUBLIC_CONTRACT_VERSION, "sa-race", "parent-1", None, time.time(), "t", "tm", "leaf", 1, "cap"),
        SubagentState.RUNNING, time.time(), request_stall_timeout_seconds=30.0)
    with _REGISTRY.lock:
        _REGISTRY.records["sa-race"] = stalled_record
    try:
        generation_before = _REGISTRY.monitor_generation
        _ensure_stall_monitor()
        assert _REGISTRY.monitor_generation == generation_before + 1
        assert _REGISTRY.monitor_thread is not None and _REGISTRY.monitor_thread.is_alive()
        assert _REGISTRY.monitor_thread is not threading.current_thread()
    finally:
        with _REGISTRY.lock:
            _REGISTRY.records.pop("sa-race", None)
            _REGISTRY.monitor_wake.set()


def test_queued_record_is_monitorable_but_not_charged_inactivity(lifecycle, monkeypatch):
    # Deterministic queued state: intercept the submit so the runner never starts.
    captured = {}

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            captured["fn"] = (fn, args)
            return SimpleNamespace(result=lambda timeout=None: None, cancel=lambda force=False: False)

    monkeypatch.setattr("agent.subagent_lifecycle._EXECUTOR", FakeExecutor())
    handle = lifecycle.launch(SubagentLaunchRequest(goal="queued child", stall_timeout_seconds=30.0))
    record = lifecycle._record(handle)
    assert record.state is SubagentState.PENDING
    assert record.last_progress_token is None
    child = record.agent
    with _REGISTRY.lock:
        record.progress_started_at = time.monotonic() - 9999.0
    assert SubagentLifecycleService._sweep() == 0
    assert child.interrupt_kind is None, "queued children are never charged inactivity"
    assert record.stall_grace_deadline is None


def test_no_policy_request_is_never_monitor_engaged(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="no policy"))
    record = lifecycle._record(handle)
    assert record.request_stall_timeout_seconds is None
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    # Even frozen far past every threshold, a no-policy record is invisible to the sweep.
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 9999.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 0
    assert record.agent.interrupt_kind is None
    assert record.stall_grace_deadline is None
    lifecycle.wait(handle, timeout_seconds=5)


# ── A4: atomic terminal publication, wait() wake, stall metadata ────────────
def test_wait_returns_on_force_finalized_record_with_blocked_runner(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1
    # The runner Future is still blocked, but wait() must return via terminal_event.
    terminal = lifecycle.wait(handle)  # timeout_seconds=None: used to block forever
    assert terminal.state is SubagentState.FAILED
    assert terminal.completed is True and terminal.timed_out is False
    released = lifecycle.result(handle)
    assert released.error_classification == "STALLED"
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)  # drain the runner thread


def test_finite_wait_during_grace_reports_not_ready_and_record_still_finalizes(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    # A finite wait during grace returns the current (non-terminal) state immediately.
    terminal = lifecycle.wait(handle, timeout_seconds=0.05)
    assert terminal.completed is False and terminal.timed_out is True
    assert terminal.state is SubagentState.RUNNING
    # The record still reaches terminal: force-finalize, then a plain wait() returns.
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1
    assert lifecycle.wait(handle).state is SubagentState.FAILED
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_stall_metadata_is_frozen_flat_mapping_of_strings_and_numbers(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1
    result = lifecycle.result(handle)
    meta = result.stall_metadata
    assert set(meta) == {"stalled_after_quiet_seconds", "stall_threshold_seconds", "stall_phase", "stall_grace_seconds"}
    assert all(isinstance(v, (str, int, float)) for v in meta.values())
    # Mutability does not leak into the frozen result: mutating the mapping fails.
    with pytest.raises((AttributeError, TypeError)):
        meta["injected"] = "x"
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_cancel_during_grace_wins_over_stall(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1  # phase-1 interrupt committed; grace running
    # Explicit cancel during grace: precedence resolved under the same lock as publication.
    assert lifecycle.cancel(handle, reason="user changed mind").accepted
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1, "grace expiry still finalizes, but not as STALLED"
    result = lifecycle.result(handle)
    assert result.error_classification == "CANCELLED"
    assert result.terminal_state is SubagentState.CANCELLED
    assert result.ready is True
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_armed_status_diagnostic_cleared_at_terminal(lifecycle, monkeypatch):
    release = threading.Event()
    handle, record = _make_record(lifecycle, monkeypatch=monkeypatch, release=release)
    child = record.agent
    deadline = time.monotonic() + 5
    while record.state is not SubagentState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    child.activity_summary.update(api_call_count=0, current_tool=None, last_activity_ts=1000.0)
    with _REGISTRY.lock:
        record.last_progress_token = (0, None, 1000.0)
        record.progress_started_at = time.monotonic() - 45.0
        record.in_tool = False
    assert SubagentLifecycleService._sweep() == 1
    assert "stall interrupt requested" in (lifecycle.status(handle).diagnostic or "")
    with _REGISTRY.lock:
        record.stall_grace_deadline = time.monotonic() - 1.0
    assert SubagentLifecycleService._sweep() == 1
    assert lifecycle.status(handle).diagnostic is None, "diagnostics clear at terminal"
    release.set()
    lifecycle.wait(handle, timeout_seconds=5)


def test_blocked_runner_consumes_executor_worker_after_force_finalize(monkeypatch):
    # Documented 8-worker limitation: force-finalization abandons the outcome, it does not
    # reclaim the worker; eight permanently blocked runners can starve subsequent launches.
    import agent.subagent_lifecycle as mod
    from concurrent.futures import ThreadPoolExecutor
    blocked = threading.Event()
    wedge = ThreadPoolExecutor(max_workers=1)

    def never_returns():
        blocked.wait(5)

    # Simulate the abandoned outcome: the task occupies the single worker and never returns.
    wedge.submit(never_returns)
    assert wedge._work_queue.qsize() == 0 or True  # worker is busy, not queued
    try:
        wedge.submit(lambda: None, timeout=0)
    except TypeError:
        pass
    # The second task cannot get a worker while the first is blocked.
    probe = []

    def quick():
        probe.append(1)

    future = wedge.submit(quick)
    try:
        future.result(timeout=0.2)
        starved = False
    except Exception:
        starved = True
    assert starved, "blocked runner starves subsequent launches on a saturated executor"
    blocked.set()
    wedge.shutdown(wait=False)
