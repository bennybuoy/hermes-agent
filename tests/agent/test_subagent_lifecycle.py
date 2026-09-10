"""Contract tests for the public plugin subagent lifecycle API."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
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
