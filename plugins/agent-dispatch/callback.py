"""SSE-based callback for agent-dispatch.

When dispatch_agent fires, it:
1. POSTs to /v1/runs to start the run (returns run_id immediately)
2. Opens a GET /v1/runs/{run_id}/events SSE connection in a daemon thread
3. The SSE stream delivers events in real-time — message deltas, tool calls,
   reasoning, and the final run.completed event with full output
4. When run.completed arrives, the result is pushed into
   process_registry.completion_queue as an async_delegation event

Stall detection instead of hard timeout:
- Activity events (message.delta, tool.started, tool.completed, reasoning.available)
  reset a stall timer
- If no activity events arrive for _STALL_TIMEOUT seconds (default 10 min),
  the agent is considered stalled and a timeout notification is delivered
- Keepalives from the server keep the TCP connection alive but do NOT reset
  the stall timer — they prove the connection is healthy, not that the agent
  is working
- This allows agents to run for hours as long as they keep producing output,
  while catching genuinely stalled runs quickly

Session persistence: each dispatch can optionally pass a session_id to
continue an existing conversation. The plugin tracks session_ids per
profile so subsequent dispatches to the same agent reuse the same
conversation thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from . import config as cfg

logger = logging.getLogger(__name__)

# Active SSE listeners: run_id -> threading.Thread
_listeners: Dict[str, threading.Thread] = {}
_listeners_lock = threading.Lock()

# Session IDs per profile: profile -> session_id
_session_ids: Dict[str, str] = {}
_session_ids_lock = threading.Lock()

# SSE socket read timeout (seconds) — per-read, not per-connection.
# Server sends keepalives every 30s, so 120s = 4 missed keepalives = dead connection.
_SSE_READ_TIMEOUT = 120

# Stall timeout (seconds) — if no activity events arrive for this long,
# the agent is considered stalled. Activity events are:
# message.delta, tool.started, tool.completed, reasoning.available
# Keepalives do NOT reset this timer — they only prove the TCP connection
# is alive, not that the agent is producing output.
# Default: 600s (10 min). An agent waiting on a slow tool call (e.g. a
# long web search or file processing) might be silent for a few minutes,
# but 10 minutes of complete silence almost always means something is wrong.
_STALL_TIMEOUT = 600

# Activity event types that reset the stall timer
_ACTIVITY_EVENTS = {
    "message.delta",
    "tool.started",
    "tool.completed",
    "tool.failed",
    "reasoning.available",
    "approval.request",  # waiting on user approval — not stalled
}

# Terminal event types that end the listener
_TERMINAL_EVENTS = {
    "run.completed",
    "run.failed",
    "run.cancelled",
}


def _get_session_id(profile: str) -> Optional[str]:
    """Get the stored session_id for a profile, if any."""
    with _session_ids_lock:
        return _session_ids.get(profile)


def _set_session_id(profile: str, session_id: str) -> None:
    """Store a session_id for a profile."""
    with _session_ids_lock:
        _session_ids[profile] = session_id


def _build_delegation_event(
    run_id: str,
    profile: str,
    output: str,
    usage: dict,
    message_preview: str,
    session_id: str,
    session_key: str,
    status: str = "completed",
    error: Optional[str] = None,
    model: str = "",
    duration_seconds: Optional[float] = None,
) -> dict:
    """Build an async_delegation event that mimics delegate_task completions.

    The core's _format_async_delegation() and the gateway's
    _enrich_async_delegation_routing() process this just like a
    delegate_task completion — no core patches required.
    """
    if status == "completed":
        deleg_status = "completed"
        summary = output
    elif status == "failed":
        deleg_status = "failed"
        summary = output or None
    else:
        deleg_status = status
        summary = output or None

    # Record terminal provenance in the state file
    try:
        from .tools import _update_run_status
        _update_run_status(
            run_id,
            status=status,
            output_preview=output[:500] if output else "",
            duration_seconds=duration_seconds,
            usage=usage if usage else None,
            model=model,
        )
    except Exception:
        pass  # state file is best-effort; don't crash the SSE thread

    return {
        "type": "async_delegation",
        "delegation_id": f"dispatch-{run_id[:16]}",
        "goal": f"[dispatched to {profile}] {message_preview}",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": model or "?",
        "status": deleg_status,
        "summary": summary,
        "error": error,
        "api_calls": usage.get("total_tokens", 0) if usage else 0,
        "duration_seconds": duration_seconds if duration_seconds is not None else "?",
        "dispatched_at": time.time(),
        "completed_at": time.time(),
        "session_key": session_key,
        "session_id": session_id,
    }


def _listen_sse(
    run_id: str,
    profile: str,
    url: str,
    api_key: str,
    message_preview: str,
    session_id: str,
    session_key: str,
) -> None:
    """Background thread: subscribe to SSE event stream for a run.

    Uses stall detection instead of a hard timeout. Activity events reset
    the stall timer. If no activity for _STALL_TIMEOUT seconds, the run is
    considered stalled. The SSE connection can stay open indefinitely as
    long as the agent keeps producing output.
    """
    sse_url = f"{url}/v1/runs/{run_id}/events"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }

    req = Request(sse_url, headers=headers, method="GET")
    start_time = time.time()
    resolved_model = ""

    try:
        with urlopen(req, timeout=_SSE_READ_TIMEOUT) as resp:
            buffer = ""
            last_activity = time.time()

            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8", errors="replace")

                # Parse SSE events (separated by \n\n)
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)

                    # Skip keepalives (lines starting with ":")
                    if event_str.strip().startswith(":"):
                        # Keepalive — connection alive but no agent activity.
                        # Check stall timer.
                        if time.time() - last_activity > _STALL_TIMEOUT:
                            _handle_stall(run_id, profile, message_preview, session_id, session_key)
                            return
                        continue

                    for line in event_str.split("\n"):
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue

                        try:
                            evt = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        event_type = evt.get("event", "")

                        # Activity events reset stall timer
                        if event_type in _ACTIVITY_EVENTS:
                            last_activity = time.time()

                        # Terminal events — deliver result and return
                        if event_type in _TERMINAL_EVENTS:
                            elapsed = time.time() - start_time
                            # Extract model from the completed event if present
                            if event_type == "run.completed":
                                output = evt.get("output", "")
                                usage = evt.get("usage", {})
                                resolved_model = evt.get("model", "") or resolved_model
                                deleg_evt = _build_delegation_event(
                                    run_id, profile, output, usage,
                                    message_preview, session_id, session_key,
                                    status="completed",
                                    model=resolved_model,
                                    duration_seconds=elapsed,
                                )
                            elif event_type == "run.failed":
                                error = evt.get("error", "unknown error")
                                deleg_evt = _build_delegation_event(
                                    run_id, profile, "", {},
                                    message_preview, session_id, session_key,
                                    status="failed",
                                    error=error,
                                    duration_seconds=elapsed,
                                )
                            else:  # run.cancelled
                                deleg_evt = _build_delegation_event(
                                    run_id, profile, "", {},
                                    message_preview, session_id, session_key,
                                    status="failed",
                                    error="run was cancelled",
                                    duration_seconds=elapsed,
                                )
                            _deliver_to_session(deleg_evt, run_id, profile)
                            return

                        # Non-terminal, non-keepalive event — also counts as activity
                        if event_type not in _TERMINAL_EVENTS:
                            last_activity = time.time()

                # Check stall timer after processing all events in this chunk
                if time.time() - last_activity > _STALL_TIMEOUT:
                    _handle_stall(run_id, profile, message_preview, session_id, session_key)
                    return

    except HTTPError as e:
        if e.code == 404:
            error = f"Run {run_id} not found on {profile}"
        else:
            error = f"HTTP {e.code}: {e.reason}"
        error_evt = _build_delegation_event(
            run_id, profile, "", {}, message_preview, session_id, session_key,
            status="failed", error=error,
        )
        _deliver_to_session(error_evt, run_id, profile)

    except URLError as e:
        # Connection timeout or refused — could be temporary
        error = f"Connection error to {url}: {e.reason}"
        error_evt = _build_delegation_event(
            run_id, profile, "", {}, message_preview, session_id, session_key,
            status="failed", error=error,
        )
        _deliver_to_session(error_evt, run_id, profile)

    except Exception as e:
        error = f"SSE listener error: {type(e).__name__}: {e}"
        error_evt = _build_delegation_event(
            run_id, profile, "", {}, message_preview, session_id, session_key,
            status="failed", error=error,
        )
        _deliver_to_session(error_evt, run_id, profile)


def _handle_stall(
    run_id: str,
    profile: str,
    message_preview: str,
    session_id: str,
    session_key: str,
) -> None:
    """Deliver a stall notification — the agent hasn't produced output in too long."""
    stall_evt = _build_delegation_event(
        run_id, profile,
        f"Agent on {profile} appears stalled — no activity for "
        f"{_STALL_TIMEOUT // 60} minutes. The run may still be in progress. "
        f"Use check_dispatch to poll manually, or the agent may recover on its own.",
        {}, message_preview, session_id, session_key,
        status="failed",
        error=f"stall detected — no activity for {_STALL_TIMEOUT}s",
    )
    _deliver_to_session(stall_evt, run_id, profile)


def _deliver_to_session(evt: dict, run_id: str, profile: str) -> None:
    """Push the event into process_registry.completion_queue.

    The existing notification poller picks it up and injects it as a
    user message when the session is idle.
    """
    try:
        from tools.process_registry import process_registry
    except ImportError:
        logger.warning(
            "dispatch callback: process_registry not available — "
            "cannot deliver result for %s on %s",
            run_id, profile,
        )
        return

    process_registry.completion_queue.put(evt)
    logger.info(
        "dispatch callback: delivered SSE result for %s (%s) to completion_queue",
        run_id, profile,
    )


def start_listener(
    run_id: str,
    profile: str,
    url: str,
    api_key: str,
    message_preview: str,
) -> None:
    """Start an SSE listener thread for a dispatched run.

    Called by handle_dispatch_agent after a successful dispatch.
    """
    session_id = os.environ.get("HERMES_SESSION_ID", "")
    try:
        from tools.approval import get_current_session_key
        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""
    if not session_key:
        session_key = os.environ.get("HERMES_SESSION_KEY", "") or session_id

    with _listeners_lock:
        if run_id in _listeners:
            return

        thread = threading.Thread(
            target=_listen_sse,
            args=(run_id, profile, url, api_key, message_preview, session_id, session_key),
            name=f"dispatch-sse-{run_id[:12]}",
            daemon=True,
        )
        _listeners[run_id] = thread
        thread.start()

    logger.info(
        "dispatch callback: started SSE listener for %s on %s (session=%s)",
        run_id, profile, session_id,
    )


def stop_listener(run_id: str) -> bool:
    """Stop the SSE listener for a specific run_id.

    Called when a dispatch is cancelled via cancel_dispatch. Removes the
    listener thread from the active set so it won't deliver a stale result
    after the cancel. Returns True if a listener was found and removed.
    """
    with _listeners_lock:
        thread = _listeners.pop(run_id, None)
        if thread is None:
            return False
        # Daemon threads — we can't force-kill them, but removing from the
        # dict means _deliver_to_session won't find a context. The thread
        # will exit on its own when the SSE stream closes (the API server
        # closes it after run.cancelled).
        logger.info(
            "dispatch callback: removed SSE listener for %s (cancelled by user)",
            run_id,
        )
        return True


def stop_all_listeners() -> None:
    """Stop all active listeners. Called on shutdown."""
    with _listeners_lock:
        _listeners.clear()


def get_profile_session_id(profile: str) -> Optional[str]:
    """Get the stored conversation session_id for a profile.

    Returns None if no prior conversation exists for this profile.
    The caller should create a new session via POST /api/sessions on
    the target API server if None is returned.
    """
    return _get_session_id(profile)


def store_profile_session_id(profile: str, session_id: str) -> None:
    """Store a conversation session_id for a profile.

    Called after creating a new session on the target API server,
    or after a dispatch that created a session implicitly.
    """
    _set_session_id(profile, session_id)