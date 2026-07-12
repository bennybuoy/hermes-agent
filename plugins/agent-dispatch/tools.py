"""Tool schemas and handlers for the agent-dispatch plugin.

Four tools that wrap the Hermes API server's /v1/runs endpoints:
  - dispatch_agent: POST /v1/runs, returns run_id
  - check_dispatch: GET /v1/runs/{run_id}, returns status
  - collect_dispatches: batch GET /v1/runs/{run_id} for multiple runs
  - dispatch_status: list persisted runs from state file

All HTTP is done with urllib.request (stdlib). Handlers are synchronous.
"""

from __future__ import annotations

import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from . import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

DISPATCH_AGENT_SCHEMA: Dict[str, Any] = {
    "name": "dispatch_agent",
    "description": (
        "Dispatch an async task to another Hermes profile's API server. "
        "Returns a run_id handle immediately — the task runs in a separate "
        "session on the target profile with its own model, skills, and memory. "
        "Results are auto-delivered to this session via SSE event stream "
        "— you'll receive a [ASYNC DELEGATION COMPLETE] message when the "
        "task finishes, typically within seconds of completion. "
        "Multiple dispatches can be issued in one turn for parallel execution. "
        "For synchronous multi-turn conversations with session persistence, "
        "use dispatch_chat instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "Profile name from agent_dispatch.profiles config "
                    "(e.g. 'vincent', 'y10', 'qc-engagement')."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The task message to send to the target profile. "
                    "Be specific and self-contained — the target profile "
                    "has no context from this conversation."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional system prompt override for the target session. "
                    "Use to set role/behavior (e.g. 'You are a pedagogical "
                    "reviewer...')."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override for this run. Overrides both "
                    "the target profile's default model and any model set "
                    "in the profile config. Pass e.g. 'gemma4:31b' or "
                    "'glm-5.2'."
                ),
            },
        },
        "required": ["profile", "message"],
    },
}

CHECK_DISPATCH_SCHEMA: Dict[str, Any] = {
    "name": "check_dispatch",
    "description": (
        "Check the status of a dispatched task. Returns result if complete, "
        "or current status if still running."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by dispatch_agent.",
            },
            "profile": {
                "type": "string",
                "description": (
                    "Profile name the task was dispatched to (needed to "
                    "know which API server to poll)."
                ),
            },
        },
        "required": ["run_id", "profile"],
    },
}

COLLECT_DISPATCHES_SCHEMA: Dict[str, Any] = {
    "name": "collect_dispatches",
    "description": (
        "Check multiple dispatched tasks at once. Returns completed results "
        "and a list of still-running run_ids. Call this after dispatching "
        "multiple parallel tasks to gather whatever's done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_ids": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "profile": {"type": "string"},
                    },
                },
                "description": (
                    "List of {run_id, profile} objects to check."
                ),
            },
        },
        "required": ["run_ids"],
    },
}

DISPATCH_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "dispatch_status",
    "description": (
        "List all known dispatched runs (from state file), including ones "
        "from previous sessions. Useful for recovering orchestrator state "
        "after a restart."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

DISPATCH_CHAT_SCHEMA: Dict[str, Any] = {
    "name": "dispatch_chat",
    "description": (
        "Synchronously dispatch a task to another Hermes profile and block "
        "until the reply arrives. Uses the persistent session API endpoint "
        "(/api/sessions/{id}/chat) so conversation history is preserved — "
        "subsequent calls to the same profile continue the same conversation. "
        "The profile remembers prior turns, context, and decisions. "
        "Use this for interactive multi-turn dialogue with a teaching agent. "
        "For fire-and-forget parallel dispatch, use dispatch_agent instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "Profile name from agent_dispatch.profiles config "
                    "(e.g. 'marie', 'ada', 'richard', 'rosalind')."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The message to send to the target profile. "
                    "Be specific and self-contained."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional system prompt override for this turn. "
                    "Note: on subsequent calls to the same profile, the "
                    "system prompt from the first call is retained."
                ),
            },
            "new_session": {
                "type": "boolean",
                "description": (
                    "If true, start a fresh conversation instead of "
                    "continuing the existing session for this profile. "
                    "Default: false (continue existing session)."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override for this call. Overrides both "
                    "the target profile's default model and any model set "
                    "in the profile config. Pass e.g. 'gemma4:31b' or "
                    "'glm-5.2'. Only applies to this call; subsequent calls "
                    "without this param revert to the profile default."
                ),
            },
        },
        "required": ["profile", "message"],
    },
}

CANCEL_DISPATCH_SCHEMA: Dict[str, Any] = {
    "name": "cancel_dispatch",
    "description": (
        "Cancel a running dispatched task by sending a stop request to the "
        "target profile's API server. The agent receives an interrupt signal "
        "and the asyncio task is cancelled. Also stops the local SSE listener "
        "so no stale result is delivered. The run may take a few seconds to "
        "actually stop (the agent finishes its current step first)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by dispatch_agent.",
            },
            "profile": {
                "type": "string",
                "description": (
                    "Profile name the task was dispatched to (needed to "
                    "know which API server to contact)."
                ),
            },
        },
        "required": ["run_id", "profile"],
    },
}


# delegate_subagent schema — defined here (before ALL_SCHEMAS) so the list
# can reference it. The handler and helper functions are at the bottom of
# the file.
DELEGATE_SUBAGENT_SCHEMA: Dict[str, Any] = {
    "name": "delegate_subagent",
    "description": (
        "Spawn an in-process subagent with a per-call model override. "
        "Unlike delegate_task (which inherits the parent model), this tool "
        "lets you pick a different model for the subagent — useful for "
        "fan-out across models (e.g. review code with a fast model while "
        "the parent uses a reasoning model). The subagent runs in the same "
        "process with its own isolated context, terminal session, and "
        "toolset. Only the final summary is returned.\n\n"
        "The model name is resolved leniently via the same /model switch "
        "pipeline — bare names like 'opus', 'gpt-5', 'glm' work, as do "
        "full 'vendor/model' slugs. If the resolved provider differs from "
        "the parent's, the subagent gets fresh credentials for that provider. "
        "If it's the same provider/aggregator, credentials are inherited.\n\n"
        "Runs synchronously — the result is returned when the subagent "
        "finishes. For long tasks, consider dispatch_agent instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the subagent should accomplish. Be specific and "
                    "self-contained — the subagent knows nothing about "
                    "your conversation history."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Model for this subagent (e.g. 'gemma4:31b', 'glm-5.2', "
                    "'opus', 'gpt-5'). Resolved via the model switch pipeline. "
                    "If omitted, inherits the parent model (same as "
                    "delegate_task)."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Background information the subagent needs: file paths, "
                    "error messages, project structure, constraints."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of toolset names for the subagent. If "
                    "omitted, inherits the parent's toolsets (minus blocked "
                    "tools like delegate_task, clarify, memory, etc.)."
                ),
            },
        },
        "required": ["goal"],
    },
}


ALL_SCHEMAS = [
    DISPATCH_AGENT_SCHEMA,
    CHECK_DISPATCH_SCHEMA,
    COLLECT_DISPATCHES_SCHEMA,
    DISPATCH_STATUS_SCHEMA,
    DISPATCH_CHAT_SCHEMA,
    CANCEL_DISPATCH_SCHEMA,
    DELEGATE_SUBAGENT_SCHEMA,
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, api_key: str, body: dict, timeout: float = 30.0) -> dict:
    """POST JSON to the API server. Returns the parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


def _post_empty(url: str, api_key: str, timeout: float = 10.0) -> dict:
    """POST with no body to the API server (e.g. /stop endpoint). Returns parsed JSON."""
    req = Request(
        url,
        data=b"",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


def _get_json(url: str, api_key: str, timeout: float = 10.0) -> dict:
    """GET from the API server. Returns the parsed JSON response."""
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            return {"run_id": url.rsplit("/", 1)[-1], "status": "not_found"}
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

_MAX_STATE_ENTRIES = 200


def _load_state() -> dict:
    """Load the run-state JSON file. Returns empty structure if missing."""
    path = cfg.get_state_file_path()
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"runs": []}
        return data
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {"runs": []}


def _save_state(data: dict) -> None:
    """Atomically write the run-state JSON file."""
    path = cfg.get_state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Trim old entries
    runs = data.get("runs", [])
    if len(runs) > _MAX_STATE_ENTRIES:
        data["runs"] = runs[-_MAX_STATE_ENTRIES:]
    # Atomic write: temp file + rename
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _persist_run(run_id: str, profile: str, message_preview: str, model: str = "") -> None:
    """Add a run entry to the state file."""
    state = _load_state()
    state.setdefault("runs", []).append({
        "run_id": run_id,
        "profile": profile,
        "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message_preview": message_preview[:120],
        "session_id": os.environ.get("HERMES_SESSION_ID", ""),
        "model": model or "",
        "status": "dispatched",
        "completed_at": "",
        "duration_seconds": None,
        "output_preview": "",
        "usage": {},
    })
    _save_state(state)


def _update_run_status(
    run_id: str,
    status: str,
    output_preview: str = "",
    duration_seconds: Optional[float] = None,
    usage: Optional[dict] = None,
    model: str = "",
) -> None:
    """Update a run entry with terminal status and provenance.

    Called when a dispatch completes (via SSE callback or dispatch_chat reply),
    fails, or is cancelled. Merges the new fields into the existing run record
    without clobbering the dispatch-time metadata.
    """
    state = _load_state()
    runs = state.get("runs", [])
    for run in runs:
        if run.get("run_id") == run_id:
            run["status"] = status
            run["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if output_preview:
                run["output_preview"] = output_preview[:500]
            if duration_seconds is not None:
                run["duration_seconds"] = round(duration_seconds, 2)
            if usage:
                run["usage"] = usage
            if model:
                run["model"] = model
            break
    _save_state(state)


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _tool_error(msg: str) -> str:
    """Return a JSON error string for the tool result."""
    return json.dumps({"status": "error", "error": msg})


def _resolve_profile(profile: str) -> tuple[dict, Optional[str]]:
    """Resolve a profile name to its config dict.

    Returns (config_dict, error_message). If the profile is found,
    error_message is None. If not found, config_dict is an empty dict
    and error_message explains why.
    """
    pcfg = cfg.get_profile_config(profile) or {}
    if not pcfg:
        available = cfg.list_profiles()
        avail_str = ", ".join(available) if available else "(none configured)"
        return {}, (
            f"Profile '{profile}' not found in agent_dispatch.profiles "
            f"config. Available profiles: {avail_str}"
        )
    url = pcfg.get("url", "").strip()
    if not url:
        return {}, f"Profile '{profile}' has no 'url' configured."
    api_key = pcfg.get("api_key", "")
    if not api_key:
        return {}, (
            f"Profile '{profile}' has no 'api_key' configured (or the env "
            f"var it references is not set)."
        )
    return pcfg, None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_dispatch_agent(args: dict, **kwargs) -> str:
    """Dispatch an async task to another Hermes profile."""
    profile = args.get("profile", "").strip()
    message = args.get("message", "")
    instructions = args.get("instructions")
    model_override = args.get("model")

    if not profile:
        return _tool_error("'profile' is required.")
    if not message:
        return _tool_error("'message' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    # Build request body — omit null/empty fields
    body: Dict[str, Any] = {"input": message}
    if instructions:
        body["instructions"] = instructions
    # Model precedence: explicit model arg > profile config model > none (use target default)
    if model_override:
        body["model"] = model_override
    elif pcfg.get("model"):
        body["model"] = pcfg["model"]

    try:
        result = _post_json(
            f"{pcfg['url']}/v1/runs",
            pcfg["api_key"],
            body,
            timeout=30.0,
        )
    except RuntimeError as e:
        # Distinguish auth errors from connection errors
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"agent_dispatch.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"Dispatch to {profile} failed: {msg}")

    run_id = result.get("run_id", "")
    if not run_id:
        return _tool_error(
            f"Unexpected response from {profile}: no run_id in {json.dumps(result)}"
        )

    # Persist for recovery
    _persist_run(run_id, profile, message, model=model_override or pcfg.get("model", ""))

    # Start SSE listener to auto-deliver result when complete.
    # SSE is event-driven (no polling) — one thread blocks on a single
    # HTTP connection until the run completes, then pushes the result
    # into completion_queue as an async_delegation event.
    try:
        from .callback import start_listener
        start_listener(
            run_id=run_id,
            profile=profile,
            url=pcfg["url"],
            api_key=pcfg["api_key"],
            message_preview=message[:120],
        )
    except Exception as e:
        logger.warning("dispatch callback: could not start SSE listener for %s: %s", run_id, e)

    return json.dumps({
        "run_id": run_id,
        "profile": profile,
        "status": "dispatched",
        "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message_preview": message[:120],
        "callback": "SSE — result will be delivered automatically when complete",
    })


def handle_check_dispatch(args: dict, **kwargs) -> str:
    """Check the status of a single dispatched task."""
    run_id = args.get("run_id", "").strip()
    profile = args.get("profile", "").strip()

    if not run_id:
        return _tool_error("'run_id' is required.")
    if not profile:
        return _tool_error("'profile' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    try:
        status = _get_json(
            f"{pcfg['url']}/v1/runs/{run_id}",
            pcfg["api_key"],
            timeout=10.0,
        )
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"agent_dispatch.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"Check dispatch {run_id} on {profile} failed: {msg}")

    # If not_found, check state file for context
    if status.get("status") == "not_found":
        return _tool_error(
            f"Run {run_id} not found on {profile}. It may have been from "
            f"a previous gateway restart (runs are in-memory and don't "
            f"survive a restart). Check dispatch_status for persisted records."
        )

    # Enrich with profile name for the agent
    status["profile"] = profile
    return json.dumps(status)


def handle_collect_dispatches(args: dict, **kwargs) -> str:
    """Check multiple dispatched tasks at once."""
    run_ids = args.get("run_ids", [])
    if not run_ids:
        return _tool_error("'run_ids' is required (list of {run_id, profile} objects).")

    completed: List[dict] = []
    running: List[dict] = []
    failed: List[dict] = []

    for entry in run_ids:
        run_id = entry.get("run_id", "").strip()
        profile = entry.get("profile", "").strip()
        if not run_id or not profile:
            failed.append({
                "run_id": run_id or "?",
                "profile": profile or "?",
                "status": "invalid",
                "error": "Missing run_id or profile",
            })
            continue

        pcfg, err = _resolve_profile(profile)
        if err:
            failed.append({
                "run_id": run_id,
                "profile": profile,
                "status": "config_error",
                "error": err,
            })
            continue

        try:
            status = _get_json(
                f"{pcfg['url']}/v1/runs/{run_id}",
                pcfg["api_key"],
                timeout=10.0,
            )
        except RuntimeError as e:
            failed.append({
                "run_id": run_id,
                "profile": profile,
                "status": "error",
                "error": str(e),
            })
            continue

        status["profile"] = profile

        if status.get("status") == "completed":
            completed.append(status)
        elif status.get("status") in ("failed", "cancelled", "not_found", "invalid"):
            failed.append(status)
        else:
            # queued, running, or any other state
            running.append(status)

    return json.dumps({
        "completed": completed,
        "running": running,
        "failed": failed,
        "summary": {
            "total": len(run_ids),
            "completed": len(completed),
            "running": len(running),
            "failed": len(failed),
        },
    })


def handle_dispatch_status(args: dict, **kwargs) -> str:
    """List all known dispatched runs from the state file."""
    state = _load_state()
    runs = state.get("runs", [])
    return json.dumps({
        "total": len(runs),
        "runs": runs,
    }, indent=2)


def handle_dispatch_chat(args: dict, **kwargs) -> str:
    """Synchronously dispatch a task with session persistence.

    Uses POST /api/sessions/{id}/chat which:
    - Blocks until the agent replies (synchronous)
    - Loads and saves conversation history to state.db
    - Allows multi-turn conversations with the same agent

    Session IDs are tracked per-profile in callback.py so subsequent
    calls continue the same conversation thread.
    """
    profile = args.get("profile", "").strip()
    message = args.get("message", "")
    instructions = args.get("instructions")
    new_session = args.get("new_session", False)
    model_override = args.get("model")

    if not profile:
        return _tool_error("'profile' is required.")
    if not message:
        return _tool_error("'message' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    from .callback import get_profile_session_id, store_profile_session_id

    # Get or create a session for this profile
    session_id = get_profile_session_id(profile)

    if not session_id or new_session:
        # Create a new session on the target API server
        create_body: Dict[str, Any] = {"title": f"Dispatch to {profile}"}
        try:
            create_result = _post_json(
                f"{pcfg['url']}/api/sessions",
                pcfg["api_key"],
                create_body,
                timeout=10.0,
            )
        except RuntimeError as e:
            msg = str(e)
            if "Cannot reach" in msg:
                return _tool_error(
                    f"Cannot reach {profile} API server at {pcfg['url']}. "
                    f"Is the gateway running for that profile? ({msg})"
                )
            return _tool_error(f"Failed to create session on {profile}: {msg}")

        session_obj = create_result.get("session", {})
        session_id = session_obj.get("id", "")
        if not session_id:
            return _tool_error(
                f"Unexpected response creating session on {profile}: {json.dumps(create_result)}"
            )
        store_profile_session_id(profile, session_id)

    # Send the message via the persistent session chat endpoint
    chat_body: Dict[str, Any] = {"message": message}
    if instructions:
        chat_body["system_message"] = instructions
    if model_override:
        chat_body["model"] = model_override

    try:
        result = _post_json(
            f"{pcfg['url']}/api/sessions/{session_id}/chat",
            pcfg["api_key"],
            chat_body,
            timeout=600.0,  # 10 min — agentic tasks with tool calls can take a while
        )
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"agent_dispatch.profiles config. ({msg})"
            )
        if "404" in msg:
            # Session was deleted/expired — clear cache and retry with new session
            store_profile_session_id(profile, "")
            return _tool_error(
                f"Session {session_id} not found on {profile}. It may have "
                f"expired. Retry with new_session=true to start a fresh conversation."
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"dispatch_chat to {profile} failed: {msg}")

    # Extract the reply
    reply = result.get("message", {}).get("content", "")
    result_session_id = result.get("session_id", session_id)
    if result_session_id != session_id:
        store_profile_session_id(profile, result_session_id)

    usage = result.get("usage", {})
    effective_model = model_override or pcfg.get("model", "")

    # Record in state file for provenance
    chat_record_id = f"chat-{profile}-{int(time.time())}"
    state = _load_state()
    state.setdefault("runs", []).append({
        "run_id": chat_record_id,
        "profile": profile,
        "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message_preview": message[:120],
        "session_id": os.environ.get("HERMES_SESSION_ID", ""),
        "model": effective_model,
        "status": "completed",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": None,
        "output_preview": reply[:500] if reply else "",
        "usage": usage,
        "type": "chat",
    })
    _save_state(state)

    return json.dumps({
        "profile": profile,
        "session_id": result_session_id,
        "status": "completed",
        "reply": reply,
        "usage": usage,
    })


def handle_cancel_dispatch(args: dict, **kwargs) -> str:
    """Cancel a running dispatched task via POST /v1/runs/{run_id}/stop."""
    run_id = args.get("run_id", "").strip()
    profile = args.get("profile", "").strip()

    if not run_id:
        return _tool_error("'run_id' is required.")
    if not profile:
        return _tool_error("'profile' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    # Stop the local SSE listener so we don't deliver a stale result
    try:
        from .callback import stop_listener
        listener_removed = stop_listener(run_id)
    except Exception:
        listener_removed = False

    # Send the stop request to the target API server
    try:
        result = _post_empty(
            f"{pcfg['url']}/v1/runs/{run_id}/stop",
            pcfg["api_key"],
            timeout=15.0,
        )
    except RuntimeError as e:
        msg = str(e)
        if "404" in msg:
            # Run not found — might have already completed or been from a
            # previous gateway restart. Still report success since the listener
            # was removed and the run is effectively gone.
            return json.dumps({
                "run_id": run_id,
                "profile": profile,
                "status": "not_found",
                "message": (
                    "Run not found on target — it may have already completed "
                    "or the gateway was restarted. Local SSE listener removed."
                ),
                "listener_removed": listener_removed,
            })
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"agent_dispatch.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"Cancel dispatch {run_id} on {profile} failed: {msg}")

    return json.dumps({
        "run_id": run_id,
        "profile": profile,
        "status": result.get("status", "stopping"),
        "message": (
            "Stop signal sent. The agent will finish its current step "
            "then halt. The run.cancelled event will fire on the SSE stream."
        ),
        "listener_removed": listener_removed,
        "api_response": result,
    })


# ---------------------------------------------------------------------------
# delegate_subagent — in-process subagent with per-call model selection
# (schema is defined at the top of the file, before ALL_SCHEMAS)
# ---------------------------------------------------------------------------


def _resolve_model_creds(model_name: str, parent_agent) -> dict:
    """Resolve a model name to a credential bundle for a subagent.

    Uses the same switch_model pipeline as /model. Returns a dict with
    model, provider, base_url, api_key, api_mode — all resolved from
    the model name. When the resolved provider matches the parent's,
    provider/base_url/api_key/api_mode are None so _build_child_agent
    inherits from the parent.
    """
    name = (model_name or "").strip()
    if not name:
        return {"model": None, "provider": None, "base_url": None,
                "api_key": None, "api_mode": None}

    from hermes_cli.model_switch import switch_model

    parent_provider = getattr(parent_agent, "provider", "") or ""
    parent_model = getattr(parent_agent, "model", "") or ""
    parent_base_url = getattr(parent_agent, "base_url", "") or ""
    parent_api_key = getattr(parent_agent, "api_key", "") or ""

    user_providers = {}
    custom_providers = None
    try:
        from cli import CLI_CONFIG
        user_providers = CLI_CONFIG.get("providers") or {}
        custom_providers = CLI_CONFIG.get("custom_providers")
    except Exception:
        try:
            from hermes_cli.config import load_config
            _full = load_config()
            user_providers = _full.get("providers") or {}
            custom_providers = _full.get("custom_providers")
        except Exception:
            pass

    result = switch_model(
        raw_input=name,
        current_provider=parent_provider,
        current_model=parent_model,
        current_base_url=parent_base_url,
        current_api_key=parent_api_key,
        is_global=False,
        user_providers=user_providers,
        custom_providers=custom_providers,
    )

    if not result.success:
        raise ValueError(
            result.error_message
            or f"Could not resolve model '{name}' for this subagent."
        )

    creds = {
        "model": result.new_model or parent_model,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
    }
    # Only override credentials when the provider actually differs
    if result.target_provider and result.target_provider != parent_provider:
        creds["provider"] = result.target_provider
        creds["base_url"] = result.base_url or None
        creds["api_key"] = result.api_key or None
        creds["api_mode"] = result.api_mode or None
    return creds


def handle_delegate_subagent(args: dict, **kwargs) -> str:
    """Spawn an in-process subagent with a per-call model override.

    This tool uses the core delegate_task internals (_build_child_agent +
    _run_single_child) directly, injecting a per-call model that the
    core delegate_task schema doesn't expose. No core patches required —
    we just call the internals as a library.
    """
    goal = args.get("goal", "")
    model_name = args.get("model", "")
    context = args.get("context")
    toolsets = args.get("toolsets")
    parent_agent = kwargs.get("parent_agent")

    if not goal.strip():
        return _tool_error("'goal' is required.")
    if not parent_agent:
        return _tool_error(
            "delegate_subagent requires a parent agent context "
            "(not available in this mode)."
        )

    # Resolve the model to a credential bundle
    try:
        creds = _resolve_model_creds(model_name, parent_agent)
    except ValueError as e:
        return _tool_error(f"Could not resolve model '{model_name}': {e}")
    except Exception as e:
        return _tool_error(f"Model resolution failed: {type(e).__name__}: {e}")

    # Import core delegation internals
    try:
        from tools.delegate_tool import (
            _build_child_agent,
            _run_single_child,
            _load_config as _load_delegation_config,
            DEFAULT_MAX_ITERATIONS,
        )
    except ImportError as e:
        return _tool_error(f"Cannot import core delegation internals: {e}")

    cfg = _load_delegation_config()
    max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # Build the child agent with the resolved model + credentials
    try:
        child = _build_child_agent(
            task_index=0,
            goal=goal,
            context=context,
            toolsets=toolsets,
            model=creds["model"],
            max_iterations=max_iter,
            task_count=1,
            parent_agent=parent_agent,
            override_provider=creds["provider"],
            override_base_url=creds["base_url"],
            override_api_key=creds["api_key"],
            override_api_mode=creds["api_mode"],
            role="leaf",
        )
    except Exception as e:
        return _tool_error(f"Failed to build subagent: {type(e).__name__}: {e}")

    # Run the child synchronously (the handler runs in a tool thread,
    # and the parent conversation will see the result when it completes)
    try:
        result = _run_single_child(
            task_index=0,
            goal=goal,
            child=child,
            parent_agent=parent_agent,
        )
    except Exception as e:
        return _tool_error(f"Subagent execution failed: {type(e).__name__}: {e}")

    return json.dumps(result, default=str)