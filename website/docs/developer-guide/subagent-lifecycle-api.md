---
title: Public Subagent Lifecycle API
sidebar_label: Subagent lifecycle API
---

# Public Subagent Lifecycle API

Plugins can launch and supervise fresh Hermes child sessions without importing
`tools.delegate_tool`, gateway internals, TUI state, or `AIAgent` fields.
The service resolves its parent from the current agent turn, so it works in
CLI, gateway, non-interactive, and kanban-worker sessions. Launching outside an
active agent turn fails closed with `No active Hermes parent session`.

```python
from agent.subagent_lifecycle import SubagentLaunchRequest

def launch_review(ctx):
    # Call from a plugin tool or hook while an agent turn is active.
    service = ctx.subagent_lifecycle
    handle = service.launch(SubagentLaunchRequest(
        goal="Review this change for regressions.",
        context="Only inspect the supplied repository.",
        role="leaf",
        correlation_id="review-42",
        allowed_toolsets=("file",),
    ))
    # Persist handle.to_dict() if desired.
    if service.wait(handle, timeout_seconds=2).timed_out:
        return handle.to_dict()
    return service.result(handle)
```

`SubagentHandle` is serializable and carries a versioned, opaque capability.
Pass it back to `status`, `wait`, `cancel`, `result`, or `reconnect`; malformed
or forged handles return `UNKNOWN`/`UNKNOWN_HANDLE` and cannot access a child.

The stable states are `PENDING`, `STARTING`, `RUNNING`, `SUCCEEDED`, `FAILED`,
`INTERRUPTED`, `CANCEL_REQUESTED`, `CANCELLED`, and `UNKNOWN`.

`cancel(handle, reason=...)` is cooperative: it asks the child agent to
interrupt at its next safe boundary and returns `CANCEL_REQUESTED`; it never
claims completion until `wait` or `result` observes a terminal state. Terminal
results are immutable, idempotent, bounded to 32k characters, omit transcripts
and hidden reasoning, and include a stable result hash.

This API is lifecycle-managed asynchronous execution. Child construction and
completion use the same host-owned path as `delegate_task`, including parent
tool-resolution restoration, memory notification, serialized `subagent_stop`
hooks, resource cleanup, and child-cost rollup. It does not change the
synchronous `delegate_task` tool, batch delegation, or its gateway/TUI display.
The initial implementation retains metadata and terminal results in-process for
one hour.
After a process restart, `reconnect` returns `RECONNECT_UNAVAILABLE` and never
starts a replacement child. Running Python threads also cannot survive process
exit; callers must treat those handles as interrupted by process exit.

Requests are fail-closed: goal/context/metadata sizes are capped, unknown or
parent-broadening toolsets are rejected, and per-tool blocks and
working-directory overrides are rejected until Hermes can support them without
weakening isolation. Use `allowed_toolsets` to narrow a child; Hermes's existing
unsafe-tool block remains enforced.

## Child-activity supervision (`stall_timeout_seconds`)

`stall_timeout_seconds` optionally enables child-activity supervision. It bounds
consecutive inactivity observed by the lifecycle monitor, not total task
runtime. Observed child activity renews the inactivity window. On expiry, Hermes
requests interruption and allows a fixed grace period before publishing a
terminal failure if execution has not returned. Publishing that result does not
guarantee the underlying thread or tool has stopped. Existing transport and
configured delegation timeouts remain independently effective.

This is a liveness watchdog, not an overall deadline. `stall_timeout_seconds`
does not set an overall deadline for the child, and it does not configure the
in-tool threshold: a child that spends a long time *inside* a tool is protected
by a fixed in-tool ceiling (1200 seconds) regardless of the requested value, and
entering or leaving a tool is itself an activity transition that renews the
window. Healthy work that keeps streaming, calling tools, or completing API
calls keeps renewing the window and may legitimately run well beyond the
requested value; only a child frozen across every sample is interrupted.

```python
# Healthy child: activity renews the window, so runtime is NOT bounded by
# stall_timeout_seconds. This child works for 40 minutes of continuous activity.
handle = service.launch(SubagentLaunchRequest(
    goal="Refactor the parser module.",
    stall_timeout_seconds=30.0,   # idle watchdog only, minimum 30 seconds
))
terminal = service.wait(handle)   # may legitimately take far longer than 30s

# Frozen child: no streamed tokens, tool transitions, or API-call progress.
handle = service.launch(SubagentLaunchRequest(
    goal="Summarize the repository.",
    stall_timeout_seconds=30.0,
))
status = service.status(handle)
# status.diagnostic == "stall interrupt requested: quiet=31s threshold=30s mode=idle grace=120s ..."
```

Rules the monitor enforces:

- **30s minimum.** Values below 30.0 (and any non-finite number) are rejected at
  launch. `None` (the default) disables this monitor for that child — it
  disables *this monitor only*, not other watchdogs.
- **30s sampling cadence.** The monitor sweeps at most every 30 seconds, so
  detection latency is threshold plus one sweep. A lazily-started daemon monitor
  exits when nothing is monitorable and is re-armed by the next launch.
- **Queued children are not charged inactivity.** A child still waiting for an
  executor worker has made no progress *yet*; its window starts when it starts.
- **Fixed 1200s in-tool rule.** While a child is inside a tool, the idle
  threshold does not apply; the fixed 1200-second in-tool ceiling does.
- **Fixed 120s grace.** Once the interrupt is committed, the grace deadline is
  fixed: activity during grace does not postpone the terminal publication.
  Completion during grace still wins the terminal race normally. When the grace
  expires with the runner still blocked, the record is force-finalized as
  `FAILED` with `error_classification="STALLED"` (no new terminal state is
  introduced) plus structured `stall_metadata` (`stalled_after_quiet_seconds`,
  `stall_threshold_seconds`, `stall_phase` (`idle|in_tool`),
  `stall_grace_seconds`). A stuck provider request with a functioning
  non-streaming heartbeat can remain "alive" to this monitor indefinitely;
  transport watchdogs own that failure class.
- **`timeout_seconds` stays rejected**, including when both fields are supplied:
  a per-launch runtime deadline remains unsupported; configure delegation
  timeouts explicitly. Contrast `service.wait(handle, timeout_seconds=...)`,
  which limits the *caller's* wait, not the child's runtime.

Two documented limitations follow from the design. First, forced finalization
abandons an outcome, not the worker: the child runs on an 8-worker daemon
executor, and eight permanently blocked runners can starve subsequent launches;
the terminal publication guarantees no second user-visible terminal result, and
memory notification, cost rollup, and host hooks remain execution-owned (they
run inside the runner), but the worker is not reclaimed. Second, `result_hash`
stays an opaque, versioned integrity token: the additive `stall_metadata` field
changes the hash preimage across software versions, so cross-version equality is
not a supported property.
