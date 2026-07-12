"""agent-dispatch plugin — dispatch tasks to other Hermes profiles.

Two dispatch modes:
  - dispatch_agent: async, returns run_id immediately, SSE callback auto-delivers
  - dispatch_chat: sync, blocks until reply, session-persistent (multi-turn)

Plus polling/recovery tools:
  - check_dispatch: GET /v1/runs/{run_id}, polls a single run
  - collect_dispatches: batch-poll multiple runs at once
  - dispatch_status: list persisted runs from state file

Config (config.yaml):
  agent_dispatch:
    profiles:
      marie:
        url: http://localhost:8651
        api_key: ${MARIE_API_KEY}
        model: glm-5.2
    state_file: ~/.hermes/agent-dispatch-runs.json
"""

from __future__ import annotations

import logging

from .tools import (
    ALL_SCHEMAS,
    handle_dispatch_agent,
    handle_check_dispatch,
    handle_collect_dispatches,
    handle_dispatch_status,
    handle_dispatch_chat,
    handle_cancel_dispatch,
    handle_delegate_subagent,
)

logger = logging.getLogger(__name__)

_TOOLS = (
    ("dispatch_agent", DISPATCH_AGENT_SCHEMA := ALL_SCHEMAS[0], handle_dispatch_agent, "🚀"),
    ("check_dispatch", ALL_SCHEMAS[1], handle_check_dispatch, "🔍"),
    ("collect_dispatches", ALL_SCHEMAS[2], handle_collect_dispatches, "📥"),
    ("dispatch_status", ALL_SCHEMAS[3], handle_dispatch_status, "📋"),
    ("dispatch_chat", ALL_SCHEMAS[4], handle_dispatch_chat, "💬"),
    ("cancel_dispatch", ALL_SCHEMAS[5], handle_cancel_dispatch, "🛑"),
    ("delegate_subagent", ALL_SCHEMAS[6], handle_delegate_subagent, "🔀"),
)


def register(ctx) -> None:
    """Register all agent-dispatch tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="agent_dispatch",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )
    logger.info(
        "agent-dispatch: registered %d tools (dispatch_agent, check_dispatch, "
        "collect_dispatches, dispatch_status, dispatch_chat, cancel_dispatch, "
        "delegate_subagent)",
        len(_TOOLS),
    )