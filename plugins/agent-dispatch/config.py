"""Config loading for agent-dispatch plugin.

Reads profile endpoint config from config.yaml under the ``agent_dispatch``
key. Supports env var interpolation in ``api_key`` values via ``${VAR_NAME}``.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

# Cache the loaded config so we don't re-read config.yaml on every tool call.
# The gateway restarts to pick up config changes, so a process-lifetime cache
# is fine.
_config_cache: Optional[dict] = None


def _resolve_hermes_home() -> Path:
    """Find the active HERMES_HOME directory."""
    return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _load_config() -> dict:
    """Load the agent_dispatch section from config.yaml.

    Returns an empty dict if the section is missing or the file is unreadable.
    The result is cached for the process lifetime.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    hermes_home = _resolve_hermes_home()
    config_path = hermes_home / "config.yaml"
    try:
        import yaml
        with open(config_path) as f:
            full_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("agent-dispatch: cannot load %s: %s", config_path, e)
        _config_cache = {}
        return _config_cache

    _config_cache = full_config.get("agent_dispatch", {}) or {}
    if not isinstance(_config_cache, dict):
        _config_cache = {}
    return _config_cache


def _resolve_env_var(value: str) -> str:
    """Resolve a ``${VAR_NAME}`` string from os.environ.

    If the value doesn't match the env var pattern, return it as-is.
    If the env var is not set, return an empty string (the API server will
    reject the auth, which produces a clear error).
    """
    m = _ENV_VAR_RE.match(value.strip())
    if m:
        return os.environ.get(m.group(1), "")
    return value


def get_profile_config(profile: str) -> Optional[Dict[str, Any]]:
    """Return the endpoint config for a named profile.

    Returns ``None`` if the profile is not configured. The returned dict
    has keys: ``url``, ``api_key``, and optionally ``model``.
    """
    cfg = _load_config()
    profiles = cfg.get("profiles", {})
    entry = profiles.get(profile)
    if entry is None:
        return None

    resolved = dict(entry)
    # Resolve env var in api_key
    raw_key = entry.get("api_key", "")
    if isinstance(raw_key, str):
        resolved["api_key"] = _resolve_env_var(raw_key)
    return resolved


def list_profiles() -> list:
    """Return the list of configured profile names."""
    cfg = _load_config()
    return sorted((cfg.get("profiles") or {}).keys())


def get_default_profile() -> Optional[str]:
    """Return the configured default profile, if any."""
    cfg = _load_config()
    return cfg.get("default_profile")


def get_state_file_path() -> Path:
    """Return the path to the run-state JSON file."""
    cfg = _load_config()
    raw = cfg.get("state_file", "~/.hermes/agent-dispatch-runs.json")
    return Path(os.path.expanduser(raw))