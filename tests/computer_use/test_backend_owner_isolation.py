"""Backend ownership follows the profile scope used by provider selection."""

import json
from contextlib import contextmanager

import pytest

from agent.computer_use_provider import ComputerUseProvider
from agent.computer_use_registry import register_provider, restore_registration
from hermes_constants import hermes_home_key, set_hermes_home_override, reset_hermes_home_override
from tools.computer_use import tool


@contextmanager
def profile(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


class Backend(tool._NoopBackend):
    def __init__(self):
        super().__init__()
        self.stopped = False

    def stop(self):
        self.stopped = True


class Provider(ComputerUseProvider):
    @property
    def name(self):
        return "owner-test"

    def __init__(self):
        self.created = []
        self.cleaned = 0

    def is_available(self):
        return True

    def create_backend(self, session_id, permission_mode):
        backend = Backend()
        self.created.append(backend)
        return backend

    def emergency_cleanup(self):
        self.cleaned += 1


@pytest.fixture
def owners(tmp_path, monkeypatch):
    tool.reset_backend_for_tests()
    monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
    homes = [tmp_path / "a", tmp_path / "b"]
    providers = [Provider(), Provider()]
    for home, provider in zip(homes, providers):
        home.mkdir()
        (home / "config.yaml").write_text("computer_use:\n  provider: owner-test\n")
        register_provider(provider, scope=hermes_home_key(home))
    yield homes, providers
    tool.reset_backend_for_tests()
    for home, provider in zip(homes, providers):
        restore_registration(provider.name, provider, None, scope=hermes_home_key(home))


def dispatch(session_id):
    result = json.loads(tool.handle_computer_use({"action": "list_apps"}, session_id=session_id))
    assert "error" not in result, result
    return result


def test_empty_session_injection_is_profile_qualified(owners, monkeypatch):
    (home_a, home_b), (_, provider_b) = owners
    injected = Backend()
    monkeypatch.setattr(tool, "_backend", {hermes_home_key(home_a): injected})
    with profile(home_b):
        assert not tool.release_computer_use_session("")
        dispatch("")
        assert len(provider_b.created) == 1
        assert not injected.stopped
    with profile(home_a):
        assert tool._get_backend("") is injected
        assert tool.release_computer_use_session("")
        assert injected.stopped


@pytest.mark.parametrize("session_id", ["same-session", ""])
def test_other_profile_cannot_reuse_or_release_backend(owners, session_id):
    (home_a, home_b), (provider_a, provider_b) = owners
    with profile(home_a):
        dispatch(session_id)
        backend_a = provider_a.created[0]
    with profile(home_b):
        assert not tool.release_computer_use_session(session_id)
        assert not backend_a.stopped
        dispatch(session_id)
        assert len(provider_b.created) == 1
        backend_b = provider_b.created[0]
        assert backend_b is not backend_a
        assert tool.release_computer_use_session(session_id)
        assert backend_b.stopped
        assert not backend_a.stopped
    with profile(home_a):
        assert tool._get_backend(session_id) is backend_a
        assert tool.release_computer_use_session(session_id)
        assert backend_a.stopped
