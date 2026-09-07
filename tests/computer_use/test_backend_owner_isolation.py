"""Backend ownership follows the profile scope used by provider selection."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
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


def in_profile(home, operation, *args):
    with profile(home):
        return operation(*args)


@pytest.mark.parametrize("blocked_stage", ["create", "start"])
def test_slow_provider_does_not_block_another_owner(owners, monkeypatch, blocked_stage):
    (home_a, home_b), (provider_a, provider_b) = owners
    entered, proceed = threading.Event(), threading.Event()
    backend_a = Backend()

    def block():
        entered.set()
        assert proceed.wait(10), "test did not release the slow provider"

    def create(sid, mode):
        if blocked_stage == "create":
            block()
        return backend_a

    if blocked_stage == "start":
        backend_a.start = block
    monkeypatch.setattr(provider_a, "create_backend", create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(in_profile, home_a, dispatch, "same-session")
        try:
            assert entered.wait(5)
            pool.submit(in_profile, home_b, dispatch, "same-session").result(3)
            assert pool.submit(in_profile, home_b, tool.release_computer_use_session, "same-session").result(3)
            assert provider_b.created[0].stopped
            assert not slow.done()
        finally:
            proceed.set()
        slow.result(5)


def test_release_fences_starter_and_queued_waiter(owners, monkeypatch):
    (home_a, _), (provider_a, _) = owners
    entered, proceed, queued = threading.Event(), threading.Event(), threading.Event()
    stale, replacement = Backend(), Backend()
    candidates = iter((stale, replacement))

    def start():
        entered.set()
        assert proceed.wait(10), "test did not release stale startup"

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if self.lock.locked():
                queued.set()
            self.lock.acquire()
            return self

        def __exit__(self, *exc):
            self.lock.release()

    # Observe an actual waiter acquiring the old single-flight record, not a timing guess.
    owner = (hermes_home_key(home_a), "same-session")
    monkeypatch.setattr(tool, "_backend_start_locks", {owner: ObservedLock()}, raising=False)
    stale.start = start
    monkeypatch.setattr(provider_a, "create_backend", lambda sid, mode: next(candidates))
    with ThreadPoolExecutor(max_workers=3) as pool:
        original = pool.submit(in_profile, home_a, tool._get_backend, "same-session")
        try:
            assert entered.wait(5)
            waiter = pool.submit(in_profile, home_a, tool._get_backend, "same-session")
            assert queued.wait(3), "waiter cannot reach its owner while startup holds the global lock"
            pool.submit(in_profile, home_a, tool.release_computer_use_session, "same-session").result(3)
            assert pool.submit(in_profile, home_a, tool._get_backend, "same-session").result(3) is replacement
        finally:
            proceed.set()
        for future in (original, waiter):
            with pytest.raises(RuntimeError, match="released"):
                future.result(5)
        assert stale.stopped
        assert not replacement.stopped
        with profile(home_a):
            assert tool._get_backend("same-session") is replacement
            assert tool.release_computer_use_session("same-session")
            assert replacement.stopped
