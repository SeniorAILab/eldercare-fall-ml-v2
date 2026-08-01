from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import final

from contracts.worker_config import PulledWorkerConfig
from worker.runtime.config.restart import (
    RestartDirective,
    RestartDirectiveTracker,
    make_restart_check,
)


def _pulled(*, config_version: int, restart_epoch: int) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=config_version,
        restart_epoch=restart_epoch,
        night_window=None,
        cameras=(),
    )


@final
class _FakeClock:
    """Monotonic stand-in whose value only moves when `advance` is called."""

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now: float = start

    def __call__(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += delta


@final
class _RecordingPuller:
    """Fake `pull_config` that replays scripted results and records calls."""

    __slots__ = ("_results", "calls")

    def __init__(self, results: list[PulledWorkerConfig | None]) -> None:
        self._results: list[PulledWorkerConfig | None] = list(results)
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, relay_url: str, relay_token: str | None) -> PulledWorkerConfig | None:
        self.calls.append((relay_url, relay_token))
        return self._results.pop(0)


def _raising_puller(error: BaseException) -> Callable[[str, str | None], PulledWorkerConfig | None]:
    def _raise(_relay_url: str, _relay_token: str | None) -> PulledWorkerConfig | None:
        raise error

    return _raise


# --- RestartDirective: identity and ordering ---


def test_from_pulled_maps_restart_epoch_to_generation_and_config_version_to_version() -> None:
    pulled = _pulled(config_version=12, restart_epoch=4)

    assert RestartDirective.from_pulled(pulled) == RestartDirective(generation=4, version=12)


def test_directive_ordering_prioritizes_generation_over_version() -> None:
    newer_epoch = RestartDirective(generation=2, version=0)
    older_epoch_higher_version = RestartDirective(generation=1, version=999)

    assert newer_epoch > older_epoch_higher_version


def test_directive_ordering_breaks_ties_on_version_within_same_generation() -> None:
    assert RestartDirective(generation=1, version=5) > RestartDirective(generation=1, version=4)


def test_directive_equality_requires_matching_generation_and_version() -> None:
    assert RestartDirective(generation=1, version=5) == RestartDirective(generation=1, version=5)
    assert RestartDirective(generation=1, version=5) != RestartDirective(generation=1, version=6)
    assert RestartDirective(generation=1, version=5) != RestartDirective(generation=2, version=5)


# --- RestartDirectiveTracker: monotonic acceptance ---


def test_tracker_current_returns_initial_directive_before_any_observation() -> None:
    tracker = RestartDirectiveTracker(RestartDirective(generation=0, version=0))

    assert tracker.current == RestartDirective(generation=0, version=0)


def test_tracker_observe_advances_current_on_strictly_greater_candidate() -> None:
    tracker = RestartDirectiveTracker(RestartDirective(generation=1, version=1))

    advanced = tracker.observe(RestartDirective(generation=1, version=2))

    assert advanced is True
    assert tracker.current == RestartDirective(generation=1, version=2)


def test_tracker_observe_rejects_equal_candidate_idempotently() -> None:
    initial = RestartDirective(generation=1, version=2)
    tracker = RestartDirectiveTracker(initial)

    advanced = tracker.observe(RestartDirective(generation=1, version=2))

    assert advanced is False
    assert tracker.current == initial


def test_tracker_observe_rejects_stale_lower_candidate_monotonically() -> None:
    tracker = RestartDirectiveTracker(RestartDirective(generation=2, version=5))

    advanced = tracker.observe(RestartDirective(generation=2, version=1))

    assert advanced is False
    assert tracker.current == RestartDirective(generation=2, version=5)


def test_tracker_observe_advances_on_new_epoch_even_with_lower_version() -> None:
    tracker = RestartDirectiveTracker(RestartDirective(generation=1, version=999))

    advanced = tracker.observe(RestartDirective(generation=2, version=0))

    assert advanced is True
    assert tracker.current == RestartDirective(generation=2, version=0)


def test_tracker_observe_rejects_lower_generation_even_with_higher_version() -> None:
    tracker = RestartDirectiveTracker(RestartDirective(generation=3, version=0))

    advanced = tracker.observe(RestartDirective(generation=2, version=999))

    assert advanced is False
    assert tracker.current == RestartDirective(generation=3, version=0)


def test_tracker_observe_is_thread_safe_under_concurrent_advancing_candidates() -> None:
    tracker = RestartDirectiveTracker(RestartDirective(generation=1, version=0))
    candidates = [RestartDirective(generation=1, version=version) for version in range(1, 51)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(tracker.observe, candidates))

    assert tracker.current == RestartDirective(generation=1, version=50)
    assert any(results)


# --- make_restart_check: polling cadence, directive gating, failure handling ---


def test_restart_check_polls_immediately_on_first_call() -> None:
    clock = _FakeClock(start=100.0)
    puller = _RecordingPuller([_pulled(config_version=1, restart_epoch=0)])
    check = make_restart_check(
        "http://relay",
        "token",
        RestartDirective(generation=0, version=0),
        poll_interval_sec=60.0,
        monotonic=clock,
        pull_config=puller,
    )

    result = check()

    assert result is True
    assert puller.calls == [("http://relay", "token")]


def test_restart_check_suppresses_repeated_calls_before_poll_interval_elapses() -> None:
    clock = _FakeClock()
    puller = _RecordingPuller(
        [
            _pulled(config_version=1, restart_epoch=0),
            _pulled(config_version=2, restart_epoch=0),
        ]
    )
    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        poll_interval_sec=60.0,
        monotonic=clock,
        pull_config=puller,
    )

    first = check()
    clock.advance(10.0)
    second = check()

    assert first is True
    assert second is False
    assert len(puller.calls) == 1


def test_restart_check_polls_again_once_interval_elapses() -> None:
    clock = _FakeClock()
    puller = _RecordingPuller(
        [
            _pulled(config_version=1, restart_epoch=0),
            _pulled(config_version=2, restart_epoch=0),
        ]
    )
    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        poll_interval_sec=60.0,
        monotonic=clock,
        pull_config=puller,
    )

    first = check()
    clock.advance(60.0)
    second = check()

    assert first is True
    assert second is True
    assert len(puller.calls) == 2


def test_restart_check_returns_false_when_pulled_directive_does_not_advance() -> None:
    boot = RestartDirective(generation=2, version=5)
    puller = _RecordingPuller([_pulled(config_version=5, restart_epoch=2)])
    check = make_restart_check(
        "http://relay",
        None,
        boot,
        monotonic=_FakeClock(),
        pull_config=puller,
    )

    assert check() is False


def test_restart_check_returns_false_when_pull_returns_none() -> None:
    puller = _RecordingPuller([None])
    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        monotonic=_FakeClock(),
        pull_config=puller,
    )

    assert check() is False


def test_restart_check_swallows_oserror_and_returns_false() -> None:
    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        monotonic=_FakeClock(),
        pull_config=_raising_puller(OSError("network down")),
    )

    assert check() is False


def test_restart_check_swallows_timeouterror_and_returns_false() -> None:
    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        monotonic=_FakeClock(),
        pull_config=_raising_puller(TimeoutError()),
    )

    assert check() is False


def test_restart_check_passes_relay_url_and_token_through_to_pull_config() -> None:
    puller = _RecordingPuller([None])
    check = make_restart_check(
        "http://relay:9000",
        "secret-token",
        RestartDirective(generation=0, version=0),
        monotonic=_FakeClock(),
        pull_config=puller,
    )

    check()

    assert puller.calls == [("http://relay:9000", "secret-token")]


def test_restart_check_tracks_directive_advancement_across_multiple_successful_polls() -> None:
    clock = _FakeClock()
    puller = _RecordingPuller(
        [
            _pulled(config_version=1, restart_epoch=0),  # advances past boot -> True
            _pulled(config_version=1, restart_epoch=0),  # unchanged -> False
            _pulled(config_version=3, restart_epoch=0),  # advances again -> True
        ]
    )
    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        poll_interval_sec=10.0,
        monotonic=clock,
        pull_config=puller,
    )

    results = []
    for _ in range(3):
        results.append(check())
        clock.advance(10.0)

    assert results == [True, False, True]


def test_restart_check_ignores_a_failed_poll_and_still_advances_on_the_next_success() -> None:
    clock = _FakeClock()
    calls: list[int] = []

    def flaky_then_advancing(
        _relay_url: str, _relay_token: str | None
    ) -> PulledWorkerConfig | None:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("offline")
        return _pulled(config_version=1, restart_epoch=0)

    check = make_restart_check(
        "http://relay",
        None,
        RestartDirective(generation=0, version=0),
        poll_interval_sec=10.0,
        monotonic=clock,
        pull_config=flaky_then_advancing,
    )

    first = check()
    clock.advance(10.0)
    second = check()

    assert first is False
    assert second is True
    assert len(calls) == 2


def test_a_restarted_worker_does_not_immediately_restart_again() -> None:
    """Two consecutive boots against an unchanged relay: the second must run.

    Every test above checks the tracker in isolation, where "a higher version
    means restart" is correct. The composed property is different and was
    missing: a worker that exits *because* of a restart directive gets started
    again, and must not reach the same conclusion from the same unchanged
    relay. Otherwise it restart-loops and never opens a stream.

    That is exactly what a literal ``RestartDirective(0, 0)`` seed produced.
    The relay serves ``config_version = registry_version``, which is >= 1 the
    moment one camera is registered, so every boot saw 0 -> 1 and exited. It
    measured as 0 frames processed against a real backend while the whole suite
    stayed green.

    ``worker/__main__.py`` now seeds from the config the process actually
    booted with, matching the baseline's ``boot_registry_version``.
    """
    relay_state = _pulled(config_version=7, restart_epoch=2)

    def pull_config(_url: str, _token: str | None) -> PulledWorkerConfig:
        return relay_state

    def boot_and_poll() -> bool:
        """One process lifetime: seed from the boot pull, then poll once."""
        boot = pull_config("http://relay", "token")
        check = make_restart_check(
            "http://relay",
            "token",
            RestartDirective.from_pulled(boot),
            pull_config=pull_config,
            monotonic=_FakeClock(),
        )
        return check()

    assert boot_and_poll() is False, "first boot restarted against an unchanged relay"
    assert boot_and_poll() is False, "restarted worker restarted again: this is the loop"

    # A boot after a change adopts it rather than restarting onto it.
    relay_state = _pulled(config_version=8, restart_epoch=2)
    assert boot_and_poll() is False, "a boot must adopt current config, not restart"

    # But a change that lands *after* boot must still restart, or the fix would
    # have simply disabled restarts.
    def boot_stale_then_poll() -> bool:
        check = make_restart_check(
            "http://relay",
            "token",
            RestartDirective.from_pulled(_pulled(config_version=7, restart_epoch=2)),
            pull_config=pull_config,
            monotonic=_FakeClock(),
        )
        return check()

    assert boot_stale_then_poll() is True, "a config change after boot must still restart"
