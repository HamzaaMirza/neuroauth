"""The session state machine: EMA confidence, the challenge/recover levels, and the revoke
dwell with its skip rule, span bound, and gap rule (D-025)."""

import math
import random
from dataclasses import replace

import pytest

from neuroauth.session.logic import (
    PRE_REGISTERED_THRESHOLDS,
    TERMINAL_STATES,
    SessionState,
    SessionTransition,
    WindowObservation,
    initial_session_state,
    update_session,
)
from neuroauth.verification.metrics import SessionStateName

T = PRE_REGISTERED_THRESHOLDS
ALPHA_1S = 1.0 - 0.5 ** (1.0 / 4.0)


def seen(
    time_s: float,
    score: float | None = 0.70,
    *,
    ok: bool = True,
    gap: bool = False,
    llr: float | None = 1.0,
) -> WindowObservation:
    return WindowObservation(
        decision_time_s=time_s, score=score, llr=llr, quality_ok=ok, gap_before=gap
    )


def at(
    state: SessionStateName = "challenged",
    confidence: float = 0.50,
    *,
    scored: int = 10,
    not_ok: int = 0,
    last: float = 100.0,
    dwell: int = 0,
    started: float | None = None,
) -> SessionState:
    """A session already past the min_scored_windows gate."""
    return SessionState(state, confidence, scored, not_ok, last, dwell, started)


def drive(
    observations: list[WindowObservation], state: SessionState | None = None
) -> tuple[SessionState, list[SessionTransition | None]]:
    current = initial_session_state() if state is None else state
    transitions: list[SessionTransition | None] = []
    for observation in observations:
        current, transition = update_session(current, observation, T)
        transitions.append(transition)
    return current, transitions


def test_initial_state_has_no_prior() -> None:
    state = initial_session_state()
    assert state.state == "active"
    assert state.confidence is None
    assert (state.n_scored_windows, state.consecutive_not_ok, state.dwell_count) == (0, 0, 0)
    assert state.last_decision_time_s is None and state.dwell_started_s is None


def test_confidence_seeds_from_the_first_usable_window() -> None:
    state, transitions = drive([seen(6.0, 0.70)])
    assert state.confidence == 0.70
    assert state.n_scored_windows == 1
    assert transitions == [None]


def test_confidence_decays_towards_later_windows() -> None:
    state, _ = drive([seen(6.0, 0.70), seen(7.0, 0.50)])
    assert state.confidence == pytest.approx(0.70 + ALPHA_1S * (0.50 - 0.70))


def test_decay_measures_from_the_last_usable_window() -> None:
    """A skipped window does not move the EMA's clock, so the gap to the next one is 2 s."""
    state, _ = drive([seen(6.0, 0.70), seen(7.0, 0.20, ok=False), seen(8.0, 0.50)])
    two_second_alpha = 1.0 - 0.5 ** (2.0 / 4.0)
    assert state.confidence == pytest.approx(0.70 + two_second_alpha * (0.50 - 0.70))


def test_no_transition_before_min_scored_windows() -> None:
    """Three sub-threshold decisions complete a dwell run but cannot revoke yet."""
    state, transitions = drive([seen(6.0 + i, 0.50) for i in range(4)])
    assert transitions[:3] == [None, None, None]
    assert transitions[3] is not None and transitions[3].to_state == "revoked"
    assert state.state == "revoked"


def test_challenge_fires_on_one_decision_and_does_not_refire() -> None:
    state, transitions = drive([seen(6.0 + i, 0.57) for i in range(5)])
    assert next(t for t in transitions if t is not None).to_state == "challenged"
    assert transitions[4] is None
    assert state.state == "challenged"


def test_levels_are_exclusive_at_the_boundary() -> None:
    """Exactly at a level is not below it, and not above it."""
    _, challenge = drive([seen(101.0, 0.58)], at("active", 0.58))
    _, recover = drive([seen(101.0, 0.62)], at("challenged", 0.62))
    _, revoke = drive([seen(101.0, 0.56)], at("challenged", 0.56, dwell=2, started=99.0))
    assert challenge == recover == [None]
    assert revoke == [None]


def test_recovery_needs_confidence_above_the_level() -> None:
    state, transitions = drive([seen(101.0, 0.70)], at("challenged", 0.62))
    assert state.confidence is not None and state.confidence > T.recover_above
    assert transitions[0] is not None
    assert (transitions[0].from_state, transitions[0].to_state) == ("challenged", "active")


def test_revoke_needs_three_consecutive_sub_threshold_decisions() -> None:
    """A window that lifts confidence back over the level restarts the run."""
    state, transitions = drive(
        [seen(101.0, 0.50), seen(102.0, 0.50), seen(103.0, 0.90), seen(104.0, 0.50)],
        at("challenged", 0.50),
    )
    assert transitions == [None, None, None, None]
    assert state.dwell_count == 1
    state, transitions = drive([seen(105.0, 0.50), seen(106.0, 0.50)], state)
    assert transitions[0] is None
    assert transitions[1] is not None and transitions[1].to_state == "revoked"


def test_revoke_outranks_challenge_on_the_same_decision() -> None:
    _, transitions = drive([seen(101.0, 0.50)], at("active", 0.50, dwell=2, started=99.0))
    assert transitions[0] is not None
    assert (transitions[0].from_state, transitions[0].to_state) == ("active", "revoked")


@pytest.mark.parametrize("terminal", ["revoked", "expired", "closed"])
def test_terminal_states_absorb(terminal: SessionStateName) -> None:
    start = at(terminal, 0.50)
    state, transitions = drive([seen(101.0, 0.99), seen(102.0, None, ok=False)], start)
    assert state == start
    assert transitions == [None, None]


def test_flagged_and_unscorable_decisions_are_skipped_by_the_dwell() -> None:
    """They neither advance nor reset the run, and they do not move the confidence."""
    state, transitions = drive(
        [
            seen(101.0, 0.50),
            seen(102.0, 0.99, ok=False),
            seen(103.0, None),
            seen(104.0, 0.50),
            seen(105.0, 0.50),
        ],
        at("challenged", 0.50),
    )
    assert transitions[:4] == [None, None, None, None]
    assert transitions[4] is not None and transitions[4].to_state == "revoked"
    assert state.confidence == pytest.approx(0.50)


def test_a_run_exactly_at_the_span_bound_survives() -> None:
    """Skipped windows stretch the run: first at 10 s, third at 18 s, span exactly 8.0."""
    state, transitions = drive(
        [
            seen(10.0, 0.50),
            *[seen(11.0 + i, 0.50, ok=False) for i in range(5)],
            seen(16.0, 0.50),
            seen(17.0, 0.50, ok=False),
            seen(18.0, 0.50),
        ],
        at("challenged", 0.50, last=9.0),
    )
    assert transitions[-1] is not None and transitions[-1].to_state == "revoked"
    assert state.state == "revoked"


def test_a_run_past_the_span_bound_expires_and_the_overrunning_decision_restarts_it() -> None:
    """The same run with the third decision one second later: span 9.0, so it expires."""
    stretched = [
        seen(10.0, 0.50),
        *[seen(11.0 + i, 0.50, ok=False) for i in range(5)],
        seen(16.0, 0.50),
        *[seen(17.0 + i, 0.50, ok=False) for i in range(2)],
        seen(19.0, 0.50),
    ]
    state, transitions = drive(stretched, at("challenged", 0.50, last=9.0))
    assert all(transition is None for transition in transitions)
    assert (state.dwell_count, state.dwell_started_s) == (1, 19.0)
    state, restarted = drive([seen(20.0, 0.50), seen(21.0, 0.50)], state)
    assert restarted[0] is None
    assert restarted[1] is not None and restarted[1].to_state == "revoked"


def test_a_gap_breaks_the_run_but_keeps_the_confidence() -> None:
    state, transitions = drive(
        [seen(101.0, 0.50), seen(102.0, 0.50), seen(103.0, 0.50, gap=True)],
        at("challenged", 0.50),
    )
    assert transitions == [None, None, None]
    assert state.dwell_count == 1
    assert state.confidence == pytest.approx(0.50)
    _, later = drive([seen(104.0, 0.50), seen(105.0, 0.50)], state)
    assert later[1] is not None and later[1].to_state == "revoked"


def test_sustained_bad_signal_expires_rather_than_revokes() -> None:
    state, transitions = drive(
        [seen(101.0 + i, None, ok=False) for i in range(T.max_consecutive_not_ok + 1)],
        at("active", 0.50),
    )
    assert transitions[: T.max_consecutive_not_ok] == [None] * T.max_consecutive_not_ok
    last = transitions[T.max_consecutive_not_ok]
    assert last is not None and last.to_state == "expired"
    assert state.state == "expired"


def test_expiry_is_not_gated_by_min_scored_windows() -> None:
    """A session that never produces a usable window must still be able to end."""
    state, _ = drive([seen(6.0 + i, None, ok=False) for i in range(T.max_consecutive_not_ok + 1)])
    assert state.state == "expired"
    assert state.n_scored_windows == 0


def test_one_usable_window_resets_the_unusable_run() -> None:
    state, transitions = drive(
        [
            *[seen(101.0 + i, None, ok=False) for i in range(3)],
            seen(104.0, 0.70),
            *[seen(105.0 + i, None, ok=False) for i in range(3)],
        ],
        at("active", 0.70),
    )
    assert all(transition is None for transition in transitions)
    assert state.consecutive_not_ok == 3
    assert state.state == "active"


def test_a_non_finite_score_counts_as_unusable() -> None:
    state, transitions = drive([seen(101.0, math.nan)], at("active", 0.70))
    assert transitions == [None]
    assert (state.consecutive_not_ok, state.n_scored_windows) == (1, 10)
    assert state.confidence == 0.70


def test_the_llr_is_ignored() -> None:
    first, _ = drive([seen(101.0, 0.57, llr=-9.0)], at("active", 0.57))
    second, _ = drive([seen(101.0, 0.57, llr=9.0)], at("active", 0.57))
    assert first == second


def test_the_input_state_is_never_mutated() -> None:
    start = at("active", 0.57)
    before = replace(start)
    state, _ = drive([seen(101.0, 0.50)], start)
    assert start == before
    assert state is not start


def test_a_transition_is_returned_exactly_when_the_state_name_changes() -> None:
    """The contract the runtime writes the events table from, over randomized sequences.

    The oracle is the state itself: a transition must accompany every change of state name
    and must never accompany anything else, and it must report the names it moved between.
    Also pins that terminal states absorb whatever arrives."""
    rng = random.Random(7)
    for _ in range(200):
        state = initial_session_state()
        for step in range(40):
            score = None if rng.random() < 0.15 else rng.uniform(0.40, 0.75)
            observation = seen(6.0 + step, score, ok=rng.random() > 0.2, gap=rng.random() > 0.9)
            before = state
            state, transition = update_session(before, observation, T)
            assert (transition is not None) == (state.state != before.state)
            if transition is not None:
                assert transition.from_state == before.state
                assert transition.to_state == state.state
                assert transition.decision_time_s == observation.decision_time_s
                assert transition.confidence == state.confidence
            if before.state in TERMINAL_STATES:
                assert state == before
