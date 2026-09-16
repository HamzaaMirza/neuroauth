"""The pre-registered session parameters (D-025) and the contract around them."""

import dataclasses

from neuroauth.session import logic
from neuroauth.session.logic import PRE_REGISTERED_THRESHOLDS, WindowObservation


def test_pre_registered_session_parameters_are_unchanged() -> None:
    """Fixed on 2026-09-16 from cohort data only, before any holdout session result (D-025).

    Under the D-016 rule these are not adjustable after holdout results. If this test fails,
    the fix is a DECISIONS entry explaining the change, reporting results under both values,
    not editing the test."""
    assert PRE_REGISTERED_THRESHOLDS.ema_half_life_s == 4.0
    assert PRE_REGISTERED_THRESHOLDS.revoke_below == 0.56
    assert PRE_REGISTERED_THRESHOLDS.challenge_below == 0.58
    assert PRE_REGISTERED_THRESHOLDS.recover_above == 0.62
    assert PRE_REGISTERED_THRESHOLDS.revoke_dwell_decisions == 3
    assert PRE_REGISTERED_THRESHOLDS.revoke_dwell_max_span_s == 8.0


def test_thresholds_are_ordered_as_the_state_machine_needs() -> None:
    thresholds = PRE_REGISTERED_THRESHOLDS
    assert thresholds.revoke_below < thresholds.challenge_below < thresholds.recover_above


def test_the_logic_runs_on_the_pre_registered_parameters() -> None:
    """Behaviour lives in tests/test_session_logic.py; this only pins the wiring."""
    observation = WindowObservation(
        decision_time_s=6.0, score=0.7, llr=2.0, quality_ok=True, gap_before=False
    )
    state, transition = logic.update_session(
        logic.initial_session_state(), observation, PRE_REGISTERED_THRESHOLDS
    )
    assert state.confidence == 0.7
    assert transition is None


def test_observation_carries_the_llr_but_thresholds_are_in_score_units() -> None:
    """D-025: the session reads the protected score; the LLR travels for logging and Phase 3."""
    names = {field.name for field in dataclasses.fields(WindowObservation)}
    assert {"score", "llr"} <= names
    assert 0.0 <= PRE_REGISTERED_THRESHOLDS.revoke_below <= 1.0
    assert 0.0 <= PRE_REGISTERED_THRESHOLDS.recover_above <= 1.0
