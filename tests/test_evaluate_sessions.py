"""The session evaluation driver's pure parts: state folding, grouping, and the criteria.

The assembly paths need real EEG and are exercised by the driver itself; what is tested here
is everything that decides a reported number.
"""

import numpy as np
import pytest
from scripts.evaluate_sessions import (
    SessionSet,
    build_traces,
    judge,
    non_tail_mask,
    select,
    session_states,
)

from neuroauth.session.logic import PRE_REGISTERED_THRESHOLDS
from neuroauth.verification.metrics import (
    SESSION_DETECT_MAX_MEDIAN_S,
    SESSION_DETECT_MIN_CAUGHT,
    SESSION_MAX_GENUINE_REVOKE,
    SPLICE_CONTROL_MAX_EXCESS,
    DetectionSummary,
    FalseTransitionSummary,
    SessionTrace,
    false_transition_rate,
    summarize_time_to_detect,
)

T = PRE_REGISTERED_THRESHOLDS
TIMES = np.arange(6.0, 6.0 + 40.0, 1.0)


def sessions(
    scores: np.ndarray, *, kind: str = "genuine", quality: np.ndarray | None = None
) -> SessionSet:
    rows = scores.shape[0]
    claimed = np.arange(1, rows + 1, dtype=np.int64)
    return SessionSet(
        kind=kind,
        claimed=claimed,
        impostor=None if kind == "genuine" else claimed,
        times=TIMES[: scores.shape[1]],
        scores=scores,
        llrs=np.zeros_like(scores),
        quality=np.ones(scores.shape, dtype=np.bool_) if quality is None else quality,
        swap_at_s=None if kind == "genuine" else 30.0,
    )


def test_states_come_from_the_real_state_machine() -> None:
    """High scores stay active; the dwell and the gate mean a drop revokes on the 4th."""
    high = session_states(
        np.full(10, 0.80), np.zeros(10), np.ones(10, dtype=np.bool_), TIMES[:10], T
    )
    assert set(high) == {"active"}
    low = session_states(
        np.full(10, 0.40), np.zeros(10), np.ones(10, dtype=np.bool_), TIMES[:10], T
    )
    assert low == ("active", "active", "active", "revoked")


def test_a_terminal_state_ends_the_session() -> None:
    """The runtime stops scoring after revocation, so the trace stops there too."""
    scores = np.concatenate([np.full(6, 0.40), np.full(10, 0.90)])
    states = session_states(scores, np.zeros(16), np.ones(16, dtype=np.bool_), TIMES[:16], T)
    assert states[-1] == "revoked"
    assert len(states) == 4


def test_traces_carry_times_truncated_to_the_states() -> None:
    """time_to_detect requires the two to agree in length."""
    traces = build_traces(sessions(np.full((2, 12), 0.40), kind="swap"), T)
    for trace in traces:
        assert trace.decision_times_s.shape == (len(trace.states),)
        assert trace.impostor_subject == trace.claimed_subject


def test_flagged_decisions_are_skipped_by_the_dwell_through_the_driver() -> None:
    """The skip rule (D-025) survives the driver's observation construction.

    Unflagged, four sub-threshold decisions revoke on the fourth. Flagging the two inside the
    run delays it to the sixth: a skipped decision advances neither the dwell nor the gate."""
    quality = np.ones(12, dtype=np.bool_)
    quality[1:3] = False
    states = session_states(np.full(12, 0.40), np.zeros(12), quality, TIMES[:12], T)
    assert states[-1] == "revoked"
    assert len(states) == 6
    assert "revoked" not in states[:5]


def test_non_tail_mask_excludes_the_worst_decile() -> None:
    claimed = np.array([1, 2, 3, 4], dtype=np.int64)
    mask = non_tail_mask(claimed, np.array([2, 4], dtype=np.int64))
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_select_keeps_rows_and_shared_columns_consistent() -> None:
    chosen = select(sessions(np.full((3, 8), 0.7)), np.array([True, False, True]))
    assert chosen.claimed.size == chosen.scores.shape[0] == 2
    np.testing.assert_array_equal(chosen.times, TIMES[:8])


def detection(median: float, caught: float) -> DetectionSummary:
    return DetectionSummary(
        target_state="revoked",
        n_traces=10,
        n_censored=0,
        median_s=median,
        p90_s=median,
        fraction_detected_within_horizon=caught,
        horizon_s=25.0,
    )


def transitions(fraction: float) -> FalseTransitionSummary:
    return FalseTransitionSummary(
        target_state="revoked",
        n_sessions=10,
        fraction_of_sessions=fraction,
        events_per_hour=0.0,
        exposure_s=100.0,
    )


def test_criteria_pass_on_the_cohort_measured_values() -> None:
    """12 s, 87.8%, 5.0% and a negative splice excess are what cohort data showed (D-025)."""
    verdicts = judge(detection(12.0, 0.878), transitions(0.05), transitions(0.025))
    assert all(block["passes"] for block in verdicts.values() if isinstance(block, dict))
    assert verdicts["time_to_detect_reportable"] is True


@pytest.mark.parametrize(
    ("median", "caught", "genuine", "failing"),
    [
        (15.1, 0.878, 0.05, "median_time_to_detect_s"),
        (12.0, 0.799, 0.05, "fraction_caught_within_horizon"),
        (12.0, 0.878, 0.151, "genuine_sessions_revoked"),
    ],
)
def test_each_criterion_fails_just_past_its_registered_value(
    median: float, caught: float, genuine: float, failing: str
) -> None:
    verdicts = judge(detection(median, caught), transitions(genuine), transitions(0.0))
    assert verdicts[failing]["passes"] is False


def test_the_criteria_are_exclusive_at_their_registered_values() -> None:
    """Exactly at each registered value passes; the margin is inclusive."""
    verdicts = judge(
        detection(SESSION_DETECT_MAX_MEDIAN_S, SESSION_DETECT_MIN_CAUGHT),
        transitions(SESSION_MAX_GENUINE_REVOKE),
        transitions(SESSION_MAX_GENUINE_REVOKE + SPLICE_CONTROL_MAX_EXCESS),
    )
    assert all(block["passes"] for block in verdicts.values() if isinstance(block, dict))


def test_a_failed_self_splice_makes_time_to_detect_unreportable() -> None:
    """D-022: if the splice artifact is doing the detecting, the timing is not a result."""
    verdicts = judge(detection(12.0, 0.9), transitions(0.05), transitions(0.20))
    assert verdicts["self_splice_excess"]["passes"] is False
    assert verdicts["time_to_detect_reportable"] is False


def test_the_genuine_criterion_is_labelled_as_not_a_holdout_result() -> None:
    """The distinction D-022 records has to survive into the artifact."""
    verdicts = judge(detection(12.0, 0.9), transitions(0.05), transitions(0.0))
    assert verdicts["genuine_sessions_revoked"]["is_holdout_result"] is False


def test_summaries_accept_the_traces_the_driver_builds() -> None:
    """The driver's traces satisfy both metric contracts, including the genuine-only check."""
    swap_traces = build_traces(sessions(np.full((3, 30), 0.40), kind="swap"), T)
    summary = summarize_time_to_detect(swap_traces, "revoked", horizon_s=25.0)
    assert summary.n_traces == 3
    genuine_traces = build_traces(sessions(np.full((3, 30), 0.90)), T)
    assert false_transition_rate(genuine_traces, "revoked").fraction_of_sessions == 0.0


def test_a_genuine_trace_with_another_subject_is_refused() -> None:
    """false_transition_rate must never be handed a swap by mistake."""
    trace = SessionTrace(
        claimed_subject=1,
        decision_times_s=TIMES[:5],
        states=("active",) * 5,
        swap_at_s=30.0,
        impostor_subject=2,
    )
    with pytest.raises(ValueError, match="another subject"):
        false_transition_rate([trace], "revoked")
