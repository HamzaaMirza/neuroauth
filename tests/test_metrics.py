"""Biometric metrics: atomic ties, the EER and FRR@FAR definitions, resolution flags, the
tail and concordance verdicts, the bootstrap, time-to-detect, and the report."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from neuroauth.verification import metrics
from neuroauth.verification.metrics import (
    ScoreTable,
    SessionTrace,
    VerificationReport,
    bootstrap_eer_ci,
    check_random_pairing_control,
    check_score_table,
    det_coordinates,
    eer_value,
    equal_error_rate,
    error_rates,
    evaluate_verification,
    false_transition_rate,
    frr_at_far,
    per_subject_rows,
    phase1_concordance,
    rates_at_threshold,
    split_scores,
    summarize_tail,
    summarize_time_to_detect,
    time_to_detect,
    write_det_plot,
    write_verification_report,
)

HOLDOUT = frozenset({90, 91})


def make_table(
    rows: list[tuple[int, int, float]], *, holdout: frozenset[int] = HOLDOUT
) -> ScoreTable:
    claimed, source, scores = (np.array(column) for column in zip(*rows, strict=True))
    n_rows = len(rows)
    return ScoreTable(
        scores=scores.astype(np.float64),
        claimed_subject=claimed.astype(np.int64),
        source_subject=source.astype(np.int64),
        source_is_holdout=np.isin(source, sorted(holdout)),
        fold=np.zeros(n_rows, dtype=np.int64),
        probe_onset_s=np.arange(n_rows, dtype=np.float64),
        probe_window_ok=np.ones(n_rows, dtype=np.bool_),
        template_score_mean=np.full(n_rows, 0.8),
        template_score_std=np.full(n_rows, 0.05),
        domain="protected",
        split_kind="cross_condition",
    )


def synthetic_table(seed: int = 0) -> ScoreTable:
    """10 enrolled subjects; genuine ~0.75, holdout impostors ~0.5, cohort impostors ~0.55."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, float]] = []
    for claimed in range(1, 11):
        rows += [(claimed, claimed, float(v)) for v in rng.normal(0.75, 0.06, 20)]
        for impostor in HOLDOUT:
            rows += [(claimed, impostor, float(v)) for v in rng.normal(0.5, 0.06, 20)]
        for other in range(1, 11):
            if other != claimed:
                rows += [(claimed, other, float(v)) for v in rng.normal(0.55, 0.06, 5)]
    return make_table(rows)


def test_report_has_no_accuracy_field() -> None:
    """Hard rule 2, enforced structurally (D-012)."""
    assert not any("accuracy" in f.name for f in dataclasses.fields(VerificationReport))


def test_a_priori_thresholds_are_unchanged() -> None:
    """Fixed in 0a4abb7 before any Phase 2 result (D-016 rule). If this fails, the fix is a
    DECISIONS entry explaining the change, with results under both values, not editing the test."""
    assert metrics.FAR_TARGETS == (0.01, 0.001)
    assert metrics.HEADLINE_FAR == 0.01
    assert metrics.MIN_EXPECTED_ERRORS == 3.0
    assert metrics.RANDOM_PAIRING_CONTROL_MIN_EER == 0.40
    assert metrics.PROTECTION_MATERIAL_EER_COST == 0.02
    assert (metrics.TAIL_WORST_FRACTION, metrics.TAIL_HEAVY_MIN_RATIO) == (0.10, 2.0)
    assert metrics.TAIL_HEAVY_MIN_GAP == 0.10
    assert metrics.CONCORDANCE_CONFIRM_MAX_RHO == -0.30
    assert metrics.CONCORDANCE_REFUTE_MIN_RHO == -0.10
    assert metrics.CONCORDANCE_ALPHA == 0.05
    assert metrics.REVOCATION_AGREEMENT_BAND == (0.45, 0.55)
    assert metrics.REVOCATION_MAX_ACCEPTED_FRACTION == 0.03
    assert metrics.SPLICE_CONTROL_MAX_EXCESS == 0.10
    assert metrics.SESSION_DETECT_MAX_MEDIAN_S == 15.0
    assert metrics.SESSION_DETECT_MIN_CAUGHT == 0.80
    assert metrics.SESSION_MAX_GENUINE_REVOKE == 0.15
    assert (metrics.N_BOOTSTRAP, metrics.BOOTSTRAP_SEED) == (2000, 20260916)
    assert (metrics.N_PERMUTATIONS, metrics.PERMUTATION_SEED) == (10000, 20260917)
    assert metrics.PAIRING_CONTROL_SEED == 20260918


def test_error_rates_on_a_known_case() -> None:
    rates = error_rates(np.array([0.9, 0.8, 0.7]), np.array([0.1, 0.2, 0.75]))
    np.testing.assert_array_equal(rates.thresholds, [0.1, 0.2, 0.7, 0.75, 0.8, 0.9, np.inf])
    np.testing.assert_allclose(rates.far, [1.0, 2 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(rates.frr, [0.0, 0.0, 0.0, 1 / 3, 1 / 3, 2 / 3, 1.0])


def test_tied_scores_are_accepted_or_rejected_together() -> None:
    """No operating point between ties is invented."""
    rates = error_rates(np.array([0.5, 0.5]), np.array([0.5]))
    np.testing.assert_array_equal(rates.thresholds, [0.5, np.inf])
    point = equal_error_rate(rates)
    assert (point.threshold, point.far, point.frr) == (0.5, 1.0, 0.0)
    assert eer_value(point) == 1.0


def test_eer_is_the_smallest_achievable_max_at_the_lowest_threshold() -> None:
    rates = error_rates(np.array([0.6, 0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3, 0.65]))
    point = equal_error_rate(rates)
    assert (point.threshold, point.far, point.frr) == (0.6, 0.25, 0.0)


def test_perfect_separation_has_zero_eer() -> None:
    assert eer_value(equal_error_rate(error_rates(np.array([0.9]), np.array([0.1])))) == 0.0


def test_frr_at_far_uses_the_smallest_threshold_meeting_the_target() -> None:
    rates = error_rates(np.full(10, 2.0), np.arange(100) / 100)
    point = frr_at_far(rates, 0.05, n_impostor_pairs=1780)
    assert (point.threshold, point.far, point.frr) == (0.95, 0.05, 0.0)


def test_under_resolved_flag_counts_pairs() -> None:
    rates = error_rates(np.full(10, 2.0), np.arange(100) / 100)
    assert frr_at_far(rates, 0.001, n_impostor_pairs=1780).under_resolved is True
    assert frr_at_far(rates, 0.01, n_impostor_pairs=1780).under_resolved is False
    assert frr_at_far(rates, 0.01, n_impostor_pairs=299).under_resolved is True


def test_an_unreachable_target_rejects_everything_rather_than_raising() -> None:
    point = frr_at_far(error_rates(np.array([0.1]), np.array([0.9])), 0.5, n_impostor_pairs=1)
    assert math.isinf(point.threshold)
    assert (point.far, point.frr) == (0.0, 1.0)


def test_rates_at_a_threshold_chosen_elsewhere() -> None:
    point = rates_at_threshold(np.array([0.4, 0.6, 0.8]), np.array([0.2, 0.5, 0.7]), 0.5)
    assert (point.far, point.frr) == pytest.approx((2 / 3, 1 / 3))


def test_det_coordinates_are_finite_at_zero_and_one() -> None:
    x, y = det_coordinates(error_rates(np.array([0.9, 0.8]), np.array([0.1, 0.2])))
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_score_table_integrity_checks() -> None:
    table = synthetic_table()
    check_score_table(table, HOLDOUT)
    with pytest.raises(AssertionError, match="claimed"):
        check_score_table(make_table([(90, 90, 0.9), (90, 1, 0.1)]), HOLDOUT)
    with pytest.raises(AssertionError, match="source_is_holdout"):
        check_score_table(
            dataclasses.replace(table, source_is_holdout=~table.source_is_holdout), HOLDOUT
        )
    with pytest.raises(AssertionError, match="non-finite"):
        check_score_table(
            dataclasses.replace(table, scores=np.full_like(table.scores, np.nan)), HOLDOUT
        )


def test_split_scores_separates_impostor_sources() -> None:
    table = synthetic_table()
    genuine, holdout = split_scores(table, "impostor_holdout")
    _, cohort = split_scores(table, "cohort")
    assert (genuine.size, holdout.size, cohort.size) == (200, 400, 450)
    assert metrics.n_impostor_pairs(table, "impostor_holdout") == 20
    assert metrics.n_impostor_pairs(table, "cohort") == 90


def test_per_subject_rows_carry_counts() -> None:
    rows = per_subject_rows(synthetic_table(), "impostor_holdout")
    assert [row.subject_id for row in rows] == list(range(1, 11))
    assert {(row.n_genuine, row.n_impostor, row.n_genuine_not_ok) for row in rows} == {(20, 40, 0)}


@pytest.mark.parametrize(
    ("worst", "median", "heavy"),
    [(0.40, 0.10, True), (0.19, 0.10, False), (0.08, 0.02, False)],
    ids=["heavy", "ratio-below-2x", "gap-below-0.10"],
)
def test_heavy_tail_rule(worst: float, median: float, heavy: bool) -> None:
    per_subject = {s: median for s in range(1, 19)} | {19: worst, 20: worst}
    summary = summarize_tail(per_subject, pooled_eer=median)
    assert summary.worst_decile_subjects == (19, 20)
    assert summary.heavy_tail is heavy


def test_concordance_verdicts() -> None:
    subjects = range(1, 31)
    f1 = {s: (s - 1) / 29 for s in subjects}
    confirmed = phase1_concordance({s: 1.0 - f1[s] for s in subjects}, f1, n_permutations=500)
    assert confirmed.verdict == "confirmed"
    assert confirmed.spearman_rho == pytest.approx(-1.0)
    assert confirmed.zero_f1_subjects == (1,)
    assert confirmed.zero_f1_in_worst_quartile == 1
    refuted = phase1_concordance({s: f1[s] for s in subjects}, f1, n_permutations=500)
    assert refuted.verdict == "refuted"


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate() -> None:
    table = synthetic_table()
    genuine, impostor = split_scores(table, "impostor_holdout")
    estimate = eer_value(equal_error_rate(error_rates(genuine, impostor)))
    first = bootstrap_eer_ci(table, "impostor_holdout", n_boot=200)
    assert first == bootstrap_eer_ci(table, "impostor_holdout", n_boot=200)
    assert first[0] <= estimate <= first[1]


def trace(states: list[str], swap_at_s: float | None, impostor: int | None = 2) -> SessionTrace:
    return SessionTrace(
        claimed_subject=1,
        decision_times_s=np.arange(len(states), dtype=np.float64) + 6.0,
        states=tuple(states),  # type: ignore[arg-type]
        swap_at_s=swap_at_s,
        impostor_subject=impostor,
    )


def test_time_to_detect_measures_from_the_swap_at_decision_time() -> None:
    states = ["active"] * 30 + ["challenged"] * 4 + ["revoked"] * 3  # decisions at 6..42 s
    swapped = trace(states, swap_at_s=30.0)
    assert time_to_detect(swapped, "challenged") == 6.0
    assert time_to_detect(swapped, "revoked") == 10.0
    assert time_to_detect(trace(["active"] * 37, swap_at_s=30.0), "revoked") is None
    assert time_to_detect(trace(["revoked"] * 37, swap_at_s=30.0), "revoked") is None
    with pytest.raises(ValueError, match="swap"):
        time_to_detect(trace(states, swap_at_s=None), "revoked")


def test_detection_summary_treats_censored_traces_as_infinite() -> None:
    detected = trace(["active"] * 30 + ["revoked"] * 7, swap_at_s=30.0)
    censored = trace(["active"] * 37, swap_at_s=30.0)
    summary = summarize_time_to_detect([detected, censored, censored], "revoked", horizon_s=25.0)
    assert summary.n_censored == 2
    assert math.isinf(summary.median_s)
    assert summary.fraction_detected_within_horizon == pytest.approx(1 / 3)


def test_false_transition_rate_counts_entries_and_refuses_impostor_signal() -> None:
    genuine = trace(["active", "challenged", "active", "challenged"] + ["active"] * 6, None, None)
    summary = false_transition_rate([genuine], "challenged")
    assert summary.fraction_of_sessions == 1.0
    assert summary.events_per_hour == pytest.approx(2 / 9 * 3600)
    self_splice = trace(["active"] * 10, swap_at_s=10.0, impostor=1)
    assert false_transition_rate([self_splice], "revoked").fraction_of_sessions == 0.0
    with pytest.raises(ValueError, match="another subject"):
        false_transition_rate([trace(["active"] * 10, swap_at_s=10.0, impostor=2)], "revoked")


def test_verification_report_and_artifacts(tmp_path: Path) -> None:
    table = synthetic_table()
    report = evaluate_verification(
        table,
        impostor_holdout=HOLDOUT,
        impostor_source="impostor_holdout",
        deployed_threshold=0.62,
        random_pairing_control_eer=0.5,
        n_failure_to_enroll=0,
        representation_versions=("rep",),
        streaming_fingerprint="fp",
        bootstrap=False,
    )
    assert report.headline.target_far == metrics.HEADLINE_FAR
    assert report.headline.under_resolved is True  # 20 pairs * 0.01 < 3
    assert report.n_impostor_pairs == 20
    assert report.deployed is not None
    check_random_pairing_control(report)
    with pytest.raises(AssertionError, match="pairing"):
        check_random_pairing_control(dataclasses.replace(report, random_pairing_control_eer=0.2))

    write_verification_report(report, tmp_path, prefix="headline")
    text = (tmp_path / "headline_metrics.json").read_text("utf-8")
    assert "accuracy" not in text
    assert json.loads(text)["headline_frr_at_far"]["under_resolved"] is True
    assert (tmp_path / "headline_per_subject_eer.csv").read_text("utf-8").count("\n") == 11


def test_det_plot_writes_a_png_and_refuses_four_curves(tmp_path: Path) -> None:
    genuine, impostor = split_scores(synthetic_table(), "impostor_holdout")
    rates = error_rates(genuine, impostor)
    point = equal_error_rate(rates)
    write_det_plot(
        {"protected": rates},
        tmp_path / "det.png",
        title="t",
        subtitle="s",
        eer_points={"protected": point},
    )
    assert (tmp_path / "det.png").stat().st_size > 0
    with pytest.raises(ValueError, match="curves"):
        write_det_plot(
            {str(k): rates for k in range(4)},
            tmp_path / "x.png",
            title="t",
            subtitle="s",
            eer_points={},
        )
