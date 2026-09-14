"""Identification metrics, the control gate, and the committed artifacts."""

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from neuroauth.models.evaluation import (
    IdentificationReport,
    chance_level,
    check_shuffled_label_control,
    compare_normalizations,
    evaluate_identification,
    macro_f1,
    summarize_ablation,
    summarize_quality,
    write_quality_report,
    write_report,
)
from tests.synthetic import feature_matrix

LABELS = (1, 2, 3, 4)
Y_TRUE = [1, 1, 2, 2, 3, 3, 4, 4]
PERFECT = Y_TRUE
HALF_RIGHT = [1, 2, 2, 3, 3, 4, 4, 1]


def _report(
    y_pred: list[int],
    *,
    y_true: list[int] = Y_TRUE,
    normalization: str = "relative",
    split_kind: str = "cross_condition",
    shuffled: float | None = 0.25,
) -> IdentificationReport:
    return evaluate_identification(
        np.asarray(y_true),
        np.asarray(y_pred),
        LABELS,
        split_kind=split_kind,
        normalization=normalization,
        config_fingerprint="abc123",
        n_train=100,
        n_test_not_ok=3,
        n_subjects_enrollable=len(LABELS),
        n_subjects_impostor_holdout=20,
        shuffled_label_macro_f1=shuffled,
    )


def test_report_has_no_accuracy_field() -> None:
    """Hard rule 2, enforced structurally."""
    names = {field.name for field in dataclasses.fields(IdentificationReport)}
    assert not any("accuracy" in name for name in names)


def test_perfect_predictions() -> None:
    report = _report(PERFECT)
    assert report.macro_f1 == 1.0
    np.testing.assert_array_equal(report.per_class_f1, 1.0)
    np.testing.assert_array_equal(report.confusion, 2 * np.eye(4, dtype=np.int64))


def test_macro_f1_is_the_mean_of_per_class_f1() -> None:
    report = _report(HALF_RIGHT)
    np.testing.assert_allclose(report.per_class_f1, 0.5)
    assert report.macro_f1 == pytest.approx(0.5)
    assert macro_f1(np.asarray(Y_TRUE), np.asarray(HALF_RIGHT), LABELS) == pytest.approx(0.5)


def test_chance_level_is_one_over_n_classes() -> None:
    assert _report(PERFECT).chance_level == pytest.approx(0.25)
    assert chance_level(89) == pytest.approx(1 / 89)
    with pytest.raises(ValueError):
        chance_level(0)


def test_absent_label_still_appears_in_confusion_matrix() -> None:
    report = _report([1, 2], y_true=[1, 2])
    assert report.confusion.shape == (4, 4)
    np.testing.assert_array_equal(report.confusion[2:], 0)
    np.testing.assert_array_equal(report.per_class_f1, [1.0, 1.0, 0.0, 0.0])


def test_label_outside_the_label_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="label set"):
        _report([1, 9], y_true=[1, 2])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        _report([1, 2, 3], y_true=[1, 2])


@pytest.mark.parametrize("shuffled", [0.0, 0.25, 0.74])
def test_control_passes_within_three_times_chance(shuffled: float) -> None:
    check_shuffled_label_control(_report(PERFECT, shuffled=shuffled))


def test_control_fails_above_three_times_chance() -> None:
    with pytest.raises(AssertionError, match="shuffled-label"):
        check_shuffled_label_control(_report(PERFECT, shuffled=0.8))


def test_control_fails_when_it_was_not_run() -> None:
    with pytest.raises(AssertionError, match="not run"):
        check_shuffled_label_control(_report(PERFECT, shuffled=None))


def test_compare_normalizations_reports_a_material_gap_as_a_bound() -> None:
    """A material gap is an upper bound on session artifacts, never a decomposition."""
    summary = compare_normalizations(
        _report(HALF_RIGHT, normalization="relative"),
        _report(PERFECT, normalization="absolute_log"),
    )
    text = str(summary["interpretation"])
    assert summary["delta_absolute_minus_relative"] == pytest.approx(0.5)
    assert "upper bound" in text
    assert "cannot be separated" in text
    assert "rather than identity" not in text


def test_compare_normalizations_within_tolerance() -> None:
    summary = compare_normalizations(
        _report(PERFECT, normalization="relative"),
        _report(PERFECT, normalization="absolute_log"),
    )
    assert summary["delta_absolute_minus_relative"] == pytest.approx(0.0)
    assert "within" in str(summary["interpretation"])


def test_compare_normalizations_rejects_incomparable_reports() -> None:
    relative = _report(PERFECT, normalization="relative", split_kind="cross_condition")
    with pytest.raises(ValueError):
        compare_normalizations(
            relative, _report(PERFECT, normalization="absolute_log", split_kind="temporal")
        )
    with pytest.raises(ValueError):
        compare_normalizations(_report(PERFECT, normalization="absolute_log"), relative)


def test_summarize_ablation_material_drop_is_a_bound() -> None:
    summary = summarize_ablation(_report(PERFECT), _report(HALF_RIGHT), removed="the gamma band")
    text = str(summary["interpretation"])
    assert summary["drop_headline_minus_ablated"] == pytest.approx(0.5)
    assert summary["material"] is True
    assert "bounds" in text
    assert "cannot separate" in text
    assert "not bounded" in str(summary["scope"])


def test_summarize_ablation_immaterial_drop() -> None:
    summary = summarize_ablation(_report(PERFECT), _report(PERFECT), removed="the gamma band")
    assert summary["drop_headline_minus_ablated"] == pytest.approx(0.0)
    assert summary["material"] is False
    assert "within" in str(summary["interpretation"])


def test_summarize_ablation_improvement_is_not_material() -> None:
    summary = summarize_ablation(_report(HALF_RIGHT), _report(PERFECT), removed="the gamma band")
    assert summary["drop_headline_minus_ablated"] == pytest.approx(-0.5)
    assert summary["material"] is False
    assert "raises" in str(summary["interpretation"])


def test_summarize_ablation_rejects_incomparable_reports() -> None:
    with pytest.raises(ValueError):
        summarize_ablation(
            _report(PERFECT), _report(PERFECT, split_kind="temporal"), removed="the gamma band"
        )


def test_write_report_artifacts(tmp_path: Path) -> None:
    report = _report(HALF_RIGHT)
    write_report(report, tmp_path / "artifacts", prefix="relative_cross_condition")
    out = tmp_path / "artifacts"

    metrics_text = (out / "relative_cross_condition_metrics.json").read_text(encoding="utf-8")
    assert "accuracy" not in metrics_text
    metrics = json.loads(metrics_text)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["chance_level"] == pytest.approx(0.25)
    assert metrics["shuffled_label_macro_f1"] == pytest.approx(0.25)
    assert metrics["n_test_not_ok"] == 3

    with (out / "relative_cross_condition_per_class_f1.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["subject_id", "f1"]
    assert len(rows) == 1 + len(LABELS)

    with (out / "relative_cross_condition_confusion_matrix.csv").open(encoding="utf-8") as handle:
        grid = list(csv.reader(handle))
    assert len(grid) == 1 + len(LABELS)
    assert [int(v) for v in grid[1][1:]] == report.confusion[0].tolist()

    png = (out / "relative_cross_condition_confusion_matrix.png").read_bytes()
    assert png.startswith(b"\x89PNG")


def test_summarize_quality_counts_flags_per_recording(tmp_path: Path) -> None:
    names = ("rel:Fp1:alpha", "rel:C3:alpha")
    channel_ok = np.ones((5, 2), dtype=np.bool_)
    channel_ok[[0, 1], 0] = False  # Fp1 flagged in windows 0 and 1
    channel_ok[3, 1] = False  # C3 flagged in window 3
    window_ok = np.array([False, True, True, True, True])
    matrix = feature_matrix(
        subject_id=7,
        run=1,
        n_windows=5,
        feature_names=names,
        channel_ok=channel_ok,
        window_ok=window_ok,
    )
    (row,) = summarize_quality([matrix], frontal_channels=("Fp1",))
    assert (row.subject_id, row.run, row.condition) == (7, 1, "eyes_open")
    assert (row.n_windows, row.n_windows_not_ok) == (5, 1)
    assert (row.n_windows_any_flag, row.n_windows_frontal_flag) == (3, 2)

    write_quality_report([row], tmp_path)
    with (tmp_path / "quality_flag_rates.csv").open(encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["subject_id"] == "7"
    assert float(written[0]["fraction_frontal_flag"]) == pytest.approx(0.4)


def test_summarize_quality_rejects_missing_frontal_channel() -> None:
    """Default frontal channels absent from the features is a naming bug, not zero flags."""
    with pytest.raises(ValueError, match="Fp1"):
        summarize_quality([feature_matrix()])


def test_summarize_quality_requires_provenance() -> None:
    with pytest.raises(ValueError):
        summarize_quality([feature_matrix(subject_id=None)], frontal_channels=())


def test_a_priori_thresholds_are_unchanged() -> None:
    """All three thresholds were fixed before the runs they judge (D-016).

    If this fails, a threshold was edited. The fix is not updating this test: it is a
    new DECISIONS.md entry saying why, with results reported under both the old and
    the new value.
    """
    from neuroauth.models.evaluation import (
        ABLATION_MATERIAL_DROP,
        CONTROL_MAX_CHANCE_RATIO,
        MATERIAL_DELTA,
    )

    assert CONTROL_MAX_CHANCE_RATIO == 3.0
    assert MATERIAL_DELTA == 0.05
    assert ABLATION_MATERIAL_DROP == 0.05
