"""Phase 1 identification metrics.

IdentificationReport deliberately has no `accuracy` field, and none will be added
(CLAUDE.md hard rule 2). The closed-set framing of Phase 1 is a scaffold for the
signal pipeline; Phase 2 replaces these metrics with EER, FAR, and FRR.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IdentificationReport:
    """Closed-set identification results for the Phase 1 baseline.

    Attributes:
        macro_f1: Unweighted mean of per-class F1.
        per_class_f1: (n_classes,) aligned with `labels`.
        confusion: (n_classes, n_classes) int, rows true, columns predicted.
        labels: Subject ids, in confusion-matrix row and column order.
        chance_level: 1 / n_classes. Reported alongside macro_f1 so the number has
            context -- 0.009 for 109 classes, and a reader cannot judge a macro-F1
            without it.
        shuffled_label_macro_f1: Macro-F1 of the same pipeline trained on shuffled
            labels. Must collapse to roughly chance_level. None only when the
            control was deliberately skipped, which should never be the case for a
            committed run.
        n_train: Training rows used.
        n_test: Test rows used.
        n_low_quality_excluded: Test rows dropped by the quality mask.
        split_kind: "temporal" or "cross_condition".
        normalization: "relative" or "absolute_log". Both are run; the relative
            result is the headline.
        n_subjects_enrollable: Subjects in the enrollable cohort.
        n_subjects_impostor_holdout: Subjects reserved as impostors and untouched by
            this evaluation.
        config_fingerprint: PipelineConfig.fingerprint() of the features.
    """

    macro_f1: float
    per_class_f1: NDArray[np.float64]
    confusion: NDArray[np.int64]
    labels: tuple[int, ...]
    chance_level: float
    shuffled_label_macro_f1: float | None
    n_train: int
    n_test: int
    n_low_quality_excluded: int
    split_kind: str
    normalization: str
    n_subjects_enrollable: int
    n_subjects_impostor_holdout: int
    config_fingerprint: str


def evaluate_identification(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    labels: tuple[int, ...],
    *,
    split_kind: str,
    normalization: str,
    config_fingerprint: str,
    n_train: int,
    n_low_quality_excluded: int,
    n_subjects_enrollable: int,
    n_subjects_impostor_holdout: int,
    shuffled_label_macro_f1: float | None = None,
) -> IdentificationReport:
    """Compute macro-F1, per-class F1, the confusion matrix, and the chance level.

    Args:
        y_true: (n_test,) subject ids.
        y_pred: (n_test,) predicted subject ids.
        labels: Full label set, so classes absent from y_true still appear in the
            confusion matrix.
        split_kind: Recorded on the report for provenance.
        normalization: Recorded on the report for provenance.
        config_fingerprint: Recorded on the report for provenance.
        n_train: Recorded on the report.
        n_low_quality_excluded: Recorded on the report.
        n_subjects_enrollable: Recorded on the report.
        n_subjects_impostor_holdout: Recorded on the report.
        shuffled_label_macro_f1: Result of run_shuffled_label_control, if run.

    Returns:
        An IdentificationReport.
    """
    raise NotImplementedError("TODO(phase-1): identification metrics")


def compare_normalizations(
    relative: IdentificationReport,
    absolute: IdentificationReport,
) -> dict[str, float | str]:
    """Summarize the relative-vs-absolute band-power comparison.

    The gap between the two is a finding, not a footnote. If absolute scores
    materially higher, the honest reading is that single-session amplitude
    confounds -- electrode impedance, cap placement, amplifier gain, all perfectly
    correlated with subject in this dataset -- are doing the work, not identity
    information. Reporting that gap and naming its cause is worth more than the
    higher number would be.

    Args:
        relative: Report from the "relative" run. The headline.
        absolute: Report from the "absolute_log" run.

    Returns:
        A JSON-serializable summary: both macro-F1 values, the delta, and a
        plain-language interpretation string for the README.
    """
    raise NotImplementedError("TODO(phase-1): normalization comparison summary")


def write_report(
    report: IdentificationReport,
    out_dir: Path,
    *,
    prefix: str,
) -> None:
    """Write the committed Phase 1 artifacts.

    Writes {prefix}_metrics.json, {prefix}_per_class_f1.csv, and
    {prefix}_confusion_matrix.png. These are the Phase 1 exit criteria and are
    committed to the repo.

    Args:
        report: The report to serialize.
        out_dir: The artifacts/ directory. Created if absent.
        prefix: Distinguishes runs, e.g. "relative_cross_condition".
    """
    raise NotImplementedError("TODO(phase-1): serialize report artifacts")
