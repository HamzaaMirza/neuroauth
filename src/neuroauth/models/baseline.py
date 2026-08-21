"""Random Forest identification baseline.

PHASE 1 ONLY. This is closed-set multi-class classification over known subjects,
which CLAUDE.md hard rule 1 forbids as the framing of the system. It exists to
validate the signal pipeline and produce a baseline number, and Phase 2 replaces it
with open-set verification against a claimed identity. Nothing here should ever be
imported by verification or session code.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class BaselineConfig:
    """Random Forest hyperparameters.

    Attributes:
        n_estimators: Number of trees.
        max_depth: None for unlimited.
        min_samples_leaf: Regularization. Raise it if the temporal split and the
            cross-condition split disagree sharply.
        class_weight: "balanced" or None.
        random_state: Seeded for reproducibility.
        n_jobs: -1 for all cores.
    """

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced"
    random_state: int = 1337
    n_jobs: int = -1


def train_baseline(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    config: BaselineConfig,
) -> RandomForestClassifier:
    """Fit the Phase 1 identification baseline.

    Args:
        x_train: (n_train, n_features). Columns must match the feature_names
            recorded with the matrix they came from.
        y_train: (n_train,) subject ids.
        config: Hyperparameters.

    Returns:
        A fitted classifier. Not persisted to disk -- Phase 1 retrains on demand,
        and the MLflow model registry arrives in Phase 3. No pickle files in
        folders.
    """
    raise NotImplementedError("TODO(phase-1): fit random forest")


def predict_baseline(
    model: RandomForestClassifier,
    x: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Predict subject ids and class probabilities.

    Args:
        model: A fitted classifier.
        x: (n, n_features).

    Returns:
        (predictions, probabilities) with shapes (n,) and (n, n_classes), the
        probability columns in model.classes_ order.
    """
    raise NotImplementedError("TODO(phase-1): predict")


def run_shuffled_label_control(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    labels: tuple[int, ...],
    config: BaselineConfig,
    *,
    seed: int,
) -> float:
    """Train on shuffled labels and return the resulting macro-F1.

    The cheapest available insurance against publishing a number that is silently
    wrong. Permute y_train, refit, and evaluate: macro-F1 must collapse to roughly
    1 / n_classes. If it does not, the split is leaking and every other number from
    this run is meaningless.

    Runs on every committed evaluation, not only in tests. Costs one refit.

    Args:
        x_train: (n_train, n_features).
        y_train: (n_train,) subject ids. Permuted internally; not mutated.
        x_test: (n_test, n_features).
        y_test: (n_test,) true subject ids -- NOT shuffled. The control asks whether
            a model fit on scrambled labels can still predict the real ones.
        labels: Full label set.
        config: Hyperparameters. Reused unchanged so the control differs from the
            real run in exactly one respect.
        seed: Seeds the label permutation, separately from config.random_state.

    Returns:
        Macro-F1 of the shuffled-label model against the true test labels.
    """
    raise NotImplementedError("TODO(phase-1): shuffled-label control")


def band_importance(
    model: RandomForestClassifier,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    """Aggregate impurity-based feature importance per frequency band.

    Used to answer one specific question: does delta earn its place? At the default
    Welch resolution delta spans roughly three bins, which is thin. If it
    contributes nothing here, drop the band and record why in docs/DECISIONS.md.

    Impurity importance is biased toward high-cardinality features and correlated
    groups, so treat the ordering as a screen rather than evidence. If a band looks
    borderline, confirm by refitting without it.

    Args:
        model: A fitted classifier.
        feature_names: Length n_features, matching the training columns.

    Returns:
        Band name -> summed importance, descending. Sums to 1.0 across bands.
    """
    raise NotImplementedError("TODO(phase-1): per-band importance aggregation")
