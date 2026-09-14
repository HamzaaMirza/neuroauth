"""Random Forest identification baseline.

PHASE 1 ONLY. This is closed-set multi-class classification over known subjects,
which CLAUDE.md hard rule 1 forbids as the framing of the system. It exists to
validate the signal pipeline and produce a baseline number, and Phase 2 replaces it
with open-set verification against a claimed identity. Nothing here should ever be
imported by verification or session code.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from neuroauth.models.evaluation import macro_f1


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


def _check_design(x: NDArray[np.float64], y: NDArray[np.int64]) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be (n_rows, n_features), got shape {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y must be ({x.shape[0]},), got shape {y.shape}")
    if x.shape[0] == 0:
        raise ValueError("cannot fit on zero rows")


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

    Raises:
        ValueError: If x is not 2-D, y does not match its rows, or there are no rows.
    """
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    _check_design(x, y)
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(x, y)
    return model


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

    Raises:
        ValueError: If x is not 2-D.
    """
    rows = np.asarray(x, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"x must be (n_rows, n_features), got shape {rows.shape}")
    predictions = np.asarray(model.predict(rows), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(rows), dtype=np.float64)
    return predictions, probabilities


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

    Permute y_train, refit, and score against the true test labels. Macro-F1 must
    collapse to roughly 1 / n_classes; check_shuffled_label_control enforces this.

    What this catches: any path by which the true test labels reach the predictions
    other than through the (x_train, y_train) association. That covers test labels
    leaking into features or into the fit, row/label misalignment, and metric bugs.

    What this cannot catch: window-overlap leakage across the split. A test window
    that shares samples with a train window inherits that train window's label --
    and after permutation, that label is random. The leaky model reads as chance
    here while its real score is inflated. Overlap is ruled out structurally by
    assert_no_window_overlap instead; see D-007.

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
    shuffled = np.random.default_rng(seed).permutation(np.asarray(y_train, dtype=np.int64))
    model = train_baseline(x_train, shuffled, config)
    predictions, _ = predict_baseline(model, x_test)
    return macro_f1(np.asarray(y_test, dtype=np.int64), predictions, labels)


def _name_parts(name: str) -> tuple[str, str, str]:
    parts = name.split(":")
    if len(parts) != 3:
        raise ValueError(f"feature name {name!r} is not in mode:channel:band form")
    return parts[0], parts[1], parts[2]


def _aggregate_importance(
    model: RandomForestClassifier, feature_names: tuple[str, ...], part: int
) -> dict[str, float]:
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if importances.size != len(feature_names):
        raise ValueError(
            f"model has {importances.size} features but {len(feature_names)} names were given"
        )
    totals: dict[str, float] = {}
    for name, value in zip(feature_names, importances, strict=True):
        key = _name_parts(name)[part]
        totals[key] = totals.get(key, 0.0) + float(value)
    grand_total = sum(totals.values())
    if grand_total > 0.0:
        totals = {key: value / grand_total for key, value in totals.items()}
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


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
        feature_names: Length n_features, matching the training columns. Under
            "both" normalization the rel: and abs: columns of a band are summed.

    Returns:
        Band name -> summed importance, descending. Sums to 1.0 across bands.

    Raises:
        ValueError: If feature_names does not match the model or a name is not in
            mode:channel:band form.
    """
    return _aggregate_importance(model, feature_names, part=2)


def channel_importance(
    model: RandomForestClassifier,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    """Aggregate impurity-based feature importance per channel.

    Used for the frontal EOG check (D-004b): on the eyes-open-trained model, does
    importance concentrate on the channels above the eyes? Pair with
    importance_share for the uniform baseline. The same impurity-importance caveats
    as band_importance apply; the frontal channels are strongly correlated with each
    other, which makes their group share more trustworthy than any single channel.

    Args:
        model: A fitted classifier.
        feature_names: Length n_features, matching the training columns.

    Returns:
        Channel name -> summed importance across bands and modes, descending. Sums
        to 1.0 across channels.

    Raises:
        ValueError: If feature_names does not match the model or a name is not in
            mode:channel:band form.
    """
    return _aggregate_importance(model, feature_names, part=1)


def importance_share(
    importance: dict[str, float],
    group: Sequence[str],
) -> tuple[float, float]:
    """Share of total importance carried by a group, and the uniform baseline.

    "Concentrates" only means something against a reference. If importance were
    spread evenly across channels, a group of 4 out of 64 would carry 6.25%.

    Args:
        importance: Output of channel_importance (or band_importance).
        group: Keys to sum, e.g. config.FRONTAL_EOG_CHANNELS. Duplicates count once.

    Returns:
        (share, uniform_share): the fraction of total importance on the group, and
        the fraction it would carry under uniform importance.

    Raises:
        ValueError: If the group is empty or names a key absent from importance --
            usually a spelling mismatch such as "FP1" for "Fp1".
    """
    members = set(group)
    if not members:
        raise ValueError("group is empty")
    missing = sorted(members - importance.keys())
    if missing:
        raise ValueError(f"not present in importance: {missing}")
    total = sum(importance.values())
    share = sum(importance[key] for key in members) / total if total > 0.0 else 0.0
    return share, len(members) / len(importance)
