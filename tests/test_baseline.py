"""Random Forest baseline, shuffled-label control, and importance aggregation."""

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.config import BANDS
from neuroauth.models.baseline import (
    BaselineConfig,
    band_importance,
    channel_importance,
    importance_share,
    predict_baseline,
    run_shuffled_label_control,
    train_baseline,
)
from neuroauth.models.evaluation import chance_level, macro_f1

FAST = BaselineConfig(n_estimators=60, n_jobs=1)
CONTROL_MAX_RATIO = 3.0


def _clustered(
    *,
    n_classes: int = 20,
    per_class: int = 40,
    n_features: int = 10,
    informative: tuple[int, ...] = (0, 1, 2),
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Well-separated classes that differ only in the informative columns."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((n_classes, n_features))
    centers[:, list(informative)] = rng.normal(0.0, 8.0, size=(n_classes, len(informative)))
    y = np.repeat(np.arange(1, n_classes + 1, dtype=np.int64), per_class)
    x = centers[y - 1] + rng.normal(size=(y.size, n_features))
    return x, y


def _alternate(
    x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    return x[::2], y[::2], x[1::2], y[1::2]


def test_baseline_learns_separable_classes() -> None:
    x_train, y_train, x_test, y_test = _alternate(*_clustered())
    predictions, _ = predict_baseline(train_baseline(x_train, y_train, FAST), x_test)
    assert macro_f1(y_test, predictions, tuple(range(1, 21))) > 0.9


def test_predictions_and_probabilities_are_aligned() -> None:
    x_train, y_train, x_test, _ = _alternate(*_clustered())
    model = train_baseline(x_train, y_train, FAST)
    predictions, probabilities = predict_baseline(model, x_test)
    assert predictions.shape == (x_test.shape[0],)
    assert probabilities.shape == (x_test.shape[0], 20)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(predictions, model.classes_[probabilities.argmax(axis=1)])


def test_train_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        train_baseline(np.zeros((10, 3)), np.zeros(9, dtype=np.int64), FAST)


def test_shuffled_label_control_collapses_to_chance() -> None:
    """On data the model genuinely learns, scrambled labels still score near chance."""
    labels = tuple(range(1, 21))
    x_train, y_train, x_test, y_test = _alternate(*_clustered())
    control = run_shuffled_label_control(x_train, y_train, x_test, y_test, labels, FAST, seed=7)
    assert control <= CONTROL_MAX_RATIO * chance_level(len(labels))


def test_shuffled_label_control_is_seeded_and_does_not_mutate() -> None:
    labels = tuple(range(1, 21))
    x_train, y_train, x_test, y_test = _alternate(*_clustered())
    before = y_train.copy()
    first = run_shuffled_label_control(x_train, y_train, x_test, y_test, labels, FAST, seed=7)
    second = run_shuffled_label_control(x_train, y_train, x_test, y_test, labels, FAST, seed=7)
    np.testing.assert_array_equal(y_train, before)
    assert first == second


def test_shuffled_control_cannot_detect_window_overlap_leakage() -> None:
    """Documents a limit of the control, so nobody over-trusts it (D-007).

    Every subject's rows come from the same stationary AR(1) process, so there is no
    identity information anywhere -- a correct evaluation must score chance. Adjacent
    rows are strongly correlated (phi = 0.9), standing in for windows that share
    samples.

    - A random row split puts correlated neighbours on both sides. The model scores
      far above chance purely by memorizing neighbours: the leak.
    - A temporal split with a wide gap scores chance, as it should.
    - The shuffled-label control on the leaky split also scores chance, because each
      memorized neighbour now carries a random label. It does not see the leak.
    """
    rng = np.random.default_rng(11)
    n_subjects, n_windows, n_features, phi = 12, 60, 16, 0.9
    rows = np.empty((n_subjects, n_windows, n_features))
    rows[:, 0] = rng.normal(size=(n_subjects, n_features))
    for t in range(1, n_windows):
        innovation = rng.normal(size=(n_subjects, n_features))
        rows[:, t] = phi * rows[:, t - 1] + np.sqrt(1.0 - phi**2) * innovation
    x = rows.reshape(-1, n_features)
    y = np.repeat(np.arange(1, n_subjects + 1, dtype=np.int64), n_windows)
    labels = tuple(range(1, n_subjects + 1))
    position = np.tile(np.arange(n_windows), n_subjects)
    ceiling = CONTROL_MAX_RATIO * chance_level(n_subjects)

    leaky_train = rng.permutation(x.shape[0])[: int(0.7 * x.shape[0])]
    leaky_test = np.setdiff1d(np.arange(x.shape[0]), leaky_train)
    honest_train = np.flatnonzero(position < 30)
    honest_test = np.flatnonzero(position >= 45)

    def score(train: NDArray[np.int64], test: NDArray[np.int64]) -> float:
        predictions, _ = predict_baseline(train_baseline(x[train], y[train], FAST), x[test])
        return macro_f1(y[test], predictions, labels)

    control = run_shuffled_label_control(
        x[leaky_train], y[leaky_train], x[leaky_test], y[leaky_test], labels, FAST, seed=3
    )
    assert score(leaky_train, leaky_test) > 0.8
    assert score(honest_train, honest_test) <= ceiling
    assert control <= ceiling


def _named(channels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"rel:{channel}:{band}" for channel in channels for band in BANDS)


def test_band_importance_finds_the_informative_band() -> None:
    names = _named(("C3", "C4"))
    alpha_columns = tuple(i for i, name in enumerate(names) if name.endswith(":alpha"))
    x, y = _clustered(n_features=len(names), informative=alpha_columns)
    importance = band_importance(train_baseline(x, y, FAST), names)
    assert next(iter(importance)) == "alpha"
    assert set(importance) == set(BANDS)
    assert sum(importance.values()) == pytest.approx(1.0)


def test_channel_importance_finds_the_informative_channel() -> None:
    names = _named(("C3", "Fp1", "C4"))
    fp1_columns = tuple(i for i, name in enumerate(names) if ":Fp1:" in name)
    x, y = _clustered(n_features=len(names), informative=fp1_columns)
    importance = channel_importance(train_baseline(x, y, FAST), names)
    assert next(iter(importance)) == "Fp1"
    assert sum(importance.values()) == pytest.approx(1.0)


def test_importance_merges_normalization_modes() -> None:
    names = tuple(f"{mode}:C3:{band}" for mode in ("rel", "abs") for band in BANDS)
    x, y = _clustered(n_features=len(names))
    model = train_baseline(x, y, FAST)
    assert set(band_importance(model, names)) == set(BANDS)
    assert set(channel_importance(model, names)) == {"C3"}


def test_importance_share_reports_uniform_baseline() -> None:
    importance = {"Fp1": 0.5, "Fp2": 0.2, "C3": 0.2, "C4": 0.1}
    share, uniform = importance_share(importance, ("Fp1", "Fp2", "Fp1"))
    assert share == pytest.approx(0.7)
    assert uniform == pytest.approx(0.5)


def test_importance_share_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="FP1"):
        importance_share({"Fp1": 1.0}, ("FP1",))


def test_importance_rejects_mismatched_feature_names() -> None:
    x, y = _clustered(n_features=10)
    model = train_baseline(x, y, FAST)
    with pytest.raises(ValueError):
        band_importance(model, _named(("C3",)))
