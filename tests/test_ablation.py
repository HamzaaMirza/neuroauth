"""Feature ablations for the EMG check (D-018)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.config import BANDS, FeatureConfig, QualityConfig
from neuroauth.dsp.features import extract_features, feature_channels
from neuroauth.dsp.types import WindowSet
from neuroauth.models.ablation import bands_without, drop_channels
from tests.synthetic import CH_NAMES, SFREQ, feature_matrix, noise_windows


def _window_set(data: NDArray[np.float64], ch_names: tuple[str, ...]) -> WindowSet:
    return WindowSet(
        data=data,
        sfreq=SFREQ,
        ch_names=ch_names,
        onsets_s=np.arange(data.shape[0], dtype=np.float64),
        subject_id=1,
        run=1,
        condition="eyes_open",
    )


def _windows() -> NDArray[np.float64]:
    return noise_windows(n_windows=3, n_channels=4)


def test_feature_channels_reads_row_order() -> None:
    names = ("rel:C3:alpha", "rel:C3:beta", "rel:C4:alpha", "abs:C3:alpha")
    assert feature_channels(names) == ("C3", "C4")
    with pytest.raises(ValueError):
        feature_channels(("alpha",))


def test_a_dropped_relative_gamma_column_is_still_recoverable() -> None:
    """Why gamma is removed by re-extraction, never by dropping its column.

    Relative band powers sum to one per channel, so gamma = 1 - (delta + theta + alpha
    + beta). A model given the other four relative columns still has gamma.
    """
    matrix = extract_features(
        _window_set(_windows(), CH_NAMES[:4]),
        FeatureConfig(normalization="relative"),
        QualityConfig(),
    )
    per_channel = matrix.values.reshape(3, 4, 5)
    np.testing.assert_allclose(
        1.0 - per_channel[..., :4].sum(axis=-1), per_channel[..., 4], atol=1e-12
    )


def test_bands_without_renormalizes_over_the_remaining_bands() -> None:
    bands = bands_without(BANDS, ("gamma",))
    assert list(bands) == ["delta", "theta", "alpha", "beta"]
    matrix = extract_features(
        _window_set(_windows(), CH_NAMES[:4]),
        FeatureConfig(normalization="relative", bands=bands),
        QualityConfig(),
    )
    assert not any(name.endswith(":gamma") for name in matrix.feature_names)
    np.testing.assert_allclose(matrix.values.reshape(3, 4, 4).sum(axis=-1), 1.0)


@pytest.mark.parametrize("removed", [("gama",), tuple(BANDS)])
def test_bands_without_rejects_unknown_or_all_bands(removed: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        bands_without(BANDS, removed)


def test_drop_channels_removes_feature_and_mask_columns() -> None:
    matrix = extract_features(
        _window_set(_windows(), CH_NAMES[:4]), FeatureConfig(), QualityConfig()
    )
    dropped = drop_channels(matrix, ("Ch02", "Ch04"))
    assert feature_channels(dropped.feature_names) == ("Ch01", "Ch03")
    assert dropped.values.shape == (3, 2 * 5)
    assert dropped.quality.channel_ok.shape == (3, 2)
    np.testing.assert_array_equal(dropped.quality.window_ok, matrix.quality.window_ok)


def test_drop_channels_matches_extracting_without_those_channels() -> None:
    """Exact because relative power is per channel and CAR is off (D-006)."""
    data = _windows()
    full = extract_features(_window_set(data, CH_NAMES[:4]), FeatureConfig(), QualityConfig())
    never_had_it = extract_features(
        _window_set(data[:, [0, 2, 3], :], (CH_NAMES[0], CH_NAMES[2], CH_NAMES[3])),
        FeatureConfig(),
        QualityConfig(),
    )
    dropped = drop_channels(full, (CH_NAMES[1],))
    assert dropped.feature_names == never_had_it.feature_names
    np.testing.assert_allclose(dropped.values, never_had_it.values, rtol=1e-12)


def test_drop_channels_rejects_unknown_channel_or_empty_result() -> None:
    matrix = feature_matrix()
    with pytest.raises(ValueError, match="T9"):
        drop_channels(matrix, ("T9",))
    with pytest.raises(ValueError):
        drop_channels(matrix, ("C3", "C4"))
