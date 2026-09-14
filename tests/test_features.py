"""Band powers and feature naming."""

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.config import BANDS, FeatureConfig, Normalization, QualityConfig, WindowConfig
from neuroauth.dsp.features import (
    band_powers,
    extract_features,
    feature_names,
    normalize_band_powers,
    welch_psd,
)
from neuroauth.dsp.types import Recording, WindowSet
from neuroauth.dsp.windowing import window_stream_chunk
from tests.synthetic import CH_NAMES, SFREQ, noise_windows, sine

BAND_ORDER = list(BANDS)


def _window_set(data: NDArray[np.float64]) -> WindowSet:
    return WindowSet(
        data=data,
        sfreq=SFREQ,
        ch_names=CH_NAMES[: data.shape[1]],
        onsets_s=np.arange(data.shape[0], dtype=np.float64),
        subject_id=None,
        run=None,
        condition=None,
    )


def _features(data: NDArray[np.float64], normalization: Normalization) -> NDArray[np.float64]:
    config = FeatureConfig(normalization=normalization)
    return extract_features(_window_set(data), config, QualityConfig()).values


@pytest.mark.parametrize(
    ("freq_hz", "band"),
    [(2.5, "delta"), (6.0, "theta"), (10.5, "alpha"), (21.0, "beta"), (40.0, "gamma")],
)
def test_band_power_lands_in_the_right_band(freq_hz: float, band: str) -> None:
    """A tone inside a band puts most of its power in that band."""
    window = sine(freq_hz, duration_s=2.0)[None, :]
    freqs, psd = welch_psd(window, SFREQ, FeatureConfig())
    relative = normalize_band_powers(band_powers(freqs, psd, BANDS), "relative", 1e-20)[0]
    assert int(np.argmax(relative)) == BAND_ORDER.index(band)
    assert relative[BAND_ORDER.index(band)] > 0.7


def test_bands_tile_without_gaps() -> None:
    """The five bands sum to exactly the integral over 1-50 Hz."""
    window = np.random.default_rng(0).normal(size=(3, 320))
    freqs, psd = welch_psd(window, SFREQ, FeatureConfig())
    per_band_total = band_powers(freqs, psd, BANDS).sum(axis=-1)
    whole = band_powers(freqs, psd, {"all": (1.0, 50.0)})[..., 0]
    np.testing.assert_allclose(per_band_total, whole, rtol=1e-12)


def test_band_power_is_resolution_invariant() -> None:
    """Broadband power per band agrees at nperseg 160 (1 Hz) and 320 (0.5 Hz).

    Integrating only the bins inside each band would fail this by up to 25% in delta,
    because the dropped edge width is one bin and the bin width halves.
    """
    signal = np.random.default_rng(1).normal(size=(1, round(120 * SFREQ)))
    coarse = band_powers(
        *welch_psd(signal, SFREQ, FeatureConfig(welch_nperseg=160, welch_noverlap=80)), BANDS
    )
    fine = band_powers(
        *welch_psd(signal, SFREQ, FeatureConfig(welch_nperseg=320, welch_noverlap=160)), BANDS
    )
    np.testing.assert_allclose(fine, coarse, rtol=0.05)


def test_tone_power_is_resolution_invariant() -> None:
    """Integrated tone power per band agrees across Welch resolutions."""
    signal = sum(sine(f, duration_s=8.0) for f in (6.0, 10.5, 21.0, 40.0))[None, :]
    coarse = band_powers(
        *welch_psd(signal, SFREQ, FeatureConfig(welch_nperseg=160, welch_noverlap=80)), BANDS
    )
    fine = band_powers(
        *welch_psd(signal, SFREQ, FeatureConfig(welch_nperseg=320, welch_noverlap=160)), BANDS
    )
    # Delta holds only leakage here, so a relative tolerance on it is meaningless.
    np.testing.assert_allclose(fine[..., 1:], coarse[..., 1:], rtol=0.05)


def test_relative_powers_sum_to_one_per_channel(synthetic_recording: Recording) -> None:
    """The default bands tile 1-50 Hz, so relative values sum to 1.0."""
    windows = window_stream_chunk(
        synthetic_recording.data[:, :1600], SFREQ, CH_NAMES, WindowConfig()
    )
    matrix = extract_features(windows, FeatureConfig(normalization="relative"), QualityConfig())
    per_channel = matrix.values.reshape(windows.data.shape[0], len(CH_NAMES), len(BANDS))
    np.testing.assert_allclose(per_channel.sum(axis=-1), 1.0, rtol=1e-12)


def _scaled_pair() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.random.default_rng(2).normal(0.0, 2e-6, size=(2, 4, 320))
    scaled = data.copy()
    scaled[:, 1, :] *= 10.0
    return data, scaled


def test_relative_is_invariant_to_amplitude_scaling() -> None:
    """Scaling a channel by 10x leaves its relative band powers unchanged.

    This is the whole argument for relative being the headline: it discards exactly
    the per-channel amplitude scaling that absolute power would encode -- session
    artifact and anatomy alike, which single-session data cannot tell apart.
    """
    data, scaled = _scaled_pair()
    np.testing.assert_allclose(
        _features(scaled, "relative"), _features(data, "relative"), rtol=1e-9
    )


def test_absolute_log_is_not_invariant_to_amplitude_scaling() -> None:
    """The mirror of the above: 10x amplitude is 100x power, exactly +2 in log10."""
    data, scaled = _scaled_pair()
    delta = (_features(scaled, "absolute_log") - _features(data, "absolute_log")).reshape(2, 4, 5)
    np.testing.assert_allclose(delta[:, 1, :], 2.0, atol=1e-6)
    np.testing.assert_allclose(delta[:, [0, 2, 3], :], 0.0, atol=1e-12)


def test_feature_names_carry_the_normalization_prefix() -> None:
    """Single-mode output is still prefixed, so modes cannot be silently mixed."""
    channels = ("C3", "C4")
    bands = {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    relative = ("rel:C3:alpha", "rel:C3:beta", "rel:C4:alpha", "rel:C4:beta")
    absolute = ("abs:C3:alpha", "abs:C3:beta", "abs:C4:alpha", "abs:C4:beta")
    assert feature_names(channels, bands, "relative") == relative
    assert feature_names(channels, bands, "absolute_log") == absolute
    assert feature_names(channels, bands, "both") == relative + absolute
    with pytest.raises(ValueError):
        feature_names(channels, bands, "zscore")  # type: ignore[arg-type]


@pytest.mark.parametrize("normalization", ["relative", "absolute_log", "both"])
def test_feature_names_match_value_columns(normalization: Normalization) -> None:
    """Every named column holds the value for that exact (mode, channel, band).

    Checks the mapping, not just the count: a mode-major vs channel-major mix-up in
    "both" would keep the shape right and put every value in the wrong column.
    """
    data = np.random.default_rng(3).normal(0.0, 2e-6, size=(3, 4, 320))
    data[:, 2, :] += sine(10.0, duration_s=2.0, amplitude_v=20e-6)
    windows = _window_set(data)
    matrix = extract_features(windows, FeatureConfig(normalization=normalization), QualityConfig())
    assert matrix.values.shape == (3, len(matrix.feature_names))

    powers = band_powers(*welch_psd(data, SFREQ, FeatureConfig()), BANDS)
    expected = {
        "rel": normalize_band_powers(powers, "relative", 1e-20),
        "abs": normalize_band_powers(powers, "absolute_log", 1e-20),
    }
    for column, name in enumerate(matrix.feature_names):
        mode, channel, band = name.split(":")
        np.testing.assert_allclose(
            matrix.values[:, column],
            expected[mode][:, windows.ch_names.index(channel), BAND_ORDER.index(band)],
        )


def test_feature_order_is_deterministic() -> None:
    """Same inputs give the same order; band insertion order is respected."""
    channels = ("C3", "C4")
    assert feature_names(channels, BANDS, "both") == feature_names(channels, BANDS, "both")
    reversed_bands = dict(reversed(list(BANDS.items())))
    assert feature_names(channels, reversed_bands, "relative")[0] == "rel:C3:gamma"


def test_empty_window_set_yields_empty_matrix() -> None:
    """Zero windows produce a (0, n_features) matrix, not an error."""
    matrix = extract_features(
        _window_set(np.zeros((0, 4, 320))), FeatureConfig(normalization="both"), QualityConfig()
    )
    assert matrix.values.shape == (0, 4 * 5 * 2)
    assert len(matrix.feature_names) == 40
    assert matrix.quality.window_ok.shape == (0,)


@pytest.mark.parametrize("n_samples", [1, 100])
def test_window_shorter_than_welch_segment_stays_finite(n_samples: int) -> None:
    matrix = extract_features(
        _window_set(noise_windows(n_windows=2, n_channels=4, n_samples=n_samples)),
        FeatureConfig(),
        QualityConfig(),
    )
    assert matrix.values.shape == (2, 20)
    assert np.isfinite(matrix.values).all()


def test_provenance_is_copied_through(synthetic_recording: Recording) -> None:
    windows = window_stream_chunk(
        synthetic_recording.data[:, :1600], SFREQ, CH_NAMES, WindowConfig(), onset_offset_s=5.0
    )
    matrix = extract_features(windows, FeatureConfig(), QualityConfig())
    np.testing.assert_array_equal(matrix.onsets_s, windows.onsets_s)
    assert (matrix.subject_id, matrix.run, matrix.condition) == (None, None, None)
