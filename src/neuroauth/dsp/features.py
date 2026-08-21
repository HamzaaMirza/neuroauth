"""Welch PSD band-power features and the quality mask.

Pure functions. No I/O, no MNE, no persistence. Feature values produced here are
in-memory only: writing one to Postgres, S3, or a log line violates CLAUDE.md hard
rule 3. The cancelable transform (Phase 2) is what makes a vector storable.
"""

import numpy as np
from numpy.typing import NDArray

from neuroauth.config import FeatureConfig, Normalization, QualityConfig
from neuroauth.dsp.types import FeatureMatrix, QualityReport, WindowSet


def feature_names(
    ch_names: tuple[str, ...],
    bands: dict[str, tuple[float, float]],
    normalization: Normalization,
) -> tuple[str, ...]:
    """Build the deterministic column-name ordering.

    Channel-major, band-minor, with the normalization mode as a prefix:
    ("rel:Fc5:delta", "rel:Fc5:theta", ..., "rel:Fc3:delta"). The prefix is present
    even in single-mode output, so a matrix built under one normalization can never
    be silently fed to a model trained under the other. For "both", all "abs:"
    columns follow all "rel:" columns.

    This ordering is part of the model contract and is persisted alongside the
    model. A model must never be fed a matrix built from a different channel order.

    Args:
        ch_names: Standardized channel names, in data row order.
        bands: Name -> (low_hz, high_hz). Insertion order is respected.
        normalization: "relative", "absolute_log", or "both".

    Returns:
        Length n_features, matching FeatureMatrix.values columns.
    """
    raise NotImplementedError("TODO(phase-1): deterministic feature naming")


def welch_psd(
    window: NDArray[np.float64],
    sfreq: float,
    config: FeatureConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Power spectral density for one window, per channel.

    Args:
        window: (n_channels, n_samples), volts.
        sfreq: Hz.
        config: Welch segment length, overlap, window function, detrend.

    Returns:
        (freqs, psd) where freqs is (n_freqs,) in Hz ascending and psd is
        (n_channels, n_freqs) in V^2/Hz.
    """
    raise NotImplementedError("TODO(phase-1): welch psd")


def band_powers(
    freqs: NDArray[np.float64],
    psd: NDArray[np.float64],
    bands: dict[str, tuple[float, float]],
) -> NDArray[np.float64]:
    """Integrate a PSD over each frequency band.

    Integrates with the trapezoid rule over bins falling in [low_hz, high_hz),
    rather than summing bins, so the result does not shift when the Welch frequency
    resolution changes. A band containing no bins yields 0.0, not NaN.

    Args:
        freqs: (n_freqs,) Hz, ascending.
        psd: (n_channels, n_freqs) V^2/Hz.
        bands: Name -> (low_hz, high_hz).

    Returns:
        (n_channels, n_bands) in V^2, column order matching `bands`.
    """
    raise NotImplementedError("TODO(phase-1): trapezoid band integration")


def normalize_band_powers(
    powers: NDArray[np.float64],
    normalization: Normalization,
    log_epsilon: float,
) -> NDArray[np.float64]:
    """Apply the normalization that turns band powers into model features.

    "relative" divides each band by the total power of that channel across all
    bands. This is the headline configuration. eegmmidb is single-session, so
    electrode impedance, cap placement, and amplifier gain are perfectly confounded
    with subject identity, and absolute power hands a model those offsets directly.
    Dividing them out forces the model onto spectral shape.

    Because the default bands tile 1-50 Hz without gaps, relative values sum to 1.0
    per channel, so one band per channel is linearly dependent on the others. A tree
    ensemble is indifferent to this; a linear model would not be.

    "absolute_log" returns log10(power + log_epsilon). It is run as a comparison,
    not as the headline. If it scores materially higher, that gap is a finding about
    single-session amplitude confounds and gets reported as one.

    Args:
        powers: (..., n_channels, n_bands) in V^2.
        normalization: Which transform to apply.
        log_epsilon: Floor added before log10 so a flat channel stays finite.

    Returns:
        (..., n_channels, n_bands) for a single mode, or (..., n_channels,
        2 * n_bands) for "both", relative columns first. Always finite.
    """
    raise NotImplementedError("TODO(phase-1): relative / log-absolute normalization")


def assess_quality(
    windows: NDArray[np.float64],
    ch_names: tuple[str, ...],
    config: QualityConfig,
) -> QualityReport:
    """Flag flat, clipping, and non-finite channels.

    Never raises and never rejects. A window full of NaN produces a report with
    window_ok False and a flag string; the inference path returns a low-confidence
    result and lets session logic decide.

    Checks, per (window, channel):
        - non-finite samples present
        - within-window std below config.flat_std_v (dead electrode)
        - peak-to-peak above config.max_peak_to_peak_v (clipping or motion)

    A window is not-ok when the bad-channel fraction exceeds
    config.max_bad_channel_fraction.

    Args:
        windows: (n_windows, n_channels, n_samples), volts.
        ch_names: Length n_channels, used to name channels in the flag strings.
        config: Thresholds.

    Returns:
        A QualityReport aligned with the window axis.
    """
    raise NotImplementedError("TODO(phase-1): quality mask")


def extract_features(
    window_set: WindowSet,
    feature_config: FeatureConfig,
    quality_config: QualityConfig,
) -> FeatureMatrix:
    """Band-power features for every window, with the quality mask attached.

    Pure and total: for any input shape this returns a FeatureMatrix whose `values`
    are all finite. Bad channels still get features computed -- they are flagged in
    `quality`, not dropped -- so the matrix keeps a fixed column count regardless of
    signal quality. Non-finite intermediates are replaced with the log_epsilon
    floor, never left as NaN.

    Args:
        window_set: Preprocessed, windowed signal.
        feature_config: Welch parameters, bands, normalization.
        quality_config: Quality thresholds.

    Returns:
        A FeatureMatrix with values (n_windows, n_features), provenance copied from
        the WindowSet, and quality aligned row-wise. A WindowSet with zero windows
        yields a (0, n_features) matrix rather than an error.
    """
    raise NotImplementedError("TODO(phase-1): windows -> feature matrix")
