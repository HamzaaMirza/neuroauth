"""Welch PSD band-power features and the quality mask.

Pure functions. No I/O, no MNE, no persistence. Feature values produced here are
in-memory only: writing one to Postgres, S3, or a log line violates CLAUDE.md hard
rule 3. The cancelable transform (Phase 2) is what makes a vector storable.

Error policy: functions here raise ValueError only for structurally inconsistent
arguments (wrong array rank, ch_names not matching the channel axis, an unknown
normalization). Signal content -- NaN, inf, flat, clipping, too short -- never
raises; it is flagged in the QualityReport and the features stay finite.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from neuroauth.config import FeatureConfig, Normalization, QualityConfig
from neuroauth.dsp.types import FeatureMatrix, QualityReport, WindowSet

_MODE_PREFIXES: dict[str, tuple[str, ...]] = {
    "relative": ("rel",),
    "absolute_log": ("abs",),
    "both": ("rel", "abs"),
}


def _modes(normalization: str) -> tuple[str, ...]:
    try:
        return _MODE_PREFIXES[normalization]
    except KeyError:
        raise ValueError(f"unknown normalization {normalization!r}") from None


def feature_names(
    ch_names: tuple[str, ...],
    bands: dict[str, tuple[float, float]],
    normalization: Normalization,
) -> tuple[str, ...]:
    """Build the deterministic column-name ordering.

    Channel-major, band-minor, with the normalization mode as a prefix:
    ("rel:FC5:delta", "rel:FC5:theta", ..., "rel:FC3:delta"). The prefix is present
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

    Raises:
        ValueError: If normalization is not one of the three modes.
    """
    return tuple(
        f"{mode}:{channel}:{band}"
        for mode in _modes(normalization)
        for channel in ch_names
        for band in bands
    )


def welch_psd(
    window: NDArray[np.float64],
    sfreq: float,
    config: FeatureConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Power spectral density per channel, vectorized over any leading axes.

    A window shorter than config.welch_nperseg is estimated with a segment the
    length of the window, rather than raising. Non-finite samples propagate to a
    non-finite PSD without warnings; callers sanitize downstream.

    Args:
        window: (..., n_channels, n_samples), volts. Typically one window of shape
            (n_channels, n_samples), or a whole stack of shape
            (n_windows, n_channels, n_samples).
        sfreq: Hz.
        config: Welch segment length, overlap, window function, detrend.

    Returns:
        (freqs, psd) where freqs is (n_freqs,) in Hz ascending and psd is
        (..., n_channels, n_freqs) in V^2/Hz.

    Raises:
        ValueError: If the window has fewer than 2 samples.
    """
    x = np.asarray(window, dtype=np.float64)
    n_samples = x.shape[-1]
    if n_samples < 2:
        raise ValueError(f"need at least 2 samples for a PSD, got {n_samples}")

    nperseg = min(config.welch_nperseg, n_samples)
    noverlap = min(config.welch_noverlap, nperseg - 1)
    detrend: str | bool = False if config.detrend == "none" else config.detrend
    with np.errstate(invalid="ignore", over="ignore"):
        freqs, psd = welch(
            x,
            fs=sfreq,
            window=config.welch_window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density",
            axis=-1,
        )
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def _interpolate_at(
    freqs: NDArray[np.float64], psd: NDArray[np.float64], target_hz: float
) -> NDArray[np.float64]:
    j = int(np.clip(np.searchsorted(freqs, target_hz, side="right") - 1, 0, freqs.size - 2))
    t = (target_hz - freqs[j]) / (freqs[j + 1] - freqs[j])
    interpolated: NDArray[np.float64] = psd[..., j] * (1.0 - t) + psd[..., j + 1] * t
    return interpolated


def band_powers(
    freqs: NDArray[np.float64],
    psd: NDArray[np.float64],
    bands: dict[str, tuple[float, float]],
) -> NDArray[np.float64]:
    """Integrate a PSD over each frequency band.

    Integrates the piecewise-linear PSD over exactly [low_hz, high_hz], inserting
    linearly interpolated values at the two band edges before applying the
    trapezoid rule. Two properties follow, and both are tested:

    - Adjacent bands tile. The five default bands sum to the integral over 1-50 Hz
      with no gap and no double counting.
    - The result does not shift when Welch resolution changes. Integrating only the
      bins inside [low_hz, high_hz) would drop one bin-width per band -- at 1 Hz
      resolution delta would cover 1-3 Hz instead of 1-4 Hz, a 33% shortfall, and
      the shortfall would change with nperseg.

    A band lying entirely outside the PSD frequency range yields 0.0, not NaN.

    Args:
        freqs: (n_freqs,) Hz, ascending.
        psd: (..., n_channels, n_freqs) V^2/Hz.
        bands: Name -> (low_hz, high_hz).

    Returns:
        (..., n_channels, n_bands) in V^2, column order matching `bands`.
    """
    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    out = np.zeros((*p.shape[:-1], len(bands)), dtype=np.float64)
    if f.size < 2:
        return out

    for i, (low_hz, high_hz) in enumerate(bands.values()):
        low = max(float(low_hz), float(f[0]))
        high = min(float(high_hz), float(f[-1]))
        if high <= low:
            continue
        inner = (f > low) & (f < high)
        grid = np.concatenate(([low], f[inner], [high]))
        values = np.concatenate(
            (
                _interpolate_at(f, p, low)[..., None],
                p[..., inner],
                _interpolate_at(f, p, high)[..., None],
            ),
            axis=-1,
        )
        out[..., i] = np.trapezoid(values, grid, axis=-1)
    return out


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

    Non-finite or negative powers are treated as zero before either transform. A
    channel with zero total power therefore yields all-zero relative values and
    log10(log_epsilon) absolute values -- finite, and flagged by the quality mask.

    Args:
        powers: (..., n_channels, n_bands) in V^2.
        normalization: Which transform to apply.
        log_epsilon: Floor added before log10 so a flat channel stays finite.

    Returns:
        (..., n_channels, n_bands) for a single mode, or (..., n_channels,
        2 * n_bands) for "both", relative columns first. Always finite.

    Raises:
        ValueError: If normalization is unknown or log_epsilon is not positive.
    """
    modes = _modes(normalization)
    if log_epsilon <= 0.0:
        raise ValueError(f"log_epsilon must be positive, got {log_epsilon}")

    raw = np.asarray(powers, dtype=np.float64)
    p = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)

    parts: list[NDArray[np.float64]] = []
    with np.errstate(over="ignore"):
        for mode in modes:
            if mode == "rel":
                total = p.sum(axis=-1, keepdims=True)
                parts.append(np.divide(p, total, out=np.zeros_like(p), where=total > 0.0))
            else:
                parts.append(np.log10(p + log_epsilon))
    return np.concatenate(parts, axis=-1)


def _window_flags(
    non_finite: NDArray[np.bool_],
    flat: NDArray[np.bool_],
    clipping: NDArray[np.bool_],
    bad_fraction: float,
    window_ok: bool,
    ch_names: tuple[str, ...],
) -> tuple[str, ...]:
    flags: list[str] = []
    for label, row in (("non_finite", non_finite), ("flat", flat), ("clipping", clipping)):
        bad = np.flatnonzero(row)
        if bad.size:
            flags.append(f"{label}:{','.join(ch_names[k] for k in bad)}")
    if not window_ok:
        flags.append(f"bad_channel_fraction:{bad_fraction:.2f}")
    return tuple(flags)


def assess_quality(
    windows: NDArray[np.float64],
    ch_names: tuple[str, ...],
    config: QualityConfig,
) -> QualityReport:
    """Flag flat, clipping, and non-finite channels.

    Never raises on signal content and never rejects. A window full of NaN produces
    a report with window_ok False and a flag string; the inference path returns a
    low-confidence result and lets session logic decide.

    Checks, per (window, channel):
        - non-finite samples present
        - within-window std below config.flat_std_v (dead electrode)
        - peak-to-peak above config.max_peak_to_peak_v (clipping or motion)

    A window is not-ok when the bad-channel fraction exceeds
    config.max_bad_channel_fraction. A window with no samples or no channels is
    always not-ok.

    Flags name the offending channels, one entry per category, e.g.
    ("flat:FC5,C3", "clipping:Fp1", "bad_channel_fraction:0.23"). The fraction entry
    appears only when the window is not-ok.

    Args:
        windows: (n_windows, n_channels, n_samples), volts.
        ch_names: Length n_channels, used to name channels in the flag strings.
        config: Thresholds.

    Returns:
        A QualityReport aligned with the window axis.

    Raises:
        ValueError: If windows is not 3-D or ch_names does not match the channel
            axis. Structural errors only -- never signal content.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError(f"expected (n_windows, n_channels, n_samples), got shape {w.shape}")
    n_windows, n_channels, n_samples = w.shape
    if n_channels != len(ch_names):
        raise ValueError(f"windows have {n_channels} channels but {len(ch_names)} ch_names")

    if n_samples == 0 or n_channels == 0:
        empty = np.zeros((n_windows, n_channels), dtype=np.bool_)
        reason = "empty_window" if n_samples == 0 else "no_channels"
        return QualityReport(
            window_ok=np.zeros(n_windows, dtype=np.bool_),
            channel_ok=empty,
            flags=tuple((reason,) for _ in range(n_windows)),
        )

    finite_samples = np.isfinite(w)
    non_finite = ~finite_samples.all(axis=-1)
    safe = np.where(finite_samples, w, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        std = safe.std(axis=-1)
        peak_to_peak = safe.max(axis=-1) - safe.min(axis=-1)
    flat = ~non_finite & (std < config.flat_std_v)
    clipping = ~non_finite & (peak_to_peak > config.max_peak_to_peak_v)

    channel_ok = ~(non_finite | flat | clipping)
    bad_fraction = (~channel_ok).sum(axis=1) / n_channels
    window_ok = bad_fraction <= config.max_bad_channel_fraction

    flags = tuple(
        _window_flags(
            non_finite[i],
            flat[i],
            clipping[i],
            float(bad_fraction[i]),
            bool(window_ok[i]),
            ch_names,
        )
        for i in range(n_windows)
    )
    return QualityReport(window_ok=window_ok, channel_ok=channel_ok, flags=flags)


def extract_features(
    window_set: WindowSet,
    feature_config: FeatureConfig,
    quality_config: QualityConfig,
) -> FeatureMatrix:
    """Band-power features for every window, with the quality mask attached.

    Total over signal content: any well-formed WindowSet -- including zero windows,
    windows shorter than a Welch segment, and windows full of NaN, inf, or flat
    channels -- returns a FeatureMatrix whose `values` are all finite. Bad channels
    still get features computed -- they are flagged in `quality`, not dropped -- so
    the matrix keeps a fixed column count regardless of signal quality. Non-finite
    intermediates are floored (see normalize_band_powers), never left as NaN.

    Args:
        window_set: Preprocessed, windowed signal.
        feature_config: Welch parameters, bands, normalization.
        quality_config: Quality thresholds.

    Returns:
        A FeatureMatrix with values (n_windows, n_features), provenance copied from
        the WindowSet, and quality aligned row-wise. A WindowSet with zero windows
        yields a (0, n_features) matrix rather than an error.

    Raises:
        ValueError: Only if the WindowSet is structurally inconsistent (data not
            3-D, or ch_names not matching the channel axis).
    """
    data = np.asarray(window_set.data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected (n_windows, n_channels, n_samples), got shape {data.shape}")
    n_windows, n_channels, n_samples = data.shape
    bands = feature_config.bands
    n_bands = len(bands)
    normalization = feature_config.normalization

    names = feature_names(window_set.ch_names, bands, normalization)
    quality = assess_quality(data, window_set.ch_names, quality_config)

    if n_windows == 0 or n_samples < 2:
        powers = np.zeros((n_windows, n_channels, n_bands), dtype=np.float64)
    else:
        freqs, psd = welch_psd(data, window_set.sfreq, feature_config)
        powers = band_powers(freqs, psd, bands)

    normalized = normalize_band_powers(powers, normalization, feature_config.log_epsilon)
    # normalized is (n_windows, n_channels, n_modes * n_bands), modes in _modes order.
    # Flatten each mode block separately so columns come out mode-major, matching
    # feature_names.
    blocks = [
        normalized[..., k * n_bands : (k + 1) * n_bands].reshape(n_windows, n_channels * n_bands)
        for k in range(len(_modes(normalization)))
    ]

    return FeatureMatrix(
        values=np.concatenate(blocks, axis=1),
        feature_names=names,
        quality=quality,
        onsets_s=np.asarray(window_set.onsets_s, dtype=np.float64),
        subject_id=window_set.subject_id,
        run=window_set.run,
        condition=window_set.condition,
    )
