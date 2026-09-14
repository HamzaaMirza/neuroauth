"""Frozen pipeline configuration.

These objects are provenance, not just parameters. Every model artifact and every
session row records the config that produced it, so a stored score can always be
traced back to the exact transform that generated it.
"""

from dataclasses import dataclass, field
from typing import Final, Literal

BANDS: Final[dict[str, tuple[float, float]]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
"""Gamma stops at 50 Hz because the bandpass does. Anything above the passband
edge is filter roll-off, not signal.

Delta gets roughly three Welch bins at the default resolution (see
FeatureConfig.welch_nperseg). That is thin. Check its feature importance once the
baseline is trained; if it contributes nothing, drop the band and record why.
"""

FRONTAL_EOG_CHANNELS: Final[tuple[str, ...]] = ("Fp1", "Fp2", "AF7", "AF8")
"""Electrodes directly above the eyes, where blinks and eye movements dominate. On
eyes-open-trained models, importance concentrating here signals the EOG confound
(D-004b). Spelled as the loader's standardized 10-10 names."""

Normalization = Literal["relative", "absolute_log", "both"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Filter settings. Applied in order: notch, then bandpass, then reference.

    Attributes:
        notch_freq: Mains frequency, Hz. 60.0 for US-recorded eegmmidb.
        notch_q: Quality factor of the IIR notch. Higher = narrower stopband.
        notch_harmonics: Harmonics to remove, including the fundamental. Above the
            50 Hz passband edge the harmonics are already suppressed, so 1 is the
            sane default.
        bandpass_low: Hz.
        bandpass_high: Hz.
        bandpass_order: Butterworth order, realized as second-order sections.
        common_average_reference: Subtract the across-channel mean per sample.
            Defaults False for Phase 1. Note that CAR over 64 channels and CAR over
            a reduced montage are different operations -- if the channel-reduction
            experiment happens, this flag is part of that experiment's design, not a
            fixed setting. See docs/DECISIONS.md.
    """

    notch_freq: float = 60.0
    notch_q: float = 30.0
    notch_harmonics: int = 1
    bandpass_low: float = 1.0
    bandpass_high: float = 50.0
    bandpass_order: int = 4
    common_average_reference: bool = False


@dataclass(frozen=True)
class WindowConfig:
    """Windowing settings.

    Window length is also the time-to-detect floor for an impostor swap, which is a
    Phase 2 metric. Do not lengthen it to buy spectral resolution.

    A trailing remainder shorter than window_s is always dropped, never zero-padded:
    padding biases a window's PSD toward low frequencies.

    Attributes:
        window_s: Window length in seconds.
        overlap: Fraction in [0, 1). 0.5 = 50% overlap.
    """

    window_s: float = 2.0
    overlap: float = 0.5


@dataclass(frozen=True)
class FeatureConfig:
    """Welch PSD and band-power settings.

    Attributes:
        bands: Name -> (low_hz, high_hz), inclusive-low, exclusive-high.
        welch_nperseg: Samples per Welch segment. At 160 Hz, 160 gives 1 Hz
            resolution and ~3 averaged segments inside a 2 s window. The alternative
            (320) gives 0.5 Hz resolution but a single unaveraged periodogram.
        welch_noverlap: Samples of overlap between Welch segments.
        welch_window: SciPy window name passed to scipy.signal.welch.
        detrend: Per-window detrend applied before Welch.
        normalization: "relative" = band power / total 1-50 Hz power, per channel.
            "absolute_log" = log10 of band power. "both" = concatenate.

            Defaults to "relative", and that is the headline configuration.
            eegmmidb is single-session, so every recording-specific artifact
            (electrode impedance, cap placement, amplifier gain) is perfectly
            confounded with subject identity. Absolute power carries those offsets
            directly, which lets a model score well by recognizing a recording's
            broadband amplitude without learning anything about the person.
            "absolute_log" is run as a comparison, not as the headline; the gap
            between the two is a reportable finding. See docs/DECISIONS.md.
        log_epsilon: Added before log10 to keep a flat channel finite.
    """

    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(BANDS))
    welch_nperseg: int = 160
    welch_noverlap: int = 80
    welch_window: str = "hann"
    detrend: Literal["constant", "linear", "none"] = "constant"
    normalization: Normalization = "relative"
    log_epsilon: float = 1e-20


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the quality mask. Volts throughout.

    In Phase 1 the mask is reported, not used to exclude windows from scoring
    (D-015). It starts gating session decisions in Phase 2, where it is recalibrated
    on the enrollable cohort only.

    Attributes:
        flat_std_v: A channel whose within-window std is below this is flat (dead
            electrode). 1e-7 V = 0.1 uV.
        max_peak_to_peak_v: Above this, the channel is flagged as carrying a gross
            artifact. 500e-6 V = 500 uV. The textbook 250 uV flagged 30% of windows
            on eegmmidb -- ocular activity in eyes-open, occipital alpha in
            eyes-closed -- rather than clipping (D-015).
        max_bad_channel_fraction: Fraction of bad channels above which the whole
            window is marked not-ok.
    """

    flat_std_v: float = 1e-7
    max_peak_to_peak_v: float = 500e-6
    max_bad_channel_fraction: float = 0.2


@dataclass(frozen=True)
class HoldoutConfig:
    """Impostor-holdout selection. Fixed before any Phase 1 result is looked at.

    Selection is a seeded permutation truncated to n_impostors, so the chosen set is
    nested: raising n_impostors later keeps every previously selected subject and
    only adds more. That means the count can be revised in Phase 2 against a FAR
    stability argument without invalidating the commitment made here.

    These values produced config/impostor_holdout.json. After that file exists it is
    the source of truth, and nothing re-derives the holdout from these values.

    Attributes:
        seed: Fixed. Changing it re-rolls the holdout and invalidates every Phase 2
            FAR estimate that preceded the change.
        n_impostors: Subjects reserved as impostors -- never enrolled, never trained
            on, never evaluated against until Phase 2.
    """

    seed: int = 20260821
    n_impostors: int = 20


@dataclass(frozen=True)
class PipelineConfig:
    """Everything needed to reproduce a feature vector from an EDF file."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def fingerprint(self) -> str:
        """Stable short hash of the full config.

        Recorded on model artifacts and session rows so a score can be tied to the
        exact transform that produced it. Must be stable across processes and Python
        versions, so it cannot use the built-in hash().

        Returns:
            A short hex digest.
        """
        raise NotImplementedError("TODO(phase-1): stable config fingerprint")
