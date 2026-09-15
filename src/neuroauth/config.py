"""Frozen pipeline configuration.

These objects are provenance, not just parameters. Every model artifact and every
session row records the config that produced it, so a stored score can always be
traced back to the exact transform that generated it.
"""

import dataclasses
import hashlib
import json
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

TEMPORAL_EMG_CHANNELS: Final[tuple[str, ...]] = (
    "FT7",
    "FT8",
    "T7",
    "T8",
    "TP7",
    "TP8",
    "T9",
    "T10",
)
"""The lateral temporal electrodes, over the temporalis muscle, where scalp muscle
activity (EMG) is strongest. Chosen by anatomy -- every lateral temporal site in the
montage -- rather than by importance rank, and fixed before the ablation run (D-018).
Iz, near the neck muscles, is not included."""

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
            directly, which lets a model score well partly by recognizing a
            recording's broadband amplitude. Amplitude also reflects anatomy, which
            is person-specific, and one session cannot separate the two.
            "absolute_log" is run as a comparison, not as the headline; the gap
            between the two is an upper bound on the session-artifact contribution
            (D-004).
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

        The config is serialized canonically -- dataclass fields keyed by name, dicts
        kept as ordered [key, value] pairs, floats in Python's shortest round-trip
        form -- and hashed with SHA-256. Band order is deliberately part of the
        fingerprint: it fixes feature column order, so two configs listing the same
        bands in a different order produce incompatible matrices and must not share
        a fingerprint.

        Returns:
            The first 16 hex characters of the SHA-256 digest.
        """
        canonical = {"fingerprint_version": _FINGERPRINT_VERSION, "config": _canonical(self)}
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


_FINGERPRINT_VERSION = 1
"""Bump if the serialization changes, so old and new fingerprints can never collide."""


def _canonical(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return [[str(key), _canonical(item)] for key, item in value.items()]
    if isinstance(value, tuple | list):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True)
class ContextConfig:
    """Raw context filtered around every window, on the training, enrollment, and live paths.

    Each window's features come from filtering the raw slice [start - left_margin_s,
    start + window_s + right_margin_s) with `preprocess` and cropping back to the window
    (neuroauth.dsp.streaming). Zero-phase filtering needs samples on both sides of a
    window. On a live stream the right-hand samples are in the future, so the right
    margin is also decision latency: it adds directly to time-to-detect.

    Both margins must cover the settling time of the notch-plus-bandpass impulse response,
    which is 1.58 s to 1e-3 of peak at the defaults, rounded up to the 1 s hop
    (streaming.check_margins). Measured on S001R01 against whole-recording filtering, 2 s
    margins keep every relative band power within 1.4e-3. A 1 s margin on either side
    allows 1.3e-2.

    Attributes:
        left_margin_s: Raw seconds before the window start.
        right_margin_s: Raw seconds after the window end.
    """

    left_margin_s: float = 2.0
    right_margin_s: float = 2.0


@dataclass(frozen=True)
class StreamingConfig:
    """Everything that fixes a Phase 2 feature vector: PipelineConfig plus the context margins.

    PipelineConfig is left unchanged, so every Phase 1 fingerprint still reproduces. Phase 2
    features come from bounded contexts, so they are close to but not identical to Phase 1's
    whole-recording features. They carry this fingerprint instead, and the two can never be
    confused for each other.
    """

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    def fingerprint(self) -> str:
        """Stable short hash of the pipeline and context settings.

        Serialized the same canonical way as PipelineConfig.fingerprint, but under a
        distinct namespace key, so it can never equal a PipelineConfig fingerprint even
        in principle.

        Returns:
            The first 16 hex characters of the SHA-256 digest.
        """
        raise NotImplementedError("TODO(phase-2): namespaced fingerprint over pipeline + context")
