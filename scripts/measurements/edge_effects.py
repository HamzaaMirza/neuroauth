"""Filter edge effects on short live buffers: evidence for the bounded-context design.

Backs docs/PHASE2_CONTRACTS.md section 1 (D-020 once folded into DECISIONS.md): every
window's features come from zero-phase filtering over the window plus 2 s of raw context
on each side, identically offline and live.

This is evidence for a design decision, not a result. No threshold judges anything here.
It measures signal-processing properties on one enrollable recording, S001R01. Eyes-open
is the harder case: blinks put large slow transients into delta. Nothing identity-bearing
is computed, and only summary statistics of feature differences are written, never a
feature vector (hard rule 3).

Measurements, all as |difference| in relative band power against whole-recording
`preprocess` (the Phase 1 path), over interior windows:

    1. Settling time of the notch+bandpass impulse response. Data-free.
    2. Scale reference: relative band power and its window-to-window spread.
    3. Zero-phase `preprocess` on each isolated 2 s buffer (the trap).
    4. Zero-phase `preprocess` over a bounded context, for a grid of left and right margins.
    5. Causal filtering with carried state (the rejected alternative): chunk invariance,
       distance from Phase 1 features with the cascade applied once and twice, and
       cold-start convergence.

Usage:
    python -m scripts.measurements.edge_effects
"""

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, load_holdout_record
from neuroauth.config import PipelineConfig
from neuroauth.dsp.features import extract_features
from neuroauth.dsp.io import load_recording
from neuroauth.dsp.preprocess import preprocess
from neuroauth.dsp.types import WindowSet
from scripts.train_baseline import (
    PreconditionError,
    check_holdout_committed,
    working_tree_changes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "measurements" / "edge_effects.json"

SUBJECT = 1
RUN = 1
INTERIOR_START_S = 12.0
"""Interior windows start here and end INTERIOR_END_MARGIN_S before the recording does, so
the whole-recording reference is itself far from both edges."""
INTERIOR_END_MARGIN_S = 4.0
MARGINS_S = (1.0, 2.0, 3.0)
SETTLING_TOLERANCES = (1e-2, 1e-3, 1e-4)
IMPULSE_LENGTH_S = 30.0
CHUNK_SEED = 0
MAX_CHUNK_SAMPLES = 400
COLD_START_OFFSET_S = 20.0
COLD_START_EXTRA_SAMPLES = 37
"""Puts the cold start off the hop grid, so it is not a special case."""
COLD_START_TIMES_S = (1.0, 2.0, 3.0, 4.0, 5.0)


def filter_sections(config: PipelineConfig, sfreq: float) -> NDArray[np.float64]:
    """The notch and bandpass `preprocess` applies, as one causal SOS cascade."""
    pre = config.preprocess
    notch = tf2sos(*iirnotch(pre.notch_freq, pre.notch_q, fs=sfreq))
    band = butter(
        pre.bandpass_order,
        [pre.bandpass_low, pre.bandpass_high],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    return np.vstack([notch, band])


def settling_time_s(sos: NDArray[np.float64], sfreq: float, tolerance: float) -> float:
    """Last time the impulse response envelope exceeds tolerance times its peak."""
    impulse = np.zeros(round(IMPULSE_LENGTH_S * sfreq))
    impulse[0] = 1.0
    envelope = np.abs(sosfilt(sos, impulse))
    return float(np.flatnonzero(envelope > tolerance * envelope.max()).max() / sfreq)


def relative_powers(
    windows: NDArray[np.float64], sfreq: float, ch_names: tuple[str, ...], config: PipelineConfig
) -> NDArray[np.float64]:
    """(n_windows, n_channels, n_bands) relative band power for filtered windows."""
    window_set = WindowSet(
        data=windows,
        sfreq=sfreq,
        ch_names=ch_names,
        onsets_s=np.zeros(windows.shape[0]),
        subject_id=None,
        run=None,
        condition=None,
    )
    values = extract_features(window_set, config.features, config.quality).values
    return values.reshape(windows.shape[0], len(ch_names), len(config.features.bands))


def summary(diff: NDArray[np.float64]) -> dict[str, float]:
    magnitude = np.abs(diff)
    return {
        "median": float(np.median(magnitude)),
        "p95": float(np.percentile(magnitude, 95)),
        "max": float(magnitude.max()),
    }


def per_band(diff: NDArray[np.float64], bands: Sequence[str]) -> dict[str, dict[str, float]]:
    return {band: summary(diff[..., i]) for i, band in enumerate(bands)}


def causal_filter(
    sos: NDArray[np.float64],
    data: NDArray[np.float64],
    chunk_sizes: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Causal sosfilt, state initialized to steady state at the first sample.

    With chunk_sizes, the signal is filtered chunk by chunk, carrying the filter state
    across chunk boundaries, as a live stream would.
    """
    state = np.moveaxis(sosfilt_zi(sos)[:, :, None] * data[:, 0][None, None, :], 2, 1)
    if chunk_sizes is None:
        filtered, _ = sosfilt(sos, data, axis=-1, zi=state)
        return np.asarray(filtered, dtype=np.float64)
    pieces: list[NDArray[np.float64]] = []
    position = 0
    for size in chunk_sizes:
        piece, state = sosfilt(sos, data[:, position : position + size], axis=-1, zi=state)
        pieces.append(piece)
        position += size
    return np.concatenate(pieces, axis=1)


def random_chunk_sizes(n_samples: int, seed: int, max_chunk: int) -> list[int]:
    rng = np.random.default_rng(seed)
    sizes: list[int] = []
    while sum(sizes) < n_samples:
        sizes.append(int(rng.integers(1, max_chunk)))
    return sizes


def measure(data_dir: Path, config: PipelineConfig) -> dict[str, Any]:
    recording = load_recording(SUBJECT, RUN, data_dir)
    x, sfreq, ch_names = recording.data, recording.sfreq, recording.ch_names
    n_samples = x.shape[1]
    window = round(config.window.window_s * sfreq)
    hop = round(window * (1.0 - config.window.overlap))
    bands = tuple(config.features.bands)
    starts = np.arange(
        round(INTERIOR_START_S * sfreq),
        n_samples - window - round(INTERIOR_END_MARGIN_S * sfreq),
        hop,
    )

    def windows_of(signal: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.stack([signal[:, s : s + window] for s in starts])

    def powers(windows: NDArray[np.float64]) -> NDArray[np.float64]:
        return relative_powers(windows, sfreq, ch_names, config)

    reference = powers(windows_of(preprocess(x, sfreq, config.preprocess)))
    single = filter_sections(config, sfreq)
    double = np.vstack([single, single])

    results: dict[str, Any] = {
        "settling_time_s": {
            "single_cascade": {
                f"{t:g}": settling_time_s(single, sfreq, t) for t in SETTLING_TOLERANCES
            },
            "double_cascade": {
                f"{t:g}": settling_time_s(double, sfreq, t) for t in SETTLING_TOLERANCES
            },
        },
        "scale_reference": {
            band: {
                "median_value": float(np.median(reference[..., i])),
                "median_within_recording_std": float(np.median(reference[..., i].std(axis=0))),
            }
            for i, band in enumerate(bands)
        },
    }

    naive = powers(
        np.stack([preprocess(x[:, s : s + window], sfreq, config.preprocess) for s in starts])
    )
    results["naive_per_buffer_zero_phase"] = {
        "pooled": summary(naive - reference),
        "per_band": per_band(naive - reference, bands),
    }

    grid: dict[str, Any] = {}
    for left_s in MARGINS_S:
        for right_s in MARGINS_S:
            left, right = round(left_s * sfreq), round(right_s * sfreq)
            bounded = powers(
                np.stack(
                    [
                        preprocess(x[:, s - left : s + window + right], sfreq, config.preprocess)[
                            :, left : left + window
                        ]
                        for s in starts
                    ]
                )
            )
            grid[f"left_{left_s:g}s_right_{right_s:g}s"] = {
                "pooled": summary(bounded - reference),
                "per_band": per_band(bounded - reference, bands),
            }
    results["bounded_context_zero_phase"] = grid

    one_shot = causal_filter(single, x)
    chunked = causal_filter(single, x, random_chunk_sizes(n_samples, CHUNK_SEED, MAX_CHUNK_SAMPLES))
    causal: dict[str, Any] = {
        "chunked_vs_one_shot": {
            "bit_identical": bool(np.array_equal(one_shot, chunked)),
            "max_abs_diff_v": float(np.abs(one_shot - chunked).max()),
        }
    }
    cold_offset = round(COLD_START_OFFSET_S * sfreq) + COLD_START_EXTRA_SAMPLES
    for name, sos in (("single_cascade", single), ("double_cascade", double)):
        filtered = causal_filter(sos, x)
        cold = causal_filter(sos, x[:, cold_offset:])
        cold_max: dict[str, float] = {}
        for t in COLD_START_TIMES_S:
            a = round(t * sfreq)
            diff = powers(cold[None, :, a : a + window]) - powers(
                filtered[None, :, cold_offset + a : cold_offset + a + window]
            )
            cold_max[f"{t:g}"] = float(np.abs(diff).max())
        causal[name] = {
            "vs_phase1_zero_phase": per_band(powers(windows_of(filtered)) - reference, bands),
            "cold_start_max_abs_diff_by_window_start_s": cold_max,
        }
    results["causal_stateful"] = causal

    return {
        "recording": {"subject": SUBJECT, "run": RUN, "n_samples": n_samples, "sfreq": sfreq},
        "n_interior_windows": int(starts.size),
        "results": results,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    results = payload["results"]
    print(f"settling time, single cascade: {results['settling_time_s']['single_cascade']}")
    for label, key in (
        ("naive per-buffer", "naive_per_buffer_zero_phase"),
        ("bounded 2 s + 2 s", None),
    ):
        entry = results[key] if key else results["bounded_context_zero_phase"]["left_2s_right_2s"]
        pooled = entry["pooled"]
        delta = entry["per_band"]["delta"]
        print(
            f"{label:<20} pooled p95 {pooled['p95']:.1e} max {pooled['max']:.1e}   "
            f"delta median {delta['median']:.4f} p95 {delta['p95']:.4f}"
        )
    for name, entry in results["bounded_context_zero_phase"].items():
        print(
            f"  {name:<22} pooled p95 {entry['pooled']['p95']:.1e} max {entry['pooled']['max']:.1e}"
        )
    causal = results["causal_stateful"]
    identical = causal["chunked_vs_one_shot"]["bit_identical"]
    print(f"causal chunked vs one-shot bit-identical: {identical}")
    for name in ("single_cascade", "double_cascade"):
        delta = causal[name]["vs_phase1_zero_phase"]["delta"]
        print(
            f"causal {name:<15} vs Phase 1 delta "
            f"median {delta['median']:.4f} p95 {delta['p95']:.4f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure filter edge effects on S001R01.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / DEFAULT_HOLDOUT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    try:
        git_head = check_holdout_committed(args.holdout, REPO_ROOT)
        changes = working_tree_changes(REPO_ROOT)
    except PreconditionError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    if SUBJECT in load_holdout_record(args.holdout).impostor_holdout:
        print(f"refusing to run: subject {SUBJECT} is in the impostor holdout", file=sys.stderr)
        return 2

    config = PipelineConfig()
    payload = {
        "what_this_is": (
            "Evidence for the bounded-context filtering design (docs/PHASE2_CONTRACTS.md "
            "section 1). Signal-processing properties of one enrollable recording; not a "
            "result, and judged by no threshold. Values are |differences| in relative band "
            "power against whole-recording zero-phase filtering."
        ),
        "git_head": git_head,
        "git_dirty": bool(changes),
        "config_fingerprint": config.fingerprint(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        **measure(args.data_dir, config),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(payload)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
