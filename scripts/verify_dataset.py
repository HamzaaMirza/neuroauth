"""
Fetch and verify the PhysioNet EEG Motor Movement/Imagery baseline runs.

Downloads only R01 (eyes-open) and R02 (eyes-closed) for all 109 subjects —
roughly 218 one-minute recordings, a small fraction of the full corpus.

Usage:
    python verify_dataset.py              # all 109 subjects
    python verify_dataset.py --subjects 5 # first 5 only, for a quick check

Data:  PhysioNet EEG Motor Movement/Imagery Database v1.0.0
License: Open Data Commons Attribution (ODC-By) v1.0
"""

import argparse
import sys

import mne
from mne.datasets import eegbci

BASELINE_RUNS = [1, 2]  # R01 eyes-open, R02 eyes-closed
EXPECTED_CHANNELS = 64
EXPECTED_SFREQ = 160.0
DATA_DIR = "./data"


def check_subject(subject: int) -> dict:
    """Download (if needed) and validate one subject's baseline runs."""
    result = {"subject": subject, "ok": True, "issues": [], "runs": []}

    try:
        paths = eegbci.load_data(
            subjects=[subject], runs=BASELINE_RUNS, path=DATA_DIR, verbose="ERROR"
        )
    except Exception as exc:
        result["ok"] = False
        result["issues"].append(f"download failed: {exc}")
        return result

    for path in paths:
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        except Exception as exc:
            result["ok"] = False
            result["issues"].append(f"{path}: read failed: {exc}")
            continue

        n_ch = len(raw.ch_names)
        sfreq = raw.info["sfreq"]
        duration = raw.n_times / sfreq

        if n_ch != EXPECTED_CHANNELS:
            result["ok"] = False
            result["issues"].append(f"{path}: {n_ch} channels, expected {EXPECTED_CHANNELS}")
        if sfreq != EXPECTED_SFREQ:
            result["ok"] = False
            result["issues"].append(f"{path}: {sfreq} Hz, expected {EXPECTED_SFREQ}")
        if duration < 50:
            result["ok"] = False
            result["issues"].append(f"{path}: only {duration:.1f}s, expected ~60s")

        result["runs"].append({"path": str(path), "n_ch": n_ch, "duration_s": round(duration, 1)})

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=109, help="how many subjects to fetch")
    args = parser.parse_args()

    print(f"Fetching baseline runs for subjects 1-{args.subjects} into {DATA_DIR}/")
    print("First run downloads; later runs use the cache.\n")

    results = []
    for subject in range(1, args.subjects + 1):
        res = check_subject(subject)
        results.append(res)
        status = "ok " if res["ok"] else "FAIL"
        print(f"  [{status}] subject {subject:3d}  ({len(res['runs'])} runs)", flush=True)
        for issue in res["issues"]:
            print(f"         {issue}")

    good = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    print(f"\n{'=' * 60}")
    print(f"Usable subjects: {len(good)}/{len(results)}")
    if bad:
        print(f"Problem subjects: {[r['subject'] for r in bad]}")
        print("Note: a handful of subjects are known to have irregular recordings.")
        print("Excluding them is normal and documented in the literature — record")
        print("which ones you dropped and why in docs/DECISIONS.md.")

    # Channel-name sanity check on the first good subject.
    if good:
        raw = mne.io.read_raw_edf(good[0]["runs"][0]["path"], preload=False, verbose="ERROR")
        print(f"\nRaw channel names (first 8): {raw.ch_names[:8]}")
        eegbci.standardize(raw)
        print(f"After eegbci.standardize: {raw.ch_names[:8]}")
        print("\nHeads up: this dataset ships channel names with trailing dots")
        print("('Fc5.', 'Fcz.'). Call eegbci.standardize(raw) before setting a")
        print("montage or the 10-10 positions will not match.")

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
