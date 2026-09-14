"""Run the Phase 1 identification baseline and write the committed artifacts.

Runs four evaluations, because one number would not be interpretable:

    1. relative     x cross-condition   <- the headline
    2. relative     x temporal
    3. absolute_log x cross-condition   <- the confound comparison (D-004)
    4. absolute_log x temporal

The impostor holdout is loaded from config/impostor_holdout.json -- never re-derived
from the seed -- and its exclusion is asserted rather than assumed
(neuroauth.cohorts.assert_holdout_excluded).

Every run also fires the shuffled-label control and records the chance level. If the
control does not collapse to within 3x chance, the script aborts rather than writing
artifacts. The control catches label leakage outside the train association;
window-overlap leakage is ruled out structurally by assert_no_window_overlap (D-007).

Windows flagged by the quality mask are scored, not excluded. Their count is reported
next to each score, and per-recording flag rates go to artifacts/quality_flag_rates.csv
(D-015).

For the eyes-open-trained models, the share of importance on Fp1/Fp2/AF7/AF8 is
reported against its uniform baseline, to check for the EOG confound (D-004b).

Usage:
    python -m scripts.train_baseline [--subjects N] [--out artifacts/]
"""


def main() -> int:
    """Entry point. Returns a process exit code."""
    raise NotImplementedError("TODO(phase-1): baseline training + evaluation driver")


if __name__ == "__main__":
    raise SystemExit(main())
