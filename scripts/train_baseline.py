"""Run the Phase 1 identification baseline and write the committed artifacts.

Runs four evaluations, because one number would not be interpretable:

    1. relative     x cross-condition   <- the headline
    2. relative     x temporal
    3. absolute_log x cross-condition   <- the confound comparison
    4. absolute_log x temporal

Every run also fires the shuffled-label control and records the chance level, so a
leaking split cannot produce a number that looks plausible. If any control fails to
collapse to roughly chance, the script aborts rather than writing artifacts.

The impostor holdout is excluded throughout, and the exclusion is asserted rather
than assumed (neuroauth.cohorts.assert_holdout_excluded).

Usage:
    python -m scripts.train_baseline [--subjects N] [--out artifacts/]
"""


def main() -> int:
    """Entry point. Returns a process exit code."""
    raise NotImplementedError("TODO(phase-1): baseline training + evaluation driver")


if __name__ == "__main__":
    raise SystemExit(main())
