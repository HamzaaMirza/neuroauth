"""Register subjects in Postgres and fix the impostor holdout. Run once, first.

Assigns every subject a cohort using the seeded, nested selection in
neuroauth.cohorts, writes the subjects rows, and emits a 'cohort_assigned' event per
subject so the assignment has provenance.

This must run before any Phase 1 training, and the resulting impostor list is copied
into docs/DECISIONS.md. Re-running with a different seed is a destructive act: it
re-rolls the holdout and invalidates every FAR estimate that came before.

Usage:
    python -m scripts.ingest_subjects [--dry-run]
"""


def main() -> int:
    """Entry point. Returns a process exit code."""
    raise NotImplementedError("TODO(phase-1): subject ingest + cohort assignment")


if __name__ == "__main__":
    raise SystemExit(main())
