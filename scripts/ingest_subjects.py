"""Register subjects in Postgres from the committed holdout file. Deferred with Docker.

Reads config/impostor_holdout.json -- written once by scripts/select_holdout.py and
committed on its own before any results -- and writes one subjects row per candidate
with its cohort, emitting a 'cohort_assigned' event per subject for provenance.

It never re-derives the selection from the seed. On every run it asserts that the
cohort of every existing subjects row matches the file, and aborts on any mismatch.
A seed change, a numpy version change, or a code change therefore cannot silently
produce a different holdout in the database.

Usage:
    python -m scripts.ingest_subjects [--dry-run]
"""


def main() -> int:
    """Entry point. Returns a process exit code."""
    raise NotImplementedError("TODO(phase-1): subject ingest from the committed holdout file")


if __name__ == "__main__":
    raise SystemExit(main())
