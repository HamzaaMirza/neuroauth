"""Fix the impostor holdout by writing config/impostor_holdout.json. Run exactly once.

Must run before any identification or verification result exists (D-008). From then
on the file -- not the seed -- is the source of truth: every consumer loads the
subject list from it, and scripts/ingest_subjects.py asserts the database agrees
with it. Nothing ever re-derives the selection from the seed, so a numpy or code
change cannot silently produce a different holdout.

All 109 subjects are candidates: every baseline recording passed the structural
check (D-014).

Refuses to overwrite an existing file. This script commits nothing. Commit the file
on its own, before any results, with a message marking it pre-results.

Usage:
    python -m scripts.select_holdout
"""

import sys
from pathlib import Path

from neuroauth.cohorts import DEFAULT_HOLDOUT_PATH, create_holdout_record, write_holdout_record
from neuroauth.config import HoldoutConfig
from neuroauth.dsp.io import N_SUBJECTS

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Entry point. Returns a process exit code."""
    path = REPO_ROOT / DEFAULT_HOLDOUT_PATH
    record = create_holdout_record(range(1, N_SUBJECTS + 1), HoldoutConfig())
    try:
        write_holdout_record(record, path)
    except FileExistsError:
        print(f"{path} already exists. The holdout is fixed; not overwriting.", file=sys.stderr)
        return 1

    print(f"wrote {path}")
    print(f"  seed:             {record.seed}")
    print(f"  n_impostors:      {record.n_impostors}")
    print(f"  selected_at_utc:  {record.selected_at_utc}")
    print(f"  impostor_holdout: {list(record.impostor_holdout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
