"""Impostor-holdout selection is deterministic and nested; the committed file wins."""

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuroauth.cohorts import (
    DEFAULT_HOLDOUT_PATH,
    assert_holdout_excluded,
    create_holdout_record,
    load_holdout_record,
    partition_cohorts,
    select_impostor_holdout,
    write_holdout_record,
)
from neuroauth.config import HoldoutConfig

SUBJECTS = tuple(range(1, 110))
CONFIG = HoldoutConfig(seed=20260821, n_impostors=20)
FIXED_NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, Any]:
    record = create_holdout_record(SUBJECTS, CONFIG, now=FIXED_NOW)
    payload: dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(record)))
    return payload


def test_selection_is_deterministic() -> None:
    """Same seed and count give the same holdout on every call."""
    first = select_impostor_holdout(SUBJECTS, CONFIG)
    assert first == select_impostor_holdout(SUBJECTS, CONFIG)
    assert len(first) == 20
    assert first <= set(SUBJECTS)


def test_different_seed_gives_different_holdout() -> None:
    other = HoldoutConfig(seed=CONFIG.seed + 1, n_impostors=20)
    assert select_impostor_holdout(SUBJECTS, other) != select_impostor_holdout(SUBJECTS, CONFIG)


def test_selection_is_order_independent() -> None:
    """Shuffling the input subject list does not change the result."""
    shuffled = [int(s) for s in np.random.default_rng(0).permutation(SUBJECTS)]
    assert select_impostor_holdout(shuffled, CONFIG) == select_impostor_holdout(SUBJECTS, CONFIG)


def test_selection_is_nested_in_count() -> None:
    """Raising n_impostors keeps every previously selected subject.

    This is what lets Phase 2 revise the count against a FAR stability argument
    without re-rolling a commitment already made.
    """
    smaller = select_impostor_holdout(SUBJECTS, HoldoutConfig(seed=CONFIG.seed, n_impostors=20))
    larger = select_impostor_holdout(SUBJECTS, HoldoutConfig(seed=CONFIG.seed, n_impostors=30))
    assert smaller < larger


def test_cohorts_partition_exactly() -> None:
    """Enrollable and holdout are disjoint and together cover every subject."""
    enrollable, holdout = partition_cohorts(SUBJECTS, CONFIG)
    assert set(enrollable).isdisjoint(holdout)
    assert sorted(enrollable + holdout) == list(SUBJECTS)
    assert list(enrollable) == sorted(enrollable)
    assert list(holdout) == sorted(holdout)
    assert (len(enrollable), len(holdout)) == (89, 20)


def test_duplicate_subject_ids_are_rejected() -> None:
    """Duplicates raise rather than being silently de-duplicated."""
    with pytest.raises(ValueError, match="duplicate"):
        select_impostor_holdout((1, 2, 2, 3), HoldoutConfig(seed=1, n_impostors=1))


@pytest.mark.parametrize("n_impostors", [0, 109, 200])
def test_both_cohorts_must_be_non_empty(n_impostors: int) -> None:
    with pytest.raises(ValueError):
        select_impostor_holdout(SUBJECTS, HoldoutConfig(seed=1, n_impostors=n_impostors))


def test_record_round_trips_through_json(tmp_path: Path) -> None:
    record = create_holdout_record(SUBJECTS, CONFIG, now=FIXED_NOW)
    path = tmp_path / "config" / "impostor_holdout.json"
    write_holdout_record(record, path)
    assert load_holdout_record(path) == record
    assert record.selected_at_utc == "2026-09-14T12:00:00+00:00"
    assert b"\r\n" not in path.read_bytes()


def test_timestamp_is_normalized_to_utc() -> None:
    eastern = datetime(2026, 9, 14, 8, 0, 0, tzinfo=timezone(timedelta(hours=-4)))
    record = create_holdout_record(SUBJECTS, CONFIG, now=eastern)
    assert record.selected_at_utc == "2026-09-14T12:00:00+00:00"


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        create_holdout_record(SUBJECTS, CONFIG, now=datetime(2026, 9, 14, 12, 0, 0))


def test_existing_record_is_never_overwritten(tmp_path: Path) -> None:
    """A second run cannot re-roll the holdout, even with different settings."""
    path = tmp_path / "impostor_holdout.json"
    write_holdout_record(create_holdout_record(SUBJECTS, CONFIG, now=FIXED_NOW), path)
    original = path.read_bytes()
    rerolled = create_holdout_record(SUBJECTS, HoldoutConfig(seed=1, n_impostors=20))
    with pytest.raises(FileExistsError):
        write_holdout_record(rerolled, path)
    assert path.read_bytes() == original


def test_loading_does_not_rederive_from_seed(tmp_path: Path) -> None:
    """The stored list wins even when the seed would now derive something else.

    Simulates seed or numpy drift: the record's seed no longer produces its list, and
    the list is still loaded exactly as written.
    """
    payload = _payload()
    payload["seed"] = 1
    path = tmp_path / "impostor_holdout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_holdout_record(path)
    assert loaded.impostor_holdout == tuple(payload["impostor_holdout"])
    assert set(loaded.impostor_holdout) != select_impostor_holdout(
        SUBJECTS, HoldoutConfig(seed=1, n_impostors=20)
    )


def _tamper(payload: dict[str, Any], how: str) -> None:
    if how == "overlap":
        payload["enrollable"] = sorted(payload["enrollable"] + payload["impostor_holdout"][:1])
    elif how == "wrong_count":
        payload["n_impostors"] = 21
    elif how == "missing_subject":
        payload["enrollable"] = payload["enrollable"][1:]
    elif how == "unsorted":
        payload["impostor_holdout"] = list(reversed(payload["impostor_holdout"]))
    elif how == "not_utc":
        payload["selected_at_utc"] = "2026-09-14T08:00:00-04:00"
    elif how == "unknown_field":
        payload["note"] = "hand-edited"


@pytest.mark.parametrize(
    "how", ["overlap", "wrong_count", "missing_subject", "unsorted", "not_utc", "unknown_field"]
)
def test_inconsistent_record_is_rejected(tmp_path: Path, how: str) -> None:
    payload = _payload()
    _tamper(payload, how)
    path = tmp_path / "impostor_holdout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_holdout_record(path)


def test_assert_holdout_excluded_names_leaked_subjects() -> None:
    with pytest.raises(AssertionError, match=r"\[4, 7\]"):
        assert_holdout_excluded([1, 4, 7, 7], frozenset({4, 7, 9}))
    assert_holdout_excluded(np.array([1, 2, 3]), frozenset({4}))


def test_committed_holdout_record_is_consistent() -> None:
    """Validates the committed file once it exists. Never re-derives from the seed."""
    path = REPO_ROOT / DEFAULT_HOLDOUT_PATH
    if not path.exists():
        pytest.skip("impostor holdout not selected yet")
    record = load_holdout_record(path)
    assert record.candidate_subjects == SUBJECTS
    assert len(record.enrollable) + len(record.impostor_holdout) == len(SUBJECTS)
