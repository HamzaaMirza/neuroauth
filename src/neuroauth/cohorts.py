"""Impostor-holdout selection, and the committed record that fixes it.

CLAUDE.md hard rule 1: a held-out set of impostor subjects -- never enrolled, never
trained on -- must exist at all times. This module is what makes that set exist.

Picking the holdout after seeing results would be selection bias, and it would
compromise the Phase 2 EER before Phase 2 starts. So the selection runs once, before
any result exists, and is written to config/impostor_holdout.json, which is committed
on its own. From then on the file is the source of truth: every consumer loads the
subject list from it and nothing re-derives it from the seed. The seed documents how
the list was made, but a numpy upgrade or a code change must never be able to
silently produce a different list (D-008).
"""

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from neuroauth.config import HoldoutConfig

DEFAULT_HOLDOUT_PATH = Path("config") / "impostor_holdout.json"
"""Relative to the repository root."""

SELECTION_METHOD = "numpy.random.default_rng(seed).permutation(sorted(candidates))[:n_impostors]"


@dataclass(frozen=True)
class HoldoutRecord:
    """The committed impostor-holdout decision.

    Attributes:
        seed: HoldoutConfig.seed used for the selection.
        n_impostors: HoldoutConfig.n_impostors used for the selection.
        candidate_subjects: Every subject considered, sorted.
        impostor_holdout: The selected impostors, sorted. Never enrolled, never
            trained on, never evaluated against until Phase 2.
        enrollable: candidate_subjects minus impostor_holdout, sorted.
        selected_at_utc: ISO 8601 timestamp with a +00:00 offset.
        selection_method: How the list was derived, for a human reader. Never
            executed.
        numpy_version: numpy version at selection time. numpy does not promise that
            a seeded permutation stays identical across versions, which is one reason
            the list itself is stored.
    """

    seed: int
    n_impostors: int
    candidate_subjects: tuple[int, ...]
    impostor_holdout: tuple[int, ...]
    enrollable: tuple[int, ...]
    selected_at_utc: str
    selection_method: str
    numpy_version: str


def _sorted_candidates(subject_ids: Iterable[int], n_impostors: int) -> list[int]:
    ids = [int(subject) for subject in subject_ids]
    duplicates = sorted(subject for subject, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate subject ids: {duplicates}")
    if not 0 < n_impostors < len(ids):
        raise ValueError(
            f"n_impostors must be in (0, {len(ids)}) so both cohorts are non-empty, "
            f"got {n_impostors}"
        )
    return sorted(ids)


def select_impostor_holdout(
    subject_ids: Sequence[int],
    config: HoldoutConfig,
) -> frozenset[int]:
    """Choose the impostor-holdout subjects deterministically.

    Implemented as a seeded permutation of the sorted subject ids, truncated to
    config.n_impostors. Sorting first means the result does not depend on the order
    the caller happened to discover subjects in. Truncating a permutation means the
    selection is nested in n_impostors, so the count can be revised later without
    re-rolling subjects already committed to the holdout.

    Call this once, through scripts/select_holdout.py. Everything afterwards reads
    the committed record via load_holdout_record.

    Args:
        subject_ids: All available subject numbers. Duplicates are an error, not
            something to silently de-duplicate.
        config: Seed and holdout size.

    Returns:
        The impostor subject ids.

    Raises:
        ValueError: If subject_ids contains duplicates, or n_impostors is not in
            (0, len(subject_ids)).
    """
    candidates = _sorted_candidates(subject_ids, config.n_impostors)
    permuted = np.random.default_rng(config.seed).permutation(
        np.asarray(candidates, dtype=np.int64)
    )
    return frozenset(int(subject) for subject in permuted[: config.n_impostors])


def partition_cohorts(
    subject_ids: Sequence[int],
    config: HoldoutConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split all subjects into the enrollable cohort and the impostor holdout.

    Phase 1's identification baseline trains and evaluates on the enrollable cohort
    only. The holdout subjects are not touched -- not for training, not for
    evaluation, not for a quick sanity check -- until Phase 2.

    Args:
        subject_ids: All available subject numbers.
        config: Seed and holdout size.

    Returns:
        (enrollable, impostor_holdout), both sorted ascending.

    Raises:
        ValueError: As select_impostor_holdout.
    """
    holdout = select_impostor_holdout(subject_ids, config)
    candidates = sorted(int(subject) for subject in subject_ids)
    enrollable = tuple(subject for subject in candidates if subject not in holdout)
    return enrollable, tuple(sorted(holdout))


def create_holdout_record(
    subject_ids: Sequence[int],
    config: HoldoutConfig,
    *,
    now: datetime | None = None,
) -> HoldoutRecord:
    """Select the holdout and package it with its provenance.

    Args:
        subject_ids: All available subject numbers.
        config: Seed and holdout size.
        now: Selection time, normalized to UTC. Defaults to the current time. Must
            be timezone-aware.

    Returns:
        A consistent HoldoutRecord, not yet written anywhere.

    Raises:
        ValueError: As select_impostor_holdout, or if now is timezone-naive.
    """
    moment = datetime.now(UTC) if now is None else now
    if moment.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    enrollable, holdout = partition_cohorts(subject_ids, config)
    return HoldoutRecord(
        seed=config.seed,
        n_impostors=config.n_impostors,
        candidate_subjects=tuple(sorted(enrollable + holdout)),
        impostor_holdout=holdout,
        enrollable=enrollable,
        selected_at_utc=moment.astimezone(UTC).isoformat(timespec="seconds"),
        selection_method=SELECTION_METHOD,
        numpy_version=np.__version__,
    )


def _check_record(record: HoldoutRecord) -> None:
    for name in ("candidate_subjects", "impostor_holdout", "enrollable"):
        values = list(getattr(record, name))
        if values != sorted(set(values)):
            raise ValueError(f"{name} must be sorted and free of duplicates")
    if len(record.impostor_holdout) != record.n_impostors:
        raise ValueError(
            f"impostor_holdout has {len(record.impostor_holdout)} subjects but "
            f"n_impostors is {record.n_impostors}"
        )
    holdout, enrollable = set(record.impostor_holdout), set(record.enrollable)
    both = sorted(holdout & enrollable)
    if both:
        raise ValueError(f"subjects in both cohorts: {both}")
    if holdout | enrollable != set(record.candidate_subjects):
        raise ValueError("impostor_holdout and enrollable do not partition candidate_subjects")
    try:
        stamp = datetime.fromisoformat(record.selected_at_utc)
    except ValueError as exc:
        raise ValueError(f"selected_at_utc is not ISO 8601: {exc}") from None
    if stamp.utcoffset() != timedelta(0):
        raise ValueError(f"selected_at_utc must be UTC, got {record.selected_at_utc!r}")


def write_holdout_record(record: HoldoutRecord, path: Path) -> None:
    """Write the record as JSON, refusing to overwrite an existing file.

    The file is opened in exclusive-create mode, so an existing holdout is never
    touched -- not even truncated -- by a second run.

    Args:
        record: The record to write. Checked for internal consistency first.
        path: Destination. Parent directories are created.

    Raises:
        FileExistsError: If path already exists.
        ValueError: If the record is internally inconsistent.
    """
    _check_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(record), handle, indent=2)
        handle.write("\n")


def load_holdout_record(path: Path) -> HoldoutRecord:
    """Load and validate the committed holdout record.

    Validation is structural only: sorted, duplicate-free, a clean partition, the
    stated count, a UTC timestamp. It deliberately does NOT re-derive the list from
    the seed. If the seed and the list ever disagree -- after a numpy upgrade, say --
    the stored list wins.

    Args:
        path: Location of the JSON record.

    Returns:
        The record exactly as written.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file is not valid JSON, has missing or unexpected fields,
            or is internally inconsistent.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from None

    expected = {field.name for field in fields(HoldoutRecord)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{path}: expected exactly the fields {sorted(expected)}")
    try:
        record = HoldoutRecord(
            seed=int(payload["seed"]),
            n_impostors=int(payload["n_impostors"]),
            candidate_subjects=tuple(int(s) for s in payload["candidate_subjects"]),
            impostor_holdout=tuple(int(s) for s in payload["impostor_holdout"]),
            enrollable=tuple(int(s) for s in payload["enrollable"]),
            selected_at_utc=str(payload["selected_at_utc"]),
            selection_method=str(payload["selection_method"]),
            numpy_version=str(payload["numpy_version"]),
        )
        _check_record(record)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid holdout record: {exc}") from None
    return record


def assert_holdout_excluded(
    used_subject_ids: Iterable[int],
    impostor_holdout: frozenset[int],
) -> None:
    """Assert that no impostor-holdout subject appears in a training or eval set.

    Called on every training-set construction, not only in tests -- the same posture
    hard rule 5 takes for holdout integrity in Phase 3. This is the Phase 1
    ancestor of the assertions that will live in build_training_set. Raises
    explicitly rather than using an `assert` statement, so it survives `python -O`.

    Args:
        used_subject_ids: Subjects appearing in the data about to be used. Repeats
            are fine, so a label array can be passed directly.
        impostor_holdout: The reserved impostor set.

    Raises:
        AssertionError: If the two intersect, naming the leaked subjects.
    """
    leaked = sorted({int(subject) for subject in used_subject_ids} & impostor_holdout)
    if leaked:
        raise AssertionError(
            f"impostor-holdout subjects present in data about to be used: {leaked}"
        )
