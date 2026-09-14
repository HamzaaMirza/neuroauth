"""Train/test splitting for the Phase 1 identification baseline.

Windows overlap by 50%. A random split therefore puts windows that share half their
samples on both sides of the boundary, and the resulting score measures memorization
of shared samples rather than subject identity. Every function here splits on
contiguous time or on whole recordings, never at random.

This is why the published macro-F1 will sit below most numbers reported on eegmmidb.
That is the intended outcome, and the reason is worth stating in the README.

Integrity checks raise AssertionError explicitly instead of using an `assert`
statement, so they still run under `python -O`.
"""

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import Condition, FeatureMatrix

_ONSET_TOLERANCE_S = 1e-9
"""Float slack when comparing window onsets. Far below one sample at any rate."""


def _row_offsets(matrices: list[FeatureMatrix]) -> NDArray[np.int64]:
    sizes = np.array([matrix.values.shape[0] for matrix in matrices], dtype=np.int64)
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes, dtype=np.int64)))


def _join(parts: list[NDArray[np.int64]]) -> NDArray[np.int64]:
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def _describe(matrix: FeatureMatrix, index: int) -> str:
    return f"recording {index} (subject {matrix.subject_id}, run {matrix.run})"


def temporal_split(
    matrices: list[FeatureMatrix],
    train_fraction: float,
    *,
    window_s: float,
    guard_s: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split each recording into a leading train block and a trailing test block.

    For each recording, the covered timeline runs from its first onset to its last
    onset plus window_s. The split point sits at train_fraction of that timeline.
    Train takes windows starting before the split point; test takes windows starting
    at least guard_s after it. Windows in the gap are discarded.

    Because a train window can start arbitrarily close to the split point, the only
    guard that rules out shared samples for every hop size is guard_s >= window_s.

    assert_no_window_overlap runs on the result before it is returned.

    Args:
        matrices: One per recording, in a fixed order. Row indices in the returned
            arrays refer to the concatenation of matrices in that order.
        train_fraction: Fraction of the timeline of each recording used for train.
        window_s: Window length in seconds. Needed because onsets record where
            windows start, not how long they are.
        guard_s: Seconds discarded after the split point. With a 2 s window, 2.0 is
            the minimum and the default choice.

    Returns:
        (train_idx, test_idx), sorted, into the concatenated feature matrix. Disjoint,
        and no train window overlaps any test window in time.

    Raises:
        ValueError: If train_fraction is outside (0, 1), window_s is not positive,
            guard_s is smaller than window_s, or any non-empty recording would
            contribute no windows to train or to test -- which would silently drop
            that subject from one side of the evaluation.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if guard_s < window_s:
        raise ValueError(
            f"guard_s ({guard_s} s) is shorter than the window ({window_s} s): the last "
            "train window and the first test window could share samples"
        )

    offsets = _row_offsets(matrices)
    train_parts: list[NDArray[np.int64]] = []
    test_parts: list[NDArray[np.int64]] = []
    for k, matrix in enumerate(matrices):
        onsets = np.asarray(matrix.onsets_s, dtype=np.float64)
        if onsets.size == 0:
            continue
        start = float(onsets.min())
        end = float(onsets.max()) + window_s
        split_s = start + train_fraction * (end - start)

        rows = np.arange(onsets.size, dtype=np.int64) + offsets[k]
        train_rows = rows[onsets < split_s]
        test_rows = rows[onsets >= split_s + guard_s]
        if train_rows.size == 0 or test_rows.size == 0:
            raise ValueError(
                f"{_describe(matrix, k)} contributes {train_rows.size} train and "
                f"{test_rows.size} test windows; adjust train_fraction or guard_s"
            )
        train_parts.append(train_rows)
        test_parts.append(test_rows)

    train_idx, test_idx = _join(train_parts), _join(test_parts)
    assert_no_window_overlap(matrices, train_idx, test_idx, window_s)
    return train_idx, test_idx


def cross_condition_split(
    matrices: list[FeatureMatrix],
    train_condition: Condition,
    test_condition: Condition,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Train on one baseline condition, test on the other. The headline split.

    This is not about a condition shortcut. Every subject has both R01 and R02, so
    condition is balanced across labels and carries no information about which
    subject a window belongs to -- pooling the two runs offers a classifier nothing
    to cheat with.

    What this split measures is whether identity features survive a change in brain
    state, which is the actual deployment question. A user enrolls calm and alert,
    then authenticates tired at the end of the day. Training on eyes-open and
    testing on eyes-closed is the closest approximation this dataset allows, and a
    model that only works within a single brain state is not an authenticator.

    Train and test come from different recordings by construction, so no window can
    share samples across the split. Recordings of any other condition are ignored.

    Args:
        matrices: One per recording, in a fixed order.
        train_condition: "eyes_open" or "eyes_closed".
        test_condition: The other one.

    Returns:
        (train_idx, test_idx), sorted, into the concatenated feature matrix.

    Raises:
        ValueError: If the two conditions are equal, either condition is absent from
            the matrices, a matrix in either condition has no subject_id, or some
            subject lacks a recording in one of the two conditions.
    """
    if train_condition == test_condition:
        raise ValueError(f"train and test condition are both {train_condition!r}")

    offsets = _row_offsets(matrices)
    train_parts: list[NDArray[np.int64]] = []
    test_parts: list[NDArray[np.int64]] = []
    train_subjects: set[int] = set()
    test_subjects: set[int] = set()
    for k, matrix in enumerate(matrices):
        if matrix.condition not in (train_condition, test_condition):
            continue
        if matrix.subject_id is None:
            raise ValueError(f"{_describe(matrix, k)} has no subject_id")
        rows = np.arange(offsets[k], offsets[k + 1], dtype=np.int64)
        if matrix.condition == train_condition:
            train_parts.append(rows)
            train_subjects.add(matrix.subject_id)
        else:
            test_parts.append(rows)
            test_subjects.add(matrix.subject_id)

    for condition, subjects in ((train_condition, train_subjects), (test_condition, test_subjects)):
        if not subjects:
            raise ValueError(f"no recordings with condition {condition!r}")
    unpaired = sorted(train_subjects ^ test_subjects)
    if unpaired:
        raise ValueError(
            f"subjects missing a {train_condition!r} or {test_condition!r} recording: {unpaired}"
        )
    return _join(train_parts), _join(test_parts)


def assert_no_window_overlap(
    matrices: list[FeatureMatrix],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    window_s: float,
) -> None:
    """Assert that no train window shares samples with any test window.

    Runs on every split, not only in tests -- the same posture hard rule 5 takes for
    holdout integrity in Phase 3. temporal_split calls it before returning.

    Two windows [a, a + window_s) and [b, b + window_s) from the same recording share
    samples exactly when |a - b| < window_s. Windows from different recordings never
    do.

    Args:
        matrices: The matrices the indices refer to.
        train_idx: Row indices.
        test_idx: Row indices.
        window_s: Window length in seconds.

    Raises:
        ValueError: If window_s is not positive.
        AssertionError: On an out-of-range index, any index present on both sides, or
            any pair of windows from the same recording whose time spans intersect.
            The message names the offending recording and onsets.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")

    offsets = _row_offsets(matrices)
    n_rows = int(offsets[-1])
    train = np.asarray(train_idx, dtype=np.int64)
    test = np.asarray(test_idx, dtype=np.int64)
    for side, idx in (("train", train), ("test", test)):
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_rows):
            raise AssertionError(f"{side} indices fall outside the {n_rows} available rows")

    collisions = np.intersect1d(train, test)
    if collisions.size:
        raise AssertionError(
            f"{collisions.size} rows are in both train and test, e.g. {collisions[:5].tolist()}"
        )

    train_recording = np.searchsorted(offsets, train, side="right") - 1
    test_recording = np.searchsorted(offsets, test, side="right") - 1
    for k, matrix in enumerate(matrices):
        train_onsets = np.sort(matrix.onsets_s[train[train_recording == k] - offsets[k]])
        test_onsets = matrix.onsets_s[test[test_recording == k] - offsets[k]]
        if train_onsets.size == 0 or test_onsets.size == 0:
            continue
        position = np.searchsorted(train_onsets, test_onsets)
        before = train_onsets[np.clip(position - 1, 0, train_onsets.size - 1)]
        after = train_onsets[np.clip(position, 0, train_onsets.size - 1)]
        gap = np.minimum(np.abs(test_onsets - before), np.abs(test_onsets - after))
        overlapping = np.flatnonzero(gap < window_s - _ONSET_TOLERANCE_S)
        if overlapping.size:
            j = int(overlapping[0])
            raise AssertionError(
                f"{_describe(matrix, k)}: test window at {test_onsets[j]:.3f} s shares "
                f"samples with a train window {gap[j]:.3f} s away (window is {window_s} s); "
                f"{overlapping.size} test windows affected"
            )


def concatenate(
    matrices: list[FeatureMatrix],
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    """Stack per-recording matrices into one design matrix with subject labels.

    Args:
        matrices: One per recording, in a fixed order. All must share the same
            feature_names, and none may have subject_id None.

    Returns:
        (x, y, feature_names) with x of shape (n_rows, n_features) and y of shape
        (n_rows,) holding subject ids.

    Raises:
        ValueError: On an empty list, mismatched feature_names, or a matrix with
            subject_id None.
    """
    if not matrices:
        raise ValueError("no feature matrices to concatenate")
    names = matrices[0].feature_names
    labels: list[NDArray[np.int64]] = []
    for k, matrix in enumerate(matrices):
        if matrix.feature_names != names:
            raise ValueError(
                f"{_describe(matrix, k)} has different feature_names from recording 0; "
                "matrices built under different configs cannot be stacked"
            )
        if matrix.subject_id is None:
            raise ValueError(f"{_describe(matrix, k)} has no subject_id")
        labels.append(np.full(matrix.values.shape[0], matrix.subject_id, dtype=np.int64))

    x = np.concatenate([matrix.values for matrix in matrices], axis=0).astype(np.float64)
    return x, np.concatenate(labels), names
