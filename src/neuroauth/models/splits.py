"""Train/test splitting for the Phase 1 identification baseline.

Windows overlap by 50%. A random split therefore puts windows that share half their
samples on both sides of the boundary, and the resulting score measures memorization
of shared samples rather than subject identity. Every function here splits on
contiguous time, never at random.

This is why the published macro-F1 will sit below most numbers reported on eegmmidb.
That is the intended outcome, and the reason is worth stating in the README.
"""

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import Condition, FeatureMatrix


def temporal_split(
    matrices: list[FeatureMatrix],
    train_fraction: float,
    *,
    guard_s: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split each recording into a leading train block and a trailing test block.

    Args:
        matrices: One per recording, in a fixed order. Row indices in the returned
            arrays refer to the concatenation of matrices in that order.
        train_fraction: Fraction of the timeline of each recording used for train.
        guard_s: Seconds discarded at the boundary. Must be at least the window
            length, or the last train window and the first test window share
            samples. With a 2 s window, 2.0 is the minimum and the default choice.

    Returns:
        (train_idx, test_idx) into the concatenated feature matrix. Disjoint, and no
        train window overlaps any test window in time.

    Raises:
        ValueError: If guard_s is smaller than the window length implied by the
            onsets, or train_fraction is outside (0, 1).
    """
    raise NotImplementedError("TODO(phase-1): guarded temporal split")


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

    Args:
        matrices: One per recording, in a fixed order.
        train_condition: "eyes_open" or "eyes_closed".
        test_condition: The other one.

    Returns:
        (train_idx, test_idx) into the concatenated feature matrix.

    Raises:
        ValueError: If either condition is absent from the matrices, or if some
            subject lacks a recording in one of the two conditions.
    """
    raise NotImplementedError("TODO(phase-1): cross-condition split")


def assert_no_window_overlap(
    matrices: list[FeatureMatrix],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    window_s: float,
) -> None:
    """Assert that no train window shares samples with any test window.

    Runs on every split, not only in tests -- the same posture hard rule 5 takes for
    holdout integrity in Phase 3.

    Args:
        matrices: The matrices the indices refer to.
        train_idx: Row indices.
        test_idx: Row indices.
        window_s: Window length in seconds.

    Raises:
        AssertionError: On any index collision, or any pair of windows from the same
            recording whose time spans intersect. The message names the offending
            recording and onsets.
    """
    raise NotImplementedError("TODO(phase-1): overlap assertion")


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
        ValueError: On mismatched feature_names, or a matrix with subject_id None.
    """
    raise NotImplementedError("TODO(phase-1): concatenate feature matrices")
