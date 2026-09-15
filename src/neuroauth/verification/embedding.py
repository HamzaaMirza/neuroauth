"""Shared identity embedding, fitted on training subjects and never on the subjects it verifies.

Why a shared embedding with template matching, and not a per-subject classifier: the stored
template must be protected by a keyed, non-invertible transform (hard rule 3), so a score has
to be a comparison inside that keyed space. A per-subject one-vs-rest model cannot live
there. Its fitted parameters are an unprotected template. Adding a user would also mean
retraining. With this design, a new user enrolls against a fixed model, the way deployment
works.

The model is standardization, then shrinkage LDA over subject labels, then centering. This is
the classic LDA-plus-cosine pipeline from speaker verification. It is linear, which reopens
D-004's known property: relative band powers sum to one per channel, so the raw columns are
linearly dependent. Features are therefore floored and log10-transformed first, and LDA is
fitted with shrinkage, so the within-class covariance is always invertible.

Phase 2 persists no model. The served model is refitted deterministically at startup and
identified by a content hash, and templates are bound to that hash. The MLflow registry takes
over in Phase 3.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import FeatureMatrix

EMBEDDING_ALGORITHM: Final = "log-relative/standardize/shrinkage-lda/center-v1"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding hyperparameters. Fixed before any Phase 2 result, not tuned (D-016 rule).

    If tuning is ever wanted, it happens by inner cross-validation over training subjects
    only, and the impostor holdout is never involved.

    Attributes:
        relative_floor: Relative band power is clipped up to this before log10, so a flat
            channel (relative power 0) stays finite.
        n_components: Embedding dimension, capped at n_train_subjects - 1.
        shrinkage: Passed to LDA with the eigen solver. "auto" is Ledoit-Wolf.
    """

    relative_floor: float = 1e-6
    n_components: int = 64
    shrinkage: float | Literal["auto"] = "auto"


@dataclass(frozen=True)
class EmbeddingModel:
    """A fitted embedding, holding population statistics only.

    Deliberately not the sklearn estimator. LinearDiscriminantAnalysis keeps `means_`, the
    mean feature vector of every training subject: 70-odd unprotected raw-feature templates
    inside the model object, which hard rule 3 forbids. fit_embedding copies out only what
    `embed` needs and discards the estimator before returning.

    Attributes:
        feature_names: Column contract; `embed` refuses anything else.
        feature_mean: (n_features,) pooled mean over all training windows, after the log.
        feature_scale: (n_features,) pooled standard deviation, floored away from 0.
        projection: (n_components, n_features) LDA scalings.
        embedding_center: (n_components,) mean embedded training window. Subtracting it
            centers the embedding so sign bits are balanced in the cancelable transform.
        train_subjects: Sorted ids the model was fitted on. Recorded so every consumer can
            assert disjointness from the subjects it evaluates.
        config: Hyperparameters used.
        streaming_fingerprint: StreamingConfig.fingerprint() of the training features.
        model_version: "sha256:" plus 16 hex digits over the arrays, names, subjects,
            config, algorithm, and fingerprint. Templates record it, and a mismatch
            refuses verification.
    """

    feature_names: tuple[str, ...]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    projection: NDArray[np.float64]
    embedding_center: NDArray[np.float64]
    train_subjects: tuple[int, ...]
    config: EmbeddingConfig
    streaming_fingerprint: str
    model_version: str


def fit_embedding(
    matrices: list[FeatureMatrix],
    *,
    config: EmbeddingConfig,
    streaming_fingerprint: str,
    impostor_holdout: frozenset[int],
    evaluated_subjects: frozenset[int],
) -> EmbeddingModel:
    """Fit the shared embedding on training subjects' windows.

    Every window of every given recording is used, both conditions, labelled by subject.
    Within-subject variation therefore includes the eyes-open/eyes-closed change, and LDA
    learns directions that suppress it, taken from subjects other than the ones evaluated.
    Windows flagged not-ok are used as well, as in D-015.

    Args:
        matrices: Per-recording features from process_recording_bounded, all with
            subject_id set and identical "rel:" feature_names.
        config: Hyperparameters.
        streaming_fingerprint: Recorded on the model.
        impostor_holdout: Must not intersect the training subjects.
        evaluated_subjects: The subjects this model will enroll and verify, e.g. the
            current fold. Must not intersect the training subjects.

    Returns:
        An EmbeddingModel with no per-subject statistics in it.

    Raises:
        AssertionError: If any training subject is in impostor_holdout or
            evaluated_subjects. Raised explicitly so it survives `python -O` (hard rules 1
            and 5).
        ValueError: If fewer than 3 training subjects, feature_names differ across
            matrices, any name lacks the "rel:" prefix, or a subject_id is None.
    """
    raise NotImplementedError("TODO(phase-2): fit log/standardize/shrinkage-LDA embedding")


def embed(model: EmbeddingModel, features: FeatureMatrix) -> NDArray[np.float64]:
    """Map feature rows to centered embeddings.

    Pure and always finite, because FeatureMatrix values are finite by contract and the
    floor keeps the log finite.

    Args:
        model: A fitted embedding.
        features: Rows to embed, with feature_names equal to model.feature_names.

    Returns:
        (n_windows, n_components). Never returned outside the in-memory scoring path, and
        never persisted (hard rule 3).

    Raises:
        ValueError: If feature_names differ from the model's. The verifier checks this
            once at session start (load_verifier), so the per-window path never raises.
    """
    raise NotImplementedError("TODO(phase-2): apply embedding")
