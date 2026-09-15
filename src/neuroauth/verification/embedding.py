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

The embedding is part of the frozen representation (D-021): it is fixed when a template is
enrolled and never retrained by the correction loop. Phase 2 persists no model. The served
model is refitted deterministically at startup and identified by a content hash, and
templates are bound to that hash.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from neuroauth.dsp.types import FeatureMatrix

EMBEDDING_ALGORITHM: Final = "log-relative/standardize/shrinkage-lda/center-v1"

_MIN_SCALE: Final = 1e-12
"""A standardized column with a smaller standard deviation is left unscaled rather than
divided by almost zero."""


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
        embedding_version: "sha256:" plus 16 hex digits over the arrays, names, subjects,
            config, algorithm, and fingerprint. One component of the representation version
            templates are bound to (D-021).
    """

    feature_names: tuple[str, ...]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    projection: NDArray[np.float64]
    embedding_center: NDArray[np.float64]
    train_subjects: tuple[int, ...]
    config: EmbeddingConfig
    streaming_fingerprint: str
    embedding_version: str

    @property
    def n_components(self) -> int:
        return int(self.projection.shape[0])


def _log_relative(values: NDArray[np.float64], floor: float) -> NDArray[np.float64]:
    logged: NDArray[np.float64] = np.log10(np.maximum(np.asarray(values, dtype=np.float64), floor))
    return logged


def _embedding_version(
    *,
    feature_names: tuple[str, ...],
    arrays: tuple[NDArray[np.float64], ...],
    train_subjects: tuple[int, ...],
    config: EmbeddingConfig,
    streaming_fingerprint: str,
) -> str:
    header = {
        "algorithm": EMBEDDING_ALGORITHM,
        "config": {
            "relative_floor": config.relative_floor,
            "n_components": config.n_components,
            "shrinkage": config.shrinkage,
        },
        "feature_names": list(feature_names),
        "streaming_fingerprint": streaming_fingerprint,
        "train_subjects": list(train_subjects),
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8"))
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return f"sha256:{digest.hexdigest()[:16]}"


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
            matrices, any name lacks the "rel:" prefix, a subject_id is None, or the
            config is invalid.
    """
    if not matrices:
        raise ValueError("no training matrices")
    if config.n_components < 1 or config.relative_floor <= 0.0:
        raise ValueError("n_components must be >= 1 and relative_floor positive")
    names = matrices[0].feature_names
    labels: list[NDArray[np.int64]] = []
    for matrix in matrices:
        if matrix.feature_names != names:
            raise ValueError("training matrices have different feature_names")
        if matrix.subject_id is None:
            raise ValueError("every training matrix needs a subject_id")
        labels.append(np.full(matrix.values.shape[0], matrix.subject_id, dtype=np.int64))
    if not all(name.startswith("rel:") for name in names):
        raise ValueError("the embedding is defined on relative band power ('rel:' columns) only")

    subjects = tuple(
        sorted({matrix.subject_id for matrix in matrices if matrix.subject_id is not None})
    )
    leaked = sorted(set(subjects) & impostor_holdout)
    if leaked:
        raise AssertionError(f"impostor-holdout subjects in embedding training data: {leaked}")
    evaluated = sorted(set(subjects) & evaluated_subjects)
    if evaluated:
        raise AssertionError(f"evaluated subjects in embedding training data: {evaluated}")
    if len(subjects) < 3:
        raise ValueError(f"need at least 3 training subjects, got {len(subjects)}")

    x = _log_relative(np.concatenate([matrix.values for matrix in matrices]), config.relative_floor)
    y = np.concatenate(labels)
    feature_mean = x.mean(axis=0)
    raw_scale = x.std(axis=0)
    feature_scale = np.where(raw_scale > _MIN_SCALE, raw_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale

    n_components = min(config.n_components, len(subjects) - 1, x.shape[1])
    lda = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage=config.shrinkage, n_components=n_components
    )
    lda.fit(standardized, y)
    # Copy out the projection only. The estimator, with its per-subject means_, goes out of
    # scope here and is never returned.
    projection = np.ascontiguousarray(
        np.asarray(lda.scalings_, dtype=np.float64)[:, :n_components].T
    )
    del lda
    center = (standardized @ projection.T).mean(axis=0)

    return EmbeddingModel(
        feature_names=names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        projection=projection,
        embedding_center=center,
        train_subjects=subjects,
        config=config,
        streaming_fingerprint=streaming_fingerprint,
        embedding_version=_embedding_version(
            feature_names=names,
            arrays=(feature_mean, feature_scale, projection, center),
            train_subjects=subjects,
            config=config,
            streaming_fingerprint=streaming_fingerprint,
        ),
    )


def embed_values(model: EmbeddingModel, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map raw feature rows, in model.feature_names order, to centered embeddings.

    Raises:
        ValueError: If values is not (n, n_features).
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.feature_mean.shape[0]:
        raise ValueError(
            f"expected (n, {model.feature_mean.shape[0]}) feature rows, got shape {x.shape}"
        )
    standardized = (_log_relative(x, model.config.relative_floor) - model.feature_mean) / (
        model.feature_scale
    )
    embedded: NDArray[np.float64] = standardized @ model.projection.T - model.embedding_center
    return embedded


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
    if features.feature_names != model.feature_names:
        raise ValueError("feature_names differ from the embedding model's")
    return embed_values(model, features.values)
