"""The shared embedding: holdout and fold assertions, population statistics only, and
separation of subjects it never saw."""

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from neuroauth.dsp.types import FeatureMatrix
from neuroauth.verification.embedding import EmbeddingConfig, EmbeddingModel, embed, fit_embedding
from tests.synthetic import IDENTITY_FEATURE_NAMES, identity_feature_matrix

N_FEATURES = len(IDENTITY_FEATURE_NAMES)
CONFIG = EmbeddingConfig()


def cohort(subjects: range) -> list[FeatureMatrix]:
    return [identity_feature_matrix(subject, run) for subject in subjects for run in (1, 2)]


def fit(
    subjects: range,
    *,
    impostor_holdout: frozenset[int] = frozenset(),
    evaluated_subjects: frozenset[int] = frozenset(),
) -> EmbeddingModel:
    return fit_embedding(
        cohort(subjects),
        config=CONFIG,
        streaming_fingerprint="fp",
        impostor_holdout=impostor_holdout,
        evaluated_subjects=evaluated_subjects,
    )


def test_training_on_a_holdout_subject_raises() -> None:
    with pytest.raises(AssertionError, match="impostor-holdout"):
        fit(range(1, 6), impostor_holdout=frozenset({3}))


def test_training_on_an_evaluated_subject_raises() -> None:
    with pytest.raises(AssertionError, match="evaluated"):
        fit(range(1, 6), evaluated_subjects=frozenset({5, 40}))


def test_model_carries_population_statistics_only() -> None:
    """Hard rule 3: no per-subject vector survives into the model (LDA's means_ is dropped)."""
    model = fit(range(1, 8))
    assert {field.name for field in dataclasses.fields(EmbeddingModel)} == {
        "feature_names",
        "feature_mean",
        "feature_scale",
        "projection",
        "embedding_center",
        "train_subjects",
        "config",
        "streaming_fingerprint",
        "embedding_version",
    }
    assert model.feature_mean.shape == (N_FEATURES,)
    assert model.feature_scale.shape == (N_FEATURES,)
    assert model.projection.shape == (6, N_FEATURES)  # capped at 7 subjects - 1
    assert model.embedding_center.shape == (6,)
    assert model.train_subjects == tuple(range(1, 8))


def test_subjects_the_model_never_saw_are_separable() -> None:
    """Enroll on eyes-open, probe on eyes-closed, for two subjects outside training."""
    model = fit(range(1, 13))

    def unit(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        return rows / np.linalg.norm(rows, axis=-1, keepdims=True)

    templates = {
        s: unit(embed(model, identity_feature_matrix(s, 1)).mean(axis=0)) for s in (20, 21)
    }
    probes = {s: unit(embed(model, identity_feature_matrix(s, 2))) for s in (20, 21)}
    for claimed, other in ((20, 21), (21, 20)):
        genuine = float((probes[claimed] @ templates[claimed]).mean())
        impostor = float((probes[other] @ templates[claimed]).mean())
        assert genuine > impostor + 0.2


def test_embed_rejects_other_feature_names() -> None:
    model = fit(range(1, 6))
    renamed = dataclasses.replace(
        identity_feature_matrix(20),
        feature_names=tuple(name.replace("C00", "X00") for name in IDENTITY_FEATURE_NAMES),
    )
    with pytest.raises(ValueError, match="feature_names"):
        embed(model, renamed)


def test_absolute_power_columns_are_refused() -> None:
    absolute = tuple(name.replace("rel:", "abs:") for name in IDENTITY_FEATURE_NAMES)
    matrices = [dataclasses.replace(m, feature_names=absolute) for m in cohort(range(1, 5))]
    with pytest.raises(ValueError, match="rel:"):
        fit_embedding(
            matrices,
            config=CONFIG,
            streaming_fingerprint="fp",
            impostor_holdout=frozenset(),
            evaluated_subjects=frozenset(),
        )


def test_embedding_version_is_deterministic_and_content_sensitive() -> None:
    first = fit(range(1, 8)).embedding_version
    assert first.startswith("sha256:")
    assert fit(range(1, 8)).embedding_version == first
    assert fit(range(1, 9)).embedding_version != first


def test_embeddings_are_finite_for_flat_channels() -> None:
    model = fit(range(1, 6))
    probe = identity_feature_matrix(20)
    assert np.isfinite(
        embed(model, dataclasses.replace(probe, values=np.zeros_like(probe.values)))
    ).all()
