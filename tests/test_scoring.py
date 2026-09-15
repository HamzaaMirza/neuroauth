"""Scoring live windows: the claimed identity's key, version checks at session start, and an
inference path that never raises."""

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest

from neuroauth.config import ContextConfig, StreamingConfig
from neuroauth.templates.cancelable import (
    derive_key,
    hamming_similarity,
    projection_matrix,
    protect,
)
from neuroauth.templates.enrollment import (
    ProtectedTemplate,
    enroll_subject,
    representation_version,
    subject_ref,
)
from neuroauth.verification.decision import DecisionLayer, fit_decision_layer
from neuroauth.verification.embedding import EmbeddingConfig, EmbeddingModel, embed, fit_embedding
from neuroauth.verification.metrics import ScoreTable
from neuroauth.verification.scoring import (
    TemplateMismatchError,
    Verifier,
    load_verifier,
    score_windows,
)
from tests.synthetic import identity_feature_matrix

SECRET = bytes(range(32))
STREAMING = StreamingConfig()
HOLDOUT = frozenset({90})


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    return fit_embedding(
        [identity_feature_matrix(s, run) for s in range(1, 21) for run in (1, 2)],
        config=EmbeddingConfig(),
        streaming_fingerprint=STREAMING.fingerprint(),
        impostor_holdout=HOLDOUT,
        evaluated_subjects=frozenset({30, 31}),
    )


@pytest.fixture(scope="module")
def template(model: EmbeddingModel) -> ProtectedTemplate:
    return enroll_subject(
        identity_feature_matrix(30, 1),
        model,
        SECRET,
        subject_id=30,
        subject_ref=subject_ref(30),
        impostor_holdout=HOLDOUT,
        now=datetime(2026, 9, 15, tzinfo=UTC),
    )


@pytest.fixture(scope="module")
def decision(model: EmbeddingModel) -> DecisionLayer:
    rng = np.random.default_rng(1)
    scores = np.concatenate((rng.normal(0.8, 0.05, 200), rng.normal(0.5, 0.05, 200)))
    source = np.repeat(np.array([30, 31], dtype=np.int64), 200)
    n_rows = scores.size
    training = ScoreTable(
        scores=scores,
        claimed_subject=np.full(n_rows, 30, dtype=np.int64),
        source_subject=source,
        source_is_holdout=np.zeros(n_rows, dtype=np.bool_),
        fold=np.zeros(n_rows, dtype=np.int64),
        probe_onset_s=np.arange(n_rows, dtype=np.float64),
        probe_window_ok=np.ones(n_rows, dtype=np.bool_),
        template_score_mean=np.full(n_rows, 0.85),
        template_score_std=np.full(n_rows, 0.05),
        domain="protected",
        split_kind="cross_condition",
    )
    return fit_decision_layer(
        training,
        n_bits=model.n_components,
        representation_versions=(representation_version(model),),
    )


@pytest.fixture(scope="module")
def verifier(
    model: EmbeddingModel, template: ProtectedTemplate, decision: DecisionLayer
) -> Verifier:
    return load_verifier(template, model, decision, SECRET, STREAMING)


def test_genuine_windows_outscore_impostor_windows(verifier: Verifier) -> None:
    genuine = score_windows(verifier, identity_feature_matrix(30, 2))
    impostor = score_windows(verifier, identity_feature_matrix(31, 2))
    genuine_scores = [w.score for w in genuine if w.score is not None]
    impostor_scores = [w.score for w in impostor if w.score is not None]
    assert np.mean(genuine_scores) > np.mean(impostor_scores) + 0.1
    genuine_llr = [w.llr for w in genuine if w.llr is not None]
    impostor_llr = [w.llr for w in impostor if w.llr is not None]
    assert np.mean(genuine_llr) > np.mean(impostor_llr)


def test_scores_use_the_claimed_identitys_key(
    verifier: Verifier, model: EmbeddingModel, template: ProtectedTemplate
) -> None:
    """The unstolen-token error would transform an impostor's probe under the impostor's key."""
    probe = identity_feature_matrix(31, 2)
    scored = np.array([w.score for w in score_windows(verifier, probe)])
    embeddings = embed(model, probe)

    def under(reference: str) -> np.ndarray:
        projection = projection_matrix(derive_key(SECRET, reference, 1), 19, 19)
        return hamming_similarity(protect(embeddings, projection), template.bits)

    np.testing.assert_array_equal(scored, under(subject_ref(30)))
    assert not np.array_equal(scored, under(subject_ref(31)))


def test_decision_time_includes_window_and_right_margin(verifier: Verifier) -> None:
    windows = score_windows(verifier, identity_feature_matrix(30, 2))
    assert all(w.decision_time_s == pytest.approx(w.onset_s + 4.0) for w in windows)


def test_a_feature_contract_mismatch_is_flagged_not_raised(verifier: Verifier) -> None:
    probe = identity_feature_matrix(30, 2)
    renamed = dataclasses.replace(probe, feature_names=tuple(f"x{n}" for n in probe.feature_names))
    windows = score_windows(verifier, renamed)
    assert len(windows) == probe.values.shape[0]
    assert all(w.score is None and "feature_contract_mismatch" in w.flags for w in windows)


def test_non_finite_values_are_flagged_not_raised(verifier: Verifier) -> None:
    probe = identity_feature_matrix(30, 2)
    values = probe.values.copy()
    values[3, 5] = np.nan
    windows = score_windows(verifier, dataclasses.replace(probe, values=values))
    assert windows[3].score is None and "non_finite_embedding" in windows[3].flags
    assert windows[4].score is not None


def test_no_rows_give_no_scores(verifier: Verifier) -> None:
    probe = identity_feature_matrix(30, 2, n_windows=1)
    empty = dataclasses.replace(
        probe,
        values=probe.values[:0],
        onsets_s=probe.onsets_s[:0],
        quality=dataclasses.replace(
            probe.quality,
            window_ok=probe.quality.window_ok[:0],
            channel_ok=probe.quality.channel_ok[:0],
            flags=(),
        ),
    )
    assert score_windows(verifier, empty) == ()


def test_session_start_refuses_mismatched_versions(
    model: EmbeddingModel, template: ProtectedTemplate, decision: DecisionLayer
) -> None:
    with pytest.raises(TemplateMismatchError, match="embedding version"):
        load_verifier(
            dataclasses.replace(template, embedding_version="sha256:other"),
            model,
            decision,
            SECRET,
            STREAMING,
        )
    with pytest.raises(TemplateMismatchError, match="streaming fingerprint"):
        load_verifier(
            template,
            model,
            decision,
            SECRET,
            StreamingConfig(context=ContextConfig(right_margin_s=3.0)),
        )
    with pytest.raises(TemplateMismatchError, match="decision layer"):
        load_verifier(
            template,
            model,
            dataclasses.replace(decision, representation_versions=("other",)),
            SECRET,
            STREAMING,
        )


def test_verifier_repr_does_not_expose_key_material(verifier: Verifier) -> None:
    assert "projection" not in repr(verifier)
