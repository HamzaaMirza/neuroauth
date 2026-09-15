"""Enrollment and revoke-and-reissue: only a protected template leaves, holdout subjects cannot
enroll, failures to enroll are refused loudly, and a reissued template is unlinkable."""

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest

from neuroauth.templates.cancelable import TRANSFORM_VERSION, hamming_similarity
from neuroauth.templates.enrollment import (
    EnrollmentRefusedError,
    ProtectedTemplate,
    enroll_subject,
    representation_version,
    revoke_and_reissue,
    subject_ref,
)
from neuroauth.verification.embedding import EmbeddingConfig, EmbeddingModel, fit_embedding
from tests.synthetic import identity_feature_matrix

SECRET = bytes(range(32))
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
HOLDOUT = frozenset({90})
EVALUATED = frozenset(range(101, 131))


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    return fit_embedding(
        [identity_feature_matrix(s, run) for s in range(1, 31) for run in (1, 2)],
        config=EmbeddingConfig(),
        streaming_fingerprint="fp",
        impostor_holdout=HOLDOUT,
        evaluated_subjects=EVALUATED,
    )


def enroll(model: EmbeddingModel, subject: int = 101, n_windows: int = 40) -> ProtectedTemplate:
    return enroll_subject(
        identity_feature_matrix(subject, 1, n_windows=n_windows),
        model,
        SECRET,
        subject_id=subject,
        subject_ref=subject_ref(subject),
        impostor_holdout=HOLDOUT,
        now=NOW,
    )


def test_template_holds_bits_and_scalars_only(model: EmbeddingModel) -> None:
    """Hard rule 3: no key, seed, embedding, or feature vector in the storable object."""
    template = enroll(model)
    arrays = {
        field.name
        for field in dataclasses.fields(ProtectedTemplate)
        if isinstance(getattr(template, field.name), np.ndarray)
    }
    assert arrays == {"bits"}
    assert template.bits.dtype == np.bool_
    assert template.bits.shape == (model.n_components,)
    assert not any("key" in name and name != "key_version" for name in template.__dict__)
    assert not any("seed" in name for name in template.__dict__)


def test_template_records_its_representation(model: EmbeddingModel) -> None:
    template = enroll(model)
    assert template.subject_ref == "eegmmidb-S101"
    assert template.key_version == 1
    assert template.transform_version == TRANSFORM_VERSION
    assert template.representation_version == representation_version(model)
    assert template.embedding_version == model.embedding_version
    assert template.enrolled_at_utc == "2026-09-15T12:00:00+00:00"
    assert template.n_enrollment_windows == 40


def test_enrollment_statistics_are_leave_one_out_similarities(model: EmbeddingModel) -> None:
    template = enroll(model)
    assert 0.5 < template.enrollment_score_mean <= 1.0
    assert 0.0 <= template.enrollment_score_std < 0.5


def test_enrollment_is_deterministic(model: EmbeddingModel) -> None:
    np.testing.assert_array_equal(enroll(model).bits, enroll(model).bits)


def test_a_holdout_subject_cannot_enroll(model: EmbeddingModel) -> None:
    with pytest.raises(AssertionError, match="impostor holdout"):
        enroll_subject(
            identity_feature_matrix(90, 1),
            model,
            SECRET,
            subject_id=90,
            subject_ref=subject_ref(90),
            impostor_holdout=HOLDOUT,
            now=NOW,
        )


def test_too_few_windows_is_a_refused_enrollment(model: EmbeddingModel) -> None:
    with pytest.raises(EnrollmentRefusedError, match="at least 20"):
        enroll(model, n_windows=19)


def test_naive_times_and_mismatched_subjects_are_refused(model: EmbeddingModel) -> None:
    features = identity_feature_matrix(101, 1)
    with pytest.raises(ValueError, match="timezone"):
        enroll_subject(
            features,
            model,
            SECRET,
            subject_id=101,
            subject_ref=subject_ref(101),
            impostor_holdout=HOLDOUT,
            now=datetime(2026, 9, 15),
        )
    with pytest.raises(ValueError, match="belong to subject 101"):
        enroll_subject(
            features,
            model,
            SECRET,
            subject_id=102,
            subject_ref=subject_ref(102),
            impostor_holdout=HOLDOUT,
            now=NOW,
        )


def test_reissue_increments_the_key_version_and_records_provenance(model: EmbeddingModel) -> None:
    old = enroll(model)
    record, new = revoke_and_reissue(
        old,
        identity_feature_matrix(101, 1),
        model,
        SECRET,
        subject_id=101,
        impostor_holdout=HOLDOUT,
        reason="suspected leak",
        actor="admin:test",
        now=NOW,
    )
    assert new.key_version == 2
    assert (record.revoked_key_version, record.reissued_key_version) == (1, 2)
    assert (record.reason, record.actor, record.subject_ref) == (
        "suspected leak",
        "admin:test",
        "eegmmidb-S101",
    )
    assert not any(isinstance(value, np.ndarray) for value in record.__dict__.values())


def test_reissued_templates_are_unlinkable_to_revoked_ones(model: EmbeddingModel) -> None:
    """Same enrollment data, new key: bit agreement sits near chance, not near 1."""
    agreements = []
    for subject in sorted(EVALUATED):
        old = enroll(model, subject)
        _, new = revoke_and_reissue(
            old,
            identity_feature_matrix(subject, 1),
            model,
            SECRET,
            subject_id=subject,
            impostor_holdout=HOLDOUT,
            reason="test",
            actor="test",
            now=NOW,
        )
        agreements.append(float(hamming_similarity(old.bits[None, :], new.bits)[0]))
    assert 0.4 <= float(np.mean(agreements)) <= 0.6
