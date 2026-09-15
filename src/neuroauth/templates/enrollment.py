"""Enrollment and revoke-and-reissue. A ProtectedTemplate is the only thing that leaves.

There is deliberately no `rekey(template)` function. Re-keying stored bits would mean
inverting the transform, and if that function could be written, the template would not be
cancelable. Reissue therefore takes fresh enrollment features. With one session per
eegmmidb subject, the demo reissues from the same enrollment recording, so the
unlinkability shown comes entirely from the key. That is exactly the property that needs
demonstrating, and the README says so.

Templates are bound to a representation version (D-021): embedding, transform, and
streaming configuration together. The representation is frozen at enrollment and is never
retrained by the correction loop, so templates never go stale.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import FeatureMatrix
from neuroauth.templates.cancelable import (
    TRANSFORM_VERSION,
    derive_key,
    projection_matrix,
    protect,
)
from neuroauth.verification.embedding import EmbeddingModel, embed

MIN_ENROLLMENT_WINDOWS: Final = 20
"""Fewer windows than this is a failure to enroll, counted and reported, never dropped
silently. A recording with full context yields about 56. Fixed before any Phase 2 result."""


class EnrollmentRefusedError(ValueError):
    """Enrollment is not the inference path, so refusing a bad enrollment is correct.

    Every refusal is counted as a failure to enroll (FTE) in the verification report.
    """


def subject_ref(subject_id: int) -> str:
    """The stable external reference for an eegmmidb subject, as in subjects.external_ref."""
    return f"eegmmidb-S{subject_id:03d}"


def representation_version(model: EmbeddingModel) -> str:
    """The version templates are bound to: embedding, transform, bits, and streaming config.

    D-021: this whole representation is frozen at enrollment. A decision layer declares
    which representation versions its inputs came from.
    """
    header = {
        "embedding_version": model.embedding_version,
        "n_bits": model.n_components,
        "streaming_fingerprint": model.streaming_fingerprint,
        "transform_version": TRANSFORM_VERSION,
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


@dataclass(frozen=True)
class ProtectedTemplate:
    """The storable template. Holds no key, no seed, no embedding, and no feature vector.

    Attributes:
        subject_ref: e.g. "eegmmidb-S001".
        bits: (n_bits,) bool, protect(mean enrollment embedding).
        key_version: Re-derive the key from the master secret with this.
        transform_version: cancelable.TRANSFORM_VERSION at enrollment.
        representation_version: representation_version(model) at enrollment (D-021).
        embedding_version: EmbeddingModel.embedding_version the bits were computed under.
        streaming_fingerprint: StreamingConfig.fingerprint() of the enrollment features.
        n_enrollment_windows: Windows averaged.
        n_enrollment_windows_not_ok: Of those, windows the quality mask flagged. They are
            included, not excluded: the mask gates nothing until it is recalibrated on the
            enrollable cohort (D-015).
        enrollment_score_mean: Mean similarity of each enrollment window to a
            leave-one-out template built without it. Input to the decision layer (D-021).
            A scalar summary of scores, not invertible, and impossible to compute later,
            because the enrollment windows are gone.
        enrollment_score_std: Standard deviation of the same similarities.
        enrolled_at_utc: ISO 8601 with a +00:00 offset.
    """

    subject_ref: str
    bits: NDArray[np.bool_]
    key_version: int
    transform_version: str
    representation_version: str
    embedding_version: str
    streaming_fingerprint: str
    n_enrollment_windows: int
    n_enrollment_windows_not_ok: int
    enrollment_score_mean: float
    enrollment_score_std: float
    enrolled_at_utc: str


@dataclass(frozen=True)
class RevocationRecord:
    """Provenance for a revoke-and-reissue, destined for the events table. Contains no bits.

    Attributes:
        subject_ref: Whose template.
        revoked_key_version: No longer accepted.
        reissued_key_version: revoked_key_version + 1.
        representation_version: Representation the reissued template is bound to.
        transform_version: Transform the reissued template uses.
        reason: Free text, e.g. "suspected template leak".
        actor: "admin:<uuid>", "script:<name>", or "system".
        revoked_at_utc: ISO 8601 with a +00:00 offset.
    """

    subject_ref: str
    revoked_key_version: int
    reissued_key_version: int
    representation_version: str
    transform_version: str
    reason: str
    actor: str
    revoked_at_utc: str


def _utc(now: datetime) -> str:
    if now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(UTC).isoformat(timespec="seconds")


def _enroll(
    features: FeatureMatrix,
    model: EmbeddingModel,
    master_secret: bytes,
    *,
    subject_id: int,
    reference: str,
    impostor_holdout: frozenset[int],
    now: datetime,
    key_version: int,
    min_windows: int,
) -> ProtectedTemplate:
    if subject_id in impostor_holdout:
        raise AssertionError(f"subject {subject_id} is in the impostor holdout and cannot enroll")
    if features.subject_id is not None and features.subject_id != subject_id:
        raise ValueError(
            f"features belong to subject {features.subject_id}, not subject {subject_id}"
        )
    if min_windows < 2:
        raise ValueError(f"min_windows must be at least 2, got {min_windows}")
    enrolled_at = _utc(now)
    n_windows = int(features.values.shape[0])
    if n_windows < min_windows:
        raise EnrollmentRefusedError(
            f"subject {subject_id}: {n_windows} enrollment windows, at least {min_windows} needed"
        )

    embeddings = embed(model, features)
    key = derive_key(master_secret, reference, key_version)
    projection = projection_matrix(key, model.n_components, model.n_components)
    bits = protect(embeddings.mean(axis=0, keepdims=True), projection)[0]

    leave_one_out = (embeddings.sum(axis=0, keepdims=True) - embeddings) / (n_windows - 1)
    agreement = np.count_nonzero(
        protect(embeddings, projection) == protect(leave_one_out, projection), axis=1
    )
    self_scores = agreement / float(model.n_components)

    return ProtectedTemplate(
        subject_ref=reference,
        bits=bits,
        key_version=key_version,
        transform_version=TRANSFORM_VERSION,
        representation_version=representation_version(model),
        embedding_version=model.embedding_version,
        streaming_fingerprint=model.streaming_fingerprint,
        n_enrollment_windows=n_windows,
        n_enrollment_windows_not_ok=int((~features.quality.window_ok).sum()),
        enrollment_score_mean=float(self_scores.mean()),
        enrollment_score_std=float(self_scores.std()),
        enrolled_at_utc=enrolled_at,
    )


def enroll_subject(
    features: FeatureMatrix,
    model: EmbeddingModel,
    master_secret: bytes,
    *,
    subject_id: int,
    subject_ref: str,
    impostor_holdout: frozenset[int],
    now: datetime,
    key_version: int = 1,
    min_windows: int = MIN_ENROLLMENT_WINDOWS,
) -> ProtectedTemplate:
    """Enroll one subject from in-memory enrollment windows.

    Embeds every window, averages the embeddings, and protects the mean under the subject's
    key. The mean is taken before sign(): the sign of an average is steadier than a majority
    vote over noisy per-window bits. The leave-one-out self-similarity statistics are
    computed in the same pass. Per-window embeddings and their mean are locals, and only the
    protected template is returned.

    Args:
        features: Enrollment windows, e.g. the eyes-open recording.
        model: Embedding the template is bound to. Must not have been fitted on this
            subject when used for evaluation, which protocol.score_fold asserts.
        master_secret: For key derivation.
        subject_id: Numeric id. Must equal features.subject_id when that is set.
        subject_ref: External reference for the key and template.
        impostor_holdout: The committed holdout.
        now: Enrollment time. Must be timezone-aware.
        key_version: 1 for a first enrollment.
        min_windows: Refuse below this many windows.

    Returns:
        The protected template.

    Raises:
        AssertionError: If subject_id is in impostor_holdout (hard rule 1). This is layer
            one; the holdout_never_enrolled CHECK is layer two.
        EnrollmentRefusedError: If fewer than min_windows windows are given.
        ValueError: If feature_names differ from the model's, subject ids disagree, now is
            naive, or min_windows is below 2.
    """
    return _enroll(
        features,
        model,
        master_secret,
        subject_id=subject_id,
        reference=subject_ref,
        impostor_holdout=impostor_holdout,
        now=now,
        key_version=key_version,
        min_windows=min_windows,
    )


def revoke_and_reissue(
    old: ProtectedTemplate,
    features: FeatureMatrix,
    model: EmbeddingModel,
    master_secret: bytes,
    *,
    subject_id: int,
    impostor_holdout: frozenset[int],
    reason: str,
    actor: str,
    now: datetime,
    min_windows: int = MIN_ENROLLMENT_WINDOWS,
) -> tuple[RevocationRecord, ProtectedTemplate]:
    """Revoke a template and enroll the subject again under the next key version.

    Pure. Marking the old key_version revoked in the database, and refusing it at session
    start, is the persistence layer's job (Phase 2 item 5).

    Args:
        old: The template being revoked.
        features: Fresh enrollment windows.
        model: Model for the reissued template.
        master_secret: For key derivation.
        subject_id: Numeric id of old.subject_ref.
        impostor_holdout: As enroll_subject.
        reason: Recorded.
        actor: Recorded.
        now: Timezone-aware.
        min_windows: As enroll_subject.

    Returns:
        (record, template) with template.key_version == old.key_version + 1.

    Raises:
        As enroll_subject.
    """
    reissued = _enroll(
        features,
        model,
        master_secret,
        subject_id=subject_id,
        reference=old.subject_ref,
        impostor_holdout=impostor_holdout,
        now=now,
        key_version=old.key_version + 1,
        min_windows=min_windows,
    )
    record = RevocationRecord(
        subject_ref=old.subject_ref,
        revoked_key_version=old.key_version,
        reissued_key_version=reissued.key_version,
        representation_version=reissued.representation_version,
        transform_version=reissued.transform_version,
        reason=reason,
        actor=actor,
        revoked_at_utc=reissued.enrolled_at_utc,
    )
    return record, reissued
