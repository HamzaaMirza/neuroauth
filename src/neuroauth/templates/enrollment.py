"""Enrollment and revoke-and-reissue. A ProtectedTemplate is the only thing that leaves.

There is deliberately no `rekey(template)` function. Re-keying stored bits would mean
inverting the transform, and if that function could be written, the template would not be
cancelable. Reissue therefore takes fresh enrollment features. With one session per
eegmmidb subject, the demo reissues from the same enrollment recording, so the
unlinkability shown comes entirely from the key. That is exactly the property that needs
demonstrating, and the README says so.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neuroauth.dsp.types import FeatureMatrix
from neuroauth.verification.embedding import EmbeddingModel

MIN_ENROLLMENT_WINDOWS: Final = 20
"""Fewer windows than this is a failure to enroll, counted and reported, never dropped
silently. A recording with full context yields about 56. Fixed before any Phase 2 result."""


class EnrollmentRefusedError(ValueError):
    """Enrollment is not the inference path, so refusing a bad enrollment is correct.

    Every refusal is counted as a failure to enroll (FTE) in the verification report.
    """


@dataclass(frozen=True)
class ProtectedTemplate:
    """The storable template. Holds no key, no seed, no embedding, and no feature vector.

    Attributes:
        subject_ref: e.g. "eegmmidb-S001".
        bits: (n_bits,) bool, protect(mean enrollment embedding).
        key_version: Re-derive the key from the master secret with this.
        transform_version: cancelable.TRANSFORM_VERSION at enrollment.
        model_version: EmbeddingModel.model_version the bits were computed under.
        streaming_fingerprint: StreamingConfig.fingerprint() of the enrollment features.
        n_enrollment_windows: Windows averaged.
        n_enrollment_windows_not_ok: Of those, windows the quality mask flagged. They are
            included, not excluded: the mask gates nothing until it is recalibrated on the
            enrollable cohort (D-015).
        enrolled_at_utc: ISO 8601 with a +00:00 offset.
    """

    subject_ref: str
    bits: NDArray[np.bool_]
    key_version: int
    transform_version: str
    model_version: str
    streaming_fingerprint: str
    n_enrollment_windows: int
    n_enrollment_windows_not_ok: int
    enrolled_at_utc: str


@dataclass(frozen=True)
class RevocationRecord:
    """Provenance for a revoke-and-reissue, destined for the events table. Contains no bits.

    Attributes:
        subject_ref: Whose template.
        revoked_key_version: No longer accepted.
        reissued_key_version: revoked_key_version + 1.
        model_version: Model the reissued template is bound to.
        transform_version: Transform the reissued template uses.
        reason: Free text, e.g. "suspected template leak".
        actor: "admin:<uuid>", "script:<name>", or "system".
        revoked_at_utc: ISO 8601 with a +00:00 offset.
    """

    subject_ref: str
    revoked_key_version: int
    reissued_key_version: int
    model_version: str
    transform_version: str
    reason: str
    actor: str
    revoked_at_utc: str


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
    vote over noisy per-window bits. Per-window embeddings and their mean are locals, and
    only the protected template is returned.

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
        ValueError: If feature_names differ from the model's, subject ids disagree, or now
            is naive.
    """
    raise NotImplementedError("TODO(phase-2): enroll -> protected template only")


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
    start, is the persistence layer's job (Phase 2 item 5). Reissue under a different model
    version is allowed, because that is how templates migrate when the model changes.

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
    raise NotImplementedError("TODO(phase-2): revoke and reissue under key_version + 1")
