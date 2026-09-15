"""Keyed BioHashing: bits = sign(R_k @ e), with R_k derived from a per-subject key k.

What each property rests on, stated exactly, because this is where the questions land:

- **Non-invertible with respect to features.** The embedding maps 320 features to at most
  64 dimensions, leaving a null space of 256 or more dimensions, and sign() then discards
  magnitude. Even with the key, the feature vector cannot be recovered.
- **Revocable and unlinkable.** A new key version gives an independent R. Two templates of
  the same embedding under independent keys agree on about 50% of bits, the same as
  unrelated templates.
- **Not spoof-resistant if the key and the template both leak.** With R and the bits, an
  attacker can build an embedding that reproduces them (R.T @ bits points into the right
  orthant). That is a pre-image, not an inversion, and it stays valid after re-keying,
  because it approximates the person's embedding rather than the template. Re-keying
  protects against a leaked template alone, not against a leaked template plus key. This
  is a stated limit for THREAT_MODEL.md (Phase 6).

Key handling: no key or seed is stored anywhere. key = HMAC-SHA256(master_secret,
domain-separated subject_ref and key_version). The database holds only key_version, and the
master secret lives in .env or Secrets Manager. A database leak alone yields bits without R.

Why R is not drawn from numpy's Generator: under NEP 19, Generator's distribution streams
may change between numpy versions. A silent change to R after an upgrade would make every
stored template fail to verify, a mass false rejection with no error. This is the same
reason D-008 stores the holdout list instead of re-deriving it. R is built from SHAKE-256
output, and a golden-value test pins it.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

TRANSFORM_VERSION: Final = "biohash-shake256-v1"

MIN_MASTER_SECRET_BYTES: Final = 32

N_BITS: Final = 64
"""Equal to EmbeddingConfig.n_components, so R is square and orthogonal.

Fewer bits than dimensions would discard more of the embedding for protection the
320 -> 64 LDA reduction and sign() already provide against feature recovery. More bits than
dimensions cannot have orthonormal rows and would leak more about the embedding's
direction. Fixed before any Phase 2 result."""


@dataclass(frozen=True, repr=False)
class TemplateKey:
    """Per-subject key material. In memory only: never persisted, logged, or returned by an API.

    repr=False, so an accidental log line or traceback shows an object address, never the
    bytes.

    Attributes:
        material: 32 bytes of HMAC output.
        key_version: Starts at 1 and increments on every revoke-and-reissue.
    """

    material: bytes
    key_version: int


def derive_key(master_secret: bytes, subject_ref: str, key_version: int) -> TemplateKey:
    """Derive a subject's template key for a given version.

    HMAC-SHA256 keyed by the master secret, over the UTF-8 message
    "neuroauth/template-key/{TRANSFORM_VERSION}/{subject_ref}/{key_version}". Including the
    transform version means a future transform can never reuse a v1 key.

    Args:
        master_secret: At least MIN_MASTER_SECRET_BYTES.
        subject_ref: Stable external reference, e.g. "eegmmidb-S001". Not the database
            uuid, so keys survive a re-ingest.
        key_version: 1 or greater.

    Returns:
        The key.

    Raises:
        ValueError: If the secret is too short, subject_ref is empty, or key_version is
            below 1.
    """
    raise NotImplementedError("TODO(phase-2): HMAC key derivation")


def projection_matrix(key: TemplateKey, n_bits: int, n_dims: int) -> NDArray[np.float64]:
    """The keyed projection R, with orthonormal rows.

    Standard normal entries are generated from SHAKE-256(key.material || n_bits || n_dims)
    through an in-repo inverse-CDF transform, not numpy's Generator, then orthonormalized by
    QR with the sign of R's diagonal fixed positive so the factorization is unique. A
    golden-value test pins a known key's matrix to 1e-12, so any change in generation fails
    CI before it can invalidate stored templates.

    Args:
        key: Template key.
        n_bits: Output rows. At most n_dims.
        n_dims: Embedding dimension.

    Returns:
        (n_bits, n_dims) with R @ R.T equal to the identity within 1e-12.

    Raises:
        ValueError: If n_bits is below 1 or above n_dims.
    """
    raise NotImplementedError("TODO(phase-2): SHAKE-256 Gaussian matrix + QR")


def protect(embeddings: NDArray[np.float64], projection: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Apply the cancelable transform: bit = (R @ e >= 0).

    A projection of exactly 0 maps to True, so the mapping is deterministic.

    Args:
        embeddings: (n, n_dims), centered (verification.embedding.embed).
        projection: (n_bits, n_dims) from projection_matrix.

    Returns:
        (n, n_bits) bool.

    Raises:
        ValueError: On a shape mismatch. Structural only; embeddings are finite by
            contract.
    """
    raise NotImplementedError("TODO(phase-2): sign of keyed projection")


def hamming_similarity(
    probe_bits: NDArray[np.bool_],
    template_bits: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Fraction of agreeing bits, 1 - Hamming distance / n_bits.

    Higher means more similar, so "accept iff score >= threshold" holds for every score in
    this project.

    Args:
        probe_bits: (n, n_bits).
        template_bits: (n_bits,).

    Returns:
        (n,) in [0, 1], in steps of 1 / n_bits. Scores tie often, and the metrics handle
        ties explicitly.

    Raises:
        ValueError: On a shape mismatch.
    """
    raise NotImplementedError("TODO(phase-2): normalized Hamming similarity")
