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
output through scipy's ndtri, and a golden-value test pins it.

The number of bits equals the embedding dimension (64 at the defaults), so R is square and
orthogonal. Fewer bits would discard more of the embedding for protection the 320 -> 64
reduction and sign() already provide against feature recovery; more bits than dimensions
cannot have orthonormal rows and would leak more about the embedding's direction.
"""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

TRANSFORM_VERSION: Final = "biohash-shake256-v1"

MIN_MASTER_SECRET_BYTES: Final = 32

_KEY_DOMAIN: Final = "neuroauth/template-key"
_PROJECTION_DOMAIN: Final = b"neuroauth/projection"


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
    if len(master_secret) < MIN_MASTER_SECRET_BYTES:
        raise ValueError(f"master_secret must be at least {MIN_MASTER_SECRET_BYTES} bytes")
    if not subject_ref:
        raise ValueError("subject_ref is empty")
    if key_version < 1:
        raise ValueError(f"key_version must be at least 1, got {key_version}")
    message = f"{_KEY_DOMAIN}/{TRANSFORM_VERSION}/{subject_ref}/{key_version}".encode()
    return TemplateKey(
        material=hmac.new(master_secret, message, hashlib.sha256).digest(),
        key_version=key_version,
    )


def projection_matrix(key: TemplateKey, n_bits: int, n_dims: int) -> NDArray[np.float64]:
    """The keyed projection R, with orthonormal rows.

    Standard normal entries come from SHAKE-256 over the key material and shape: each 8
    output bytes become a uniform in (0, 1) from their top 53 bits, then a normal through
    scipy.special.ndtri. They are orthonormalized by QR with the sign of R's diagonal fixed
    positive, so the factorization is unique. A golden-value test pins a known key's matrix
    to 1e-12, so any change in generation fails CI before it can invalidate stored
    templates.

    Args:
        key: Template key.
        n_bits: Output rows. At most n_dims.
        n_dims: Embedding dimension.

    Returns:
        (n_bits, n_dims) with R @ R.T equal to the identity within 1e-12.

    Raises:
        ValueError: If n_bits is below 1 or above n_dims.
    """
    if not 1 <= n_bits <= n_dims:
        raise ValueError(f"need 1 <= n_bits <= n_dims, got {n_bits} and {n_dims}")
    seed_material = (
        _PROJECTION_DOMAIN
        + b"/"
        + TRANSFORM_VERSION.encode()
        + b"/"
        + key.material
        + n_bits.to_bytes(4, "big")
        + n_dims.to_bytes(4, "big")
    )
    count = n_bits * n_dims
    raw = np.frombuffer(hashlib.shake_256(seed_material).digest(8 * count), dtype=">u8")
    uniform = ((raw >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    gaussian = np.asarray(ndtri(uniform), dtype=np.float64).reshape(n_bits, n_dims)
    q, r = np.linalg.qr(gaussian.T)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return np.ascontiguousarray((q * signs).T)


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
    e = np.asarray(embeddings, dtype=np.float64)
    p = np.asarray(projection, dtype=np.float64)
    if e.ndim != 2 or p.ndim != 2 or e.shape[1] != p.shape[1]:
        raise ValueError(f"cannot project embeddings {e.shape} with a {p.shape} matrix")
    bits: NDArray[np.bool_] = (e @ p.T) >= 0.0
    return bits


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
    probe = np.asarray(probe_bits, dtype=np.bool_)
    template = np.asarray(template_bits, dtype=np.bool_)
    if probe.ndim != 2 or template.ndim != 1 or probe.shape[1] != template.shape[0]:
        raise ValueError(f"cannot compare probe bits {probe.shape} with template {template.shape}")
    if template.size == 0:
        raise ValueError("template has no bits")
    agreement = np.count_nonzero(probe == template[None, :], axis=1)
    similarity: NDArray[np.float64] = agreement / float(template.size)
    return similarity
