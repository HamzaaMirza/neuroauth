"""The cancelable transform: stable keys and projections, orthonormal rows, and unlinkability."""

import numpy as np
import pytest

from neuroauth.templates.cancelable import (
    MIN_MASTER_SECRET_BYTES,
    TRANSFORM_VERSION,
    TemplateKey,
    derive_key,
    hamming_similarity,
    projection_matrix,
    protect,
)

SECRET = bytes(range(32))

GOLDEN_KEY_V1 = "f4d2c5e5adbc933123ba64567f7b056f55ce2ffda0a9ab9534140cf799def2dd"
GOLDEN_KEY_V2 = "8f8c75fd717613dae991c18b2c17a93e061924485db7bbacf9b02c235c5c4904"
GOLDEN_PROJECTION = np.array(
    [
        [0.3402620399773242, 0.555379792421226, -0.6973434563615607, -0.2991440024308319],
        [-0.728843498638442, -0.320898992913044, -0.6046289067706883, -0.015325662698618396],
        [-0.39915650606521685, 0.6903060759302362, 0.09970236977081419, 0.5951563178373318],
    ]
)
"""Pinned at biohash-shake256-v1. If this test fails, every stored template would silently stop
verifying. The fix is a new TRANSFORM_VERSION and re-enrollment, never editing these values."""


def test_transform_version_is_pinned() -> None:
    assert TRANSFORM_VERSION == "biohash-shake256-v1"


def test_key_derivation_is_stable_and_version_separated() -> None:
    assert derive_key(SECRET, "eegmmidb-S001", 1).material.hex() == GOLDEN_KEY_V1
    assert derive_key(SECRET, "eegmmidb-S001", 2).material.hex() == GOLDEN_KEY_V2
    assert (
        derive_key(SECRET, "eegmmidb-S002", 1).material
        != derive_key(SECRET, "eegmmidb-S001", 1).material
    )


def test_projection_is_stable_across_numpy_versions() -> None:
    """Built from SHAKE-256, not np.random.Generator, whose streams NEP 19 does not promise."""
    key = derive_key(SECRET, "eegmmidb-S001", 1)
    np.testing.assert_allclose(projection_matrix(key, 3, 4), GOLDEN_PROJECTION, atol=1e-12, rtol=0)


@pytest.mark.parametrize(("n_bits", "n_dims"), [(64, 64), (8, 20), (1, 5)])
def test_projection_rows_are_orthonormal(n_bits: int, n_dims: int) -> None:
    projection = projection_matrix(derive_key(SECRET, "eegmmidb-S001", 1), n_bits, n_dims)
    assert projection.shape == (n_bits, n_dims)
    np.testing.assert_allclose(projection @ projection.T, np.eye(n_bits), atol=1e-12)


def test_key_derivation_refuses_weak_inputs() -> None:
    with pytest.raises(ValueError, match="bytes"):
        derive_key(SECRET[: MIN_MASTER_SECRET_BYTES - 1], "eegmmidb-S001", 1)
    with pytest.raises(ValueError, match="subject_ref"):
        derive_key(SECRET, "", 1)
    with pytest.raises(ValueError, match="key_version"):
        derive_key(SECRET, "eegmmidb-S001", 0)
    with pytest.raises(ValueError, match="n_bits"):
        projection_matrix(derive_key(SECRET, "eegmmidb-S001", 1), 5, 4)


def test_key_repr_never_shows_material() -> None:
    key = derive_key(SECRET, "eegmmidb-S001", 1)
    assert key.material.hex() not in repr(key)
    assert "material" not in repr(key)
    assert isinstance(key, TemplateKey)


def test_protect_and_similarity() -> None:
    projection = np.eye(3)
    bits = protect(np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 2.0]]), projection)
    np.testing.assert_array_equal(bits, [[True, False, True], [False, True, True]])
    np.testing.assert_allclose(hamming_similarity(bits, bits[0]), [1.0, 1 / 3])
    with pytest.raises(ValueError, match="project"):
        protect(np.ones((2, 4)), projection)
    with pytest.raises(ValueError, match="compare"):
        hamming_similarity(bits, np.ones(4, dtype=np.bool_))


def test_the_same_embedding_under_two_keys_is_unlinkable() -> None:
    """Independent keys give bits that agree at chance, even for identical input."""
    embeddings = np.random.default_rng(0).normal(size=(200, 64))
    first = protect(embeddings, projection_matrix(derive_key(SECRET, "eegmmidb-S001", 1), 64, 64))
    second = protect(embeddings, projection_matrix(derive_key(SECRET, "eegmmidb-S001", 2), 64, 64))
    agreement = np.mean(first == second)
    assert 0.45 <= agreement <= 0.55
