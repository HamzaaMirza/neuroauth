"""Feature ablations for the EMG check (D-018).

Two ways to take a candidate confound out of the model input, each built so the
removed information is actually gone rather than merely hidden from one column.
"""

import dataclasses
from collections.abc import Sequence

from neuroauth.dsp.features import feature_channels
from neuroauth.dsp.types import FeatureMatrix


def bands_without(
    bands: dict[str, tuple[float, float]],
    removed: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """Return the band table with the named bands removed, order preserved.

    Remove a band here, in FeatureConfig.bands, and re-extract -- never by dropping its
    columns afterwards. Relative band powers sum to one per channel, so a dropped
    relative column is still recoverable as one minus the others. Re-extracting
    renormalizes relative power over the bands that remain, so the removed band's
    share is gone from every feature.

    Args:
        bands: Name -> (low_hz, high_hz), e.g. config.BANDS.
        removed: Band names to remove.

    Returns:
        A new band table without the named bands.

    Raises:
        ValueError: If a named band is absent, or no band would remain.
    """
    missing = sorted(set(removed) - bands.keys())
    if missing:
        raise ValueError(f"bands not present: {missing}")
    kept = {name: edges for name, edges in bands.items() if name not in set(removed)}
    if not kept:
        raise ValueError("removing these bands leaves none")
    return kept


def drop_channels(matrix: FeatureMatrix, channels: Sequence[str]) -> FeatureMatrix:
    """Remove every feature column, and quality-mask column, for the named channels.

    Exact for these features: relative normalization is per channel and common average
    referencing is off (D-006), so the features of the remaining channels are identical
    to extracting them from a montage that never had the removed ones. With CAR on,
    every channel depends on all the others and this would no longer hold.

    window_ok and the flag strings are kept as computed over the full montage: they
    describe the recording, not the model input.

    Args:
        matrix: A FeatureMatrix following the mode:channel:band naming contract.
        channels: Channels to remove. Duplicates count once.

    Returns:
        A new FeatureMatrix without those channels.

    Raises:
        ValueError: If a named channel is absent -- usually a spelling mismatch --, no
            channel would remain, or the quality mask disagrees with the features.
    """
    order = feature_channels(matrix.feature_names)
    if len(order) != matrix.quality.channel_ok.shape[1]:
        raise ValueError(
            f"{len(order)} channels in feature_names but "
            f"{matrix.quality.channel_ok.shape[1]} in the quality mask"
        )
    removed = set(channels)
    missing = sorted(removed - set(order))
    if missing:
        raise ValueError(f"channels not present in the features: {missing}")
    kept_channels = [i for i, channel in enumerate(order) if channel not in removed]
    if not kept_channels:
        raise ValueError("removing these channels leaves none")
    kept_columns = [
        i for i, name in enumerate(matrix.feature_names) if name.split(":")[1] not in removed
    ]
    return dataclasses.replace(
        matrix,
        values=matrix.values[:, kept_columns],
        feature_names=tuple(matrix.feature_names[i] for i in kept_columns),
        quality=dataclasses.replace(
            matrix.quality, channel_ok=matrix.quality.channel_ok[:, kept_channels]
        ),
    )
