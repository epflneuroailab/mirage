"""Layer and temporal pooling utilities for feature arrays.

Layer pooling
-------------
Pretrained encoders expose many layers.  We select a sparse subset and then
optionally aggregate them.

``select_layers(features, layer_fractions)``
    Select layers by fractional depth semantics
    (e.g. [0.5, 0.75, 1.0] -> indices computed as ``int(f * (n_layers - 1))``).
    Duplicate indices are collapsed before selection.

``group_mean_pool(features, n_groups)``
    Split selected layers into n_groups equal groups and average within each
    group.  This matches the ``layer_aggregation="group_mean"`` strategy used
    by the input config fragments.

Temporal pooling
----------------
``align_to_tr(features, src_hz, target_times, method)``
    Resample a (n_dim, n_src_frames) feature array to a set of target TR
    timestamps.  ``target_times`` are in seconds (start of each TR window).
"""


import numpy as np


def _fractional_layer_indices(
    n_layers: int,
    layer_fractions: list[float],
) -> list[int]:
    """Map fractional depths to layer indices."""
    if n_layers <= 0:
        return []
    if n_layers == 1:
        return [0 for _ in layer_fractions]
    return [
        max(0, min(int(f * (n_layers - 1)), n_layers - 1))
        for f in layer_fractions
    ]


def _unique_fractional_layer_indices(
    n_layers: int,
    layer_fractions: list[float],
) -> list[int]:
    """Map fractional depths to sorted unique layer indices."""
    return list(np.unique(_fractional_layer_indices(n_layers, layer_fractions)))


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------


def select_layers(
    all_layers: np.ndarray,
    layer_fractions: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Select specific layers from a full-depth feature array.

    Parameters
    ----------
    all_layers:
        (n_total_layers, ...) array from a pretrained encoder.
    layer_fractions:
        Fractional positions in [0, 1], e.g. [0.5, 0.75, 1.0].

    Returns
    -------
    selected : (n_selected, ...) array
    layer_axis : (n_selected,) array of the actual fractions used
    """
    n = all_layers.shape[0]
    indices = _unique_fractional_layer_indices(n, layer_fractions)
    selected = all_layers[indices]
    if n <= 1:
        actual_fracs = np.zeros(len(indices), dtype=np.float32)
    else:
        actual_fracs = np.array([i / (n - 1) for i in indices], dtype=np.float32)
    return selected, actual_fracs


# ---------------------------------------------------------------------------
# Layer aggregation
# ---------------------------------------------------------------------------


def group_mean_pool(
    features: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Average layers within equal-size groups.

    Parameters
    ----------
    features:
        (n_layers, ...) array — layers already selected via ``select_layers``.
    n_groups:
        Number of groups.  ``n_layers`` must be divisible by ``n_groups``.

    Returns
    -------
    (n_groups, ...) array
    """
    n_layers = features.shape[0]
    if n_layers % n_groups != 0:
        raise ValueError(
            f"n_layers={n_layers} is not divisible by n_groups={n_groups}"
        )
    grouped = features.reshape(n_groups, n_layers // n_groups, *features.shape[1:])
    return grouped.mean(axis=1)


def pool_layers(
    all_layers: np.ndarray,
    layer_fractions: list[float],
    strategy: str | None = "group_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Select and aggregate layers from fractional-depth boundaries.

    For ``"group_mean"``:
      1. Compute boundary indices as ``int(f * (n_layers - 1))`` for each
         fraction, then increment the last index by 1 to make it exclusive.
      2. Average the layer slice between each consecutive pair of boundaries.
      3. Return ``(n_fractions - 1, ...)`` — one group per inter-boundary range.

    For ``None``:
      select the exact fractional-depth layers and return them without further
      aggregation.

    For ``"mean"`` / ``"cat"`` falls back to ``select_layers`` +
    ``aggregate_layers``.

    Parameters
    ----------
    all_layers:
        Full ``(n_total_layers, ...)`` array from a pretrained encoder,
        including the initial embedding output (index 0).
    layer_fractions:
        Fractional boundary positions, e.g. ``[0.5, 0.75, 1.0]``.
    strategy:
        Aggregation strategy.

    Returns
    -------
    aggregated : array, shape ``(n_groups, ...)``
    layer_axis : ``(n_groups,)`` float32 — upper-boundary fraction per group
    """
    if strategy == "group_mean":
        n = all_layers.shape[0]
        boundaries = _unique_fractional_layer_indices(n, layer_fractions)
        if not boundaries:
            empty = all_layers[:0]
            return empty, np.zeros((0,), dtype=np.float32)
        if len(boundaries) == 1:
            idx = boundaries[0]
            if n <= 1:
                layer_axis = np.zeros((1,), dtype=np.float32)
            else:
                layer_axis = np.array([idx / (n - 1)], dtype=np.float32)
            # Keep an explicit layer axis so downstream batching/model code
            # continues to treat this as a layered tensor.
            return all_layers[[idx]], layer_axis
        boundaries[-1] += 1  # exclusive upper bound for the last group
        groups = []
        for l1, l2 in zip(boundaries[:-1], boundaries[1:]):
            groups.append(all_layers[l1:l2].mean(axis=0))
        agg = np.stack(groups)
        # Use the upper-boundary fraction as the label for each group
        n_fracs = len(layer_fractions)
        layer_axis = np.array(layer_fractions[n_fracs - len(groups):], dtype=np.float32)
        return agg, layer_axis
    else:
        sel, layer_axis = select_layers(all_layers, layer_fractions)
        if strategy is None:
            return sel, layer_axis
        return aggregate_layers(sel, strategy, layer_fractions), layer_axis


def aggregate_layers(
    features: np.ndarray,
    strategy: str,
    layer_fractions: list[float] | None = None,
    n_groups: int | None = None,
) -> np.ndarray:
    """High-level layer aggregation dispatcher.

    Strategies
    ----------
    ``"mean"``:      Average across all selected layers → (1, ...).
    ``"cat"``:       Keep layers as-is → (n_layers, ...).
    ``"group_mean"``: group_mean_pool into ``n_groups`` groups.

    Parameters
    ----------
    features:
        (n_layers, ...) array already selected to target layers.
    strategy:
        One of "mean", "cat", "group_mean".
    layer_fractions:
        Used only for ``"group_mean"`` to determine ``n_groups`` when
        ``n_groups`` is None.
    n_groups:
        Explicit group count for ``"group_mean"``.
    """
    if strategy == "mean":
        return features.mean(axis=0, keepdims=True)
    elif strategy == "cat":
        return features
    elif strategy == "group_mean":
        if n_groups is None:
            n_groups = len(layer_fractions) if layer_fractions else features.shape[0]
        return group_mean_pool(features, n_groups)
    else:
        raise ValueError(f"Unknown layer aggregation strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------


def align_to_tr(
    features: np.ndarray,
    src_hz: float,
    tr_times: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Resample feature time-series to fMRI TR timestamps.

    Parameters
    ----------
    features:
        (n_dim, n_src_frames) float array.
    src_hz:
        Native sampling rate of ``features`` in Hz.
    tr_times:
        (n_trs,) 1-D array of TR onset times in seconds (absolute, same
        reference frame as the feature time axis).
    method:
        ``"linear"`` (default) — linear interpolation between adjacent frames.
        ``"nearest"`` — nearest-frame lookup.

    Returns
    -------
    (n_dim, n_trs) float array aligned to the TR grid.
    """
    n_src = features.shape[-1]
    src_times = np.arange(n_src) / src_hz  # assumes features start at t=0

    n_trs = len(tr_times)
    out = np.zeros((features.shape[0], n_trs), dtype=features.dtype)

    for i, t in enumerate(tr_times):
        if method == "nearest":
            idx = int(round(t * src_hz))
            idx = max(0, min(n_src - 1, idx))
            out[:, i] = features[:, idx]
        else:  # linear
            lo = int(t * src_hz)
            hi = lo + 1
            if lo < 0:
                out[:, i] = features[:, 0]
            elif lo >= n_src - 1:
                out[:, i] = features[:, -1]
            else:
                alpha = t * src_hz - lo
                out[:, i] = (1 - alpha) * features[:, lo] + alpha * features[:, hi]
    return out
