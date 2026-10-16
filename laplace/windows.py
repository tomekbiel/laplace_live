"""
Day-bounded windows.

_find_open_blocks and create_windows are 3_prepare_windows.py unchanged: the
same block detection, the same loop bounds, the same z-score clipping, the
same extreme flag. A block is a maximal run of open-market rows, and no
window ever crosses one, so no window spans a weekend or an overnight break.

last_window is new and exists only for live use. The backtest walks a stored
array; the live loop needs the single most recent window and, just as much,
needs to know whether that window is sound — a window ending now is useless
if the session only opened four bars ago.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from .features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Clipping threshold for the z-score feature, as in 3_prepare_windows.
ZSCORE_CLIP = 5.0

# Windows whose maximum absolute z-score reaches this are flagged. In the
# research code the flag filtered the training set and left the test set
# intact; live it is recorded and never used to skip a bar.
ZSCORE_FILTER = 10.0


class WindowError(RuntimeError):
    """No sound window can be built from the data supplied."""


@dataclass(frozen=True)
class LiveWindow:
    """The newest window plus everything needed to judge it.

    Attributes:
        x: Shape (lookback, n_features), ready for the model.
        time: Timestamp of the bar the window ends on.
        price: Mid at that bar.
        block_length: Bars in the continuous session this window sits in.
        max_abs_z: Largest absolute z-score inside the window.
        is_extreme: Whether that exceeds ZSCORE_FILTER.
    """

    x: np.ndarray
    time: datetime
    price: float
    block_length: int
    max_abs_z: float
    is_extreme: bool


def _find_open_blocks(market_closed) -> list[tuple[int, int]]:
    """Maximal contiguous runs of open-market rows (market_closed == 0).

    Each block is a self-contained trading session: no interpolated row, no
    weekend, no overnight break inside it. Windows never cross a block
    boundary, so no window ever spans a non-trading gap.
    Returns list of (start, end) with end exclusive.
    """
    is_open = (np.asarray(market_closed) == 0)
    blocks = []
    start = None
    for i, open_now in enumerate(is_open):
        if open_now and start is None:
            start = i
        elif not open_now and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(is_open)))
    return blocks


def create_windows(
    df: pl.DataFrame,
    lookback: int = 20,
    predict_step: int = 1,
    feature_cols: tuple[str, ...] | None = None,
    zscore_clip: float = ZSCORE_CLIP,
):
    """Build every day-bounded window in a frame.

    Unchanged from 3_prepare_windows.create_windows, with the clipping step
    folded in so a caller cannot forget it.

    Returns:
        X          (n, lookback, n_features)
        y          (n,)   log_return at the prediction step
        timestamps (n,)   timestamp of the predicted step
        mids       (n,)   mid price at the predicted step
        block_ids  (n,)   index of the source block
        max_abs_z  (n,)   maximum absolute z-score in each window
    """
    cols = list(feature_cols or FEATURE_COLUMNS)
    pdf = df.to_pandas().sort_values("timestamp").reset_index(drop=True)

    feats = pdf[cols].values.astype(np.float32)
    log_ret = pdf["log_return"].values.astype(np.float32)
    ts_all = pdf["timestamp"].values
    mid_all = pdf["mid"].values
    zsc = pdf["modified_z_score"].values.astype(np.float32)

    blocks = _find_open_blocks(pdf["market_closed"].values)
    min_len = lookback + predict_step

    X_parts, y_parts, ts_parts, mid_parts, blk_parts, zmax_parts = [], [], [], [], [], []
    skipped = 0

    for b, (s, e) in enumerate(blocks):
        if (e - s) < min_len:
            skipped += 1
            continue
        for i in range(s, e - lookback - predict_step + 1):
            t = i + lookback + predict_step - 1
            X_parts.append(feats[i:i + lookback])
            y_parts.append(log_ret[t])
            ts_parts.append(ts_all[t])
            mid_parts.append(mid_all[t])
            blk_parts.append(b)
            zmax_parts.append(np.abs(zsc[i:i + lookback]).max())

    X = np.asarray(X_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.float32)

    if X.size and "modified_z_score" in cols:
        z_index = cols.index("modified_z_score")
        X[:, :, z_index] = np.clip(X[:, :, z_index], -zscore_clip, zscore_clip)

    lens = [e - s for s, e in blocks]
    if lens:
        logger.info("Open-market blocks: %d (skipped as too short: %d); "
                    "length min=%d median=%d max=%d",
                    len(blocks), skipped, min(lens), int(np.median(lens)), max(lens))
    logger.info("Created X %s, y %s", X.shape, y.shape)

    return (X, y, np.array(ts_parts), np.array(mid_parts),
            np.array(blk_parts, dtype=np.int32), np.array(zmax_parts))


def last_window(
    df: pl.DataFrame,
    lookback: int = 20,
    feature_cols: tuple[str, ...] | None = None,
    zscore_clip: float = ZSCORE_CLIP,
) -> LiveWindow:
    """Build the single newest window, for a prediction about the next bar.

    This differs from create_windows in what it is for. create_windows builds
    training examples, so every window has a target that already happened and
    sits at i + lookback + predict_step - 1. Here there is no target: the
    window ends on the last closed bar and the prediction concerns the bar
    that has not formed yet. The window is therefore the final `lookback`
    rows of the last open block, with nothing reserved beyond it.

    Args:
        df: The prepared frame, features and market_closed present.
        lookback: Bars per window.
        feature_cols: Column order; the module default when omitted.
        zscore_clip: Clipping applied to the z-score column.

    Raises:
        WindowError: The frame is empty, its last row is a closed-market row,
            or the final block is shorter than the lookback.
    """
    cols = list(feature_cols or FEATURE_COLUMNS)
    pdf = df.to_pandas().sort_values("timestamp").reset_index(drop=True)

    if len(pdf) == 0:
        raise WindowError("The prepared frame is empty")

    blocks = _find_open_blocks(pdf["market_closed"].values)
    if not blocks:
        raise WindowError("No open-market rows in the frame")

    start, end = blocks[-1]

    # The last block must reach the end of the frame; if it does not, the
    # newest rows are interpolated, meaning the market is shut right now.
    if end != len(pdf):
        raise WindowError(
            f"The newest bar is a closed-market row "
            f"(last open block ends at {pdf['timestamp'].iloc[end - 1]})"
        )

    block_length = end - start
    if block_length < lookback:
        raise WindowError(
            f"The current session holds {block_length} bar(s), "
            f"fewer than the {lookback} a window needs"
        )

    tail = pdf.iloc[end - lookback:end]
    x = tail[cols].values.astype(np.float32)

    if "modified_z_score" in cols:
        z_index = cols.index("modified_z_score")
        x[:, z_index] = np.clip(x[:, z_index], -zscore_clip, zscore_clip)

    max_abs_z = float(np.abs(tail["modified_z_score"].values).max())

    return LiveWindow(
        x=x,
        time=pdf["timestamp"].iloc[end - 1],
        price=float(pdf["mid"].iloc[end - 1]),
        block_length=int(block_length),
        max_abs_z=max_abs_z,
        is_extreme=max_abs_z >= ZSCORE_FILTER,
    )
