"""
The seven model features.

This is 2_build_features.py turned from a script into functions. The
arithmetic is unchanged, constant for constant: the same rolling median, the
same 0.6745 scaling, the same interquartile range, the same cyclical hour
encoding, the same EPS guard. What is gone is the __main__ block, the module
level paths and the parquet write — the live loop holds its frame in memory
and never touches disk between fetching and predicting.

FEATURE ORDER IS LOAD-BEARING. The model reads positions, not names, and
core.gaps reads market_closed by its index. In the research code the order
lived in one file and the index in another, with nothing tying them together;
a reordering would have gone unnoticed until the model quietly received the
wrong column. _check_feature_order asserts the two agree, at import.
"""

from __future__ import annotations

import logging
import math

import polars as pl

from .core.gaps import (
    MARKET_CLOSED_COL_INDEX,
    detect_interval,
    fill_missing_intervals,
    remove_data_before_long_gap,
)

logger = logging.getLogger(__name__)

# The order the model was trained with. Do not reorder.
FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "lag1_return",
    "modified_z_score",
    "robust_volatility",
    "hour_sin",
    "hour_cos",
    "market_closed",
)

# Numerical guard for the MAD denominator, as in the research code.
EPS = 1e-8

# Columns the window builder needs beyond the features themselves.
CARRIED_COLUMNS: tuple[str, ...] = ("timestamp", "instrument", "mid")


def _check_feature_order() -> None:
    """Fail at import if the feature order and the gap index disagree."""
    position = FEATURE_COLUMNS.index("market_closed")
    if position != MARKET_CLOSED_COL_INDEX:
        raise RuntimeError(
            f"market_closed sits at position {position} in FEATURE_COLUMNS but "
            f"core.gaps.MARKET_CLOSED_COL_INDEX is {MARKET_CLOSED_COL_INDEX}. "
            f"The runtime market-closed guard would read the wrong column."
        )


_check_feature_order()


def build_features(df: pl.DataFrame, window_size: int = 20) -> pl.DataFrame:
    """Compute the seven features from a frame of timestamp, instrument, mid.

    Lifted from 2_build_features.build_features without a change of
    behaviour. Rows whose rolling features are still null are dropped, which
    costs the first `window_size` bars of any series.

    Args:
        df: Frame carrying at least timestamp and mid, sorted ascending, with
            the market_closed flag already present.
        window_size: Rolling window for the robust statistics.

    Returns:
        The frame with the feature columns added.
    """
    logger.info("Building features for %d rows", df.height)

    # Step 1 — log return and its lag.
    df = df.with_columns([
        (pl.col("mid").log() - pl.col("mid").shift(1).log()).alias("log_return"),
        pl.col("mid").shift(1).alias("mid_lag1"),
    ])
    df = df.with_columns(pl.col("log_return").shift(1).alias("lag1_return"))

    # Step 2 — modified z-score of the price against a rolling median.
    df = df.with_columns(
        pl.col("mid")
        .rolling_median(window_size=window_size, min_samples=window_size)
        .alias("rolling_med")
    )
    df = df.with_columns(
        (pl.col("mid") - pl.col("rolling_med")).abs().alias("abs_dev")
    )
    df = df.with_columns(
        pl.col("abs_dev")
        .rolling_median(window_size=window_size, min_samples=window_size)
        .alias("rolling_mad")
    )
    df = df.with_columns(
        (0.6745 * (pl.col("mid") - pl.col("rolling_med"))
         / (pl.col("rolling_mad") + EPS)).alias("modified_z_score")
    )

    # Step 3 — robust volatility as the interquartile range of the returns.
    df = df.with_columns([
        pl.col("log_return")
        .rolling_quantile(quantile=0.75, window_size=window_size, min_samples=window_size)
        .alias("q75"),
        pl.col("log_return")
        .rolling_quantile(quantile=0.25, window_size=window_size, min_samples=window_size)
        .alias("q25"),
    ])
    df = df.with_columns((pl.col("q75") - pl.col("q25")).alias("robust_volatility"))

    # Step 4 — cyclical encoding of the hour, in BROKER time. The timestamps
    # come from the terminal and are never converted; the model learned this
    # clock, and a conversion here would rotate every window by the offset.
    df = df.with_columns([
        pl.col("timestamp").dt.hour().cast(pl.Float32).alias("hour"),
        pl.col("timestamp").dt.minute().cast(pl.Float32).alias("minute"),
    ])
    df = df.with_columns(
        (pl.col("hour") + pl.col("minute") / 60.0).alias("hour_decimal")
    )
    df = df.with_columns([
        (2 * math.pi * pl.col("hour_decimal") / 24.0).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("hour_decimal") / 24.0).cos().alias("hour_cos"),
    ])

    # Step 5 — drop the warm-up rows the rolling statistics could not fill.
    before = df.height
    df = df.drop_nulls(
        subset=["log_return", "lag1_return", "modified_z_score", "robust_volatility"]
    )
    logger.info("Dropped %d warm-up row(s)", before - df.height)

    return df


def prepare(
    df: pl.DataFrame,
    window_size: int = 20,
    gap_threshold_days: int = 7,
    interval_minutes: int | None = None,
) -> pl.DataFrame:
    """Run the whole preparation: gap handling, interpolation, features.

    The order matches 2_build_features.main step for step — long-gap removal,
    interval detection, interpolation with the market_closed flag, then the
    features. Interpolation is what creates that flag, and the flag is what
    the window builder uses to find continuous sessions, so none of these can
    be skipped for a live frame.

    Args:
        df: timestamp, instrument, mid, ascending.
        window_size: Rolling window for the robust statistics.
        gap_threshold_days: Breaks longer than this cut off earlier history.
        interval_minutes: Bar length; detected from the data when omitted.

    Returns:
        A frame with the feature columns and market_closed, ready for windows.
    """
    df = df.select(["timestamp", "instrument", "mid"]).sort("timestamp")

    df = remove_data_before_long_gap(df, gap_threshold_days)

    interval = interval_minutes if interval_minutes is not None else detect_interval(df)
    df = fill_missing_intervals(df, interval_minutes=interval)

    df = build_features(df, window_size)

    return df.select([*CARRIED_COLUMNS, *FEATURE_COLUMNS])
