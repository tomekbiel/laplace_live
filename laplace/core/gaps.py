"""
core/gaps.py
Single source for gap handling: data preparation (Layer 1) and
simulation detection (Layer 2, added later).
Phase 2a: Layer 1 moved 1:1 from 2_build_features.py — zero behaviour change.
Interval stays hardcoded (5) for now; detection added in Phase 2b.
"""

import polars as pl
import pandas as pd


# ==============================================================================
# CONSTANTS
# ==============================================================================

GAP_THRESHOLD_MIN = 10          # time-gap threshold for backtest guards (Layer 2)
MARKET_CLOSED_COL_INDEX = 6     # index of market_closed flag in the feature window
DEFAULT_INTERVAL_MIN = 5        # fallback interval; detection added in Phase 2b


# ==============================================================================
# LAYER 1 — DATA PREPARATION (used by 2_build_features)
# ==============================================================================

def remove_data_before_long_gap(df, gap_threshold_days=7):
    """
    Detects the first gap longer than gap_threshold_days and removes data before it.
    """
    print("Wykrywanie nienormalnie długich przerw...")

    # Compute time differences in seconds
    df = df.with_columns([
        pl.col("timestamp").diff().dt.total_seconds().alias("time_diff_sec")
    ])

    # Find the first gap > threshold (in seconds)
    threshold_sec = gap_threshold_days * 24 * 3600
    long_gaps = df.filter(pl.col("time_diff_sec") > threshold_sec)

    if long_gaps.height > 0:
        # Get the timestamp of the first long gap
        first_gap_row = long_gaps.head(1)
        gap_timestamp = first_gap_row.select(pl.col("timestamp")).item()
        gap_duration = first_gap_row.select(pl.col("time_diff_sec")).item() / 3600

        print(f"Znaleziono długą przerwę: {gap_timestamp}")
        print(f"Różnica: {gap_duration:.1f} godzin")
        print(f"Usuwanie danych przed {gap_timestamp}...")

        # Cut data (keep from this timestamp onward)
        df = df.filter(pl.col("timestamp") >= gap_timestamp)
    else:
        print("Nie znaleziono długich przerw.")

    # Drop helper column
    df = df.drop("time_diff_sec")

    return df


def fill_missing_intervals(df, interval_minutes=5):
    """
    Detects missing intervals and fills them with linear interpolation.
    Adds a market_closed flag for filled intervals.
    """
    print(f"Wykrywanie brakujących interwałów ({interval_minutes} minut)...")

    # Convert to pandas for interpolation (Polars has resampling limits)
    pdf = df.to_pandas()
    pdf = pdf.set_index("timestamp")
    pdf = pdf.sort_index()

    # Build full time range every interval_minutes minutes
    full_range = pd.date_range(
        start=pdf.index.min(),
        end=pdf.index.max(),
        freq=f"{interval_minutes}min"
    )

    # Reindex to full range
    pdf_full = pdf.reindex(full_range)

    # Remember which rows are original (not-NA) before interpolation
    original_mask = pdf_full["mid"].notna()

    # Linear interpolation for missing values
    pdf_full["mid"] = pdf_full["mid"].interpolate(method="linear")

    # Forward/backward fill for any remaining edge gaps
    pdf_full["mid"] = pdf_full["mid"].ffill().bfill()

    # Add instrument column (fill with constant value)
    if "instrument" in pdf_full.columns:
        pdf_full["instrument"] = pdf_full["instrument"].ffill().bfill()
    else:
        pdf_full["instrument"] = df.select(pl.col("instrument").first()).item()

    # Add market_closed flag: 1 for filled intervals, 0 for original
    pdf_full["market_closed"] = (~original_mask).astype(int)

    # Convert back to polars
    df_filled = pl.from_pandas(pdf_full.reset_index().rename(columns={"index": "timestamp"}))

    print(f"Dodano {pdf_full['market_closed'].sum()} uzupełnionych interwałów.")

    return df_filled

# ==============================================================================
# LAYER 2 — SIMULATION DETECTION (used by 8_backtest)
# ==============================================================================
# These return only a boolean "is there a gap / is market closed".
# The REACTION (closing the position, recording the trade, continue)
# stays in 8_backtest — core detects, the agent decides what to do.

def detect_time_gap(ts_current, ts_previous, threshold_min=GAP_THRESHOLD_MIN):
    """
    True if the time gap between two consecutive bars exceeds the threshold.
    ts_current, ts_previous: pandas Timestamps.
    """
    gap_min = (ts_current - ts_previous).total_seconds() / 60
    return gap_min > threshold_min


def is_market_closed(X_window, col_index=MARKET_CLOSED_COL_INDEX):
    """
    True if the market_closed flag is set for the current step.
    X_window: single window (shape [lookback, features]); reads last row, col_index.
    """
    return X_window[-1, col_index] == 1

# ==============================================================================
# LAYER 3 — INTERVAL DETECTION (used before interpolation)
# ==============================================================================
# The bar interval is inferred from the data itself: the most common time
# difference between consecutive bars. This lets the same pipeline handle
# 1-min, 5-min, hourly data without a hardcoded value.
# Assumes a single, consistent interval (no mixed-frequency data).

def detect_interval(df):
    """
    Detect the bar interval (in minutes) from the most common time gap
    between consecutive timestamps.
    df: Polars DataFrame with a 'timestamp' column (sorted).
    Returns: interval in minutes (int).
    """
    # Time differences between consecutive bars, in minutes
    diffs = (
        df.select(pl.col("timestamp").diff().dt.total_seconds() / 60)
        .drop_nulls()
        .to_series()
    )

    # Most common difference = the real bar interval
    # (weekend/holiday gaps are rare, so the mode is the normal interval)
    interval = int(diffs.mode().to_list()[0])

    print(f"Wykryty interwał: {interval} min")
    return interval


def gap_threshold_for_interval(interval_min):
    """
    Time-gap threshold for a given bar interval.
    A gap larger than 2x the normal interval counts as a real time gap.
    For 5-min -> 10, for 1-min -> 2, for hourly -> 120.
    """
    return 2 * interval_min