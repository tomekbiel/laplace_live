"""
Reading candles from the MT4 bridge.

This replaces 1_parse.py. That script read headerless seven-column exports
from the MetaTrader History Center and glued a timestamp from separate date
and time fields; the bridge writes eight columns with a header and a single
timestamp. Only the bytes differ — the frame this hands back is the one the
research pipeline expects, with the same three columns and the same meaning.

    timestamp   broker server time, as the terminal reported it
    instrument  symbol name
    mid         the price the model was trained on

About 'mid': MetaTrader builds its bars from BID prices, so the close of a
bar is a bid close and there is no ask anywhere in the series. The research
pipeline called that column 'mid' and trained on it. The name is kept so the
downstream code is untouched, but anything feeding live quotes into this
pipeline must use the bid for the same reason — a mid quote would sit half a
spread away from everything the model has ever seen.

The bridge's 'spread' column is always zero on this broker, so the ask cannot
be reconstructed from stored history even if it were wanted.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# The bridge's canonical timestamp format, as mt4bridge.series writes it.
TIME_FORMAT = "%Y.%m.%d %H:%M"

# Spellings seen across files assembled over several years. The bridge emits
# the first; the others appear in older exports that may be merged in.
READ_FORMATS = (
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)

OHLC_COLUMNS = ("open", "high", "low", "close")


class ReaderError(RuntimeError):
    """The series file is missing, empty, or not in the expected shape."""


def read_series(
    path: Path,
    instrument: str,
    price_column: str = "close",
    tail: int | None = None,
) -> pl.DataFrame:
    """Read a bridge series file into the frame the feature builder expects.

    Args:
        path: The CSV the bridge wrote.
        instrument: Name to carry in the 'instrument' column.
        price_column: Which OHLC column becomes 'mid'.
        tail: Keep only the last N rows. The live loop needs a few hundred
            bars; reading a file of hundreds of thousands every five minutes
            would be waste.

    Returns:
        A frame of timestamp, instrument and mid, sorted and deduplicated.

    Raises:
        ReaderError: The file is absent, empty, or lacks the needed columns.
    """
    if not path.is_file():
        raise ReaderError(f"Series file not found: {path}")

    frame = pl.read_csv(path, infer_schema_length=0)

    if "time" not in frame.columns:
        raise ReaderError(f"{path.name} has no 'time' column; got {frame.columns}")
    if price_column not in frame.columns:
        raise ReaderError(
            f"{path.name} has no '{price_column}' column; got {frame.columns}"
        )

    if tail is not None and frame.height > tail:
        frame = frame.tail(tail)

    frame = frame.with_columns(_parse_time(pl.col("time")).alias("timestamp"))

    unreadable = frame.filter(pl.col("timestamp").is_null()).height
    if unreadable:
        logger.warning("%s: dropping %d row(s) with an unreadable timestamp",
                       path.name, unreadable)

    frame = (
        frame
        .filter(pl.col("timestamp").is_not_null())
        .with_columns([
            pl.lit(instrument).alias("instrument"),
            pl.col(price_column).cast(pl.Float64, strict=False).alias("mid"),
        ])
        .filter(pl.col("mid").is_not_null())
        .select(["timestamp", "instrument", "mid"])
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )

    if frame.height == 0:
        raise ReaderError(f"{path.name} holds no readable rows")

    logger.info("Read %d bars from %s (%s .. %s)", frame.height, path.name,
                frame["timestamp"][0], frame["timestamp"][-1])
    return frame


def _parse_time(column: pl.Expr) -> pl.Expr:
    """Try each accepted spelling in turn, keeping the first that parses."""
    parsed = column.str.to_datetime(READ_FORMATS[0], strict=False)
    for fmt in READ_FORMATS[1:]:
        parsed = parsed.fill_null(column.str.to_datetime(fmt, strict=False))
    return parsed


def frame_from_candles(candles, instrument: str, price_column: str = "close") -> pl.DataFrame:
    """Build the same frame from bridge Candle objects held in memory.

    The live loop fetches candles over the socket and never writes them to
    disk, so it needs this path rather than read_series. Both produce an
    identical frame, which is what keeps the live and stored routes honest.

    Args:
        candles: A sequence of mt4bridge.series.Candle.
        instrument: Name for the 'instrument' column.
        price_column: Attribute of the candle to use as 'mid'.
    """
    if not candles:
        raise ReaderError("No candles to build a frame from")

    rows = {
        "timestamp": [c.time for c in candles],
        "instrument": [instrument] * len(candles),
        "mid": [float(getattr(c, price_column)) for c in candles],
    }
    return (
        pl.DataFrame(rows)
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )


def last_bar_time(frame: pl.DataFrame) -> datetime:
    """Timestamp of the newest bar in a frame."""
    return frame["timestamp"][-1]


def drop_forming_bar(frame: pl.DataFrame, now: datetime, step_minutes: int) -> pl.DataFrame:
    """Remove the bar that is still being built.

    CopyRates returns the current, incomplete bar alongside the finished
    ones: a request made at 01:18 comes back with a bar stamped 01:15 whose
    high, low and close are still moving. Feeding that to the model means
    predicting from a number that will change.

    Args:
        frame: Bars in chronological order.
        now: Current broker time.
        step_minutes: Bar length in minutes.

    Returns:
        The frame without its final bar when that bar has not closed yet.
    """
    if frame.height == 0:
        return frame

    newest = frame["timestamp"][-1]
    elapsed_min = (now - newest).total_seconds() / 60.0

    if elapsed_min < step_minutes:
        logger.debug("Dropping the forming bar at %s (%.1f min old)", newest, elapsed_min)
        return frame.head(frame.height - 1)
    return frame
