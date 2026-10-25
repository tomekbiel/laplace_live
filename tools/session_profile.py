"""Intraday volatility profile for one instrument and timeframe.

Answers one question: which minutes of the day carry elevated volatility,
so that later tests can stratify bars by session phase instead of pooling
the whole day together.

Writes a per-bucket table to stdout and, with --out, a JSON file holding
the detected session phases for downstream scripts to read.

Example:
    python tools/session_profile.py ^
        --csv C:\\python\\mt4_bridge\\data\\rates\\US.100+ ^
        --timeframe 5 --bucket-minutes 30 --clock eastern ^
        --out data/meta/session_phases.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Consistency constant that puts a median absolute deviation on the same
# scale as a standard deviation for normally distributed data.
MAD_TO_SIGMA = 0.6745

# Header names differ between exports, so every column is resolved by
# trying a list of candidates instead of assuming one spelling.
TIME_CANDIDATES = ("time", "datetime", "date_time", "timestamp", "date")
PRICE_CANDIDATES = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "mid"),
}

# MT4 exports name files after the period, so a folder can hold one CSV
# per timeframe; this maps the minute count onto that naming.
TIMEFRAME_NAMES = {
    1: "M1", 5: "M5", 15: "M15", 30: "M30",
    60: "H1", 240: "H4", 1440: "D1",
}


@dataclass(frozen=True)
class Phase:
    """One contiguous run of buckets whose volatility exceeds the threshold."""

    start: str
    end: str
    buckets: int
    median_abs_bp: float


def resolve_source(path: Path, timeframe_minutes: int) -> Path:
    """Accept either a CSV file or a folder tree holding one per timeframe.

    The search is recursive because the rates folder nests one directory
    per symbol and sometimes per period, so the timeframe token may sit in
    a parent folder name rather than in the file name. Matching therefore
    runs over the whole path relative to the folder given.

    An ambiguous or missing match raises and lists what was actually
    there, so no run silently reads M1 data while the output says M5.
    """
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"No such file or folder: {path}")

    wanted = TIMEFRAME_NAMES.get(timeframe_minutes, f"M{timeframe_minutes}").lower()
    candidates = sorted(path.rglob("*.csv"))
    if not candidates:
        found = sorted(item.relative_to(path).as_posix() for item in path.rglob("*"))
        raise SystemExit(
            f"No CSV files under {path}. Found instead: "
            + (", ".join(found[:40]) if found else "nothing")
        )

    def tokens(item: Path) -> list[str]:
        """Split the whole relative path on anything but letters and digits."""
        return re.split(r"[^A-Za-z0-9]+", item.relative_to(path).as_posix().lower())

    matches = [item for item in candidates if wanted in tokens(item)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"No {wanted.upper()} file under {path}; found: "
            + ", ".join(item.relative_to(path).as_posix() for item in candidates[:40])
        )
    raise SystemExit(
        f"Several files match {wanted.upper()}: "
        + ", ".join(item.relative_to(path).as_posix() for item in matches)
    )


def resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Return the real column name matching one of the candidates.

    Matching ignores case and surrounding whitespace. Raises instead of
    guessing, because silently picking the wrong column would produce a
    plausible-looking table built on the wrong numbers.
    """
    lookup = {str(name).strip().lower(): name for name in frame.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise KeyError(
        f"No column for {label}: tried {candidates}, found {list(frame.columns)}"
    )


def load_bars(path: Path, timeframe_minutes: int) -> pd.DataFrame:
    """Load the candle file and keep only bars usable for return maths.

    Two filters matter here and both change the result:

    * bars flagged as market_closed are dropped when that column exists,
      because a return measured across a closed market is not a return;
    * a return is kept only when the gap to the previous bar equals the
      timeframe, so weekend and holiday jumps never enter the statistics.
    """
    frame = pd.read_csv(path)

    time_col = resolve_column(frame, TIME_CANDIDATES, "time")
    price_cols = {
        field: resolve_column(frame, candidates, field)
        for field, candidates in PRICE_CANDIDATES.items()
    }

    bars = pd.DataFrame(
        {
            "time": pd.to_datetime(frame[time_col]),
            "high": pd.to_numeric(frame[price_cols["high"]]),
            "low": pd.to_numeric(frame[price_cols["low"]]),
            "close": pd.to_numeric(frame[price_cols["close"]]),
        }
    )

    closed_col = next(
        (name for name in frame.columns if str(name).strip().lower() == "market_closed"),
        None,
    )
    if closed_col is not None:
        bars = bars.loc[pd.to_numeric(frame[closed_col]) == 0].copy()

    bars = bars.sort_values("time").reset_index(drop=True)

    expected_step = pd.Timedelta(minutes=timeframe_minutes)
    step = bars["time"].diff()
    contiguous = step.eq(expected_step)

    log_return = np.log(bars["close"] / bars["close"].shift(1))
    bars["log_return"] = log_return.where(contiguous)
    bars["abs_bp"] = bars["log_return"].abs() * 1e4
    bars["range_bp"] = (bars["high"] - bars["low"]) / bars["close"] * 1e4

    return bars.dropna(subset=["log_return"]).reset_index(drop=True)


def to_clock(bars: pd.DataFrame, broker_offset_hours: float, clock: str) -> pd.Series:
    """Return bar times on the requested clock.

    Broker timestamps sit on a fixed offset from UTC, while the events
    that move an index future are tied to New York, which does observe
    daylight saving. Measuring in broker time therefore smears the cash
    open across two buckets an hour apart, one per season, and mixes it
    with the 08:30 macro release slot. Converting to Eastern time pins
    both to a fixed hour all year.
    """
    if clock == "broker":
        return bars["time"]

    offset = timezone(timedelta(hours=broker_offset_hours))
    return bars["time"].dt.tz_localize(offset).dt.tz_convert("America/New_York")


def add_buckets(
    bars: pd.DataFrame,
    bucket_minutes: int,
    broker_offset_hours: float = 2.0,
    clock: str = "broker",
) -> pd.DataFrame:
    """Tag every bar with the minute-of-day bucket it falls into."""
    local = to_clock(bars, broker_offset_hours, clock)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    bars = bars.copy()
    bars["bucket"] = (minute_of_day // bucket_minutes) * bucket_minutes
    return bars


def label(bucket: int) -> str:
    """Render a minute-of-day offset as HH:MM on the chosen clock."""
    return f"{bucket // 60:02d}:{bucket % 60:02d}"


def profile(bars: pd.DataFrame, tail_k: float) -> pd.DataFrame:
    """Aggregate one row per bucket.

    Robust statistics throughout: the median and the median absolute
    deviation, because a single spike would drag a mean and a standard
    deviation far enough to hide the very structure being measured.
    """
    global_median = bars["abs_bp"].median()
    global_mad = (bars["abs_bp"] - global_median).abs().median()
    tail_threshold = global_median + tail_k * global_mad / MAD_TO_SIGMA

    grouped = bars.groupby("bucket", sort=True)
    table = pd.DataFrame(
        {
            "bars": grouped.size(),
            "median_abs_bp": grouped["abs_bp"].median(),
            "median_range_bp": grouped["range_bp"].median(),
            "tail_share": grouped["abs_bp"].apply(
                lambda values: float((values >= tail_threshold).mean())
            ),
        }
    )
    table.index.name = "bucket"
    return table.reset_index()


def detect_phases(
    table: pd.DataFrame, threshold_ratio: float, min_buckets: int
) -> list[Phase]:
    """Find contiguous runs of buckets above threshold_ratio of the day median.

    The day median is the median *of the bucket medians*, not of all bars,
    so a long quiet night cannot drag the reference down by sheer count.
    Runs are not wrapped around midnight; a phase straddling 00:00 would
    show up as two.
    """
    day_median = table["median_abs_bp"].median()
    threshold = threshold_ratio * day_median
    hot = table["median_abs_bp"] >= threshold

    phases: list[Phase] = []
    start_index: int | None = None

    for position in range(len(table) + 1):
        is_hot = bool(hot.iloc[position]) if position < len(table) else False
        if is_hot and start_index is None:
            start_index = position
        elif not is_hot and start_index is not None:
            run = table.iloc[start_index:position]
            if len(run) >= min_buckets:
                phases.append(
                    Phase(
                        start=label(int(run["bucket"].iloc[0])),
                        end=label(int(run["bucket"].iloc[-1])),
                        buckets=len(run),
                        median_abs_bp=round(float(run["median_abs_bp"].median()), 2),
                    )
                )
            start_index = None

    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, required=True, help="candle file, or the folder holding it"
    )
    parser.add_argument(
        "--timeframe", type=int, default=5, help="bar length in minutes"
    )
    parser.add_argument(
        "--bucket-minutes", type=int, default=30, help="width of one profile bucket"
    )
    parser.add_argument(
        "--clock",
        choices=("broker", "eastern"),
        default="broker",
        help="time axis for the buckets",
    )
    parser.add_argument(
        "--broker-utc-offset",
        type=float,
        default=2.0,
        help="fixed offset of the broker clock from UTC",
    )
    parser.add_argument(
        "--tail-k",
        type=float,
        default=3.0,
        help="how many robust sigmas count as a tail bar",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=1.25,
        help="bucket is 'hot' above this multiple of the day median",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=200,
        help="drop buckets thinner than this before profiling",
    )
    parser.add_argument(
        "--min-buckets", type=int, default=2, help="shortest run kept as a phase"
    )
    parser.add_argument("--out", type=Path, help="write detected phases as JSON")
    args = parser.parse_args()

    source = resolve_source(args.csv, args.timeframe)

    bars = load_bars(source, args.timeframe)
    if bars.empty:
        raise SystemExit("No usable bars after filtering; check --timeframe.")

    bars = add_buckets(bars, args.bucket_minutes, args.broker_utc_offset, args.clock)
    table = profile(bars, args.tail_k)
    table = table.loc[table["bars"] >= args.min_bars].reset_index(drop=True)
    if table.empty:
        raise SystemExit("Every bucket fell below --min-bars; lower it.")

    phases = detect_phases(table, args.threshold_ratio, args.min_buckets)
    day_median = float(table["median_abs_bp"].median())

    print(
        f"{source.name}: {len(bars)} usable bars, "
        f"{bars['time'].min()} to {bars['time'].max()}"
    )
    print(f"clock = {args.clock}, buckets = {args.bucket_minutes} min")
    print(f"day median |return| = {day_median:.2f} bp (median of bucket medians)\n")
    print(f"{'bucket':>7} {'bars':>7} {'|ret| bp':>9} {'range bp':>9} {'tail%':>7}  hot")
    for row in table.itertuples(index=False):
        hot = "*" if row.median_abs_bp >= args.threshold_ratio * day_median else ""
        print(
            f"{label(int(row.bucket)):>7} {int(row.bars):>7} "
            f"{row.median_abs_bp:>9.2f} {row.median_range_bp:>9.2f} "
            f"{row.tail_share * 100:>6.1f}%  {hot}"
        )

    print("\ndetected phases:")
    if not phases:
        print("  none above threshold")
    for phase in phases:
        print(
            f"  {phase.start}-{phase.end}  "
            f"{phase.buckets} buckets  median {phase.median_abs_bp:.2f} bp"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(source),
            "timeframe_minutes": args.timeframe,
            "bucket_minutes": args.bucket_minutes,
            "clock": args.clock,
            "broker_utc_offset": args.broker_utc_offset,
            "bars_used": int(len(bars)),
            "day_median_abs_bp": round(day_median, 4),
            "threshold_ratio": args.threshold_ratio,
            "phases": [asdict(phase) for phase in phases],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()