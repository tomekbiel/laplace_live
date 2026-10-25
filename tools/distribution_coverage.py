"""Tail coverage test: normal against Laplace, stratified by session phase.

Answers one question: at a nominal tail probability, what fraction of
returns actually falls beyond the threshold each distribution predicts.
A well-calibrated model gives a ratio near 1; a model whose tails are
too thin gives a ratio far above 1, and the gap widens as the nominal
probability shrinks.

Reuses the loader from session_profile.py, so both scripts read the data
the same way and a fix in one applies to both.

Example:
    python tools/distribution_coverage.py ^
        --csv C:\\python\\mt4_bridge\\data\\rates\\US.100+ ^
        --timeframe 5 --clock eastern ^
        --out data/meta/coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Running "python tools/<script>.py" puts tools/ on sys.path, so the
# sibling module imports by bare name. Keeping one loader avoids the two
# scripts silently disagreeing about which bars are usable.
from session_profile import add_buckets, load_bars, resolve_source

try:
    from scipy.stats import norm
except ImportError as error:  # pragma: no cover - environment guard
    raise SystemExit("scipy is required for the normal quantiles") from error

# Nominal two-sided tail probabilities to test, coarse to extreme.
NOMINAL_LEVELS = (0.05, 0.01, 0.001, 0.0001)

# Session phases on the Eastern clock, read off the volatility profile.
# start is inclusive, end exclusive; a stratum whose start is later than
# its end wraps around midnight.
DEFAULT_STRATA = (
    ("overnight", "18:00", "08:00"),
    ("macro", "08:00", "09:30"),
    ("open", "09:30", "10:30"),
    ("session", "10:30", "16:00"),
    ("late", "16:00", "18:00"),
)


def to_minutes(hhmm: str) -> int:
    """Turn 'HH:MM' into minutes past midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def in_stratum(bucket: pd.Series, start: str, end: str) -> pd.Series:
    """Boolean mask for bars falling inside one phase.

    A phase that wraps midnight is the union of two ranges, which is why
    the comparison flips when start sits after end.
    """
    first, last = to_minutes(start), to_minutes(end)
    if first < last:
        return (bucket >= first) & (bucket < last)
    return (bucket >= first) | (bucket < last)


def fit_normal(values: np.ndarray) -> tuple[float, float]:
    """Maximum likelihood fit of a normal: the mean and the deviation."""
    return float(values.mean()), float(values.std(ddof=1))


def fit_laplace(values: np.ndarray) -> tuple[float, float]:
    """Maximum likelihood fit of a Laplace: the median and the mean |deviation|.

    The median, not the mean, because the Laplace likelihood is driven by
    absolute error, and the point minimising absolute error is the median.
    That is the same estimator the density network's head predicts.
    """
    location = float(np.median(values))
    scale = float(np.mean(np.abs(values - location)))
    return location, scale


def normal_threshold(scale: float, level: float) -> float:
    """Distance from the centre holding `level` of two-sided probability."""
    return float(scale * norm.ppf(1.0 - level / 2.0))


def laplace_threshold(scale: float, level: float) -> float:
    """Same, for a Laplace: P(|X - mu| > t) = exp(-t / b), so t = -b ln(level)."""
    return float(-scale * np.log(level))


def coverage(values: np.ndarray, centre: float, threshold: float) -> float:
    """Fraction of observations beyond the threshold on either side."""
    return float((np.abs(values - centre) > threshold).mean())


def analyse(values: np.ndarray) -> list[dict]:
    """Build one row per nominal level for a single stratum."""
    normal_centre, normal_scale = fit_normal(values)
    laplace_centre, laplace_scale = fit_laplace(values)

    rows: list[dict] = []
    for level in NOMINAL_LEVELS:
        normal_thr = normal_threshold(normal_scale, level)
        laplace_thr = laplace_threshold(laplace_scale, level)
        empirical_thr = float(
            np.quantile(np.abs(values - laplace_centre), 1.0 - level)
        )

        normal_actual = coverage(values, normal_centre, normal_thr)
        laplace_actual = coverage(values, laplace_centre, laplace_thr)

        rows.append(
            {
                "nominal": level,
                "normal_threshold_bp": normal_thr,
                "normal_actual": normal_actual,
                "normal_ratio": normal_actual / level,
                "laplace_threshold_bp": laplace_thr,
                "laplace_actual": laplace_actual,
                "laplace_ratio": laplace_actual / level,
                "empirical_threshold_bp": empirical_thr,
            }
        )
    return rows


def print_block(name: str, count: int, rows: list[dict]) -> None:
    """Print one stratum as a table."""
    print(f"\n{name}  (n = {count})")
    print(
        f"{'nominal':>8} "
        f"{'norm thr':>9} {'norm act':>9} {'ratio':>7}   "
        f"{'lapl thr':>9} {'lapl act':>9} {'ratio':>7}   "
        f"{'empir thr':>10}"
    )
    for row in rows:
        print(
            f"{row['nominal'] * 100:>7.2f}% "
            f"{row['normal_threshold_bp']:>9.2f} "
            f"{row['normal_actual'] * 100:>8.3f}% "
            f"{row['normal_ratio']:>7.2f}   "
            f"{row['laplace_threshold_bp']:>9.2f} "
            f"{row['laplace_actual'] * 100:>8.3f}% "
            f"{row['laplace_ratio']:>7.2f}   "
            f"{row['empirical_threshold_bp']:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, required=True, help="candle file, or the folder holding it"
    )
    parser.add_argument("--timeframe", type=int, default=5, help="bar length in minutes")
    parser.add_argument(
        "--clock", choices=("broker", "eastern"), default="eastern", help="time axis"
    )
    parser.add_argument(
        "--broker-utc-offset",
        type=float,
        default=2.0,
        help="fixed offset of the broker clock from UTC",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=1000,
        help="skip a stratum thinner than this",
    )
    parser.add_argument("--out", type=Path, help="write the full table as JSON")
    args = parser.parse_args()

    source = resolve_source(args.csv, args.timeframe)
    bars = load_bars(source, args.timeframe)
    # One-minute buckets so a phase boundary lands exactly where stated.
    bars = add_buckets(bars, 1, args.broker_utc_offset, args.clock)
    bars["return_bp"] = bars["log_return"] * 1e4

    print(f"{source.name}: {len(bars)} usable bars, clock = {args.clock}")
    print("ratio = actual exceedances / nominal; 1.00 means calibrated")

    report: dict[str, object] = {
        "source": str(source),
        "timeframe_minutes": args.timeframe,
        "clock": args.clock,
        "strata": {},
    }

    everything = bars["return_bp"].to_numpy()
    rows = analyse(everything)
    print_block("all bars", len(everything), rows)
    report["strata"]["all"] = {"count": len(everything), "levels": rows}

    for name, start, end in DEFAULT_STRATA:
        mask = in_stratum(bars["bucket"], start, end)
        values = bars.loc[mask, "return_bp"].to_numpy()
        if len(values) < args.min_bars:
            print(f"\n{name}  (n = {len(values)}) — skipped, below --min-bars")
            continue
        rows = analyse(values)
        print_block(f"{name}  {start}-{end}", len(values), rows)
        report["strata"][name] = {
            "window": [start, end],
            "count": len(values),
            "levels": rows,
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()