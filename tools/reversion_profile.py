"""Does price revert after a deviation, and is the payoff asymmetric?

No model is involved. The deviation is measured the same way features.py
measures it — a robust z-score of price against its rolling median — so
the answer describes the data, not the network, and cannot be invalidated
by anything that happens to the model later.

For every event the script reports what a mean-reversion trade would have
booked: short the upper side, long the lower side, entered at the close of
the triggering bar and held for a fixed number of bars. A positive figure
means reversion paid. Maximum adverse excursion is reported alongside,
because at leverage the path decides survival, not the endpoint.

Three splits are available and each answers a different question:
    --by-phase  does the open behave like the night
    --by-trend  does a deviation against the trend behave like one with it
    --stride    thins overlapping events, which is what inflates t

Usage:
    python tools/reversion_profile.py --csv C:\\python\\mt4_bridge\\data\\rates\\US.100+
    python tools/reversion_profile.py --csv <path> --by-phase
    python tools/reversion_profile.py --csv <path> --by-trend --stride 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from session_profile import add_buckets, load_bars, resolve_source
from distribution_coverage import DEFAULT_STRATA, in_stratum

# Matches the constant features.py uses, so the deviation measured here is
# the same quantity the model is fed.
MAD_TO_SIGMA = 0.6745
EPS = 1e-12

DEFAULT_EDGES = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_HORIZONS = (1, 3, 6, 12)


@dataclass(frozen=True)
class Group:
    """One row of the output: a bucket of events on one side."""

    label: str
    count: int
    means: dict[int, float]
    hits: dict[int, float]
    mae: float
    t_stat: float


def robust_z(close: pd.Series, window: int) -> pd.Series:
    """Deviation of price from its rolling median, in robust sigmas.

    The same formula as features.py: a median absolute deviation scaled by
    0.6745 so the unit matches a standard deviation under normality. The
    median rather than the mean, because one spike must not move the
    reference the spike is being measured against.
    """
    rolling_median = close.rolling(window, min_periods=window).median()
    deviation = (close - rolling_median).abs()
    rolling_mad = deviation.rolling(window, min_periods=window).median()
    return MAD_TO_SIGMA * (close - rolling_median) / (rolling_mad + EPS)


def forward_returns(
    close: np.ndarray, times: np.ndarray, step_ns: int, horizons: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Log return h bars ahead, NaN where the bars are not contiguous.

    A forward return that spans a weekend is not a forward return. The gap
    is detected from the timestamps rather than from a session calendar, so
    it catches holidays and technical breaks without a list of dates.
    """
    out: dict[int, np.ndarray] = {}
    size = len(close)
    for horizon in horizons:
        values = np.full(size, np.nan)
        if horizon >= size:
            out[horizon] = values
            continue
        span = times[horizon:] - times[:-horizon]
        contiguous = span == horizon * step_ns
        ahead = np.log(close[horizon:] / close[:-horizon])
        values[:-horizon] = np.where(contiguous, ahead, np.nan)
        out[horizon] = values
    return out


def excursions(
    high: np.ndarray, low: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Highest high and lowest low over the next `horizon` bars.

    Built by folding one shifted array at a time, which keeps the memory
    flat and avoids a Python loop over sixty thousand bars.
    """
    size = len(high)
    run_max = np.full(size, -np.inf)
    run_min = np.full(size, np.inf)
    for shift in range(1, horizon + 1):
        if shift >= size:
            break
        run_max[: size - shift] = np.maximum(run_max[: size - shift], high[shift:])
        run_min[: size - shift] = np.minimum(run_min[: size - shift], low[shift:])
    return run_max, run_min


def trend_state(close: np.ndarray, window: int, threshold_bp: float) -> np.ndarray:
    """Label each bar up, down or flat by the move over the last `window`.

    Deliberately crude: the question is whether splitting by trend changes
    the answer at all, and a crude split that changes nothing settles that
    as well as a sophisticated one.
    """
    size = len(close)
    state = np.full(size, "flat", dtype=object)
    if window >= size:
        return state
    move_bp = np.full(size, np.nan)
    move_bp[window:] = (close[window:] / close[:-window] - 1.0) * 1e4
    state[move_bp > threshold_bp] = "up"
    state[move_bp < -threshold_bp] = "down"
    return state


def summarise(
    mask: np.ndarray,
    sign: float,
    label: str,
    forwards: dict[int, np.ndarray],
    adverse_bp: np.ndarray,
    stride: int,
) -> Group | None:
    """Reduce one set of events to a row.

    `sign` turns a raw forward return into what the trade would book: minus
    one on the upper side, where reversion is a short, plus one below.
    """
    index = np.flatnonzero(mask)
    if stride > 1:
        index = index[::stride]
    if index.size == 0:
        return None

    means: dict[int, float] = {}
    hits: dict[int, float] = {}
    t_stat = float("nan")
    longest = max(forwards)

    for horizon, values in forwards.items():
        booked = sign * values[index] * 1e4
        booked = booked[~np.isnan(booked)]
        if booked.size == 0:
            means[horizon] = float("nan")
            hits[horizon] = float("nan")
            continue
        means[horizon] = float(booked.mean())
        hits[horizon] = float((booked > 0).mean() * 100)
        if horizon == longest and booked.size > 1:
            error = booked.std(ddof=1) / np.sqrt(booked.size)
            t_stat = float(booked.mean() / error) if error > 0 else float("nan")

    adverse = adverse_bp[index]
    adverse = adverse[~np.isnan(adverse)]

    return Group(
        label=label,
        count=int(index.size),
        means=means,
        hits=hits,
        mae=float(adverse.mean()) if adverse.size else float("nan"),
        t_stat=t_stat,
    )


def print_block(title: str, groups: list[Group], horizons: tuple[int, ...]) -> None:
    """Print one side, one row per bucket."""
    print(f"\n{title}")
    header = f"{'bucket':>14} {'n':>6}"
    for horizon in horizons:
        header += f" {f'h{horizon} bp':>8} {f'h{horizon} %':>7}"
    header += f" {'MAE bp':>9} {'t':>7}"
    print(header)
    print("-" * len(header))

    for group in groups:
        line = f"{group.label:>14} {group.count:>6}"
        for horizon in horizons:
            line += f" {group.means[horizon]:>8.2f} {group.hits[horizon]:>7.1f}"
        line += f" {group.mae:>9.1f} {group.t_stat:>7.2f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, required=True, help="candle file, or the folder holding it"
    )
    parser.add_argument("--timeframe", type=int, default=5, help="bar length in minutes")
    parser.add_argument(
        "--z-window", type=int, default=20, help="rolling window for the z-score"
    )
    parser.add_argument(
        "--edges",
        type=float,
        nargs="+",
        default=list(DEFAULT_EDGES),
        help="bucket edges on the absolute z-score",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="forward horizons in bars",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every k-th event; k near the horizon removes most overlap",
    )
    parser.add_argument(
        "--all-bars",
        action="store_true",
        help="count every bar above the threshold, not just the first of a run",
    )
    parser.add_argument("--by-phase", action="store_true", help="split by session phase")
    parser.add_argument("--by-trend", action="store_true", help="split by trend state")
    parser.add_argument(
        "--trend-window", type=int, default=72, help="bars used to label the trend"
    )
    parser.add_argument(
        "--trend-threshold-bp",
        type=float,
        default=20.0,
        help="move above which the trend is called",
    )
    parser.add_argument(
        "--broker-utc-offset", type=float, default=2.0, help="broker clock offset"
    )
    parser.add_argument("--min-events", type=int, default=30, help="skip thinner groups")
    parser.add_argument("--out", type=Path, help="write the whole table as JSON")
    args = parser.parse_args()

    horizons = tuple(sorted(args.horizons))
    edges = tuple(sorted(args.edges))

    source = resolve_source(args.csv, args.timeframe)
    bars = load_bars(source, args.timeframe)
    bars = add_buckets(bars, 1, args.broker_utc_offset, "eastern")

    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    times = bars["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    step_ns = args.timeframe * 60 * 1_000_000_000

    z = robust_z(bars["close"], args.z_window).to_numpy()
    forwards = forward_returns(close, times, step_ns, horizons)

    longest = max(horizons)
    run_max, run_min = excursions(high, low, longest)
    # Adverse means against the reversion trade: upward for a short above,
    # downward for a long below. Both are reported as positive basis points.
    adverse_upper = (run_max - close) / close * 1e4
    adverse_lower = (close - run_min) / close * 1e4

    trend = trend_state(close, args.trend_window, args.trend_threshold_bp)

    absolute = np.abs(z)
    above = absolute >= edges[0]
    if args.all_bars:
        fresh = above
    else:
        # One excursion is one event; without this a slow drift above the
        # threshold counts as twenty, and every statistic below inherits
        # the duplication.
        fresh = above & ~np.concatenate(([False], above[:-1]))

    print(f"{source.name}: {len(bars)} usable bars")
    print(f"z-window {args.z_window}, stride {args.stride}, "
          f"events = {'all bars above' if args.all_bars else 'first bar of each run'}")
    print("Returns are signed as a mean-reversion trade would book them: "
          "short above, long below.")
    print("Overlapping events inflate |t|; raise --stride to thin them.")

    report: dict[str, object] = {
        "source": str(source),
        "z_window": args.z_window,
        "edges": list(edges),
        "horizons": list(horizons),
        "stride": args.stride,
        "blocks": {},
    }

    def bucket_masks(base: np.ndarray) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        for low_edge, high_edge in zip(edges, list(edges[1:]) + [np.inf]):
            label = f"{low_edge:g}-{high_edge:g}" if np.isfinite(high_edge) else f">{low_edge:g}"
            out.append((label, base & (absolute >= low_edge) & (absolute < high_edge)))
        return out

    def run_side(name: str, side_mask: np.ndarray, sign: float, adverse: np.ndarray,
                 title: str) -> None:
        groups: list[Group] = []
        for label, mask in bucket_masks(side_mask):
            group = summarise(mask, sign, label, forwards, adverse, args.stride)
            if group is not None and group.count >= args.min_events:
                groups.append(group)
        if not groups:
            print(f"\n{title}: no bucket reached --min-events")
            return
        print_block(title, groups, horizons)
        report["blocks"][title] = [
            {
                "bucket": g.label, "n": g.count, "mae_bp": g.mae, "t": g.t_stat,
                "mean_bp": g.means, "hit_pct": g.hits,
            }
            for g in groups
        ]

    upper = fresh & (z > 0)
    lower = fresh & (z < 0)

    run_side("upper", upper, -1.0, adverse_upper, "UPPER deviation — short")
    run_side("lower", lower, +1.0, adverse_lower, "LOWER deviation — long")

    if args.by_phase:
        for phase, start, end in DEFAULT_STRATA:
            phase_mask = in_stratum(bars["bucket"], start, end).to_numpy()
            run_side(phase, upper & phase_mask, -1.0, adverse_upper,
                     f"UPPER — {phase} {start}-{end}")
            run_side(phase, lower & phase_mask, +1.0, adverse_lower,
                     f"LOWER — {phase} {start}-{end}")

    if args.by_trend:
        for state in ("up", "flat", "down"):
            state_mask = trend == state
            run_side(state, upper & state_mask, -1.0, adverse_upper,
                     f"UPPER — trend {state}")
            run_side(state, lower & state_mask, +1.0, adverse_lower,
                     f"LOWER — trend {state}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()