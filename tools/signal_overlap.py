"""
Timing comparison between two entry signals.

The question is narrow: when the %K rule fires, does RSI fire as well,
and does it fire earlier or later. Nothing is traded here and no exit
rule is involved, so the answer describes the signals themselves rather
than any strategy built on them.

Both thresholds are calibrated to the same number of events, because two
signals firing at different rates cannot be compared on timing: the one
that fires more often appears to lead simply by being present more.

Usage:
    python tools/signal_overlap.py
    python tools/signal_overlap.py --k-level 0.90 --rsi-window 14 --match-window 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from laplace import features as feature_builder
from laplace import model as model_io
from laplace import oscillators as osc
from laplace import reader, windows
from laplace.config import load_config, setup_logging


def load_series(cfg, model, tail: int):
    """Bars and block identifiers, exactly as the backtest sees them."""
    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column,
        tail=tail if tail > 0 else None,
    )
    prepared = feature_builder.prepare(
        frame,
        window_size=model.rolling_window or cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )
    _, _, ts, mids, block_ids, _ = windows.create_windows(
        prepared,
        lookback=model.lookback or cfg.windows.lookback,
        predict_step=model.predict_step or cfg.windows.predict_step,
        feature_cols=model.feature_columns or None,
        zscore_clip=model.zscore_clip,
    )
    return (pd.DatetimeIndex(pd.to_datetime(ts)),
            np.asarray(mids, dtype=float),
            np.asarray(block_ids))


def first_crossings(reading: np.ndarray, level: float,
                    block_ids: np.ndarray) -> np.ndarray:
    """Bars where |reading| reaches the level after being below it.

    Only the first bar of each excursion counts, so a reading that sits
    past the threshold for twenty bars produces one event rather than
    twenty. A block boundary resets the state.
    """
    values = np.asarray(reading, dtype=float)
    events = []
    inside = False
    for i in range(1, len(values)):
        if block_ids[i] != block_ids[i - 1]:
            inside = False
        if not np.isfinite(values[i]):
            continue
        past = abs(values[i]) >= level
        if past and not inside:
            events.append(i)
        inside = past
    return np.asarray(events, dtype=int)


def level_for_count(reading: np.ndarray, block_ids: np.ndarray,
                    target: int) -> tuple[float, int]:
    """Threshold whose event count comes closest to `target`.

    The count is not monotone in the threshold, so this scans a grid of
    quantiles rather than bisecting.
    """
    finite = np.abs(reading[np.isfinite(reading)])
    best = (float("nan"), -1, np.inf)
    for q in np.arange(0.50, 0.999, 0.005):
        candidate = float(np.quantile(finite, q))
        count = len(first_crossings(reading, candidate, block_ids))
        gap = abs(count - target)
        if gap < best[2]:
            best = (candidate, count, gap)
    return best[0], best[1]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed without scipy."""
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a[good])).astype(float)
    rb = np.argsort(np.argsort(b[good])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def match_events(primary: np.ndarray, secondary: np.ndarray,
                 primary_sign: np.ndarray, secondary_sign: np.ndarray,
                 block_ids: np.ndarray, span: int):
    """Nearest same-sign counterpart for each primary event.

    Returns the offset in bars for every matched pair, negative when the
    secondary signal came first, and the count of primary events with no
    counterpart inside the window.
    """
    offsets = []
    missed = 0
    for k, i in enumerate(primary):
        window = np.where(
            (np.abs(secondary - i) <= span)
            & (secondary_sign == primary_sign[k])
            & (block_ids[secondary] == block_ids[i])
        )[0]
        if window.size == 0:
            missed += 1
            continue
        nearest = window[np.argmin(np.abs(secondary[window] - i))]
        offsets.append(int(secondary[nearest] - i))
    return np.asarray(offsets, dtype=int), missed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the timing of two entry signals.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=0,
                        help="bars to read; 0 reads everything")
    parser.add_argument("--k-window", type=int, default=20)
    parser.add_argument("--k-level", type=float, default=0.90)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--match-window", type=int, default=24,
                        help="bars either side counted as the same event")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    model_path = (Path(args.model) if args.model
                  else cfg.experiment.resolved_model_path)
    model = model_io.load_scaled(model_path)

    ts, mids, block_ids = load_series(cfg, model, args.tail)

    k = osc.stochastic_k(price=mids, window=args.k_window)
    r = osc.rsi(price=mids, window=args.rsi_window)

    k_events = first_crossings(k, args.k_level, block_ids)
    rsi_level, rsi_count = level_for_count(r, block_ids, len(k_events))
    r_events = first_crossings(r, rsi_level, block_ids)

    k_sign = np.sign(k[k_events])
    r_sign = np.sign(r[r_events])

    offsets, missed = match_events(k_events, r_events, k_sign, r_sign,
                                   block_ids, args.match_window)
    back, missed_back = match_events(r_events, k_events, r_sign, k_sign,
                                     block_ids, args.match_window)

    both = np.isfinite(k) & np.isfinite(r)
    pearson = float(np.corrcoef(k[both], r[both])[0, 1])
    rank = spearman(k, r)
    sign_match = float(np.mean(np.sign(k[both]) == np.sign(r[both])))

    print("=" * 72)
    print(f"bars            {len(mids)}  {ts[0]:%Y-%m-%d} .. {ts[-1]:%Y-%m-%d}")
    print(f"%K              window {args.k_window}, level {args.k_level:.3f}")
    print(f"RSI             window {args.rsi_window}, level {rsi_level:.3f}"
          f"  (calibrated to match the event count)")
    print("=" * 72)

    print("\nreadings, bar by bar")
    print(f"  Pearson                    {pearson:+.3f}")
    print(f"  Spearman                   {rank:+.3f}")
    print(f"  same sign                  {100.0 * sign_match:.1f} %")

    print("\nevents")
    print(f"  %K events                  {len(k_events)}")
    print(f"  RSI events                 {len(r_events)}")
    print(f"  %K with an RSI match       {len(offsets)} "
          f"({100.0 * len(offsets) / max(len(k_events), 1):.1f} %)")
    print(f"  %K with none               {missed}")
    print(f"  RSI with a %K match        {len(back)} "
          f"({100.0 * len(back) / max(len(r_events), 1):.1f} %)")
    print(f"  RSI with none              {missed_back}")

    if offsets.size:
        earlier = int((offsets < 0).sum())
        same = int((offsets == 0).sum())
        later = int((offsets > 0).sum())
        print("\ntiming of RSI relative to %K, in bars")
        print(f"  median                     {np.median(offsets):+.1f}")
        print(f"  quartiles                  "
              f"{np.percentile(offsets, 25):+.1f} .. "
              f"{np.percentile(offsets, 75):+.1f}")
        print(f"  RSI earlier                {earlier} "
              f"({100.0 * earlier / len(offsets):.1f} %)")
        print(f"  same bar                   {same} "
              f"({100.0 * same / len(offsets):.1f} %)")
        print(f"  RSI later                  {later} "
              f"({100.0 * later / len(offsets):.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())