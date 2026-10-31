"""
The picture: a tunnel that follows the trend, and where fading it paid.

Three things on one chart, for a stretch of recent bars.

The TUNNEL is drawn around a trailing regression line at plus and minus
k * b * sqrt(horizon). The square root is the whole point: b is the scale
of one bar, and a band meant to contain a move over h bars has to widen
with the square root of h, the way a random walk spreads. For two hours,
which is 24 bars on M5, that is 4.9 times the one-bar width. This is how a
tunnel built for five-minute noise becomes a tunnel for the swing you can
actually see on the chart.

The BACKGROUND says what the market was doing: trending up, trending down,
or neither, from the signed drift over a short window. The threshold is a
quantile of that drift's own magnitude, so "a trend" means "stronger than
most of this market's hours" instead of a number that depends on b being
correctly scaled.

The MARKS show where a contrarian entry would have paid: a dot wherever
fading the deviation from the trend line went on to earn at least
`--payoff` points within the horizon, without first losing more than
`--tolerance`. That second condition matters, because a trade that
eventually works after a hundred points against it is not the same trade.

b is multiplied by `--b-scale`. The scale check measured mean|y| / mean(b)
at 0.45, so b appears to be about 2.2 times the realised move and every
band drawn from it is that much too wide. Passing 0.45 shows the picture
the data implies; the default of 1.0 shows what the model says.

Usage:
    python tools/plot_bands.py --bars 600
    python tools/plot_bands.py --bars 900 --horizon 36 --b-scale 0.45
    python tools/plot_bands.py --bars 600 --out runs/bands_today.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laplace import features as feature_builder
from laplace import model as model_io
from laplace import oscillators as osc
from laplace import reader, swings as sw, windows
from laplace.config import load_config, setup_logging

# Chosen for colour-vision deficiency: blue and vermillion stay distinct.
TREND_COLOUR = {"up": "#cfe6cf", "down": "#f6d4d0", "flat": "#eeeeee"}
BAND_COLOUR = "#0072B2"
MARK_COLOUR = "#D55E00"


def forward_payoff(price, block_ids, horizon: int, side: np.ndarray):
    """Best and worst the given side ever shows within `horizon` bars."""
    price = np.asarray(price, dtype=float)
    blocks = np.asarray(block_ids)
    n = len(price)
    best = np.full(n, np.nan)
    worst = np.full(n, np.nan)
    for i in range(n):
        stop = min(i + horizon, n - 1)
        if stop <= i:
            continue
        same = blocks[i + 1:stop + 1] == blocks[i]
        if not same.any():
            continue
        path = side[i] * (price[i + 1:stop + 1][same] - price[i])
        if path.size:
            best[i] = path.max()
            worst[i] = path.min()
    return best, worst


def regime_spans(state: np.ndarray, threshold: float):
    """Contiguous runs of up-trend, down-trend and no trend."""
    sign = np.where(np.isfinite(state),
                    np.sign(np.where(np.abs(state) >= threshold, state, 0.0)),
                    0.0)
    spans = []
    start = 0
    for i in range(1, len(sign) + 1):
        if i == len(sign) or sign[i] != sign[start]:
            spans.append((start, i - 1,
                          {1.0: "up", -1.0: "down", 0.0: "flat"}[float(sign[start])]))
            start = i
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description="Trend, tunnel and payoff.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bars", type=int, default=600,
                        help="bars to draw, counting back from the latest")
    parser.add_argument("--trend-window", type=int, default=120,
                        help="bars in the regression the tunnel is drawn around")
    parser.add_argument("--horizon", type=int, default=24,
                        help="bars the tunnel is meant to contain; the band "
                             "widens as its square root")
    parser.add_argument("--drift-window", type=int, default=48)
    parser.add_argument("--trend-quantile", type=float, default=0.70,
                        help="|drift| above this quantile counts as a trend")
    parser.add_argument("--payoff", type=float, default=60.0,
                        help="index points a fade must earn to be marked")
    parser.add_argument("--tolerance", type=float, default=40.0,
                        help="points it may lose first and still be marked")
    parser.add_argument("--b-scale", type=float, default=1.0,
                        help="multiplier on b; 0.45 matches the realised move")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    model_path = (Path(args.model) if args.model
                  else cfg.experiment.resolved_model_path)
    model = model_io.load_scaled(model_path)

    frame = reader.read_series(cfg.data.series_path, cfg.data.symbol,
                               cfg.data.price_column,
                               tail=max(args.bars * 4, 4000))
    prepared = feature_builder.prepare(
        frame,
        window_size=model.rolling_window or cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )
    X, y, ts, mids, block_ids, _ = windows.create_windows(
        prepared,
        lookback=model.lookback or cfg.windows.lookback,
        predict_step=model.predict_step or cfg.windows.predict_step,
        feature_cols=model.feature_columns or None,
        zscore_clip=model.zscore_clip,
    )
    _, b = model.predict_batch(X, ts)

    y = np.asarray(y, dtype=float)
    b = np.asarray(b, dtype=float) * args.b_scale
    mids = np.asarray(mids, dtype=float)
    block_ids = np.asarray(block_ids)
    stamps = pd.DatetimeIndex(pd.to_datetime(ts))

    fit = sw.rolling_trend(mids, args.trend_window)
    state = osc.drift(y=y, b=b, window=args.drift_window)
    # A fade is a bet against the current deviation from the line.
    side = -np.sign(mids - fit["fitted"])
    best, worst = forward_payoff(mids, block_ids, args.horizon, side)
    band = b * mids * np.sqrt(args.horizon)

    cut = slice(max(len(mids) - args.bars, 0), len(mids))
    finite_state = np.abs(state[np.isfinite(state)])
    threshold = (float(np.quantile(finite_state, args.trend_quantile))
                 if finite_state.size else 0.0)

    price_v = mids[cut]
    fitted_v = fit["fitted"][cut]
    band_v = band[cut]
    best_v, worst_v = best[cut], worst[cut]
    times = stamps[cut]
    x = np.arange(len(price_v))

    paid = (np.isfinite(best_v) & (best_v >= args.payoff)
            & (worst_v >= -args.tolerance))

    fig, ax = plt.subplots(figsize=(16, 7))
    for start, stop, label in regime_spans(state[cut], threshold):
        ax.axvspan(start, stop, color=TREND_COLOUR[label], alpha=0.75, lw=0)

    for k, alpha in ((1, 0.26), (2, 0.16), (3, 0.09)):
        ax.fill_between(x, fitted_v - k * band_v, fitted_v + k * band_v,
                        color=BAND_COLOUR, alpha=alpha, lw=0,
                        label=f"±{k}·b·√{args.horizon}" if k == 1 else None)
    ax.plot(x, fitted_v, lw=1.1, color=BAND_COLOUR,
            label=f"trend, {args.trend_window} bars")
    ax.plot(x, price_v, lw=0.9, color="black", label="mid")
    ax.scatter(x[paid], price_v[paid], s=10, color=MARK_COLOUR, zorder=3,
               label=f"fade earned ≥{args.payoff:.0f} pts within "
                     f"{args.horizon} bars")

    ticks = np.linspace(0, len(x) - 1, 10, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([times[i].strftime("%d %b %H:%M") for i in ticks],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("index points")
    ax.set_title(
        f"{cfg.data.symbol} M5 — tunnel over {args.horizon} bars "
        f"({args.horizon * 5} min), b x {args.b_scale:g}; "
        f"green = up-trend, red = down-trend, grey = neither "
        f"(|drift| above its {args.trend_quantile:.0%} quantile = {threshold:.2f})"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.margins(x=0)

    out = Path(args.out) if args.out else (
        cfg.journal.runs_dir / f"bands_{times[-1]:%Y-%m-%d_%H%M}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")

    share = 100.0 * np.mean(paid)
    print(f"drawn  : {len(price_v)} bars, {times[0]} .. {times[-1]}")
    print(f"band   : median ±{np.nanmedian(band_v):.1f} points at 1 sigma "
          f"over {args.horizon} bars")
    print(f"marked : {int(paid.sum())} bars ({share:.1f}%) where a fade "
          f"earned ≥{args.payoff:.0f} without losing {args.tolerance:.0f} first")
    for label in ("up", "down", "flat"):
        spans = [s for s in regime_spans(state[cut], threshold) if s[2] == label]
        covered = sum(stop - start + 1 for start, stop, _ in spans)
        inside = paid[np.concatenate([np.arange(a, bq + 1)
                                      for a, bq, _ in spans])] if spans else []
        rate = 100.0 * np.mean(inside) if len(inside) else float("nan")
        print(f"  {label:<5}{covered:>6} bars  fade paid on {rate:5.1f}% of them")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())