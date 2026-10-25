"""The tunnel as it is defined: a two-sided band, and the points inside it.

mu +/- k*b is a band around the signed return, not a half-width. Folding
the two sides together hides whether misses are symmetric and makes it
impossible to see which bars the tunnel actually contained, which is the
only question a tunnel answers.

So: one panel per model, the signed return of every bar drawn against that
model's bands at k = 1, 3 and 5, with each point coloured by the innermost
band that holds it. Measured coverage is printed in the corner beside the
Laplace figure it is supposed to match, because the gap between the two is
the whole finding.

Features are rebuilt per model from its own card — feature set, clipping,
rolling window, intraday profile — so models trained under different
settings are compared on the same bars without any of them being
reconstructed wrongly.

Usage:
    python tools/plot_tunnel.py --models models/*.keras --bars 600
    python tools/plot_tunnel.py --models models/bielu_ou_seed42_level_deseason_noclip_r20.keras ^
                                --bars 300 --ks 1 3 5 --out runs/tunnel.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laplace import features as feature_builder
from laplace import model as model_io
from laplace import reader, windows
from laplace.config import load_config, setup_logging

try:
    from laplace.core.losses import MIN_B
except ImportError:
    MIN_B = 1e-9

INK = "#1a2027"
MUTED = "#5b6673"
GRID = "#dce0e6"
BAND = "#8a97a5"

# Containment is ordinal, so it is encoded twice: hue and size. The two hues
# were checked for colour-vision separation; the grey is the quiet default.
INSIDE = "#9aa6b2"
BETWEEN = "#0072B2"
OUTSIDE = "#D55E00"


def read_card(path: Path) -> dict:
    """The model's own record of how it was trained."""
    card_path = path.with_suffix(".json")
    if not card_path.is_file():
        raise SystemExit(
            f"No card beside {path.name}. A model without one cannot be "
            f"reconstructed, which is the reason cards exist."
        )
    return json.loads(card_path.read_text(encoding="utf-8"))


def eastern_minutes(timestamps: np.ndarray, broker_offset_hours: float) -> np.ndarray:
    """Minute of day on the New York clock, from broker timestamps."""
    series = pd.Series(pd.to_datetime(timestamps))
    offset = timezone(timedelta(hours=broker_offset_hours))
    local = series.dt.tz_localize(offset).dt.tz_convert("America/New_York")
    return (local.dt.hour * 60 + local.dt.minute).to_numpy()


def profile_scale(card: dict, timestamps: np.ndarray) -> np.ndarray:
    """Rebuild the seasonal multiplier this model's b has to be scaled by.

    Interpolated across bucket centres rather than held as steps: the
    profile is a smooth daily shape, and a step function puts cliffs into b
    at bucket edges that no market event explains.
    """
    if not card.get("deseasonalised"):
        return np.ones(len(timestamps))

    table = {int(k): float(v) for k, v in card["intraday_profile"].items()}
    bucket_minutes = int(card.get("profile_bucket_minutes", 15))
    minutes = eastern_minutes(timestamps, float(card.get("broker_utc_offset", 2.0)))

    centres = np.array(sorted(table), dtype=float) + bucket_minutes / 2.0
    values = np.array([table[int(c - bucket_minutes / 2.0)] for c in centres])

    # Wrap at midnight so the interpolation closes the circle.
    extended_x = np.concatenate(([centres[-1] - 1440.0], centres, [centres[0] + 1440.0]))
    extended_y = np.concatenate(([values[-1]], values, [values[0]]))

    scale = np.interp(minutes.astype(float), extended_x, extended_y)
    return np.maximum(scale, 1e-6)


def score(path: Path, cfg, frame) -> dict:
    """Run one model over the frame and return everything the plot needs.

    Features are rebuilt here rather than once for all models: two models
    may have been trained on different rolling windows, and drawing one of
    them on the other's features would be a silent lie.
    """
    import tensorflow as tf

    card = read_card(path)
    columns = tuple(card["features"])
    clip = card.get("zscore_clip")
    clip = float("inf") if clip is None else float(clip)

    prepared = feature_builder.prepare(
        frame,
        window_size=int(card.get("rolling_window", cfg.windows.rolling_window)),
        gap_threshold_days=cfg.data.gap_threshold_days,
    )

    X, y, ts, mids, _, _ = windows.create_windows(
        prepared,
        lookback=int(card.get("lookback", cfg.windows.lookback)),
        predict_step=int(card.get("predict_step", cfg.windows.predict_step)),
        feature_cols=columns,
        zscore_clip=clip,
    )

    model = model_io.load_model(path)
    raw = model.predict(X, verbose=0)
    scale = profile_scale(card, ts)

    return {
        "name": path.stem,
        "mu": raw[:, 0] * scale,
        "b": (tf.nn.softplus(raw[:, 1]).numpy() + MIN_B) * scale,
        "y": y,
        "ts": ts,
        "mid": mids,
    }


def draw_model(axis, item: dict, view: slice, ks: tuple[float, ...], index) -> dict:
    """One panel: the bands, the points, and the coverage they imply."""
    y = item["y"][view] * 1e4
    mu = item["mu"][view] * 1e4
    b = item["b"][view] * 1e4

    deviation = np.abs(y - mu)
    coverage = {k: float(np.mean(deviation <= k * b)) for k in ks}

    # Widest band first, so the narrower ones stay visible on top of it.
    for depth, k in enumerate(sorted(ks, reverse=True)):
        axis.fill_between(
            index, mu - k * b, mu + k * b,
            color=BAND, alpha=0.10 + 0.07 * depth, linewidth=0,
        )
    inner = min(ks)
    axis.plot(index, mu + inner * b, color=BAND, linewidth=1.0)
    axis.plot(index, mu - inner * b, color=BAND, linewidth=1.0)
    axis.axhline(0.0, color=GRID, linewidth=0.8)

    middle = sorted(ks)[1] if len(ks) > 1 else inner
    inside = deviation <= inner * b
    between = (~inside) & (deviation <= middle * b)
    beyond = deviation > middle * b

    axis.scatter(index[inside], y[inside], s=5, color=INSIDE, alpha=0.55, linewidths=0)
    axis.scatter(index[between], y[between], s=14, color=BETWEEN, alpha=0.85, linewidths=0)
    axis.scatter(index[beyond], y[beyond], s=30, color=OUTSIDE, alpha=0.95, linewidths=0)

    parts = [f"k{k:g} {coverage[k] * 100:.1f}%" for k in sorted(ks)]
    theory = [f"{(1 - np.exp(-k)) * 100:.0f}%" for k in sorted(ks)]
    axis.text(
        0.006, 0.95,
        f"{item['name']}\nmeasured  {' · '.join(parts)}\nLaplace   {' · '.join(theory)}",
        transform=axis.transAxes, va="top", ha="left",
        fontsize=8.5, color=MUTED, family="monospace", linespacing=1.5,
    )
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=600)
    parser.add_argument("--ks", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    parser.add_argument("--out", type=Path, default=Path("runs/tunnel.png"))
    parser.add_argument("--tail", type=int, default=200000)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ks = tuple(sorted(args.ks))

    cfg = load_config(args.config)
    setup_logging(cfg)

    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=args.tail
    )

    scored = [score(path, cfg, frame) for path in args.models]

    count = min(args.bars, min(len(item["y"]) for item in scored))
    view = slice(-count, None)
    index = np.arange(count)

    reference = scored[0]
    times = pd.to_datetime(reference["ts"][view])

    rows = 1 + len(scored)
    heights = [0.8] + [1.0] * len(scored)
    figure, axes = plt.subplots(
        rows, 1, figsize=(13, 2.6 * rows), sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.10},
    )
    figure.patch.set_facecolor("white")
    axes = np.atleast_1d(axes)

    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(True, axis="y", color=GRID, linewidth=0.7)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=9)

    axes[0].plot(index, reference["mid"][view], color=INK, linewidth=1.1)
    axes[0].set_ylabel("price", color=MUTED, fontsize=10)
    axes[0].set_title(
        f"{cfg.data.symbol} {cfg.data.timeframe} · tunnel mu ± k·b, "
        f"k = {', '.join(f'{k:g}' for k in ks)}",
        color=INK, fontsize=13, loc="left", pad=12,
    )

    limit = 0.0
    for axis, item in zip(axes[1:], scored):
        draw_model(axis, item, view, ks, index)
        axis.set_ylabel("return, bp", color=MUTED, fontsize=10)
        y = item["y"][view] * 1e4
        limit = max(limit, float(np.quantile(np.abs(y), 0.999)) * 1.15)

    for axis in axes[1:]:
        axis.set_ylim(-limit, limit)

    middle = sorted(ks)[1] if len(ks) > 1 else min(ks)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=4,
                   color=INSIDE, label=f"inside k{min(ks):g}"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                   color=BETWEEN, label=f"k{min(ks):g} to k{middle:g}"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   color=OUTSIDE, label=f"beyond k{middle:g}"),
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=9,
                    labelcolor=MUTED, loc="lower left", ncols=3)

    ticks = np.linspace(0, count - 1, 7, dtype=int)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([times[i].strftime("%d %b %H:%M") for i in ticks], fontsize=9)
    for axis in axes:
        axis.set_xlim(0, count - 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    print(f"bars  {count}  ({times[0]} .. {times[-1]})\n")

    print(f"{'model':<52}" + "".join(f"{f'k{k:g}':>9}" for k in ks))
    for item in scored:
        y = item["y"][view] * 1e4
        mu = item["mu"][view] * 1e4
        b = item["b"][view] * 1e4
        deviation = np.abs(y - mu)
        row = "".join(f"{np.mean(deviation <= k * b) * 100:>8.1f}%" for k in ks)
        print(f"{item['name']:<52}{row}")
    print(f"{'Laplace theory':<52}" +
          "".join(f"{(1 - np.exp(-k)) * 100:>8.0f}%" for k in ks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
