"""
What happens after a tunnel breach, if you take the other side.

The event: the realized M5 return breaches k times the predicted Laplace
scale, |y| / b >= k, where b was predicted from a window that closed before
that bar. The event is therefore known at the bar's close, and the forward
path starts at the next bar. Nothing here can see its own outcome.

The position: contrarian, sign = -sign(y). An up-breach is sold, a
down-breach is bought.

What it measures, per breach bucket and per direction:

    opposite      share of next bars moving against the breach. This is the
                  50% test: below it, there is nothing to trade.
    mean / median forward return of the contrarian position at each horizon
    MFE           Maximum Favourable Excursion, the best the position ever
                  showed inside the horizon. This sizes a take-profit and
                  says whether the asymmetry exists at all.
    MAE           Maximum Adverse Excursion, the worst it showed. This sizes
                  a stop, and must be read on the WINNING trades too: a stop
                  tighter than their MAE cuts the trades that paid.

A control runs the same statistics on bars inside k1, so a 55% share can be
compared against what the series does anyway.

Two breach kinds are reported separately, because they are different objects:

    volatility    |y| / b, an increment outlier. Carries no direction of its
                  own beyond the sign of that one bar.
    conjunction   the same bar also sitting at a level extreme,
                  |modified_z_score| >= level_threshold. This is the case
                  where the increment and the level agree.

Costs are not subtracted. Everything reported is gross.

Usage:
    python tools/breach_direction.py
    python tools/breach_direction.py --horizon 12 --level-threshold 3.0
    python tools/breach_direction.py --out runs/breach.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from laplace import features as feature_builder
from laplace import model as model_io
from laplace import reader, windows
from laplace.config import load_config, setup_logging

BUCKET_EDGES = (3.0, 4.0, 5.0, np.inf)
HORIZONS = (1, 2, 3, 4, 6, 8, 12)


def to_bp(x: float) -> float:
    """Log return to basis points."""
    return float(x) * 10000.0


def forward_matrix(y: np.ndarray, block_ids: np.ndarray, horizon: int) -> np.ndarray:
    """Cumulative forward returns, NaN where the path leaves its block.

    Row i holds the cumulative return of bars i+1 .. i+horizon. A path that
    would cross a weekend or an overnight break is dropped rather than
    stitched, because the price it would need does not exist.
    """
    n = len(y)
    paths = np.full((n, horizon), np.nan)
    for step in range(1, horizon + 1):
        idx = np.arange(n) + step
        valid = idx < n
        same = np.zeros(n, dtype=bool)
        same[valid] = block_ids[idx[valid]] == block_ids[:n][valid]
        prev = paths[:, step - 2] if step > 1 else np.zeros(n)
        contrib = np.full(n, np.nan)
        contrib[same] = y[idx[same]]
        paths[:, step - 1] = prev + contrib
    return paths


def describe(signed: np.ndarray, label: str, count: int) -> dict:
    """Summarise one set of signed forward paths at the final horizon."""
    final = signed[:, -1]
    good = np.isfinite(final)
    if good.sum() == 0:
        return {}

    run = signed[good]
    final = final[good]
    mfe = np.nanmax(run, axis=1)
    mae = np.nanmin(run, axis=1)
    won = final > 0

    return {
        "label": label,
        "n": int(good.sum()),
        "share_of_bars": 100.0 * count / len(signed) if len(signed) else 0.0,
        "win_rate": 100.0 * float(won.mean()),
        "mean_bp": to_bp(np.mean(final)),
        "median_bp": to_bp(np.median(final)),
        "mfe_median_bp": to_bp(np.median(mfe)),
        "mfe_q75_bp": to_bp(np.quantile(mfe, 0.75)),
        "mae_median_bp": to_bp(np.median(mae)),
        "mae_q10_bp": to_bp(np.quantile(mae, 0.10)),
        # The stop has to clear the adverse excursion of the trades that
        # eventually paid, or it removes exactly those.
        "mae_winners_q10_bp": (to_bp(np.quantile(mae[won], 0.10))
                               if won.sum() else float("nan")),
        "edge_ratio": (float(np.median(mfe) / abs(np.median(mae)))
                       if np.median(mae) != 0 else float("nan")),
    }


def horizon_table(signed: np.ndarray, horizons: tuple[int, ...]) -> list[dict]:
    """Mean and win rate at each horizon, to find where reversion stops."""
    rows = []
    for h in horizons:
        if h > signed.shape[1]:
            continue
        col = signed[:, h - 1]
        good = np.isfinite(col)
        if good.sum() < 20:
            continue
        values = col[good]
        rows.append({
            "bars": h,
            "minutes": h * 5,
            "n": int(good.sum()),
            "win_rate": 100.0 * float((values > 0).mean()),
            "mean_bp": to_bp(np.mean(values)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Direction and excursion after a tunnel breach."
    )
    parser.add_argument("--model", default=None, help="a .keras file")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=100000)
    parser.add_argument("--horizon", type=int, default=12,
                        help="bars to follow the contrarian position")
    parser.add_argument("--level-threshold", type=float, default=3.0,
                        help="|modified_z_score| that counts as a level extreme")
    parser.add_argument("--out", default=None, help="write the summary as CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    model_path = (Path(args.model) if args.model
                  else cfg.experiment.resolved_model_path)
    model = model_io.load_scaled(model_path)

    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=args.tail
    )
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
    mu, b = model.predict_batch(X, ts)

    y = np.asarray(y, dtype=float)
    b = np.asarray(b, dtype=float)
    block_ids = np.asarray(block_ids)

    # Standardised increment, known at the close of its own bar.
    z_std = y / np.where(b > 0, b, np.nan)

    # The level at the breach bar sits at the end of the NEXT window, since
    # window i closes one bar before y[i]. Indexed from the card, never from
    # FEATURE_COLUMNS, because this model's window has two columns.
    cols = list(model.feature_columns or ())
    level = np.full(len(y), np.nan)
    if "modified_z_score" in cols:
        k = cols.index("modified_z_score")
        level[:-1] = X[1:, -1, k]

    paths = forward_matrix(y, block_ids, args.horizon)
    side = -np.sign(y)                      # contrarian
    signed = paths * side[:, None]

    print("=" * 96)
    print(f"model      : {model_path.name}")
    print(f"windows    : {len(y)}  ({ts[0]} .. {ts[-1]})")
    print(f"horizon    : {args.horizon} bars ({args.horizon * 5} min)")
    print(f"level test : |modified_z_score| >= {args.level_threshold}")
    print("=" * 96)

    rows: list[dict] = []
    finite = np.isfinite(z_std)

    def report(mask: np.ndarray, label: str) -> None:
        mask = mask & finite & np.isfinite(signed[:, -1])
        if mask.sum() < 20:
            print(f"\n{label}: {int(mask.sum())} events, too few to read")
            return
        stats = describe(signed[mask], label, int(mask.sum()))
        stats["share_of_bars"] = 100.0 * mask.sum() / finite.sum()
        rows.append(stats)

        opposite = 100.0 * float(
            (np.sign(y[mask]) != np.sign(np.nan_to_num(paths[mask, 0]))).mean()
        )
        print(f"\n{label}")
        print(f"  events            {stats['n']:>8}   "
              f"({stats['share_of_bars']:.2f}% of bars)")
        print(f"  opposite next bar {opposite:>8.1f}%   (the 50% test)")
        print(f"  win rate @{args.horizon:<3}      {stats['win_rate']:>8.1f}%")
        print(f"  mean              {stats['mean_bp']:>8.2f} bp")
        print(f"  median            {stats['median_bp']:>8.2f} bp")
        print(f"  MFE median / q75  {stats['mfe_median_bp']:>8.2f} / "
              f"{stats['mfe_q75_bp']:.2f} bp")
        print(f"  MAE median / q10  {stats['mae_median_bp']:>8.2f} / "
              f"{stats['mae_q10_bp']:.2f} bp")
        print(f"  MAE of winners q10{stats['mae_winners_q10_bp']:>8.2f} bp   "
              f"(a stop must clear this)")
        print(f"  edge ratio        {stats['edge_ratio']:>8.2f}     (MFE/|MAE|, median)")

        for row in horizon_table(signed[mask], HORIZONS):
            print(f"      {row['bars']:>2} bars ({row['minutes']:>3} min)  "
                  f"n {row['n']:>5}  win {row['win_rate']:>5.1f}%  "
                  f"mean {row['mean_bp']:>7.2f} bp")

    # Control: bars the tunnel contains.
    report(np.abs(z_std) < 1.0, "control, inside k1")

    # Volatility breaches, by bucket and by direction.
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        band = (np.abs(z_std) >= lo) & (np.abs(z_std) < hi)
        name = f"k{lo:.0f}+" if np.isinf(hi) else f"k{lo:.0f}-k{hi:.0f}"
        report(band, f"breach {name}, both directions")
        report(band & (y > 0), f"breach {name}, up-breach sold")
        report(band & (y < 0), f"breach {name}, down-breach bought")

    # The conjunction: increment outlier at a level extreme.
    if np.isfinite(level).any():
        extreme = np.abs(level) >= args.level_threshold
        beyond = np.abs(z_std) >= 3.0
        report(beyond & extreme, "breach k3+ AND level extreme")
        report(beyond & ~extreme, "breach k3+ at a middling level")
        # Agreement of sign matters: a level already high with an up-breach
        # is a different setup from a level low with an up-breach.
        report(beyond & extreme & (np.sign(y) == np.sign(level)),
               "breach k3+, level agrees with the breach")
        report(beyond & extreme & (np.sign(y) != np.sign(level)),
               "breach k3+, level opposes the breach")

    if args.out and rows:
        import csv

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten: {out}")

    print("\nRead it this way: the contrarian side exists only where the "
          "opposite share clears 50% and the mean clears zero. The edge "
          "ratio says whether the asymmetry you are after is there; below "
          "about 1.5 a stop and a target cannot both fit inside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())