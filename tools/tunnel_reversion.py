"""
Does a breach of the Laplace tunnel predict anything about what follows?

The strategy was conceived around a tunnel breach reverting. This measures
whether that is in the data, separately from whether the code exploits it.

What is measured
----------------
For every bar whose return leaves the tunnel, mu +/- k*b, the forward return
over the next h bars is accumulated and signed AGAINST the breach:

    reversion(h) = -sign(breach) * sum(y[t+1 .. t+h])

Positive means price came back. Negative means it carried on. Windows never
cross a session boundary, so a forward sum is cut at the end of its block
rather than jumping a weekend.

Three comparisons, because a number on its own says nothing
-----------------------------------------------------------
* The unconditional forward return, which is what any bar gives.
* A shuffle control: the same statistic on permuted returns, which destroys
  every time dependence while keeping the distribution. Its spread is what
  zero looks like at this sample size. A conditional figure inside that band
  is noise however large it looks.
* The spread, 4.5 bp on US.100+, which a reversion trade pays.

A separate question, often confused with the first
--------------------------------------------------
"Price returns to the tunnel" is close to a tautology. The tunnel describes
where the NEXT return lands, and under the model that draw is independent of
the last one, so after a breach the next bar is back inside with the tunnel's
own coverage, 63 per cent at k=1 and 95 per cent at k=3, whether or not
anything reverted. The report shows that probability beside its unconditional
value: equal numbers mean the breach carried no information. What would be
tradeable is the signed forward return above, not the return to the band.

Usage:
    python tools/tunnel_reversion.py
    python tools/tunnel_reversion.py --k 1 2 3 --horizons 1 2 3 6 12
    python tools/tunnel_reversion.py --tail 60000 --shuffles 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

DEFAULT_SPREAD_BP = 4.5


def forward_sums(y: np.ndarray, block_ids: np.ndarray, horizon: int) -> np.ndarray:
    """Sum of the next `horizon` returns, NaN where the session ends first.

    A forward window must stay inside one trading session: a sum reaching
    across a break carries the opening jump as if it were a price move, which
    is the mistake that once produced a catastrophic drawdown in this project.
    """
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(n - horizon):
        if block_ids[i] != block_ids[i + horizon]:
            continue
        out[i] = y[i + 1:i + 1 + horizon].sum()
    return out


def reversion_stats(
    y: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray,
    block_ids: np.ndarray,
    k: float,
    horizon: int,
) -> dict:
    """Signed forward return after a tunnel breach, against the breach."""
    deviation = y - mu
    up = deviation > k * b
    down = deviation < -k * b
    breach = up | down

    forward = forward_sums(y, block_ids, horizon)
    usable = breach & ~np.isnan(forward)

    if usable.sum() == 0:
        return {"n": 0}

    direction = np.where(up[usable], 1.0, -1.0)
    reversion = -direction * forward[usable]

    unconditional = forward[~np.isnan(forward)]

    return {
        "n": int(usable.sum()),
        "n_up": int((up & ~np.isnan(forward)).sum()),
        "n_down": int((down & ~np.isnan(forward)).sum()),
        "mean_bp": float(reversion.mean()) * 10000,
        "median_bp": float(np.median(reversion)) * 10000,
        "positive_pct": float((reversion > 0).mean()) * 100,
        "t_stat": float(reversion.mean() / (reversion.std(ddof=1) / np.sqrt(len(reversion))))
        if reversion.std(ddof=1) > 0 else 0.0,
        "unconditional_bp": float(unconditional.mean()) * 10000,
    }


def shuffle_band(
    y: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray,
    block_ids: np.ndarray,
    k: float,
    horizon: int,
    shuffles: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Spread of the statistic when every time dependence is destroyed.

    The returns are permuted inside their own session, so the distribution,
    the session structure and the tunnel widths all survive while the order
    does not. Whatever the real figure is, it has to clear this band.
    """
    results = []
    for _ in range(shuffles):
        shuffled = y.copy()
        for block in np.unique(block_ids):
            mask = block_ids == block
            shuffled[mask] = rng.permutation(y[mask])
        stats = reversion_stats(shuffled, mu, b, block_ids, k, horizon)
        if stats["n"]:
            results.append(stats["mean_bp"])
    if not results:
        return 0.0, 0.0
    return float(np.mean(results)), float(np.std(results))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure what follows a Laplace tunnel breach."
    )
    parser.add_argument("--k", type=float, nargs="+", default=[1, 2, 3],
                        help="tunnel widths to test, default 1 2 3")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6, 12],
                        help="forward horizons in bars, default 1 2 3 6 12")
    parser.add_argument("--tail", type=int, default=100000,
                        help="rows to read from the series, default 100000")
    parser.add_argument("--shuffles", type=int, default=100,
                        help="permutations for the control band, default 100")
    parser.add_argument("--spread-bp", type=float, default=DEFAULT_SPREAD_BP,
                        help=f"round-turn cost, default {DEFAULT_SPREAD_BP}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="path to strategy.json")
    args = parser.parse_args()

    from laplace import features as feature_builder
    from laplace import model as model_io
    from laplace import reader, windows
    from laplace.config import load_config, setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg)

    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=args.tail
    )
    prepared = feature_builder.prepare(
        frame,
        window_size=cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )
    X, y, ts, mids, block_ids, _ = windows.create_windows(
        prepared,
        lookback=cfg.windows.lookback,
        predict_step=cfg.windows.predict_step,
        zscore_clip=cfg.windows.zscore_clip,
    )
    if len(X) == 0:
        print("No windows to measure.")
        return 1

    model = model_io.load_model(cfg.experiment.resolved_model_path)
    mu, b = model_io.predict_batch(model, X)

    print(f"\nmodel    : {cfg.experiment.resolved_model_path.name}")
    print(f"bars     : {len(y)}  ({ts[0]} .. {ts[-1]})")
    print(f"sessions : {len(np.unique(block_ids))}")
    print(f"b        : mean {b.mean():.8f}, std {b.std():.2e}"
          + ("   (CONSTANT — the tunnel has one width everywhere)"
             if b.std() < 1e-5 else ""))
    print(f"spread   : {args.spread_bp:.1f} bp per round turn")

    rng = np.random.default_rng(args.seed)

    for k in args.k:
        deviation = y - mu
        breaches = int(((deviation > k * b) | (deviation < -k * b)).sum())
        rate = breaches / len(y) * 100
        theory = np.exp(-k) * 100
        print(f"\n{'=' * 74}")
        print(f"k = {k:g}   breaches: {breaches} of {len(y)} bars "
              f"({rate:.2f}%, Laplace theory {theory:.2f}%)")
        print(f"{'=' * 74}")
        print(f"  {'horizon':>8}{'n':>7}{'mean bp':>10}{'median':>9}{'pos %':>8}"
              f"{'t':>7}{'shuffle sd':>12}{'net bp':>9}")

        for horizon in args.horizons:
            stats = reversion_stats(y, mu, b, block_ids, k, horizon)
            if not stats["n"]:
                continue
            _, sd = shuffle_band(y, mu, b, block_ids, k, horizon,
                                 args.shuffles, rng)
            net = stats["mean_bp"] - args.spread_bp
            print(f"  {horizon:>8}{stats['n']:>7}{stats['mean_bp']:>10.2f}"
                  f"{stats['median_bp']:>9.2f}{stats['positive_pct']:>8.1f}"
                  f"{stats['t_stat']:>7.2f}{sd:>12.2f}{net:>9.2f}")

        # The tautology check, shown so it cannot be mistaken for a finding.
        deviation = y - mu
        inside = np.abs(deviation) < k * b
        breach = ~inside
        after_breach = inside[1:][breach[:-1]]
        print(f"\n  next bar back inside the tunnel: "
              f"{after_breach.mean() * 100:.1f}% after a breach, "
              f"{inside.mean() * 100:.1f}% unconditionally")
        print(f"  (equal figures mean the breach said nothing about the next bar)")

    print(f"\n{'=' * 74}")
    print("  Read the mean against the shuffle sd. Inside roughly two of them")
    print("  the figure is noise, whatever its size. Read the net column")
    print("  against zero: that is what a reversion trade would have kept.")
    print(f"{'=' * 74}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
