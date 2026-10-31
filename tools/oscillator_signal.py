"""
The signal table: enter on an excursion, exit on the way back, see what happens.

This answers the trading question directly. For every oscillator and every
excursion size, it opens a position against the move at the bar the reading
crosses the threshold, closes it when the reading comes back, and reports
what the round trip paid.

WIN RATE IS NOT THE TEST, AND THIS IS THE REASON THE CONTROL COLUMN EXISTS.

On a synthetic random walk with no structure whatsoever, this same rule
shows a 65% win rate and loses money. The exit rule manufactures it: a
position closed when the reading returns to zero books a small gain
whenever the move reverts, and holds on when it does not, so wins are
frequent and losses are large. Any rule with that shape wins most of the
time on noise.

What has to clear is the mean, against cost. A 65% win rate at +1 bp a
trade is a losing strategy at a 4.5 bp round turn; a 35% win rate at
+20 bp is not.

So every row is run twice: once on the market, once on the same shocks
permuted inside each session. The permuted run has no structure by
construction, and its figures are what the rule produces from nothing.
Read the difference.

Thresholds are quantiles of the reading's own magnitude, fitted ON THE
TRAINING SLICE ONLY and applied everywhere, so "0.80" means an excursion
larger than eighty per cent of that oscillator's excursions. That makes
one number comparable across a bounded oscillator like %K and an unbounded
one like the accumulator, and it keeps the test slice out of the choice.

Exit rules, all defined by the oscillator rather than by a fixed horizon:

    zero      the reading crosses back through zero
    half      the magnitude falls below half the entry threshold
    opposite  the reading reaches the threshold on the other side

with a hard bar limit and the session end as backstops.

Usage:
    python tools/oscillator_signal.py
    python tools/oscillator_signal.py --exit opposite --max-hold 24
    python tools/oscillator_signal.py --momentum --out runs/signal_mom.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from laplace import events as ev
from laplace import features as feature_builder
from laplace import model as model_io
from laplace import oscillators as osc
from laplace import reader, windows
from laplace.config import load_config, setup_logging
from laplace.signals import EXIT_RULES, describe, simulate

QUANTILES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)


def permuted_returns(y, b, block_ids, rng):
    """Shocks reordered inside each session; everything else preserved."""
    with np.errstate(divide="ignore", invalid="ignore"):
        shock = osc._normalised(y, b)
    out = shock.copy()
    for block in np.unique(block_ids):
        mask = block_ids == block
        values = out[mask]
        finite = np.isfinite(values)
        if finite.sum() > 1:
            drawn = values[finite]
            rng.shuffle(drawn)
            values[finite] = drawn
            out[mask] = values
    return out * b


def main() -> int:
    parser = argparse.ArgumentParser(
        description="What a threshold rule on each oscillator actually pays."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=100000)
    parser.add_argument("--exit", dest="exit_rule", choices=EXIT_RULES,
                        default="zero", help="how the round trip is closed")
    parser.add_argument("--momentum", action="store_true",
                        help="trade with the excursion instead of against it")
    parser.add_argument("--max-hold", type=int, default=48,
                        help="bars before the position is abandoned")
    parser.add_argument("--window", type=int, default=20,
                        help="shared window for the oscillators that take one")
    parser.add_argument("--cusum-h", type=float, default=3.0)
    parser.add_argument("--oscillators", nargs="+", default=None,
                        help="a subset, by name; all of them when omitted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
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
    _, b = model.predict_batch(X, ts)

    y = np.asarray(y, dtype=float)
    b = np.asarray(b, dtype=float)
    mids = np.asarray(mids, dtype=float)
    block_ids = np.asarray(block_ids)

    splits = ev.split_masks(len(y))
    rng = np.random.default_rng(args.seed)
    y_control = permuted_returns(y, b, block_ids, rng)
    price_control = np.empty(len(y))
    for block in np.unique(block_ids):
        mask = block_ids == block
        start = mids[mask][0] if np.isfinite(mids[mask][0]) else 1.0
        price_control[mask] = start * np.exp(
            np.cumsum(np.nan_to_num(y_control[mask]))
        )

    registry = osc.build_registry(window=args.window, cusum_h=args.cusum_h)
    if args.oscillators:
        registry = {k: v for k, v in registry.items() if k in args.oscillators}

    side_name = "momentum" if args.momentum else "contrarian"
    print("=" * 104)
    print(f"model     : {model_path.name}")
    print(f"bars      : {len(y)}  ({ts[0]} .. {ts[-1]})")
    print(f"rule      : {side_name}, exit on '{args.exit_rule}', "
          f"max hold {args.max_hold} bars ({args.max_hold * 5} min)")
    print(f"thresholds: quantiles of |reading| fitted on the training slice")
    print(f"control   : the same rule on shocks permuted within each session")
    print("=" * 104)

    rows: list[dict] = []
    for name, candidate in registry.items():
        reading = candidate(y=y, b=b, price=mids)
        reading_control = candidate(y=y_control, b=b, price=price_control)

        train_values = np.abs(reading[splits["train"]])
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size < 100:
            print(f"\n{name}: not enough finite readings")
            continue

        print(f"\n{name}  ({candidate.family})")
        print(f"  {'q':>6}{'level':>9}{'n':>7}{'win%':>7}{'mean':>8}"
              f"{'t':>7}{'bars':>7}{'MFE':>8}{'MAE':>8}"
              f"{'| ctrl n':>9}{'win%':>7}{'mean':>8}")
        print("  " + "-" * 98)

        for q in QUANTILES:
            level = float(np.quantile(train_values, q))
            trades = simulate(reading, y, block_ids, level,
                              contrarian=not args.momentum,
                              exit_rule=args.exit_rule, max_hold=args.max_hold)
            control = simulate(reading_control, y_control, block_ids, level,
                               contrarian=not args.momentum,
                               exit_rule=args.exit_rule, max_hold=args.max_hold)
            stats = describe(trades)
            ctrl = describe(control)
            if "win" not in stats:
                continue

            # Out-of-sample figures for the same threshold, by entry bar.
            in_test = splits["test"][trades["entry"]]
            test_stats = describe(trades, mask=in_test)

            rows.append({
                "oscillator": name, "family": candidate.family,
                "side": side_name, "exit": args.exit_rule, "quantile": q,
                "level": level, **{f"all_{k}": v for k, v in stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                **{f"ctrl_{k}": v for k, v in ctrl.items()},
            })
            print(f"  {q:>6.2f}{level:>9.3f}{stats['n']:>7}{stats['win']:>7.1f}"
                  f"{stats['mean']:>+8.2f}{stats['t']:>+7.2f}"
                  f"{stats['bars']:>7.1f}{stats['mfe']:>+8.1f}{stats['mae']:>+8.1f}"
                  f"{ctrl.get('n', 0):>9}{ctrl.get('win', float('nan')):>7.1f}"
                  f"{ctrl.get('mean', float('nan')):>+8.2f}")

    if rows:
        print("\n" + "=" * 104)
        print("Ranked by mean minus the control's mean, which is the part the "
              "market supplied.")
        print(f"  {'oscillator':<20}{'q':>6}{'n':>7}{'win%':>7}{'mean':>8}"
              f"{'ctrl':>8}{'excess':>9}{'test n':>8}"
              f"{'test mean':>11}")
        print("  " + "-" * 96)
        ranked = sorted(
            rows,
            key=lambda r: -(r.get("all_mean", 0) - r.get("ctrl_mean", 0)),
        )
        for row in ranked[:15]:
            excess = row.get("all_mean", 0) - row.get("ctrl_mean", 0)
            print(f"  {row['oscillator']:<20}{row['quantile']:>6.2f}"
                  f"{row.get('all_n', 0):>7}{row.get('all_win', 0):>7.1f}"
                  f"{row.get('all_mean', 0):>+8.2f}{row.get('ctrl_mean', 0):>+8.2f}"
                  f"{excess:>+9.2f}"
                  f"{row.get('test_n', 0):>8}"
                  f"{row.get('test_mean', float('nan')):>+11.2f}")

    if args.out and rows:
        import csv

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten: {out}")

    print("\nRead the excess column, not win%. The control run has no "
          "structure, so whatever it earns is what the exit rule earns from "
          "noise. A row is tradeable when the excess clears zero and the "
          "test column agrees with the full sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())