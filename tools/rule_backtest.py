"""
One rule, walked through the data the way an account would experience it.

The bake-off ranks candidates. This takes the one that won and answers the
questions an account asks: how many positions, how often, how long held,
how much of the session spent exposed, and what the sequence of outcomes
looks like month by month.

Results are reported in index points always, and in account currency when
config/instruments.json carries a complete specification. Points are the
honest unit until the contract size is known: a profit figure built on a
guessed contract size is wrong by whatever factor the broker chose, and
nothing about it looks wrong.

The control run is kept, on shocks permuted inside each session. It is not
a benchmark that can be traded; it is the answer to "would this exit rule
have produced these figures from a series with no structure", and it
belongs next to the result for exactly as long as the result is small.

Usage:
    python tools/rule_backtest.py
    python tools/rule_backtest.py --oscillator cci --window 40 --exit half
    python tools/rule_backtest.py --lots 0.30 --balance 10058.30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from laplace import events as ev
from laplace import features as feature_builder
from laplace import model as model_io
from laplace import money as money_layer
from laplace import oscillators as osc
from laplace import reader, windows
from laplace.config import load_config, load_instruments, setup_logging
from laplace.signals import describe, simulate

BARS_PER_SESSION = 276


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


def rebuild_price(y_control, mids, block_ids):
    """A price path consistent with the permuted shocks."""
    out = np.empty(len(y_control))
    for block in np.unique(block_ids):
        mask = block_ids == block
        start = mids[mask][0]
        start = start if np.isfinite(start) and start > 0 else 1.0
        out[mask] = start * np.exp(np.cumsum(np.nan_to_num(y_control[mask])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="What one rule does to an account."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=100000)
    parser.add_argument("--oscillator", default="stochastic_k",
                        help="which candidate to run")
    parser.add_argument("--window", type=int, default=20,
                        help="its window, where it takes one")
    parser.add_argument("--cusum-h", type=float, default=3.0)
    parser.add_argument("--quantile", type=float, default=0.70,
                        help="entry level, as a quantile of |reading| on train")
    parser.add_argument("--level", type=float, default=None,
                       help="entry threshold given directly, bypassing the quantile")
    parser.add_argument("--exit", dest="exit_rule", default="zero",
                        choices=("zero", "half", "opposite", "level"))
    parser.add_argument("--momentum", action="store_true")
    parser.add_argument("--max-hold", type=int, default=48)
    parser.add_argument("--target", type=float, default=None,
                        help="take profit in index points; none by default")
    parser.add_argument("--stop", type=float, default=None,
                        help="stop loss in index points; none by default")
    parser.add_argument("--trail", type=float, default=None,
                        help="trailing stop in index points, measured back "
                             "from the best the position has shown")
    parser.add_argument("--trail-arm", type=float, default=None,
                        help="points of profit before the trailing stop "
                             "activates; below it the position is left alone")
    parser.add_argument("--trail-after-signal", action="store_true",
                       help="arm a trailing stop at the oscillator exit instead of closing")
    parser.add_argument("--trail-start", type=float, default=88.0,
                       help="stop distance in points at the moment of arming")
    parser.add_argument("--trail-min", type=float, default=20.0,
                       help="floor the stop distance narrows to")
    parser.add_argument("--trail-tau", type=int, default=24,
                       help="bars over which the stop narrows from start to floor")
    parser.add_argument("--hold-after-arm", type=int, default=None,
                       help="bars allowed after arming, replacing the remaining max_hold")
    parser.add_argument("--trend-window", type=int, default=200,
                        help="bars in the detrending regression")
    parser.add_argument("--trend-method", choices=("ols", "holt"),
                        default="ols")
    parser.add_argument("--amplitude-b", type=float, default=5.87,
                        help="median swing in units of b, measured on history")
    parser.add_argument("--exit-z", type=float, default=None,
                        help="close when |reading| falls to this; implies "
                             "--exit level")
    parser.add_argument("--drift-window", type=int, default=24,
                        help="bars in the trend-state measurement")
    parser.add_argument("--max-drift", type=float, default=None,
                        help="refuse entries when |trend state| exceeds this")
    parser.add_argument("--control-seeds", type=int, default=8,
                        help="permutations averaged into the baseline")
    parser.add_argument("--lots", type=float, default=0.30)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="write the trades as CSV")
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
    stamps = pd.DatetimeIndex(pd.to_datetime(ts))

    if args.oscillator == "trend_deviation":
        # Built by hand because it needs the session boundaries, which the
        # shared Candidate signature does not carry.
        candidate = osc.Candidate(
            osc.trend_deviation,
            {"block_ids": block_ids, "method": args.trend_method,
             "window": args.trend_window, "amplitude_b": args.amplitude_b},
            "trend",
        )
    else:
        registry = osc.build_registry(window=args.window, cusum_h=args.cusum_h)
        if args.oscillator not in registry:
            raise SystemExit(f"unknown oscillator: {args.oscillator}")
        candidate = registry[args.oscillator]

    reading = candidate(y=y, b=b, price=mids)
    splits = ev.split_masks_by_date(ts)
    train_values = np.abs(reading[splits["train"]])
    train_values = train_values[np.isfinite(train_values)]
    level = (args.level if args.level is not None
             else float(np.quantile(train_values, args.quantile)))

    entry_mask = None
    if args.max_drift is not None:
        state = osc.trend_state(y=y, b=b, window=args.drift_window)
        entry_mask = np.isfinite(state) & (np.abs(state) <= args.max_drift)
        print(f"drift filter : |state| <= {args.max_drift} over "
              f"{args.drift_window} bars keeps "
              f"{100.0 * entry_mask.mean():.1f}% of bars")
    exit_rule = "level" if args.exit_z is not None else args.exit_rule

    trades = simulate(reading, y, block_ids, level, price=mids,
                      contrarian=not args.momentum,
                      exit_rule=exit_rule, max_hold=args.max_hold,
                      target_points=args.target, stop_points=args.stop,
                      trail_points=args.trail, trail_arm=args.trail_arm,
                      trail_after_signal=args.trail_after_signal,
                      trail_start=args.trail_start, trail_min=args.trail_min,
                      trail_tau=args.trail_tau, hold_after_arm=args.hold_after_arm,
                      exit_level=args.exit_z, entry_mask=entry_mask)

    # One permutation is not a baseline. On the same bars three seeds gave
    # +1890, +2303 and +3386 points, a spread the size of the excess being
    # measured, so the sign of that excess could be decided by the draw.
    # The average over several seeds, with its own standard error, is what
    # the rule has to beat.
    control_totals: list[float] = []
    control_means: list[float] = []
    control_wins: list[float] = []
    for seed in range(args.seed, args.seed + args.control_seeds):
        rng = np.random.default_rng(seed)
        y_ctrl = permuted_returns(y, b, block_ids, rng)
        price_ctrl = rebuild_price(y_ctrl, mids, block_ids)
        reading_ctrl = candidate(y=y_ctrl, b=b, price=price_ctrl)
        run = simulate(reading_ctrl, y_ctrl, block_ids, level,
                       price=price_ctrl, contrarian=not args.momentum,
                       exit_rule=exit_rule, max_hold=args.max_hold,
                       target_points=args.target, stop_points=args.stop,
                       trail_points=args.trail, trail_arm=args.trail_arm,
                       trail_after_signal=args.trail_after_signal,
                       trail_start=args.trail_start, trail_min=args.trail_min,
                       trail_tau=args.trail_tau, hold_after_arm=args.hold_after_arm,
                       exit_level=args.exit_z, entry_mask=entry_mask)
        stats_ctrl = describe(run)
        if "mean" not in stats_ctrl:
            continue
        control_means.append(stats_ctrl["mean"])
        control_wins.append(stats_ctrl["win"])
        control_totals.append(float((run["side"] * run["points"]).sum()))

    control_mean = float(np.mean(control_means)) if control_means else np.nan
    control_se = (float(np.std(control_means, ddof=1) / np.sqrt(len(control_means)))
                  if len(control_means) > 1 else float("nan"))

    sessions = len(np.unique(block_ids))
    span_days = (stamps[-1] - stamps[0]).days
    n = len(trades["ret"])
    exposure = float(trades["bars"].sum()) / len(y) * 100.0

    stats = describe(trades)
    side = trades["side"]
    signed_points = side * trades["points"]

    print("=" * 96)
    print(f"model      : {model_path.name}")
    window_val = args.trend_window if args.oscillator == "trend_deviation" else args.window
    print(f"rule       : {args.oscillator}, window {window_val}, "
          f"q {args.quantile:.2f} (level {level:.3f}), "
          f"{'momentum' if args.momentum else 'contrarian'}, "
          f"exit '{exit_rule}', max hold {args.max_hold}"
          + (f", target {args.target}" if args.target else "")
          + (f", stop {args.stop}" if args.stop else "")
          + (f", trail {args.trail}" if args.trail else ""))
    print(f"data       : {len(y)} bars, {sessions} sessions, "
          f"{stamps[0].date()} .. {stamps[-1].date()} ({span_days} days)")
    print("=" * 96)

    print("\nhow often")
    print(f"  trades                {n:>10}")
    print(f"  per session           {n / sessions:>10.1f}")
    print(f"  per week (5 sessions) {n / sessions * 5:>10.1f}")
    print(f"  mean hold             {stats['bars']:>10.1f} bars "
          f"({stats['bars'] * 5:.0f} min)")
    print(f"  longest hold          {trades['bars'].max():>10} bars")
    print(f"  time in a position    {exposure:>10.1f} %")
    reasons = Counter(trades["reason"])
    print("  exits                 " + ", ".join(
        f"{k} {v}" for k, v in reasons.most_common()))

    print("\nresult, in index points")
    print(f"  {'':22}{'rule':>12}")
    print(f"  {'mean per trade':22}{signed_points.mean():>12.3f}")
    print(f"  {'total':22}{signed_points.sum():>12.1f}")
    print(f"  {'win rate %':22}{stats['win']:>12.1f}")
    print(f"  {'mean bp':22}{stats['mean']:>12.3f}")
    excess = stats["mean"] - control_mean
    se_excess = float(np.hypot(stats["se"], control_se))
    print(f"  {'control mean bp':22}{control_mean:>12.3f}"
          f"   se {control_se:.3f} over {len(control_means)} seeds")
    print(f"  {'control total':22}{np.mean(control_totals):>12.1f}"
          f"   spread {np.min(control_totals):.0f} .. "
          f"{np.max(control_totals):.0f}")
    print(f"  {'excess bp':22}{excess:>12.3f}"
          f"   se {se_excess:.3f}   t {excess / se_excess:+.2f}")

    # Month by month, so a result carried by one window is visible.
    print("\nby month, in index points")
    entry_times = stamps[trades["entry"]]
    monthly = pd.DataFrame({
        "month": entry_times.to_period("M").astype(str),
        "points": signed_points,
        "bp": trades["ret"] * 10000.0,
    })
    print(f"  {'month':<10}{'trades':>8}{'points':>10}{'mean bp':>10}{'win %':>8}")
    for month, group in monthly.groupby("month"):
        print(f"  {month:<10}{len(group):>8}{group['points'].sum():>10.1f}"
              f"{group['bp'].mean():>10.2f}"
              f"{100.0 * (group['bp'] > 0).mean():>8.1f}")

    print("\nby split, in index points")
    for name, mask in splits.items():
        chosen = mask[trades["entry"]]
        if chosen.sum() < 5:
            continue
        part = describe(trades, mask=chosen)
        print(f"  {name:<8}{part['n']:>8} trades"
              f"{signed_points[chosen].sum():>12.1f} points"
              f"{part['mean']:>10.2f} bp{part['win']:>8.1f} %")

    # Anatomy of a trade, split by outcome. A low win rate is the price of
    # letting winners run and says nothing on its own; what matters is
    # whether the losers ever showed a profit.
    #
    #   losers that never went our way      -> the entry is wrong
    #   losers that were well in profit     -> the exit is late
    #   winners giving back most of the MFE -> the exit is late
    final = signed_points
    mfe_points = trades["mfe"] * trades["entry_price"]
    mae_points = trades["mae"] * trades["entry_price"]
    won = final > 0
    print("\nanatomy, in index points")
    print(f"  {'':20}{'n':>7}{'mean':>9}{'MFE med':>10}{'MAE med':>10}"
          f"{'bars':>7}")
    for label, mask in (("winners", won), ("losers", ~won)):
        if mask.sum() < 5:
            continue
        print(f"  {label:<20}{int(mask.sum()):>7}"
              f"{final[mask].mean():>9.1f}"
              f"{np.median(mfe_points[mask]):>10.1f}"
              f"{np.median(mae_points[mask]):>10.1f}"
              f"{trades['bars'][mask].mean():>7.1f}")

    for level in (10.0, 20.0, 40.0):
        share = 100.0 * np.mean(mfe_points[~won] >= level)
        print(f"  losers that were ever {level:>3.0f}+ points ahead: "
              f"{share:5.1f}%")
    given_back = np.median(mfe_points[won] - final[won])
    print(f"  median points a winner reached and handed back: "
          f"{given_back:.1f}")
    print(f"  median points a loser gave up beyond its MAE at exit: "
          f"{np.median(final[~won] - mae_points[~won]):.1f}")

    if args.trail_after_signal:
        armed = trades["armed"]
        extra = trades["after_arm"]
        print("\ntrailing stop after the exit signal")
        print(f"  armed            {armed.sum()} of {len(armed)} trades "
              f"({100.0 * armed.mean():.1f}%)")
        if armed.any():
            gained = extra[armed]
            print(f"  points after arm total {gained.sum():+.1f}  "
                  f"mean {gained.mean():+.1f}  median {np.median(gained):+.1f}")
            print(f"  held past the signal   {100.0 * (gained > 0).mean():.1f}% of armed")
            top = np.sort(gained)[::-1][:10]
            print("  ten best continuations " +
                  " ".join(f"{v:+.0f}" for v in top))
            positive_sum = gained[gained > 0].sum()
            print(f"  top ten share of total {100.0 * top.sum() / max(positive_sum, 1e-9):.1f}%")
        reasons, counts = np.unique(trades["reason"], return_counts=True)
        print("  exit reasons     " +
              "  ".join(f"{r}={c}" for r, c in zip(reasons, counts)))

    # Capital growth, the way a CEM run reported it: one number saying what
    # the starting balance became. Position size is expressed as leverage
    # rather than lots, so this needs no contract specification: each trade
    # risks the same fraction of running equity, which is what a fixed
    # percentage of margin does.
    trade_return = signed_points / trades["entry_price"]
    print(f"\ncapital, compounded, starting from {args.balance:.2f}")
    print(f"  {'leverage':>9}{'final':>14}{'multiple':>11}"
          f"{'max DD %':>10}{'worst trade %':>15}")
    for lev in (1, 2, 5, 10, 20):
        equity = args.balance
        path = [equity]
        for r in trade_return:
            equity *= (1.0 + lev * float(r))
            if equity <= 0:
                equity = 0.0
                path.append(equity)
                break
            path.append(equity)
        path = np.array(path)
        peak = np.maximum.accumulate(path)
        drawdown = np.where(peak > 0, (peak - path) / peak, 0.0)
        print(f"  {lev:>9}{path[-1]:>14.2f}{path[-1] / args.balance:>11.2f}"
              f"{100.0 * drawdown.max():>10.1f}"
              f"{100.0 * lev * float(trade_return.min()):>15.2f}")

    # Account currency, when the specification allows it.
    try:
        spec = money_layer.ContractSpec.from_mapping(
            cfg.data.symbol, load_instruments()[cfg.data.symbol]
        )
    except (KeyError, money_layer.SpecError) as exc:
        print(f"\nAccount currency: not available. {exc}")
        spec = None

    if spec is not None:
        account = money_layer.run_account(
            signed_points, trades["entry_price"], spec,
            balance=args.balance, lots=args.lots,
        )
        m = account["metrics"]
        print(f"\nresult, in {spec.currency}, {args.lots} lots on "
              f"{args.balance:.2f}")
        print(f"  value per point       {spec.value_per_point:>12.4f} "
              f"{spec.currency} per lot")
        print(f"  margin per {args.lots} lot  "
              f"{float(money_layer.margin(mids[np.isfinite(mids)].mean(), args.lots, spec)):>12.2f}")
        for key in ("trades", "end_balance", "total_profit", "return_pct",
                    "win_rate", "avg_trade", "avg_win", "avg_loss",
                    "profit_factor", "max_drawdown_pct", "max_drawdown_abs",
                    "largest_win", "largest_loss"):
            value = m.get(key)
            if isinstance(value, float):
                print(f"  {key:<22}{value:>12.2f}")
            else:
                print(f"  {key:<22}{value:>12}")
        if account["skipped"]:
            print(f"  skipped for margin    {account['skipped']:>12}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "entry_time": entry_times,
            "entry_bar": trades["entry"],
            "side": side,
            "entry_price": trades["entry_price"],
            "exit_price": trades["exit_price"],
            "points": signed_points,
            "bp": trades["ret"] * 10000.0,
            "bars": trades["bars"],
            "reason": trades["reason"],
        }).to_csv(out, index=False)
        print(f"\nWritten: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())