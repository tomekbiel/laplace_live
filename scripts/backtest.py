"""
The rule engine over stored data.

This is the reference run. It drives the same Engine the live loop drives,
so any difference between the two comes from the market or from execution,
never from a second copy of the rules that drifted.

Two sources, because they answer different questions:

    --split <file>   the research project's *_split.npz. Reproduces the
                     original backtest's test set exactly, which is the only
                     way to check the figures against what was published.
    (default)        the bridge's stored candles, through the same reader,
                     feature builder and window builder the live loop uses.
                     This is the honest rehearsal: if it disagrees with the
                     split run, the disagreement is in the data path and is
                     worth finding before any money moves.

Metrics are computed the way calculate_metrics does in 8_backtest.py,
including the leverage multiplication applied after the drawdown ratio is
taken. That is why a drawdown here can exceed one hundred per cent: it is a
ratio of equity, scaled afterwards, not a fraction of an account.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --split C:/.../US.100+5_split.npz
    python scripts/backtest.py --out runs/backtest_reference
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from laplace import features as feature_builder
from laplace import model as model_io
from laplace import reader, windows
from laplace.config import load_config, setup_logging
from laplace.journal import Journal
from laplace.rules import Bar, Engine

LEVERAGE = 20
PERIODS_PER_DAY = 78


def metrics(trades, equity, leverage: int = LEVERAGE) -> dict:
    """Performance figures, transcribed from 8_backtest.calculate_metrics."""
    if not trades:
        return {"total_trades": 0, "total_return": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_return": 0.0, "avg_duration": 0.0,
                "long_trades": 0, "short_trades": 0}

    returns = [t["return"] for t in trades]
    profits = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    total_return = float(np.sum(returns)) * leverage * 100
    win_rate = len(profits) / len(trades) * 100
    profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float("inf")

    if len(returns) > 1:
        daily = np.array(returns) * leverage / PERIODS_PER_DAY
        sharpe = (float(np.mean(daily) / np.std(daily) * np.sqrt(252))
                  if np.std(daily) > 0 else 0.0)
    else:
        sharpe = 0.0

    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity) if equity.size else np.array([0.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - equity) / np.abs(peak), 0)
    max_drawdown = float(np.nanmax(drawdown)) * 100 * leverage if equity.size else 0.0

    return {
        "total_trades": len(trades),
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": float(np.mean(returns)) * 10000,
        "avg_duration": float(np.mean([t["duration"] for t in trades])) * 5,
        "long_trades": sum(1 for t in trades if t["position"] == 1),
        "short_trades": sum(1 for t in trades if t["position"] == -1),
    }


def from_split(path: Path):
    """Load windows, targets and timestamps from a research split file."""
    data = np.load(path, allow_pickle=True)
    X = data["X_test"]
    y = data["y_test"]
    ts = data["timestamps_test"]
    mids = np.zeros(len(y))  # the split carries no prices
    return X, y, ts, mids


def from_bridge(cfg, tail: int):
    """Build the same arrays from the bridge's stored candles."""
    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=tail
    )
    prepared = feature_builder.prepare(
        frame,
        window_size=cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )
    X, y, ts, mids, _, _ = windows.create_windows(
        prepared,
        lookback=cfg.windows.lookback,
        predict_step=cfg.windows.predict_step,
        zscore_clip=cfg.windows.zscore_clip,
    )
    return X, y, ts, mids


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the rules over stored data.")
    parser.add_argument("--split", default=None,
                        help="a research *_split.npz, to reproduce the original figures")
    parser.add_argument("--tail", type=int, default=100000,
                        help="rows to read from the bridge series, default 100000")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the reverse-exit draw, which is random")
    parser.add_argument("--signed-exits", dest="signed_exits",
                        action="store_true", default=None,
                        help="repair the sign defects in the exit branches")
    parser.add_argument("--original-exits", dest="signed_exits",
                        action="store_false",
                        help="keep the exit branches exactly as 8_backtest wrote them")
    parser.add_argument("--no-entry-bar", dest="book_entry_bar",
                        action="store_false", default=None,
                        help="start the trade at the entry bar's close")
    parser.add_argument("--entry-bar", dest="book_entry_bar",
                        action="store_true",
                        help="include the triggering bar, as 8_backtest does")
    parser.add_argument("--out", default=None, help="journal directory name")
    parser.add_argument("--config", default=None, help="path to strategy.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    problems = cfg.check()
    if args.split:
        problems = [p for p in problems if "candle series" not in p]
    for problem in problems:
        print(f"  config: {problem}")
    if problems:
        print("\nFix the above before running.")
        return 1

    np.random.seed(args.seed)
    params = cfg.strategy.load()

    signed = cfg.rules.signed_exits if args.signed_exits is None else args.signed_exits
    entry_bar = (cfg.rules.book_entry_bar if args.book_entry_bar is None
                 else args.book_entry_bar)
    label = f"{'signed' if signed else 'orig'}-{'entrybar' if entry_bar else 'noentrybar'}"
    model = model_io.load_model(cfg.experiment.resolved_model_path)

    if args.split:
        X, y, ts, mids = from_split(Path(args.split))
        source = args.split
    else:
        X, y, ts, mids = from_bridge(cfg, args.tail)
        source = str(cfg.data.series_path)

    if len(X) == 0:
        print("No windows to run over.")
        return 1

    print(f"source : {source}")
    print(f"exits  : {label}")
    print(f"windows: {len(X)}  ({ts[0]} .. {ts[-1]})")

    mu, b = model_io.predict_batch(model, X)

    engine = Engine(
        params=params,
        gap_threshold_min=2 * 5,
        signed_exits=signed,
        book_entry_bar=entry_bar,
    )
    journal = Journal(
        cfg.journal.runs_dir, f"{cfg.experiment.run_name}_{label}", "backtest"
    )
    journal.write_manifest({
        "model": str(cfg.experiment.resolved_model_path),
        "model_parameters": model_io.describe_parameters(model),
        "cem_params": params,
        "source": source,
        "windows": int(len(X)),
        "seed": args.seed,
        "rules": {"signed_exits": signed, "book_entry_bar": entry_bar},
    })

    for i in range(len(X)):
        bar_time = ts[i]
        if not isinstance(bar_time, datetime):
            bar_time = np.datetime64(bar_time, "s").astype("datetime64[s]").astype(datetime)

        bar = Bar(
            time=bar_time,
            y=float(y[i]),
            mu=float(mu[i]),
            b=float(b[i]),
            price=float(mids[i]),
            window=X[i],
        )
        result = engine.observe(bar)

        journal.decision({
            "bar_time": bar.time, "price": bar.price,
            "mu": bar.mu, "b": bar.b, "signal_z": result.signal_z,
            "k1_lower": bar.mu - bar.b, "k1_upper": bar.mu + bar.b,
            "k3_lower": bar.mu - 3 * bar.b, "k3_upper": bar.mu + 3 * bar.b,
            "long_trigger": result.long_trigger, "short_trigger": result.short_trigger,
            "action": result.action, "reason": result.reason,
            "position": result.position, "cum_return": result.cum_return,
            "equity": result.equity, "blocked": result.blocked,
        })
        if result.trade is not None:
            journal.trade(result.trade)

    for value in engine.equity_curve:
        journal.equity("", value)

    stats = metrics(engine.trades, engine.equity_curve)
    print("\n" + "=" * 52)
    for name, value in stats.items():
        print(f"  {name:<16}{value:>14.2f}" if isinstance(value, float)
              else f"  {name:<16}{value:>14}")
    print("=" * 52)

    reasons: dict[str, int] = {}
    for trade in engine.trades:
        reasons[trade["reason"]] = reasons.get(trade["reason"], 0) + 1
    if reasons:
        print("\nexit reasons:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<16}{count:>6}")

    journal.summarise({"metrics": stats, "exit_reasons": reasons})
    journal.close()
    print(f"\nJournal: {journal.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
