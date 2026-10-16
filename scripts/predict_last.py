"""
One prediction from stored candles.

The smoke test: read what the bridge has written, build the features, cut the
newest window, load the model, and print what comes out. Nothing is traded
and nothing is written. If the numbers here have the right order of
magnitude, the bridge, the features, the windows and the model are holding
together, and the loop is worth starting.

What to look at:

* b should sit around the size of a typical five-minute return for the
  instrument. For US.100 that is somewhere in 1e-4 to 1e-3. A b of 1e-9 means
  the scale collapsed; a b near 1 means the window is being read wrongly.
* The k=1 tunnel covers about 63 per cent of outcomes under a Laplace, so
  roughly two bars in three should land inside it on the recent history this
  prints.
* The trigger prices say where the entry rule would fire during the next bar.

Usage:
    python scripts/predict_last.py
    python scripts/predict_last.py --bars 20
    python scripts/predict_last.py --config config/strategy.json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict from the newest stored window.")
    parser.add_argument("--bars", type=int, default=10,
                        help="how many recent bars to score for context, default 10")
    parser.add_argument("--tail", type=int, default=4000,
                        help="rows to read from the end of the series, default 4000")
    parser.add_argument("--config", default=None, help="path to strategy.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    problems = cfg.check()
    blocking = [p for p in problems if "CEM" not in p]
    for problem in problems:
        print(f"  config: {problem}")
    if blocking:
        print("\nFix the above before predicting.")
        return 1

    print(cfg.describe())

    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=args.tail
    )
    prepared = feature_builder.prepare(
        frame,
        window_size=cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )

    print(f"\nprepared rows : {prepared.height}")
    print(f"range         : {prepared['timestamp'][0]} .. {prepared['timestamp'][-1]}")
    print(f"closed-market : {int(prepared['market_closed'].sum())} row(s)")

    model = model_io.load_model(cfg.experiment.resolved_model_path)
    params = model_io.describe_parameters(model)
    if params:
        print(f"activation    : {params}")

    # Score recent complete windows, which gives the context to judge the
    # single newest one against.
    X, y, ts, mids, _, _ = windows.create_windows(
        prepared,
        lookback=cfg.windows.lookback,
        predict_step=cfg.windows.predict_step,
        zscore_clip=cfg.windows.zscore_clip,
    )
    if len(X) == 0:
        print("\nNo complete window could be built.")
        return 1

    n = min(args.bars, len(X))
    mu, b = model_io.predict_batch(model, X[-n:])

    print(f"\nlast {n} scored bars")
    print(f"{'time':<18}{'price':>11}{'y':>12}{'mu':>12}{'b':>12}{'|y-mu|/b':>10}  in k1")
    for i in range(n):
        z = abs(y[-n + i] - mu[i]) / b[i]
        print(f"{str(ts[-n + i])[:16]:<18}{mids[-n + i]:>11.2f}{y[-n + i]:>12.6f}"
              f"{mu[i]:>12.6f}{b[i]:>12.6f}{z:>10.2f}  {'yes' if z <= 1 else 'no'}")

    inside = np.mean(np.abs(y[-n:] - mu) / b <= 1) * 100
    print(f"\ninside the k=1 tunnel: {inside:.0f}% of these bars (Laplace theory: 63%)")

    # The forward view: no target yet, this is the bar now forming.
    try:
        ahead = windows.last_window(
            prepared, lookback=cfg.windows.lookback, zscore_clip=cfg.windows.zscore_clip
        )
    except windows.WindowError as exc:
        print(f"\nNo forward window: {exc}")
        return 0

    forward = model_io.predict_one(model, ahead.x)
    print(f"\nforward prediction, for the bar after {ahead.time}")
    print(f"  session so far : {ahead.block_length} bars")
    print(f"  last price     : {ahead.price:.2f}")
    print(f"  mu             : {forward.mu:.8f}")
    print(f"  b              : {forward.b:.8f}")
    for k in (1, 2, 3):
        low, high = forward.tunnel(k)
        print(f"  k={k} tunnel     : [{low:+.6f}, {high:+.6f}]  "
              f"({ahead.price * np.exp(low):.2f} .. {ahead.price * np.exp(high):.2f})")

    try:
        entry_z = cfg.strategy.load()["entry_z"]
    except Exception as exc:
        print(f"\n  entry triggers unavailable: {exc}")
        return 0

    long_at, short_at = forward.trigger_prices(ahead.price, entry_z)
    print(f"\n  with entry_z = {entry_z:.2f}")
    print(f"  long if price reaches  {long_at:.2f}  (+{long_at - ahead.price:.2f})")
    print(f"  short if price reaches {short_at:.2f}  ({short_at - ahead.price:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
