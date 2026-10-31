"""
Is b the scale of what actually happens?

Everything in this project that has a threshold divides by b: the pivot
retracement, the amplitude constant, the drift filter, the tunnel. If b is
not the scale of the realised move, every one of those thresholds means
something other than what it says, and the sweeps that tuned them were
tuning the wrong quantity.

Four numbers, and for a Laplace scale fitted to the data they have known
values:

    mean(|y|) / mean(b)     1.00    the defining property of the scale
    sd(y / b)               1.41    a Laplace has variance 2b^2
    sd(drift over n bars)   1.41    independent standardised increments
    P(|y| < b)             63.2%    1 - exp(-1)

Two of these disagreeing is the interesting case, and one disagreement is
already on record: a drift filter at |state| <= 1.0 kept 87% of bars where
1.41 implies 52%, while the tunnel's coverage of 70.8% implies b is only
about a quarter too large. A mass of exact zeros reconciles them, because
bars with no tick repeat the previous close and push the distribution's
centre up and its spread down at the same time. So the zero share is
printed, and every figure is repeated with those bars removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from laplace import events as ev
from laplace import features as feature_builder
from laplace import model as model_io
from laplace import oscillators as osc
from laplace import reader, windows
from laplace.config import load_config, setup_logging


def report(label: str, y: np.ndarray, b: np.ndarray, window: int) -> None:
    """The four numbers, for one subset."""
    good = np.isfinite(y) & np.isfinite(b) & (b > 0)
    y, b = y[good], b[good]
    if y.size < 100:
        print(f"  {label:<22}{y.size:>8}  too few")
        return
    standardised = y / b
    state = osc.drift(y=y, b=b, window=window)
    state = state[np.isfinite(state)]
    print(f"  {label:<22}{y.size:>8}"
          f"{np.mean(np.abs(y)) / np.mean(b):>12.3f}"
          f"{np.std(standardised):>12.3f}"
          f"{np.std(state):>12.3f}"
          f"{100.0 * np.mean(np.abs(y) < b):>11.1f}%"
          f"{100.0 * np.mean(np.abs(state) <= 1.0):>11.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Is b the right scale.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=100000)
    parser.add_argument("--drift-window", type=int, default=24)
    parser.add_argument("--broker-offset", type=float, default=0.0)
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
    stamps = pd.DatetimeIndex(pd.to_datetime(ts))

    zero = np.abs(y) < 1e-12
    print("=" * 96)
    print(f"model : {model_path.name}")
    print(f"bars  : {len(y)}  ({stamps[0].date()} .. {stamps[-1].date()})")
    print(f"exact zero returns: {zero.sum()}  ({100.0 * zero.mean():.2f}%)")
    print("=" * 96)
    print("\nFor a correctly scaled Laplace: ratio 1.00, sd 1.41, drift sd "
          "1.41, coverage 63.2%, keep 52%")
    print(f"\n  {'subset':<22}{'bars':>8}{'|y|/b':>12}{'sd(y/b)':>12}"
          f"{'sd(drift)':>12}{'P(|y|<b)':>12}{'keep@1':>12}")
    print("  " + "-" * 90)

    report("all", y, b, args.drift_window)
    report("non-zero only", y[~zero], b[~zero], args.drift_window)

    splits = ev.split_masks_by_date(ts)
    for name, mask in splits.items():
        report(name, y[mask], b[mask], args.drift_window)

    bucket = ev.session_labels(stamps, args.broker_offset)
    for name in ("night", "europe", "macro", "cash", "late"):
        mask = bucket == name
        share = 100.0 * np.mean(zero[mask]) if mask.sum() else float("nan")
        report(f"{name} (zeros {share:.0f}%)", y[mask], b[mask],
               args.drift_window)

    # The same four numbers without the seasonal factor. The loss the model
    # minimised has its optimum at b = mean|y|, so a converged network
    # cannot produce a ratio of 0.45 on its own; if removing the profile
    # brings the ratio to one, the defect is in how the profile is applied
    # rather than in the weights.
    raw = model.model.predict(X, verbose=0)
    b_raw = np.log1p(np.exp(raw[:, 1])) + model_io.MIN_B
    report("raw b, no profile", y, b_raw, args.drift_window)

    # If the floor added to the softplus is of the same order as the
    # scale being predicted, every b carries it and the network's own
    # output barely matters. A share near the floor says the floor is
    # the model.
    print(f"\nMIN_B = {model_io.MIN_B:.3e}")
    print(f"  b quantiles 1/25/50/75/99: "
          + " ".join(f"{np.quantile(b, q):.3e}"
                     for q in (0.01, 0.25, 0.50, 0.75, 0.99)))
    print(f"  share of b within 1% of MIN_B: "
          f"{100.0 * np.mean(b <= model_io.MIN_B * 1.01):.1f}%")
    print(f"  mean|y| = {np.mean(np.abs(y)):.3e}")

    # The inflation factor is 2.22 and sqrt(5) is 2.236. A scale that
    # belongs to a five-bar move would divide by exactly that against a
    # one-bar move, so the horizon is worth ruling in or out before
    # anything else is blamed.
    print("\nhorizon check: mean|cumulative y over h bars| / mean(b)")
    for h in range(1, 9):
        rolled = pd.Series(y).rolling(h).sum().to_numpy()
        ok = np.isfinite(rolled) & np.isfinite(b)
        print(f"  h = {h}: {np.mean(np.abs(rolled[ok])) / np.mean(b[ok]):.3f}")

    print("\nRead it this way. A ratio near 1 with sd well under 1.41 means "
          "the distribution is more peaked than a Laplace, which a large "
          "zero share produces. A ratio far from 1 means b is simply "
          "mis-scaled and every threshold divided by it is off by that "
          "factor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())