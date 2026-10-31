"""
Step A: measure the swings before building anything on them.

Three questions, in order, and the second one can end the project's whole
adaptive branch:

    1. What trend window leaves the cycle intact? A regression fitted over
       a window near the period follows the wave and the turns it finds are
       features of the fit rather than of the market. The sweep prints the
       price amplitude against the residual amplitude for several windows;
       they agree only once the window is long enough, and that is the
       window everything else should use.

    2. How long is a swing, how far does it travel, and is any of it
       PREDICTABLE? A median is not a calibration. What matters is whether
       the last few swings forecast the next one better than the long-run
       median does. If they do not, an adaptive window has nothing to adapt
       from and a fixed window is the honest choice.

    3. Does it differ by session and by month, and does any of it survive
       the control?

The control permutes the standardised shocks inside each session and
rebuilds the price path. A random walk has turning points too, with swing
lengths that look cyclical to the eye and follow a known distribution, so
every figure here is printed next to what the same procedure extracts from
a series with no structure at all.

Costs do not appear. This measures the market, not a strategy.

Usage:
    python tools/swing_profile.py --sweep
    python tools/swing_profile.py --trend-window 200 --pivot-k 3
    python tools/swing_profile.py --trend-window 200 --out runs/swings.csv
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
from laplace import reader, swings as sw, windows
from laplace.config import load_config, setup_logging

SWEEP = (40, 80, 120, 200, 300, 400)


# ----------------------------------------------------------------------
# Statistics over a list of swings
# ----------------------------------------------------------------------

def autocorrelation(values: np.ndarray) -> float:
    """Correlation between consecutive entries. Zero means no memory."""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    values = values[good]
    if values.size < 10:
        return float("nan")
    first, second = values[:-1], values[1:]
    if first.std() < 1e-12 or second.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def predictive_skill(values: np.ndarray, memory: int = 5) -> dict:
    """Does the recent median beat the long-run median at forecasting?

    For every swing past the warm-up, three forecasts of its size are
    compared against what it turned out to be:

        local    the median of the previous `memory` swings
        global   the median of everything before it
        last     the previous swing alone

    The skill is one minus the local error over the global error. Above
    zero the recent past carries information the long run does not, which
    is the only condition under which a rolling calibration is worth
    anything. At or below zero, a fixed parameter fitted once on history
    is the better estimator and an adaptive window is self-deception.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < memory + 20:
        return {"n": int(values.size)}

    local_err, global_err, last_err = [], [], []
    for i in range(memory, values.size):
        actual = values[i]
        local_err.append(abs(np.median(values[i - memory:i]) - actual))
        global_err.append(abs(np.median(values[:i]) - actual))
        last_err.append(abs(values[i - 1] - actual))

    local = float(np.mean(local_err))
    overall = float(np.mean(global_err))
    return {
        "n": int(values.size),
        "mae_local": local,
        "mae_global": overall,
        "mae_last": float(np.mean(last_err)),
        "skill": 1.0 - local / overall if overall > 0 else float("nan"),
    }


def summarise(swings: list[sw.Swing], memory: int = 5) -> dict:
    """One row describing a set of swings."""
    if len(swings) < 10:
        return {"swings": len(swings)}

    duration = np.array([s.duration for s in swings], dtype=float)
    amp_price = np.array([abs(s.amplitude) for s in swings], dtype=float)
    amp_series = np.array([abs(s.amplitude_series) for s in swings], dtype=float)
    amp_b = np.array([abs(s.amplitude_b) for s in swings], dtype=float)
    sharp = np.array([s.sharpness for s in swings], dtype=float)
    lag = np.array([s.confirmed - s.end for s in swings], dtype=float)
    slope_t = np.array([abs(s.slope_t) for s in swings], dtype=float)

    duration_skill = predictive_skill(duration, memory)
    amplitude_skill = predictive_skill(amp_series, memory)

    return {
        "swings": len(swings),
        "dur_median": float(np.nanmedian(duration)),
        "dur_iqr": float(np.nanpercentile(duration, 75)
                         - np.nanpercentile(duration, 25)),
        "dur_autocorr": autocorrelation(duration),
        "dur_skill": duration_skill.get("skill", float("nan")),
        "amp_price_median": float(np.nanmedian(amp_price)),
        "amp_series_median": float(np.nanmedian(amp_series)),
        "amp_b_median": float(np.nanmedian(amp_b)),
        "amp_autocorr": autocorrelation(amp_series),
        "amp_skill": amplitude_skill.get("skill", float("nan")),
        "sharp_median": float(np.nanmedian(sharp)),
        "sharp_q90": float(np.nanpercentile(sharp, 90)),
        "lag_median": float(np.nanmedian(lag)),
        "slope_t_median": float(np.nanmedian(slope_t)),
    }


def print_row(label: str, row: dict) -> None:
    if "dur_median" not in row:
        print(f"  {label:<14}{row.get('swings', 0):>7}  too few")
        return
    print(f"  {label:<14}{row['swings']:>7}"
          f"{row['dur_median']:>8.1f}{row['dur_iqr']:>7.1f}"
          f"{row['dur_autocorr']:>+9.3f}{row['dur_skill']:>+8.3f}"
          f"{row['amp_b_median']:>9.2f}{row['amp_autocorr']:>+9.3f}"
          f"{row['amp_skill']:>+8.3f}{row['sharp_median']:>8.2f}"
          f"{row['lag_median']:>7.1f}")


def header() -> None:
    print(f"  {'group':<14}{'swings':>7}{'dur':>8}{'IQR':>7}"
          f"{'dur ac':>9}{'skill':>8}{'amp(b)':>9}{'amp ac':>9}"
          f"{'skill':>8}{'sharp':>8}{'lag':>7}")
    print("  " + "-" * 94)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def permuted_price(y, b, mids, block_ids, rng):
    """A price path from the same shocks in a destroyed order."""
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
    returns = out * b

    price = np.empty(len(y))
    for block in np.unique(block_ids):
        mask = block_ids == block
        start = mids[mask][0]
        start = start if np.isfinite(start) and start > 0 else 1.0
        price[mask] = start * np.exp(np.cumsum(np.nan_to_num(returns[mask])))
    return price


def main() -> int:
    parser = argparse.ArgumentParser(
        description="How long is a swing, how far, and is it predictable."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=100000)
    parser.add_argument("--trend-window", type=int, default=200)
    parser.add_argument("--pivot-k", type=float, default=3.0,
                        help="retracement that confirms a turn, in units of b")
    parser.add_argument("--memory", type=int, default=5,
                        help="swings in the rolling calibration")
    parser.add_argument("--sweep", action="store_true",
                        help="only run the trend-window sweep and stop")
    parser.add_argument("--broker-offset", type=float, default=0.0,
                        help="broker clock offset in hours, for the session "
                             "buckets; the terminal now stamps close to UTC")
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
    stamps = pd.DatetimeIndex(pd.to_datetime(ts))

    rng = np.random.default_rng(args.seed)
    control_price = permuted_price(y, b, mids, block_ids, rng)

    print("=" * 104)
    print(f"model   : {model_path.name}")
    print(f"bars    : {len(y)}  ({stamps[0].date()} .. {stamps[-1].date()})")
    print(f"pivot   : {args.pivot_k} x b of retracement confirms a turn")
    print(f"b median: {np.nanmedian(b) * np.nanmean(mids):.2f} index points "
          f"per bar")
    print("=" * 104)

    # ------------------------------------------------------------------
    # 1. Which trend window leaves the cycle intact
    # ------------------------------------------------------------------
    print("\ntrend window sweep")
    print(f"  {'window':>7}{'swings':>8}{'duration':>10}{'price amp':>11}"
          f"{'resid amp':>11}{'ratio':>8}   verdict")
    print("  " + "-" * 72)
    for window in SWEEP:
        table = sw.swing_table(mids, b, block_ids, trend_window=window,
                               pivot_k=args.pivot_k)
        if len(table) < 10:
            print(f"  {window:>7}{len(table):>8}  too few swings")
            continue
        duration = np.median([s.duration for s in table])
        price_amp = np.median([abs(s.amplitude) for s in table])
        resid_amp = np.median([abs(s.amplitude_series) for s in table])
        ratio = price_amp / resid_amp if resid_amp > 0 else float("nan")
        verdict = ("the fit is eating the cycle" if ratio < 0.6
                   else "usable" if ratio < 1.4 else "little trend removed")
        print(f"  {window:>7}{len(table):>8}{duration:>10.1f}"
              f"{price_amp:>11.1f}{resid_amp:>11.1f}{ratio:>8.2f}   {verdict}")
    print("\n  Use the shortest window whose ratio is near one: the turns are"
          "\n  then features of the market and the move between them is real.")

    if args.sweep:
        return 0

    # ------------------------------------------------------------------
    # 2 and 3. The profile at the chosen window
    # ------------------------------------------------------------------
    table = sw.swing_table(mids, b, block_ids, args.trend_window, args.pivot_k)
    control = sw.swing_table(control_price, b, block_ids,
                             args.trend_window, args.pivot_k)
    if len(table) < 20:
        print(f"\nOnly {len(table)} swings at window {args.trend_window}; "
              "lower --pivot-k or shorten the window.")
        return 1

    print(f"\nprofile at trend window {args.trend_window}")
    print("  dur ac / amp ac: correlation between consecutive swings.")
    print("  skill: 1 - (error of the last-5 median) / (error of the running "
          "median).")
    print("  Above zero the recent past calibrates; at or below zero it does "
          "not.\n")
    header()
    print_row("all", summarise(table, args.memory))
    print_row("CONTROL", summarise(control, args.memory))

    bucket = ev.session_labels(stamps, args.broker_offset)
    print()
    header()
    for name in ("night", "europe", "macro", "cash", "late"):
        chosen = [s for s in table if bucket[s.end] == name]
        print_row(name, summarise(chosen, args.memory))
        shadow = [s for s in control if bucket[s.end] == name]
        print_row(f"  ctrl {name[:6]}", summarise(shadow, args.memory))

    print()
    header()
    months = stamps.to_period("M").astype(str)
    for month in sorted(set(months)):
        chosen = [s for s in table if months[s.end] == month]
        print_row(month, summarise(chosen, args.memory))

    # ------------------------------------------------------------------
    # Sharpness: the slow walk to an extreme against the one-bar jump
    # ------------------------------------------------------------------
    sharp = np.array([s.sharpness for s in table], dtype=float)
    sharp = sharp[np.isfinite(sharp)]
    control_sharp = np.array([s.sharpness for s in control], dtype=float)
    control_sharp = control_sharp[np.isfinite(control_sharp)]
    print("\nsharpness, the largest single bar over the swing's amplitude")
    print(f"  {'quantile':<10}{'market':>10}{'control':>10}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        print(f"  {q:<10.2f}{np.quantile(sharp, q):>10.3f}"
              f"{np.quantile(control_sharp, q):>10.3f}")
    print(f"  share above 0.8: market "
          f"{100.0 * np.mean(sharp > 0.8):.1f}%, control "
          f"{100.0 * np.mean(control_sharp > 0.8):.1f}%")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "start_time": stamps[s.start], "end_time": stamps[s.end],
            "confirmed_time": stamps[s.confirmed],
            "duration": s.duration, "lag": s.confirmed - s.end,
            "amplitude": s.amplitude, "amplitude_series": s.amplitude_series,
            "amplitude_b": s.amplitude_b, "sharpness": s.sharpness,
            "offset_b": s.offset_b, "slope": s.slope, "slope_t": s.slope_t,
            "session": bucket[s.end], "month": months[s.end],
        } for s in table]).to_csv(out, index=False)
        print(f"\nWritten: {out}")

    print("\nWhat decides the next step: the skill columns. Positive on the "
          "market and near zero on the control means the recent swings "
          "carry information and an adaptive window is worth building. "
          "Equal on both, or negative, means a fixed window fitted once is "
          "the better estimator and the adaptive branch closes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())