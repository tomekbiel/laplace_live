"""
Step one: where, if anywhere, does mean reversion exist.

This runs before the oscillator bake-off and can end it. An oscillator is
a way of reading a reverting process; if no bucket of this market reverts,
no reading of it will, and the twelve candidates are twelve ways of
measuring noise.

Three series are fitted, because "mean reversion" names three different
claims and they can disagree:

    increment   y / b, the standardised bar return. A negative phi is a
                bounce: an up bar tends to be followed by a down bar. This
                is the claim a contrarian entry on a single bar depends on.
    level       log price, demeaned inside each session. phi near one is a
                random walk; the Dickey-Fuller statistic tests it. This is
                the claim a fade of a level extreme depends on.
    cusum       the accumulator, which resets and is therefore stationary
                by construction. Its phi measures how fast an accumulated
                excursion is walked back, which is the claim the whole
                event-based plan depends on.

Each is fitted inside every session bucket and every split slice, pooling
the adjacent pairs that are legal: same block, same bucket. A bucket is not
contiguous - the night bucket is three hundred separate stretches - so a
rolling fit would invent a transition from one night's close to the next
night's open and measure that instead.

READ THE EXCESS COLUMN, NOT PHI.

Every one of these three series produces a phi that looks like mean
reversion for reasons that have nothing to do with the market. Demeaning a
random walk inside a window pulls it toward zero and makes the
Dickey-Fuller statistic reject. The accumulator resets at h, so its decay
is mostly the reset. Least squares is biased in finite samples. A control
therefore runs beside every fit, built by permuting the standardised shocks
inside each block: the distribution of shocks, the intraday profile of b
and the block structure all survive, only the ordering is destroyed.
Whatever the control still shows is made by the procedure. The claim about
the market is phi minus the control, and the verdict column is computed
from that difference rather than from phi.

The t statistics are not calibrated. Adjacent AR(1) pairs share a point, so
they are not independent and the standard errors are too small. Rank
buckets by them; do not read them as p-values.

The split columns matter for a reason that is not statistical. The model's
b is in-sample over the training slice, so a reading standardised by it is
tighter there than it will ever be live. A result that holds on train and
dies on test has told you about the fit, not about the market.

Usage:
    python tools/ou_screen.py
    python tools/ou_screen.py --cusum-h 5 --out runs/ou_screen_h5.csv
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

# Dickey-Fuller critical values, constant and no trend, large sample. Kept
# for reference; the verdict is taken from the excess over the control.
DF_CRITICAL = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
BUCKET_ORDER = ("all", "night", "europe", "macro", "cash", "late")

# How many standard errors of the difference count as a real excess. Three
# rather than two, because the pairs overlap and the standard error is
# optimistic.
EXCESS_T = 3.0


# ----------------------------------------------------------------------
# Series
# ----------------------------------------------------------------------

def usable_pairs(block_ids: np.ndarray, bucket: np.ndarray,
                 wanted: str, split: np.ndarray) -> np.ndarray:
    """Adjacent pairs that may be fitted: same block, same bucket, in slice."""
    n = len(block_ids)
    out = np.zeros(n, dtype=bool)
    out[1:] = block_ids[1:] == block_ids[:-1]
    if wanted != "all":
        in_bucket = bucket == wanted
        out[1:] &= in_bucket[1:] & in_bucket[:-1]
    out &= split
    return out


def demean_by_block(x: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
    """Subtract each block's own mean, so levels can be pooled across days.

    Without this, a year of index prices pooled into one regression
    measures the drift from November to October and calls it a trend
    toward equilibrium. With it, a random walk is pulled toward its own
    window mean and the Dickey-Fuller statistic rejects on a series that
    has no equilibrium at all, which is why the level row is only readable
    against its control.
    """
    out = np.full(len(x), np.nan)
    for block in np.unique(block_ids):
        mask = block_ids == block
        values = x[mask]
        if np.isfinite(values).sum() < 2:
            continue
        out[mask] = values - np.nanmean(values)
    return out


def permute_within_blocks(x: np.ndarray, block_ids: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Shuffle values inside each block, keeping the blocks themselves."""
    out = x.copy()
    for block in np.unique(block_ids):
        mask = block_ids == block
        values = out[mask]
        finite = np.isfinite(values)
        if finite.sum() > 1:
            drawn = values[finite]
            rng.shuffle(drawn)
            values[finite] = drawn
            out[mask] = values
    return out


def build_series(y: np.ndarray, b: np.ndarray, mids: np.ndarray,
                 block_ids: np.ndarray, cusum_h: float) -> dict[str, np.ndarray]:
    """The three objects whose reversion is being tested."""
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = mids > 0
        log_price = np.where(positive, np.log(np.where(positive, mids, np.nan)),
                             np.nan)
    return {
        "increment": osc._normalised(y, b),
        "level": demean_by_block(log_price, block_ids),
        "cusum": osc.cusum(y, b, h=cusum_h),
    }


def control_series(y: np.ndarray, b: np.ndarray, mids: np.ndarray,
                   block_ids: np.ndarray, cusum_h: float,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The same three series, built from shocks whose order was destroyed.

    The permutation happens on the STANDARDISED shock, y / b, and the
    returns are rebuilt as u_shuffled * b. Permuting the raw return and
    dividing by the b that happens to sit at the new position is a
    different experiment: it scrambles the match between a shock and the
    volatility it occurred in, and does so hardest where b varies most
    inside a bucket - the cash session and the macro window.

    Everything except the ordering survives: the distribution of shocks,
    the intraday profile of b, and the block structure.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        shock = osc._normalised(y, b)
    shuffled = permute_within_blocks(shock, block_ids, rng)
    rebuilt_returns = shuffled * b

    series = build_series(rebuilt_returns, b, mids, block_ids, cusum_h)

    # The level has to be rebuilt from the permuted returns; the original
    # price path would otherwise walk back in unchanged.
    path = np.full(len(y), np.nan)
    for block in np.unique(block_ids):
        mask = block_ids == block
        path[mask] = np.cumsum(np.nan_to_num(rebuilt_returns[mask]))
    series["level"] = demean_by_block(path, block_ids)
    return series


# ----------------------------------------------------------------------
# Reading the fit
# ----------------------------------------------------------------------

def verdict(name: str, fit: dict, control: dict) -> str:
    """What the fit says once its own control is subtracted.

    The control has no serial dependence, so whatever it shows is produced
    by the procedure: the finite-sample bias of least squares, the
    demeaning of a unit-root series, the reset of the accumulator. Only
    the excess over it is a claim about the market.
    """
    if "phi" not in fit or "phi" not in control:
        return "too few"
    excess = fit["phi"] - control["phi"]
    spread = fit["se"] * np.sqrt(2.0)
    t = excess / spread if spread > 0 else float("nan")
    if not np.isfinite(t) or abs(t) < EXCESS_T:
        return "as control"
    if name == "increment":
        return "bounce" if excess < 0 else "momentum"
    return "faster return" if excess < 0 else "more persistent"


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Does this market revert, and where."
    )
    parser.add_argument("--model", default=None, help="a .keras file")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=100000)
    parser.add_argument("--cusum-h", type=float, default=3.0,
                        help="reset threshold for the accumulator, in b units")
    parser.add_argument("--broker-offset", type=float, default=2.0,
                        help="the broker's fixed UTC offset, hours")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="write the table as CSV")
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

    bucket = ev.session_labels(ts, args.broker_offset)
    splits = ev.split_masks(len(y))
    splits["all"] = np.ones(len(y), dtype=bool)

    rng = np.random.default_rng(args.seed)
    series = build_series(y, b, mids, block_ids, args.cusum_h)
    control = control_series(y, b, mids, block_ids, args.cusum_h, rng)

    print("=" * 112)
    print(f"model    : {model_path.name}")
    print(f"bars     : {len(y)}  ({ts[0]} .. {ts[-1]})")
    print(f"cusum h  : {args.cusum_h} (in units of b)")
    print(f"clock    : broker UTC+{args.broker_offset:g}, buckets on US Eastern")
    print(f"control  : standardised shocks permuted within each block, seed "
          f"{args.seed}")
    print("Dickey-Fuller critical values: "
          + ", ".join(f"{k} {v}" for k, v in DF_CRITICAL.items()))
    print("=" * 112)

    rows: list[dict] = []
    for name, values in series.items():
        print(f"\n{name}")
        print(f"  {'bucket':<8}{'split':<7}{'pairs':>8}{'phi':>9}{'se':>8}"
              f"{'t_unit':>9}{'half-life':>11}{'ctrl phi':>10}"
              f"{'excess':>9}{'t_exc':>8}  verdict")
        print("  " + "-" * 102)
        for wanted in BUCKET_ORDER:
            for split_name in ("all", "train", "val", "test"):
                mask = usable_pairs(block_ids, bucket, wanted, splits[split_name])
                fit = osc.pooled_ar1(values, mask)
                if "phi" not in fit:
                    continue
                ctrl = osc.pooled_ar1(control[name], mask)
                control_phi = ctrl.get("phi", float("nan"))
                excess = fit["phi"] - control_phi
                spread = fit["se"] * np.sqrt(2.0)
                t_excess = excess / spread if spread > 0 else float("nan")

                half = fit.get("half_life", float("nan"))
                half_text = f"{half:.1f} bar" if np.isfinite(half) else "-"

                row = {
                    "series": name, "bucket": wanted, "split": split_name,
                    "pairs": fit["n"], "phi": fit["phi"], "se": fit["se"],
                    "t_zero": fit["t_zero"], "t_unit": fit["t_unit"],
                    "half_life_bars": half,
                    "half_life_min": half * 5 if np.isfinite(half) else np.nan,
                    "control_phi": control_phi,
                    "excess": excess,
                    "t_excess": t_excess,
                    "bias": fit["bias"],
                    "cusum_h": args.cusum_h,
                    "verdict": verdict(name, fit, ctrl),
                }
                rows.append(row)
                print(f"  {wanted:<8}{split_name:<7}{fit['n']:>8}"
                      f"{fit['phi']:>9.4f}{fit['se']:>8.4f}"
                      f"{fit['t_unit']:>9.2f}{half_text:>11}"
                      f"{control_phi:>10.4f}{excess:>+9.4f}{t_excess:>+8.1f}"
                      f"  {row['verdict']}")

    if args.out and rows:
        import csv

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWritten: {out}")

    print("\nHow to use this. Read the excess column; phi on its own is "
          "mostly procedure. A bucket whose cusum excess is negative gives "
          "back accumulated moves faster than chance and is where a "
          "contrarian entry belongs; a positive excess means moves are "
          "carried further than chance and belongs to a continuation "
          "entry. Run the same screen at several --cusum-h values before "
          "trusting the half life: if it scales with h it is the reset "
          "being measured, not the market.")
    return 0


if __name__ == "__main__":
    sys.exit(main())