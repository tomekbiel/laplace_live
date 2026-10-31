"""
One harness for every measurement: events in, forward paths out.

Each study in this project asks the same question of a different trigger.
Does something happen after this moment that did not happen after an
arbitrary moment. Writing that question once, here, is what makes a tunnel
breach and an oscillator crossing comparable; writing it twice guarantees
that two numbers printed in the same table were computed differently.

The shape of a study:

    reading   = an oscillator over the bars            (oscillators.py)
    events    = where that reading crosses a threshold (first_crossing)
    side      = which way the trade faces              (-1 contrarian)
    paths     = cumulative returns after each event    (forward_matrix)
    labels    = which barrier the path hit first       (triple_barrier)
    strata    = split slice and session bucket         (split_masks, sessions)

Three disciplines are built in rather than left to the caller, because each
one was learned from a measurement that had to be thrown away.

Causality. An event at bar i is known at that bar's close and its path
starts at i+1. Nothing here shifts a reading forward, and no statistic
touches a bar the trigger could not have seen.

Independence. Overlapping forward windows share bars, so their outcomes
are correlated and a t-statistic computed over them is inflated. Events
closer together than the horizon are dropped by default, which costs
sample size and buys the right to compute a standard error.

Boundaries. A forward path that would cross a weekend or an overnight
break is dropped rather than stitched, because the price it needs does
not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

# The session buckets, on the US Eastern clock, as minutes of the day.
# 08:30 is its own bucket because the macro releases land there and were
# measured as a separate peak, distinct from the 09:30 cash open.
SESSIONS: tuple[tuple[str, int, int], ...] = (
    ("night", 18 * 60, 3 * 60),        # wraps midnight
    ("europe", 3 * 60, 8 * 60 + 30),
    ("macro", 8 * 60 + 30, 9 * 60 + 30),
    ("cash", 9 * 60 + 30, 16 * 60),
    ("late", 16 * 60, 18 * 60),
)

# Training used a chronological 70/15/15. Repeated here so a measurement
# can be reported on the same slices the model was fitted on, which is the
# only way to tell an in-sample tunnel from an out-of-sample one.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


# ----------------------------------------------------------------------
# Strata
# ----------------------------------------------------------------------

def split_masks(n: int, train: float = TRAIN_FRACTION,
                val: float = VAL_FRACTION) -> dict[str, np.ndarray]:
    """Chronological train/val/test masks, as train.py cuts them."""
    train_end = int(n * train)
    val_end = int(n * (train + val))
    index = np.arange(n)
    return {
        "train": index < train_end,
        "val": (index >= train_end) & (index < val_end),
        "test": index >= val_end,
    }


# Fixed boundaries, chosen once on the data available in October 2026 and
# never recomputed. A fractional split moves every time the file grows:
# two days of new bars shifted val from +627 points to -271 without any
# rule changing, and a test slice that drifts is no longer untouched.
SPLIT_DATES = ("2026-07-01", "2026-08-20")


def split_masks_by_date(timestamps, boundaries=SPLIT_DATES
                        ) -> dict[str, np.ndarray]:
    """Train/val/test cut at fixed calendar dates.

    The same bar belongs to the same slice in every run, forever, which is
    what makes two runs comparable and what keeps the test slice honest.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    first, second = (pd.Timestamp(boundaries[0]), pd.Timestamp(boundaries[1]))
    return {
        "train": np.asarray(stamps < first),
        "val": np.asarray((stamps >= first) & (stamps < second)),
        "test": np.asarray(stamps >= second),
    }


def eastern_minutes(timestamps, broker_offset_hours: float = 2.0) -> np.ndarray:
    """Minute of the day on the New York clock.

    The broker stamps bars in a fixed offset with no daylight saving, so the
    same wall clock time is a different market time in summer and winter.
    Converting to Eastern is what collapses the cash open onto one hour
    instead of smearing it across two. A bar stamped 15:30 by the broker is
    09:30 in New York in summer and 08:30 in winter.
    """
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    if index.tz is None:
        index = index.tz_localize(
            timezone(timedelta(hours=broker_offset_hours))
        )
    eastern = index.tz_convert("America/New_York")
    return eastern.hour.to_numpy() * 60 + eastern.minute.to_numpy()


def session_labels(timestamps, broker_offset_hours: float = 2.0) -> np.ndarray:
    """One session bucket per bar, on the Eastern clock."""
    minutes = eastern_minutes(timestamps, broker_offset_hours)
    out = np.full(len(minutes), "other", dtype=object)
    for name, start, end in SESSIONS:
        if start <= end:
            mask = (minutes >= start) & (minutes < end)
        else:  # the night bucket wraps midnight
            mask = (minutes >= start) | (minutes < end)
        out[mask] = name
    return out


# ----------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------

def first_crossing(reading: np.ndarray, threshold: float,
                   both_sides: bool = True) -> np.ndarray:
    """Bars where the reading first reaches the threshold.

    Only the first bar of an excursion counts. A reading that sits above
    the threshold for six bars is one event, not six, and counting it six
    times is the commonest way an edge is manufactured from nothing.

    For an accumulator that resets at its threshold this is redundant, since
    the reading returns to zero on the triggering bar. It costs nothing and
    makes the two kinds of oscillator interchangeable here.
    """
    x = np.asarray(reading, dtype=float)
    hot = np.abs(x) >= threshold if both_sides else x >= threshold
    hot &= np.isfinite(x)
    previous = np.zeros(len(x), dtype=bool)
    previous[1:] = hot[:-1]
    return hot & ~previous


def space_events(events: np.ndarray, min_gap: int) -> np.ndarray:
    """Thin events so no two are closer together than min_gap bars.

    Two events a bar apart produce forward paths sharing all but one bar.
    Their outcomes are then almost the same number, and averaging them
    pretends to a sample size the data does not contain. Set min_gap to the
    horizon and the surviving paths do not overlap at all.
    """
    if min_gap <= 1:
        return events.copy()
    out = np.zeros(len(events), dtype=bool)
    last = -min_gap
    for i in np.flatnonzero(events):
        if i - last >= min_gap:
            out[i] = True
            last = i
    return out


def threshold_curve(reading: np.ndarray, horizon: int = 12,
                    min_gap: int | None = None,
                    points: int = 60) -> list[tuple[float, int]]:
    """Event count against threshold, over the quantiles of |reading|.

    The curve is not monotone, which is why it is computed rather than
    searched. Lowering the threshold raises the count only while the
    reading still falls back below it between excursions; past that point
    the reading sits above the threshold permanently, the re-arming stops,
    and the count collapses toward one. A bisection assumes the monotone
    shape and returns the collapsed end.
    """
    x = np.asarray(reading, dtype=float)
    magnitude = np.abs(x[np.isfinite(x)])
    if magnitude.size == 0:
        return []
    gap = horizon if min_gap is None else min_gap
    quantiles = np.linspace(0.50, 0.9995, points)
    curve = []
    for level in np.unique(np.quantile(magnitude, quantiles)):
        events = space_events(first_crossing(x, level), gap)
        curve.append((float(level), int(events.sum())))
    return curve


def calibrate_threshold(reading: np.ndarray, target_events: int,
                        block_ids: np.ndarray | None = None,
                        horizon: int = 12, min_gap: int | None = None,
                        points: int = 60) -> float:
    """The threshold whose event count comes closest to the target.

    Every candidate is given the same event count before any return is
    looked at, because a candidate firing two hundred times and one firing
    eight thousand times differ in standard error by a factor of six, and
    that difference would otherwise be read as skill.

    Ties are broken toward the higher threshold, which keeps the events
    rarer and more clearly separated from ordinary bars.

    The spacing rule caps what any threshold can deliver at
    len(reading) / min_gap, so a target above that ceiling silently returns
    the ceiling. Check the achieved count in the results rather than
    assuming the target was met.

    The search is on the count alone. No forward return enters it, so this
    is calibration rather than selection.
    """
    curve = threshold_curve(reading, horizon, min_gap, points)
    if not curve:
        return float("nan")
    best = min(curve, key=lambda item: (abs(item[1] - target_events), -item[0]))
    return best[0]


# ----------------------------------------------------------------------
# Forward paths
# ----------------------------------------------------------------------

def forward_matrix(y: np.ndarray, block_ids: np.ndarray,
                   horizon: int) -> np.ndarray:
    """Cumulative forward returns, NaN where the path leaves its block.

    Row i holds the cumulative return of bars i+1 .. i+horizon, so nothing
    in row i was known at bar i.
    """
    y = np.asarray(y, dtype=float)
    block_ids = np.asarray(block_ids)
    n = len(y)
    paths = np.full((n, horizon), np.nan)
    base = np.arange(n)
    for step in range(1, horizon + 1):
        idx = base + step
        inside = idx < n
        same = np.zeros(n, dtype=bool)
        same[inside] = block_ids[idx[inside]] == block_ids[inside]
        previous = paths[:, step - 2] if step > 1 else np.zeros(n)
        contribution = np.full(n, np.nan)
        contribution[same] = y[idx[same]]
        paths[:, step - 1] = previous + contribution
    return paths


def triple_barrier(signed_paths: np.ndarray, target: float,
                   stop: float) -> tuple[np.ndarray, np.ndarray]:
    """Which barrier the position hit first, and when.

    Returns a label per row and the step at which it was decided:

        +1  the target was reached
        -1  the stop was reached
         0  neither, the horizon ran out

    The barriers are checked at bar closes, which is all this data holds.
    A bar whose path touched both inside its own range is recorded by
    whichever its close reached, and the ordering inside that bar is not
    observable here. That understates stops, since a position is stopped
    out intrabar in practice, and the understatement is the price of
    close-only candles rather than an assumption that can be relaxed.

    Both thresholds are positive numbers in return units; the stop is
    subtracted internally.
    """
    rows, steps = signed_paths.shape
    labels = np.zeros(rows, dtype=int)
    when = np.full(rows, steps, dtype=int)
    for i in range(rows):
        path = signed_paths[i]
        for step in range(steps):
            value = path[step]
            if not np.isfinite(value):
                when[i] = step
                break
            if value >= target:
                labels[i] = 1
                when[i] = step + 1
                break
            if value <= -stop:
                labels[i] = -1
                when[i] = step + 1
                break
    return labels, when


# ----------------------------------------------------------------------
# A study
# ----------------------------------------------------------------------

@dataclass
class Study:
    """One trigger, measured. The object a bake-off collects.

    Attributes:
        name: What produced the events.
        threshold: The calibrated level.
        events: Boolean mask over bars.
        side: +1 or -1 per event bar; -sign(reading) is contrarian.
        signed: Forward paths already multiplied by side.
        meta: Anything the caller wants printed beside the figures.
    """

    name: str
    threshold: float
    events: np.ndarray
    side: np.ndarray
    signed: np.ndarray
    meta: dict = field(default_factory=dict)

    def subset(self, mask: np.ndarray) -> np.ndarray:
        """Signed paths for the events inside a stratum."""
        selected = self.events & mask & np.isfinite(self.signed[:, -1])
        return self.signed[selected]


def build_study(name: str, reading: np.ndarray, y: np.ndarray,
                block_ids: np.ndarray, threshold: float,
                horizon: int = 12, contrarian: bool = True,
                min_gap: int | None = None, meta: dict | None = None) -> Study:
    """Turn a reading into events, sides and forward paths."""
    events = space_events(
        first_crossing(reading, threshold),
        horizon if min_gap is None else min_gap,
    )
    direction = -np.sign(reading) if contrarian else np.sign(reading)
    direction = np.where(np.isfinite(direction), direction, 0.0)
    paths = forward_matrix(y, block_ids, horizon)
    return Study(
        name=name,
        threshold=float(threshold),
        events=events,
        side=direction,
        signed=paths * direction[:, None],
        meta=dict(meta or {}),
    )


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def to_bp(x) -> float:
    """Log return to basis points."""
    return float(x) * 10000.0


def summarise(signed: np.ndarray) -> dict:
    """Outcome statistics for a set of signed forward paths.

    Reports a standard error and a t-statistic alongside the mean, because
    a mean of eight basis points over forty events and the same mean over
    four thousand are different claims, and a table that prints only the
    mean invites them to be read as the same one.
    """
    if signed.size == 0:
        return {}
    final = signed[:, -1]
    good = np.isfinite(final)
    if good.sum() < 2:
        return {}

    run = signed[good]
    final = final[good]
    mfe = np.nanmax(run, axis=1)
    mae = np.nanmin(run, axis=1)
    won = final > 0
    n = int(good.sum())

    mean_bp = to_bp(np.mean(final))
    se_bp = to_bp(np.std(final, ddof=1) / np.sqrt(n))

    return {
        "n": n,
        "win_rate": 100.0 * float(won.mean()),
        "mean_bp": mean_bp,
        "median_bp": to_bp(np.median(final)),
        "se_bp": se_bp,
        "t_stat": mean_bp / se_bp if se_bp > 0 else float("nan"),
        "mfe_median_bp": to_bp(np.median(mfe)),
        "mfe_q75_bp": to_bp(np.quantile(mfe, 0.75)),
        "mae_median_bp": to_bp(np.median(mae)),
        "mae_q10_bp": to_bp(np.quantile(mae, 0.10)),
        # A stop tighter than this removes the trades that eventually paid.
        "mae_winners_q10_bp": (to_bp(np.quantile(mae[won], 0.10))
                               if won.sum() else float("nan")),
        "edge_ratio": (float(np.median(mfe) / abs(np.median(mae)))
                       if np.median(mae) != 0 else float("nan")),
    }


def opposite_share(reading: np.ndarray, paths: np.ndarray,
                   events: np.ndarray) -> float:
    """Share of next bars moving against the trigger. The 50% test."""
    selected = events & np.isfinite(paths[:, 0]) & np.isfinite(reading)
    if selected.sum() == 0:
        return float("nan")
    trigger = np.sign(reading[selected])
    nxt = np.sign(paths[selected, 0])
    return 100.0 * float((trigger != nxt).mean())


def horizon_profile(signed: np.ndarray, horizons: tuple[int, ...]) -> list[dict]:
    """Mean and win rate at each horizon, to find where an effect stops."""
    rows = []
    for h in horizons:
        if h > signed.shape[1]:
            continue
        column = signed[:, h - 1]
        good = np.isfinite(column)
        if good.sum() < 20:
            continue
        values = column[good]
        n = int(good.sum())
        mean_bp = to_bp(np.mean(values))
        se_bp = to_bp(np.std(values, ddof=1) / np.sqrt(n))
        rows.append({
            "bars": h,
            "minutes": h * 5,
            "n": n,
            "win_rate": 100.0 * float((values > 0).mean()),
            "mean_bp": mean_bp,
            "se_bp": se_bp,
        })
    return rows


def bonferroni(p_value: float, trials: int) -> float:
    """The p-value after accounting for how many cells were looked at.

    Blunt, and deliberately so. A table of forty configurations whose best
    cell reaches p = 0.03 has found nothing, and the arithmetic that says
    so belongs next to the table rather than in a footnote.
    """
    return float(min(1.0, p_value * max(1, trials)))