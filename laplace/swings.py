"""
Swings: the trend removed, the turns found, the cycle measured.

An oscillator built on a fixed bar count measures position inside whatever
range those bars happened to cover. It has no notion of a turning point, no
notion of how long a swing lasts, and its threshold is a fraction of a
range that may be twenty points or three hundred. This module supplies what
was missing, and the oscillators keep their shape.

Five quantities per swing, and only the first two feed the signal:

    duration    bars from one turn to the next. The window an adaptive
                oscillator should use, and the time limit a position
                should carry.
    amplitude   price covered, in points and in units of b. The
                denominator that turns a threshold into a distance.
    sharpness   the largest single bar inside the swing over the swing's
                own amplitude. Near zero the market walked to its extreme;
                above one it jumped there in a single bar and turned. These
                are different events and no range oscillator separates them.
    offset      the residual at the turn, in units of b: how far the
                extreme fell from the local trend.
    slope, t    the local trend at the turn, with its own standard error,
                so "there is a trend" is a measurement rather than a guess.

The signal itself is one number, computed in signal_position():

    z = (price - trend) / (half a typical recent swing)

Everything else calibrates it or conditions it.

THE TREND WINDOW MUST BE LONGER THAN THE CYCLE, and by a wide margin. A
regression fitted over a window comparable to the period follows the wave
instead of the drift, and the residual's turning points then land where
price deviates most from the line rather than where price turns. Measured
on a synthetic 60-bar cycle of amplitude 30:

    window   20: duration  37.0   price amplitude 24.9   residual 27.3
    window   40: duration  30.0   price amplitude 11.2   residual 63.6
    window   80: duration  30.0   price amplitude 65.0   residual 72.0
    window  120: duration  30.0   price amplitude 60.3   residual 71.2

At 40 against a period of 60 the measured period is still right while the
price move between turns collapses to a sixth of the truth: the pivots are
real features of the residual and useless as trades. From 80 upward the two
amplitudes agree. Use at least twice the period, which is four half
periods, and re-fit the window once the period has been measured.

Causality. The trend is fitted on a trailing window and read at its last
point, so a residual at bar t uses bars up to t. Pivots are different: a
turn is only confirmed once the series has retraced far enough from it,
which on the same synthetic took a median of 13 bars. Pivots are therefore
a CALIBRATION instrument, never a trigger; every swing carries the bar at
which it became known, and calibrated() writes from that bar alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------------------------------------------------
# Trend
# ----------------------------------------------------------------------

def rolling_trend(price, window: int = 40) -> dict[str, np.ndarray]:
    """Least-squares line over a trailing window, read at its last bar.

    Returns the slope in price units per bar, the fitted value at the
    current bar, the residual, and the slope's t statistic.

    The t statistic is what makes "there is a trend" a measurement. With x
    running 0..n-1 the spread of the regressor is fixed, Sxx = n(n^2-1)/12,
    so the standard error of the slope is the residual standard deviation
    over its square root and nothing has to be refitted to get it.

    Checked against a pure straight line of slope 0.5: recovered to the
    last printed digit.
    """
    p = pd.Series(np.asarray(price, dtype=float))
    n = window
    x_mean = (n - 1) / 2.0
    sxx = n * (n * n - 1) / 12.0

    index = pd.Series(np.arange(len(p), dtype=float))
    mean_y = p.rolling(n, min_periods=n).mean()
    # cov(x, y) over the window, with x = 0..n-1 inside each window: using
    # the global index works because shifting x by a constant does not
    # change the covariance. pandas uses ddof=1, so the matching variance
    # of the regressor is Sxx / (n - 1), not Sxx / n.
    cov = p.rolling(n, min_periods=n).cov(index)
    slope = (cov / (sxx / (n - 1))).to_numpy()

    fitted = (mean_y.to_numpy() + slope * (n - 1 - x_mean))
    residual = p.to_numpy() - fitted

    resid_sd = pd.Series(residual).rolling(n, min_periods=n).std().to_numpy()
    se = np.where(sxx > 0, resid_sd / np.sqrt(sxx), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(np.abs(se) > EPS, slope / se, np.nan)

    return {
        "slope": slope,
        "fitted": fitted,
        "residual": residual,
        "t_stat": t_stat,
    }


# ----------------------------------------------------------------------
# Pivots
# ----------------------------------------------------------------------

def find_pivots(series, threshold, block_ids=None) -> list[dict]:
    """Turning points, confirmed once the series retraces past a threshold.

    A ZigZag. The running extreme in the current direction is carried
    forward; when the series moves back from it by more than the threshold,
    that extreme becomes a pivot and the direction flips.

    The threshold may vary per bar, which is how an amplitude expressed in
    units of b stays meaningful across quiet and busy hours.

    Each pivot carries two bar numbers. `bar` is where the turn happened.
    `confirmed` is where it became known. Only the second exists in real
    time, and the gap between them is the lag any rule inherits.
    """
    x = np.asarray(series, dtype=float)
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), x.shape)
    blocks = (np.asarray(block_ids) if block_ids is not None
              else np.zeros(len(x), dtype=int))

    pivots: list[dict] = []
    start = 0
    while start < len(x):
        stop = start
        while stop + 1 < len(x) and blocks[stop + 1] == blocks[start]:
            stop += 1
        segment = range(start, stop + 1)

        extreme_index = None
        # Start by looking for a high. If the segment opens with a fall the
        # first branch fires at once and records its opening bar as a high,
        # which it is.
        direction = +1
        for i in segment:
            if not np.isfinite(x[i]) or not np.isfinite(thr[i]):
                continue
            if extreme_index is None:
                extreme_index = i
                continue
            # The extreme is extended, or the retracement from it is
            # measured. Never both on one bar: updating the extreme and
            # then measuring the distance back to it always gives zero,
            # which is how the first version of this found no pivots at all.
            if direction > 0:
                if x[i] > x[extreme_index]:
                    extreme_index = i
                elif x[i] <= x[extreme_index] - thr[i]:
                    pivots.append({"bar": extreme_index, "confirmed": i,
                                   "price": float(x[extreme_index]),
                                   "kind": "high"})
                    direction = -1
                    extreme_index = i
            else:
                if x[i] < x[extreme_index]:
                    extreme_index = i
                elif x[i] >= x[extreme_index] + thr[i]:
                    pivots.append({"bar": extreme_index, "confirmed": i,
                                   "price": float(x[extreme_index]),
                                   "kind": "low"})
                    direction = +1
                    extreme_index = i
        start = stop + 1
    return pivots


# ----------------------------------------------------------------------
# Swings
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Swing:
    """One half cycle, from a turn to the next turn."""

    start: int
    end: int
    confirmed: int
    duration: int
    amplitude: float          # price units, signed by direction
    amplitude_series: float   # same swing on the series the pivots came from
    amplitude_b: float        # the series amplitude, in units of b
    sharpness: float          # largest single bar over the swing's amplitude
    offset_b: float           # residual at the ending turn, in b
    slope: float
    slope_t: float


def swing_table(price, b, block_ids, trend_window: int = 120,
                pivot_k: float = 3.0, detrend: bool = True) -> list[Swing]:
    """Measure every half cycle in the series.

    Args:
        price: Mid prices.
        b: Predicted scale per bar; the unit everything is expressed in.
        block_ids: Session identifier; a swing never crosses one.
        trend_window: Bars in the trailing regression. Must be well above
            the cycle's own period; see the module docstring.
        pivot_k: Retracement needed to confirm a turn, in units of b.
        detrend: Find turns on the residual. Off, they are found on price,
            which in a trend produces one long swing and no cycle at all.
    """
    price = np.asarray(price, dtype=float)
    b = np.asarray(b, dtype=float)
    # b is the scale of a LOG return; the series the pivots run on is in
    # price units. Multiplying by the price converts one into the other.
    # Without it the threshold is a thousandth of a point, every bar
    # becomes a turning point, and the measured period is one bar.
    b_points = b * price
    fit = rolling_trend(price, trend_window)
    series = fit["residual"] if detrend else price
    threshold = pivot_k * b_points

    pivots = find_pivots(series, threshold, block_ids)
    swings: list[Swing] = []
    for first, second in zip(pivots[:-1], pivots[1:]):
        i, j = first["bar"], second["bar"]
        if j <= i:
            continue
        span = series[i:j + 1]
        if not np.isfinite(span).all():
            continue
        # Two amplitudes, deliberately. The price one is what a position
        # earns; the series one is the unit the signal is divided by, and
        # mixing them makes z meaningless because numerator and denominator
        # would then come from different series.
        amplitude = float(price[j] - price[i])
        amplitude_series = float(series[j] - series[i])
        steps = np.diff(span)
        largest = float(np.max(np.abs(steps))) if steps.size else 0.0
        scale = float(np.nanmean(b_points[i:j + 1]))
        swings.append(Swing(
            start=i,
            end=j,
            confirmed=int(second["confirmed"]),
            duration=int(j - i),
            amplitude=amplitude,
            amplitude_series=amplitude_series,
            amplitude_b=(amplitude_series / scale if scale > EPS else np.nan),
            sharpness=(largest / abs(amplitude_series)
                       if abs(amplitude_series) > EPS else np.nan),
            offset_b=(float(fit["residual"][j]) / float(b_points[j])
                      if np.isfinite(b_points[j]) and b_points[j] > EPS
                      else np.nan),
            slope=(float(fit["slope"][j])
                   if np.isfinite(fit["slope"][j]) else np.nan),
            slope_t=(float(fit["t_stat"][j])
                     if np.isfinite(fit["t_stat"][j]) else np.nan),
        ))
    return swings


# ----------------------------------------------------------------------
# Calibration and signal
# ----------------------------------------------------------------------

def calibrated(swings: list[Swing], length: int, memory: int = 5
               ) -> tuple[np.ndarray, np.ndarray]:
    """Running median duration and amplitude from the last `memory` swings.

    Each value is written from the bar a swing was CONFIRMED, never from
    the bar it turned, so a calibration never uses a turn that had not yet
    happened as far as the market was concerned, and is carried forward
    until the next confirmation.
    """
    duration = np.full(length, np.nan)
    amplitude = np.full(length, np.nan)
    seen: list[Swing] = []
    for swing in swings:
        seen.append(swing)
        if len(seen) < memory:
            continue
        window = seen[-memory:]
        at = swing.confirmed
        if 0 <= at < length:
            duration[at] = float(np.median([s.duration for s in window]))
            amplitude[at] = float(
                np.median([abs(s.amplitude_series) for s in window]))
    return (pd.Series(duration).ffill().to_numpy(),
            pd.Series(amplitude).ffill().to_numpy())


def signal_position(price, b, block_ids, trend_window: int = 120,
                    pivot_k: float = 3.0, memory: int = 5,
                    min_amplitude_b: float = 1.0) -> dict[str, np.ndarray]:
    """The one number a threshold is compared against, and its inputs.

        z = (price - trend) / (half a typical recent swing)

    A reading of 0.70 says the market sits seventy per cent of the way to a
    typical recent extreme, measured from its own trend line. That is a
    distance, where %K at 0.879 was a fraction of whatever range the last
    forty bars happened to cover.

    On a synthetic 60-bar cycle this exceeded 0.70 on 41% of bars; on a
    random walk of the same volatility, on 15.6%.

    The amplitude is floored at `min_amplitude_b` units of b, because
    dividing by an amplitude measured during a dead hour turns ordinary
    noise into an extreme reading.
    """
    price = np.asarray(price, dtype=float)
    b = np.asarray(b, dtype=float)
    fit = rolling_trend(price, trend_window)
    swings = swing_table(price, b, block_ids, trend_window, pivot_k)
    duration, amplitude = calibrated(swings, len(price), memory)

    # Half the swing, so that z reaches +/- 1 at a typical extreme and a
    # threshold of 0.70 means seventy per cent of the way to one. Dividing
    # by the whole peak-to-trough distance caps |z| near 0.5, and then no
    # threshold above that can ever fire.
    half = 0.5 * amplitude
    floor = min_amplitude_b * b * price
    effective = np.where(np.isfinite(half), np.maximum(half, floor), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(effective > EPS, fit["residual"] / effective, np.nan)

    return {
        "z": z,
        "residual": fit["residual"],
        "slope": fit["slope"],
        "slope_t": fit["t_stat"],
        "half_period": duration,
        "amplitude": amplitude,
        "swings": swings,
    }
def holt_trend(price, block_ids=None, level_rate: float = 0.05,
               slope_rate: float = 0.02) -> dict[str, np.ndarray]:
    """Local linear trend, which follows a change of slope instead of lagging.

    Holt's two-rate recursion:

        level = a * x + (1 - a) * (level + slope)
        slope = c * (level - previous level) + (1 - c) * slope

    A regression over a fixed window carries the whole window equally, so a
    slope that changed forty bars ago is diluted by the hundred and sixty
    bars before it and the fitted line sits below a new rally for hours.
    The residual then holds a standing offset and a rule that fades large
    residuals sells into the move again and again.

    Measured on a synthetic series whose slope changes from zero to 0.6 per
    bar: an OLS window of 200 carried a standing residual offset peaking at
    16.4 points and needing about 150 bars to clear, while this peaked at
    7.0 and cleared in about 100. Against a half amplitude of 71 points,
    that offset is a quarter of an entry threshold, held in the trend's
    direction for half a day.

    In a settled trend the comparison reverses: least squares fits a
    straight line exactly and leaves no bias, while this one keeps a small
    lag. The choice is which error matters more, and that is a measurement.

    Both rates are per bar. 0.05 and 0.02 give a level reacting over
    roughly twenty bars and a slope over fifty.

    State resets at every session boundary: carrying a slope across an
    overnight break fits the gap rather than the market.

    Returns the same four arrays as rolling_trend, so the two are
    interchangeable.
    """
    x = np.asarray(price, dtype=float)
    blocks = (np.asarray(block_ids) if block_ids is not None
              else np.zeros(len(x), dtype=int))
    n = len(x)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)

    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and blocks[stop + 1] == blocks[start]:
            stop += 1

        current_level = np.nan
        current_slope = 0.0
        for i in range(start, stop + 1):
            if not np.isfinite(x[i]):
                continue
            if not np.isfinite(current_level):
                current_level = x[i]
                current_slope = 0.0
            else:
                previous = current_level
                current_level = (
                    level_rate * x[i]
                    + (1.0 - level_rate) * (current_level + current_slope)
                )
                current_slope = (
                    slope_rate * (current_level - previous)
                    + (1.0 - slope_rate) * current_slope
                )
            level[i] = current_level
            slope[i] = current_slope
        start = stop + 1

    residual = x - level
    span = max(int(round(1.0 / max(slope_rate, 1e-6))), 5)
    resid_sd = pd.Series(residual).rolling(span, min_periods=span).std().to_numpy()
    sxx = span * (span * span - 1) / 12.0
    se = np.where(sxx > 0, resid_sd / np.sqrt(sxx), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(np.abs(se) > EPS, slope / se, np.nan)

    return {
        "slope": slope,
        "fitted": level,
        "residual": residual,
        "t_stat": t_stat,
    }


def trend_of(price, block_ids=None, method: str = "ols", window: int = 200,
             level_rate: float = 0.05, slope_rate: float = 0.02) -> dict:
    """Pick a trend estimator by name, so callers stay method agnostic."""
    if method == "ols":
        return rolling_trend(price, window)
    if method == "holt":
        return holt_trend(price, block_ids, level_rate, slope_rate)
    raise ValueError(f"unknown trend method: {method}")