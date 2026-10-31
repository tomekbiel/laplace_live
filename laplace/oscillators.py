"""
Oscillators, as comparable candidates rather than as a toolbox.

Every function here answers the same question in a different way: how far
has the market travelled from where it was resting, measured against what
it usually travels. They therefore share one signature,

    f(y, b, price, **params) -> np.ndarray

so a bake-off is a loop over a dictionary instead of twelve scripts whose
figures cannot be compared. Each returns one reading per input bar, NaN
through the warm-up, and nothing is shifted: the reading at index i uses
bars up to and including i, so an event drawn from it is known at that
bar's close and the forward path starts at i+1.

    y      log returns, one per bar
    b      the predicted Laplace scale for that bar. Dividing by it makes a
           reading scale-free and, because b is deseasonalised, makes the
           same threshold mean the same thing at 02:00 and at 09:30.
    price  the mid series, for the oscillators defined on levels.

Three families are represented, and the difference matters when reading the
bake-off:

    accumulators    cusum, cusum_amplitude, fisher_cusum, drift. They carry
                    memory of a move until it is walked back, with no fixed
                    window. The reset is what gives them memory.
    range position  stochastic_k, williams_r, donchian. Where the price sits
                    inside the last n bars. Fixed window, no memory.
    model based     z_ou. Deviation from a fitted equilibrium, in units of
                    that process's own stationary spread. The window is a
                    fit, not a choice.

    cci, rsi, center_of_gravity and cyber_cycle sit between the first two and
    are here because they are what everyone else uses; a candidate that
    cannot beat RSI is not a finding.

    efficiency_ratio is not an oscillator. It carries no sign and is meant
    as a regime filter: near 1 the path is a march, near 0 a round trip.

Nothing here imports the project. These are functions over arrays, which is
what makes them testable without a model, a config or a bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _series(x: np.ndarray) -> pd.Series:
    """A float Series, so pandas' rolling machinery can be used."""
    return pd.Series(np.asarray(x, dtype=float))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide, returning NaN where the denominator is zero or absent."""
    den = np.asarray(den, dtype=float)
    out = np.full(len(num), np.nan)
    good = np.isfinite(den) & (np.abs(den) > EPS)
    out[good] = np.asarray(num, dtype=float)[good] / den[good]
    return out


def _normalised(y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns in units of the predicted scale."""
    return _safe_divide(np.asarray(y, dtype=float), b)


# ----------------------------------------------------------------------
# Accumulators
# ----------------------------------------------------------------------

def cusum(y, b, price=None, h: float | None = None,
          normalise: bool = True) -> np.ndarray:
    """Two-sided cumulative sum with a floor at zero, the memory oscillator.

    Two accumulators run at once:

        S+ = max(0, S+ + x)      grows while the move is up, never negative
        S- = min(0, S- + x)      grows while the move is down, never positive

    The floor is the whole mechanism. A reversal pulls S+ back toward zero
    and stops there, so the distance already travelled is remembered until
    the market has walked all of it back. Meanwhile the opposite accumulator
    starts from zero, which is why a single bar that erases two hours of
    decline shows up as one side collapsing and the other taking over, and
    not as the silence a plain sum would give.

    The reading is whichever side is currently the larger excursion, carrying
    its sign. With h set, both sides reset to zero on the bar the reading
    first reaches +/- h, so that bar is the event and the next accumulation
    starts from it. This is the symmetric CUSUM filter used for event-based
    sampling, with the increments standardised by b.

    A reading may overshoot h by up to one bar's increment, since the test
    happens after the bar is added; that overshoot is the size of the bar
    that triggered the event and is worth keeping rather than clipping.

    Args:
        h: Reset threshold in units of b. None accumulates without reset,
           which is what a continuous reading wants; a threshold is what an
           event definition wants.
        normalise: Divide increments by b. Off, the reading is in log-return
           units and the same threshold means different things by hour.
    """
    x = _normalised(y, b) if normalise else np.asarray(y, dtype=float)
    out = np.full(len(x), np.nan)
    up = 0.0
    down = 0.0
    for i in range(len(x)):
        value = x[i]
        if not np.isfinite(value):
            out[i] = np.nan
            continue
        up = max(0.0, up + value)
        down = min(0.0, down + value)
        reading = up if up >= -down else down
        out[i] = reading
        if h is not None and abs(reading) >= h:
            up = 0.0
            down = 0.0
    return out


def cusum_amplitude(y, b, price=None, h: float = 3.0, memory: int = 5,
                    min_amplitude: float = 0.25) -> np.ndarray:
    """CUSUM divided by the size of recent swings, so 1.0 means "as usual".

    The amplitude is an exponentially weighted mean of the readings at the
    last `memory` resets. A reading of 0.7 therefore says the market has
    covered seventy per cent of what its recent swings covered, which is the
    threshold in the shape it was asked for.

    The amplitude updates at every reset and decays over `memory` swings, so
    a new regime of larger moves is absorbed within a few cycles while the
    previous ones still count. Until `memory` resets have happened the
    reading is NaN, because dividing by an amplitude estimated from one
    swing produces exactly the spurious extremes this is meant to avoid.
    """
    raw = cusum(y, b, h=h)
    out = np.full(len(raw), np.nan)
    alpha = 2.0 / (memory + 1.0)
    amplitude = np.nan
    seen = 0
    for i in range(len(raw)):
        reading = raw[i]
        if not np.isfinite(reading):
            continue
        if seen >= memory and np.isfinite(amplitude):
            out[i] = reading / max(amplitude, min_amplitude)
        if abs(reading) >= h:  # a reset happened on this bar
            swing = abs(reading)
            amplitude = swing if not np.isfinite(amplitude) else (
                alpha * swing + (1 - alpha) * amplitude
            )
            seen += 1
    return out


def fisher_cusum(y, b, price=None, h: float = 3.0, memory: int = 5,
                 clip: float = 0.999) -> np.ndarray:
    """Fisher transform of the amplitude-normalised CUSUM.

    0.5 * ln((1+x)/(1-x)) stretches the ends of a bounded reading toward
    infinity and pulls the middle together, which turns a crowded band near
    +/- 1 into a spread-out tail. A threshold can then be stated in standard
    deviations instead of being picked off a chart.
    """
    x = cusum_amplitude(y, b, h=h, memory=memory)
    x = np.clip(x, -clip, clip)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * np.log((1.0 + x) / (1.0 - x))


def drift(y, b, price=None, window: int = 12) -> np.ndarray:
    """Net displacement over a window, against what it should have been.

    sum(y) / (b * sqrt(n)). Variances add, so sqrt(n) is the distance a
    driftless series covers; a reading of 3 is three times that.

    Path-blind by construction: a steady march and a fall followed by an
    equal rally give the same number. That blindness is the reason the
    accumulators above exist, and the reason this is kept in the bake-off
    as the thing they have to beat.
    """
    x = _normalised(y, b)
    total = _series(x).rolling(window, min_periods=window).sum().to_numpy()
    return total / np.sqrt(window)


def efficiency_ratio(y, b=None, price=None, window: int = 12) -> np.ndarray:
    """Net displacement over total distance walked. A filter, not a signal.

    Near 1 the window is a march in one direction; near 0 it is a round
    trip with the same endpoints. Unsigned on purpose: it says how, never
    which way.
    """
    x = _series(y)
    net = x.rolling(window, min_periods=window).sum().abs().to_numpy()
    gross = x.abs().rolling(window, min_periods=window).sum().to_numpy()
    return _safe_divide(net, gross)


# ----------------------------------------------------------------------
# Position in a range
# ----------------------------------------------------------------------

def stochastic_k(y=None, b=None, price=None, window: int = 20) -> np.ndarray:
    """Where the price sits in the last n bars, rescaled to [-1, 1].

    The textbook %K runs 0..100 with thresholds at 20 and 80. Centring it
    on zero puts it on the same footing as the accumulators, so one
    threshold table covers every candidate.
    """
    p = _series(price)
    low = p.rolling(window, min_periods=window).min().to_numpy()
    high = p.rolling(window, min_periods=window).max().to_numpy()
    span = high - low
    position = _safe_divide(p.to_numpy() - low, span)
    return 2.0 * position - 1.0


def williams_r(y=None, b=None, price=None, window: int = 20) -> np.ndarray:
    """Williams %R, which is the negated %K. Kept to show it is the same.

    If this and stochastic_k ever disagree in the bake-off, the harness is
    broken, not the market.
    """
    return -stochastic_k(price=price, window=window)


def donchian_position(y=None, b=None, price=None, window: int = 20) -> np.ndarray:
    """Position inside the Donchian channel; %K under its other name."""
    return stochastic_k(price=price, window=window)


def cci(y=None, b=None, price=None, window: int = 20,
        constant: float = 0.015) -> np.ndarray:
    """Commodity Channel Index: deviation from the mean, in mean deviations.

    Lambert's 0.015 exists so that roughly seventy per cent of readings fall
    within +/- 100; it is a convention, not a property of the data. The
    reading is returned in its own units, divided by 100, so that 1.0 is the
    conventional extreme and the scale matches the others.
    """
    p = _series(price)
    mean = p.rolling(window, min_periods=window).mean()
    deviation = p.rolling(window, min_periods=window).apply(
        lambda w: np.mean(np.abs(w - w.mean())), raw=True
    )
    out = _safe_divide((p - mean).to_numpy(), (constant * deviation).to_numpy())
    return out / 100.0


def rsi(y=None, b=None, price=None, window: int = 14) -> np.ndarray:
    """Relative Strength Index, centred and scaled to [-1, 1].

    Present as the thing to beat. A candidate that does not improve on the
    oscillator every retail platform ships by default has not earned the
    complexity it costs.
    """
    p = _series(price)
    delta = p.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = _safe_divide(avg_gain.to_numpy(), avg_loss.to_numpy())
    value = 100.0 - 100.0 / (1.0 + rs)
    # A window of pure gains leaves avg_loss at zero, where the ratio is
    # undefined rather than merely large; that is the top of the scale.
    value = np.where(
        (avg_loss.to_numpy() <= EPS) & (avg_gain.to_numpy() > EPS), 100.0, value
    )
    return (value - 50.0) / 50.0


# ----------------------------------------------------------------------
# Ehlers
# ----------------------------------------------------------------------

def center_of_gravity(y=None, b=None, price=None, window: int = 10,
                      scale_window: int = 100) -> np.ndarray:
    """Ehlers' Center of Gravity: where the window's mass sits, centred.

    The weighted mean lag of the last n prices, measured against the middle
    of the window, so zero means the mass is evenly spread and a positive
    reading means it has moved toward the recent end.

    Raw, this reading is useless on an index: the weighted mean lag of ten
    prices around 30000 that differ by a few points is pinned to the middle
    of the window, and the whole signal lives in the fourth decimal. Ehlers
    wrote it for instruments whose price varies by a large fraction of its
    level. Dividing by a rolling standard deviation of the reading restores
    the shape and puts it in its own standard deviations, which is the same
    treatment cyber_cycle needs and for the same reason.
    """
    p = np.asarray(price, dtype=float)
    n = len(p)
    raw = np.full(n, np.nan)
    weights = np.arange(1, window + 1, dtype=float)
    half = (window + 1) / 2.0
    for i in range(window - 1, n):
        chunk = p[i - window + 1:i + 1][::-1]  # most recent first
        total = chunk.sum()
        if not np.isfinite(total) or abs(total) < EPS:
            continue
        raw[i] = -((weights * chunk).sum() / total) + half
    spread = _series(raw).rolling(
        scale_window, min_periods=scale_window // 2
    ).std().to_numpy()
    return _safe_divide(raw, spread)


def cyber_cycle(y=None, b=None, price=None, alpha: float = 0.07) -> np.ndarray:
    """Ehlers' Cyber Cycle: the cyclic part after the trend is filtered out.

    A high-pass filter on a four-bar smoothed price, then a two-pole
    recursion. The trend is removed by construction, which is what makes it
    a different object from the range oscillators: it does not saturate in a
    trend, it goes quiet.

    The reading is divided by a rolling standard deviation of itself, so the
    output is in its own standard deviations and comparable with the others.
    """
    p = np.asarray(price, dtype=float)
    n = len(p)
    smooth = np.full(n, np.nan)
    for i in range(3, n):
        window = p[i - 3:i + 1]
        if not np.isfinite(window).all():
            continue
        smooth[i] = (window[3] + 2.0 * window[2] + 2.0 * window[1] + window[0]) / 6.0

    cycle = np.zeros(n)
    for i in range(6, n):
        if not np.isfinite(smooth[i - 2:i + 1]).all():
            continue
        cycle[i] = (
            (1.0 - 0.5 * alpha) ** 2 * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
            + 2.0 * (1.0 - alpha) * cycle[i - 1]
            - (1.0 - alpha) ** 2 * cycle[i - 2]
        )

    out = np.where(np.arange(n) >= 6, cycle, np.nan)
    scale = _series(out).rolling(100, min_periods=50).std().to_numpy()
    return _safe_divide(out, scale)


# ----------------------------------------------------------------------
# Model based
# ----------------------------------------------------------------------

def rolling_ou(x: np.ndarray, window: int = 120) -> dict[str, np.ndarray]:
    """Fit an Ornstein-Uhlenbeck process in a rolling window, as AR(1).

    In discrete time the process is a first-order autoregression,

        x(t) = a + phi * x(t-1) + e,   phi = exp(-kappa * dt)

    so kappa = -ln(phi) and the half life is ln2 / kappa, both in bars.
    The stationary spread is sigma / sqrt(1 - phi^2), which is the natural
    unit for "how far this process usually strays".

    phi comes from the rolling covariance of the series with its own lag
    over its lagged variance, which is the least-squares slope.

    The estimate is biased: in a window of this length the least-squares phi
    sits low by roughly 2/window, so kappa reads high and reversion looks
    faster than it is. Reported as `phi_bias` so the correction is visible
    rather than silently applied. Measured on a synthetic series with a true
    phi of 0.95 and a window of 120, the median estimate was 0.932.

    Returns theta, phi, kappa, half_life, sigma_eq and phi_bias, each one
    value per bar, NaN through the warm-up and wherever the fit is not
    mean-reverting (phi outside (0, 1)).
    """
    s = _series(x)
    lag = s.shift(1)
    cov = s.rolling(window, min_periods=window).cov(lag)
    var = lag.rolling(window, min_periods=window).var()
    phi = _safe_divide(cov.to_numpy(), var.to_numpy())

    mean = s.rolling(window, min_periods=window).mean().to_numpy()
    lag_mean = lag.rolling(window, min_periods=window).mean().to_numpy()
    intercept = mean - phi * lag_mean

    # theta is the long-run level the process is pulled toward.
    theta = _safe_divide(intercept, 1.0 - phi)

    residual = s.to_numpy() - (intercept + phi * lag.to_numpy())
    sigma = _series(residual).rolling(window, min_periods=window).std().to_numpy()

    stationary = np.full(len(phi), np.nan)
    reverting = np.isfinite(phi) & (phi > 0.0) & (phi < 1.0)
    stationary[reverting] = sigma[reverting] / np.sqrt(1.0 - phi[reverting] ** 2)

    kappa = np.full(len(phi), np.nan)
    kappa[reverting] = -np.log(phi[reverting])

    half_life = np.full(len(phi), np.nan)
    positive = reverting & (kappa > 0)
    half_life[positive] = np.log(2.0) / kappa[positive]

    return {
        "theta": theta,
        "phi": phi,
        "kappa": kappa,
        "half_life": half_life,
        "sigma_eq": stationary,
        "phi_bias": np.full(len(phi), 2.0 / window),
    }

def pooled_ar1(x: np.ndarray, usable: np.ndarray) -> dict[str, float]:
    """Fit x(t) = a + phi * x(t-1) + e over the pairs that are allowed.

    A rolling fit assumes the series is contiguous. A session bucket is not:
    the night bucket is three hundred separate stretches, and joining them
    end to end invents a transition from one night's last bar to the next
    night's first. This pools only the adjacent pairs the caller marks as
    usable, which is how a bucket can be fitted without being contiguous.

    Args:
        x: The series.
        usable: True at index i when the pair (x[i-1], x[i]) may be used.

    Returns phi with its standard error, the two t-statistics that matter,
    and the Ornstein-Uhlenbeck quantities when the fit is mean-reverting:

        t_zero  = phi / se, testing phi = 0. The question for increments:
                  is this bar related to the previous one at all, and a
                  negative value is a bounce.
        t_unit  = (phi - 1) / se, the Dickey-Fuller statistic, testing a
                  random walk against mean reversion. Its distribution is
                  not the normal one; against a constant and a large
                  sample the critical values are -3.43, -2.86 and -2.57
                  at one, five and ten per cent.

    The least-squares phi is biased low by roughly (1 + 3*phi) / n, which
    at the sample sizes this pools is a third decimal and is returned as
    `bias` rather than applied.
    """
    x = np.asarray(x, dtype=float)
    usable = np.asarray(usable, dtype=bool)
    index = np.flatnonzero(usable)
    index = index[index > 0]
    if index.size < 30:
        return {"n": int(index.size)}

    current = x[index]
    previous = x[index - 1]
    good = np.isfinite(current) & np.isfinite(previous)
    current, previous = current[good], previous[good]
    n = current.size
    if n < 30:
        return {"n": int(n)}

    mean_previous = previous.mean()
    centred = previous - mean_previous
    denominator = float((centred ** 2).sum())
    if denominator <= EPS:
        return {"n": int(n)}

    phi = float((centred * (current - current.mean())).sum() / denominator)
    intercept = float(current.mean() - phi * mean_previous)
    residual = current - (intercept + phi * previous)
    sigma = float(np.sqrt((residual ** 2).sum() / max(n - 2, 1)))
    se = sigma / np.sqrt(denominator)

    out = {
        "n": int(n),
        "phi": phi,
        "se": se,
        "t_zero": phi / se if se > 0 else float("nan"),
        "t_unit": (phi - 1.0) / se if se > 0 else float("nan"),
        "sigma": sigma,
        "bias": (1.0 + 3.0 * phi) / n,
        "theta": intercept / (1.0 - phi) if abs(1.0 - phi) > EPS else float("nan"),
    }

    if 0.0 < phi < 1.0:
        kappa = -np.log(phi)
        out["kappa"] = float(kappa)
        out["half_life"] = float(np.log(2.0) / kappa)
        out["sigma_eq"] = float(sigma / np.sqrt(1.0 - phi ** 2))
    else:
        out["kappa"] = float("nan")
        out["half_life"] = float("nan")
        out["sigma_eq"] = float("nan")
    return out

def z_ou(y=None, b=None, price=None, window: int = 120,
         on: str = "cusum", h: float | None = None) -> np.ndarray:
    """Deviation from a fitted equilibrium, in that process's own spread.

        z = (x - theta) / sigma_eq

    What `x` is matters more than the formula. Price is not stationary, so
    fitting an equilibrium to it produces a theta that chases the trend and
    a kappa that measures the chase. On a synthetic random walk this reading
    ran from -21 to +59, against -2.8 to +2.8 for the same formula on the
    accumulator. The accumulator is stationary by construction, because it
    resets, which is why `on='cusum'` is the default and `on='price'` is
    kept only to show the difference.
    """
    if on == "price":
        x = np.asarray(price, dtype=float)
    elif on == "cusum":
        x = cusum(y, b, h=h)
    else:
        raise ValueError(f"z_ou: unknown series '{on}'")

    fit = rolling_ou(x, window=window)
    return _safe_divide(x - fit["theta"], fit["sigma_eq"])


# ----------------------------------------------------------------------
# The registry the bake-off loops over
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One oscillator, with the knob a calibration run is allowed to turn.

    Attributes:
        func: The oscillator itself.
        params: Fixed arguments.
        family: accumulator, range, ehlers or model. Printed with the
            results, because a bake-off won by three members of one family
            says something different from one won by three families.
        needs_price: Defined on levels rather than on returns.
        adaptive_window: The window argument should be replaced by the
            measured half life when one is available.
    """

    func: Callable[..., np.ndarray]
    params: dict = field(default_factory=dict)
    family: str = "other"
    needs_price: bool = False
    adaptive_window: str | None = None

    def __call__(self, y, b, price) -> np.ndarray:
        return self.func(y=y, b=b, price=price, **self.params)

    def with_window(self, bars: int) -> "Candidate":
        """A copy whose window is the measured one."""
        if not self.adaptive_window:
            return self
        params = dict(self.params)
        params[self.adaptive_window] = int(max(2, bars))
        return Candidate(self.func, params, self.family,
                         self.needs_price, self.adaptive_window)


def build_registry(window: int = 20, cusum_h: float = 3.0) -> dict[str, Candidate]:
    """Every candidate, at one shared window so the comparison is fair."""
    return {
        "cusum": Candidate(cusum, {"h": cusum_h}, "accumulator"),
        "cusum_amplitude": Candidate(
            cusum_amplitude, {"h": cusum_h, "memory": 5}, "accumulator"),
        "fisher_cusum": Candidate(
            fisher_cusum, {"h": cusum_h, "memory": 5}, "accumulator"),
        "drift": Candidate(
            drift, {"window": window}, "accumulator", adaptive_window="window"),
        "stochastic_k": Candidate(
            stochastic_k, {"window": window}, "range", True, "window"),
        "williams_r": Candidate(
            williams_r, {"window": window}, "range", True, "window"),
        "cci": Candidate(cci, {"window": window}, "range", True, "window"),
        "rsi": Candidate(rsi, {"window": 14}, "range", True, "window"),
        "center_of_gravity": Candidate(
            center_of_gravity, {"window": 10}, "ehlers", True, "window"),
        "cyber_cycle": Candidate(cyber_cycle, {"alpha": 0.07}, "ehlers", True),
        "z_ou_cusum": Candidate(
            z_ou, {"window": 120, "on": "cusum", "h": cusum_h},
            "model", adaptive_window="window"),
        "z_ou_price": Candidate(
            z_ou, {"window": 120, "on": "price"}, "model", True, "window"),
    }


FILTERS: dict[str, Candidate] = {
    "efficiency_ratio": Candidate(
        efficiency_ratio, {"window": 12}, "filter", adaptive_window="window"),
}

# ----------------------------------------------------------------------
# Deviation from the local trend, in units of a measured swing
# ----------------------------------------------------------------------

def trend_deviation(y=None, b=None, price=None, block_ids=None,
                    method: str = "ols", window: int = 200,
                    amplitude_b: float = 5.87, level_rate: float = 0.05,
                    slope_rate: float = 0.02) -> np.ndarray:
    """How far price sits from its trend, as a fraction of a typical swing.

        z = (price - trend) / (0.5 * amplitude_b * b * price)

    Every other oscillator here answers "where in the recent range", which
    is a fraction of a span that may be twenty points or three hundred.
    This one answers "how far from the line, in typical swings", so a
    threshold on it is a distance.

    amplitude_b is the median peak-to-trough swing measured on history, in
    units of b: 5.87 on US.100+ M5 with an OLS window of 200, stable from
    5.13 to 6.86 across session buckets and months. It is a CONSTANT on
    purpose. A rolling calibration was tested and the last five swings
    forecast the next one no better than the long-run median does, with a
    skill at or below zero on the market and on a permuted control alike,
    so a running estimate adds noise and no information.

    The constant belongs to the method that measured it. A different trend
    estimator leaves a different residual, so amplitude_b has to be
    re-measured before 'holt' and 'ols' can be compared on equal terms.

    block_ids matters only for 'holt', whose state resets at a session
    boundary; None treats the series as one unbroken session.
    """
    price = np.asarray(price, dtype=float)
    b = np.asarray(b, dtype=float)
    # Imported here rather than at module scope: swings consumes nothing
    # from this file, and keeping the edge one-way makes import order
    # irrelevant.
    from . import swings as _swings

    fit = _swings.trend_of(price, block_ids, method, window,
                           level_rate, slope_rate)
    half = 0.5 * amplitude_b * b * price
    return _safe_divide(fit["residual"], half)


def trend_state(y=None, b=None, price=None, window: int = 24) -> np.ndarray:
    """Is there a direction right now, in units of what chance would give.

        drift = sum(y) / (b * sqrt(n))

    The same quantity as `drift`, named for the job it does here: a state
    variable rather than a signal. A short window is wanted, because the
    question is whether a direction appeared in the last couple of hours,
    and eating the cycle does not matter when nothing is being subtracted.
    """
    return drift(y=y, b=b, price=price, window=window)