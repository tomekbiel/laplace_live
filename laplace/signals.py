"""
Round trips: enter on an excursion, exit on the way back.

A position here is opened at the close of the bar whose reading crosses a
threshold, and its return starts at the next bar. The signal therefore
cannot pay itself, which is the defect that made the first backtest in
this project report thirty-eight times the account.

The exit is defined by the oscillator rather than by a fixed horizon,
because that is what a threshold rule means: you enter because the market
has gone too far and you leave when it has come back. A fixed horizon
answers a different question, and events.py answers that one.
"""

from __future__ import annotations

import numpy as np

EXIT_RULES = ("zero", "half", "opposite", "level")


def simulate(reading, y, block_ids, threshold: float, price=None,
             contrarian: bool = True, exit_rule: str = "zero",
             max_hold: int = 48, target_points: float | None = None,
             stop_points: float | None = None,
             trail_points: float | None = None,
             trail_arm: float | None = None,
             trail_after_signal=False,
             trail_start=88.0,
             trail_min=20.0,
             trail_tau=24,
             hold_after_arm=None,
             exit_level: float | None = None,
             entry_mask=None) -> dict:
    """Trade every excursion past `threshold` and record what happened.

    Entry: the first bar whose |reading| reaches the threshold while flat.
    A reading that stays beyond the threshold for six bars is one trade,
    not six, and no new position opens while one is running.

    Exit, whichever comes first:

        zero       the reading crosses back through zero
        half       |reading| falls below half the threshold
        opposite   the reading reaches the threshold on the other side
        max_hold   the clock runs out
        block_end  the session ends; the position is closed at its last bar

    When `trail_after_signal` is set, the oscillator exit arms a trailing
    stop rather than closing the position. The stop starts `trail_start` 
    points behind the best price seen and narrows linearly to `trail_min` 
    over `trail_tau` bars, so a position that keeps running is held while
    one that turns is released near its peak. `hold_after_arm` replaces the
    remaining `max_hold` budget once the stop is armed.

    Args:
        reading: The oscillator, one value per bar.
        y: Log returns, aligned with the reading.
        block_ids: Session identifier per bar; a trade never crosses one.
        threshold: Entry level, in the reading's own units.
        price: Price series, for point calculation.
        contrarian: Trade against the excursion. False trades with it.
        max_hold: Hard bar limit.

    Returns arrays of per-trade returns, durations, excursions, exit
    reasons, sides, entry indices, entry prices, exit prices and points.
    """
    r = np.asarray(reading, dtype=float)
    y = np.asarray(y, dtype=float)
    block_ids = np.asarray(block_ids)
    n = len(r)
    price = (np.asarray(price, dtype=float) if price is not None
             else np.full(n, np.nan))

    returns: list[float] = []
    durations: list[int] = []
    best_excursions: list[float] = []
    worst_excursions: list[float] = []
    reasons: list[str] = []
    sides: list[int] = []
    entries: list[int] = []
    entry_prices: list[float] = []
    exit_prices: list[float] = []
    armed_flags, after_arm_points = [], []
    best_list, width_list = [], []

    open_position = None

    i = 1
    while i < n - 1:
        if not np.isfinite(r[i]) or abs(r[i]) < threshold:
            i += 1
            continue
        if np.isfinite(r[i - 1]) and abs(r[i - 1]) >= threshold:
            i += 1          # already inside this excursion
            continue
        if entry_mask is not None and not bool(entry_mask[i]):
            # A state filter refuses this entry. The reading is left
            # untouched so a position already open still sees its exits;
            # only the opening is blocked.
            i += 1
            continue

        trigger = np.sign(r[i])
        side = -trigger if contrarian else trigger
        cumulative = 0.0
        best = 0.0
        worst = 0.0
        best_points = 0.0
        armed = False
        armed_step = 0
        points_at_arm = 0.0
        trail_width = float("nan")
        reason = "max_hold"
        j = i
        limit = int(max_hold)
        step = 0
        while step < limit:
            step += 1
            j = i + step
            if j >= n:
                # The array ran out. In a backtest this is the tail of the
                # file; live it is the position being held right now, and
                # the two must not be confused with a session ending.
                j -= 1
                reason = "data_end"
                break
            if block_ids[j] != block_ids[i]:
                j -= 1
                reason = "block_end"
                break
            if not np.isfinite(y[j]):
                j -= 1
                reason = "gap"
                break
            cumulative += side * y[j]
            best = max(best, cumulative)
            worst = min(worst, cumulative)
            entry_price = price[i] if np.isfinite(price[i]) else np.nan
            points = cumulative * entry_price
            if points > best_points:
                best_points = points
            if armed:
                armed_step += 1
                span = max(1, int(trail_tau))
                reach = min(armed_step, span)
                width = trail_start - (trail_start - trail_min) * reach / span
                trail_width = width
                if best_points - points >= width:
                    reason = "trail"
                    break
            # Price barriers, checked before the oscillator, because a
            # position that has reached its target or its stop is closed
            # whatever the reading says. Checked at bar closes, which is
            # all this data holds: an intrabar touch is invisible here, so
            # stops are understated and targets are reached late.
            if np.isfinite(price[i]) and np.isfinite(price[j]):
                moved = side * (price[j] - price[i])
                best_points = max(best_points, moved)
                if stop_points is not None and moved <= -abs(stop_points):
                    reason = "stop"
                    break
                if target_points is not None and moved >= abs(target_points):
                    reason = "target"
                    break
                if (trail_points is not None
                        and best_points >= (trail_arm or 0.0)
                        and best_points > 0
                        and moved <= best_points - abs(trail_points)):
                    reason = "trail"
                    break
            value = r[j]
            if np.isfinite(value):
                if exit_rule == "zero" and np.sign(value) != trigger:
                    if not trail_after_signal:
                        reason = "zero"
                        break
                    if not armed:
                        armed = True
                        points_at_arm = points
                        if hold_after_arm:
                            limit = step + int(hold_after_arm)
                if exit_rule == "half" and abs(value) <= 0.5 * threshold:
                    if not trail_after_signal:
                        reason = "half"
                        break
                    if not armed:
                        armed = True
                        points_at_arm = points
                        if hold_after_arm:
                            limit = step + int(hold_after_arm)
                if exit_rule == "opposite" and value * trigger <= -threshold:
                    if not trail_after_signal:
                        reason = "opposite"
                        break
                    if not armed:
                        armed = True
                        points_at_arm = points
                        if hold_after_arm:
                            limit = step + int(hold_after_arm)
                if (exit_rule == "level" and exit_level is not None
                        and abs(value) <= exit_level):
                    reason = "level"
                    break

        # The data ran out while the position was still open. In a backtest
        # this is the tail of the file; live it is the position being held
        # right now, which is the one that matters.
        still_open = (reason == "data_end" and j > i)

        if j > i and not still_open:
            returns.append(cumulative)
            durations.append(j - i)
            best_excursions.append(best)
            worst_excursions.append(worst)
            reasons.append(reason)
            sides.append(int(side))
            entries.append(i)
            entry_prices.append(price[i])
            exit_prices.append(price[j])
            armed_flags.append(armed)
            after_arm_points.append(points - points_at_arm if armed else 0.0)
            best_list.append(best_points)
            width_list.append(trail_width)
        elif still_open:
            open_position = {
                "entry": int(i), "side": int(side), "bars": int(j - i),
                "entry_price": float(price[i]), "price": float(price[j]),
                "points": float(side * (price[j] - price[i])),
                "best": float(best), "worst": float(worst),
                "best_points": float(best_points),
                "trail_width": float(trail_width),
                "armed": armed,
            }
        i = max(j, i) + 1

    return {
        "ret": np.array(returns),
        "bars": np.array(durations),
        "mfe": np.array(best_excursions),
        "mae": np.array(worst_excursions),
        "reason": np.array(reasons, dtype=object),
        "side": np.array(sides),
        "entry": np.array(entries, dtype=int),
        "entry_price": np.array(entry_prices),
        "exit_price": np.array(exit_prices),
        "points": np.array(exit_prices) - np.array(entry_prices),
        "armed": np.array(armed_flags, dtype=bool),
        "after_arm": np.array(after_arm_points),
        "best_points": np.array(best_list),
        "trail_width": np.array(width_list),
        "open_position": open_position,
    }


def describe(trades: dict, mask: np.ndarray | None = None) -> dict:
    """Win rate and expectancy for a set of trades.

    The win rate is reported because it is asked for, and it is the least
    informative number here. An exit rule that closes on the way back wins
    about sixty-five per cent of the time on a pure random walk, where
    there is nothing to win. Judge a row by the mean against its control.

    Everything is gross. Costs are a separate layer and do not belong in a
    measurement of what the market did.
    """
    ret = trades["ret"]
    bars, mfe, mae, side = (trades["bars"], trades["mfe"],
                            trades["mae"], trades["side"])
    if mask is not None:
        ret, bars, mfe, mae, side = (ret[mask], bars[mask], mfe[mask],
                                     mae[mask], side[mask])
    n = len(ret)
    if n < 5:
        return {"n": n}

    bp = ret * 10000.0
    se = float(bp.std(ddof=1) / np.sqrt(n))
    return {
        "n": n,
        "win": 100.0 * float((bp > 0).mean()),
        "mean": float(bp.mean()),
        "median": float(np.median(bp)),
        "se": se,
        "t": float(bp.mean() / se) if se > 0 else float("nan"),
        "bars": float(bars.mean()),
        "mfe": float(np.median(mfe)) * 10000.0,
        "mae": float(np.median(mae)) * 10000.0,
        "long": int((side > 0).sum()),
        "short": int((side < 0).sum()),
    }