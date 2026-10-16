"""
The entry and exit rules, as a state machine.

simulate_trading in 8_backtest.py is one long function doing three jobs at
once: walking an array, holding the position state, and applying the rules.
Live, the walking disappears — bars arrive one at a time, five minutes apart.
Only the middle job and the rules are wanted.

So the body of that loop lives here, fed one bar at a time, and the walking
stays outside it. scripts/backtest.py drives it over stored data and
laplace.live drives it over the socket, and both execute THE SAME rule code.

Two switches, both off by default
---------------------------------
Off, this reproduces 8_backtest.py exactly, including its defects, so the
figures stay comparable with what was measured before. On, each repairs one
family of defects. They are separate because their effects are separate and
both need measuring against the same data.

**signed_exits** — the sign conventions.

7_cem.py and 8_backtest.py disagree about one variable. In the optimiser,
current_z_score is the raw price move; in the backtest, cum_return already
carries the position's sign. The exit conditions were copied from one to the
other without adjusting for that, so every branch written for a short negates
an already-signed quantity. Three consequences, all three observed live on
2026-10-16:

* The timeout branch tests `position * cum_return <= -stop_loss * b / 2`.
  For a short that is `-|y_entry|`, and since entry requires
  `|y_entry| > entry_z * b` while the threshold is `stop_loss * b / 2`, the
  condition holds on the first managed bar whenever entry_z > stop_loss / 2.
  Every short closed after exactly one bar — five out of five.
* take_profit and stop_loss are swapped on the short side: a losing short
  leaves as 'take_profit', a winning one as 'stop_loss'.
* max_profit tracks `abs(cum_return)`, the absolute excursion rather than
  the favourable one. A position deep in loss arms the trailing stop with
  its own drawdown, and the stop then fires the moment the loss narrows —
  which is exactly the "closed as it was turning positive" that both longs
  did on 2026-10-16.

On, cum_return is treated as signed throughout: a profit is positive for
both directions, a loss negative, and max_profit records only favourable
movement. Every branch then reads the same for a long and for a short.

**book_entry_bar** — where the trade's return starts.

On (the original), profit is summed from y[entry_step] inclusive: the bar
whose move produced the signal. That bar closes before anyone knows a signal
exists. Measured on the research test set, it was 97.8 per cent of everything
the backtest booked. 7_cem.py does NOT do this — it measures from the entry
bar's close — so the parameters were optimised under one accounting and
applied under another. Off, the trade starts at the entry bar's close, which
is what a live run can actually capture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .core.gaps import detect_time_gap, is_market_closed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    """One closed bar, with the model's view of it.

    Attributes:
        time: Bar timestamp, broker server time.
        y: Realised log return at this step.
        mu: Predicted conditional median for this step.
        b: Predicted Laplace scale for this step.
        price: Mid at this step.
        window: The feature window, needed for the market-closed guard.
    """

    time: datetime
    y: float
    mu: float
    b: float
    price: float
    window: np.ndarray


@dataclass
class StepResult:
    """What the engine did with one bar."""

    action: str = "none"
    reason: str = ""
    position: int = 0
    signal_z: float = 0.0
    cum_return: float = 0.0
    equity: float = 0.0
    trade: dict[str, Any] | None = None
    long_trigger: float = 0.0
    short_trigger: float = 0.0
    blocked: str = ""


@dataclass
class _Snapshot:
    """Enough state to undo a close that the broker refused."""

    position: int
    entry_time: datetime | None
    entry_step: int
    entry_price: float
    max_profit: float
    returns: list[float]
    equity: float
    equity_points: int


@dataclass
class Engine:
    """Position state and the rules that change it.

    Args:
        params: The thirteen CEM parameters.
        gap_threshold_min: Minutes above which a break force-closes the
            position and blocks entry.
        signed_exits: Treat cum_return as signed in every exit branch.
        book_entry_bar: Include the triggering bar's return in the trade.
    """

    params: dict[str, float]
    gap_threshold_min: float = 10.0
    signed_exits: bool = False
    book_entry_bar: bool = True

    position: int = 0
    entry_time: datetime | None = None
    entry_step: int = 0
    entry_price: float = 0.0
    equity: float = 0.0
    step: int = 0
    max_profit: float = 0.0

    _prev_time: datetime | None = None
    _returns: list[float] = field(default_factory=list)
    _last_close: _Snapshot | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def observe(self, bar: Bar) -> StepResult:
        """Feed one closed bar and get the resulting action.

        The order of checks is the order in simulate_trading: the time-gap
        guard, the market-closed guard, then the rules.
        """
        result = StepResult(position=self.position, equity=self.equity)
        self.step += 1

        if self._prev_time is not None and detect_time_gap(
            bar.time, self._prev_time, self.gap_threshold_min
        ):
            self._prev_time = bar.time
            return self._force_close(result, bar, "gap_close")

        if is_market_closed(bar.window):
            self._prev_time = bar.time
            return self._force_close(result, bar, "market_closed")

        self._prev_time = bar.time

        current_b = bar.b
        result.signal_z = bar.y / current_b if current_b else 0.0

        if self.params["volatility_adapt"]:
            current_b = current_b * self.params["risk_multiplier"]

        # Where the entry threshold sits in price terms, for anyone placing
        # the order by hand. Costs nothing and changes nothing.
        result.long_trigger = bar.price * float(np.exp(self.params["entry_z"] * bar.b))
        result.short_trigger = bar.price * float(np.exp(-self.params["entry_z"] * bar.b))

        if self.position == 0:
            return self._maybe_open(result, bar)
        return self._manage(result, bar, current_b)

    def close_out(self, bar: Bar, reason: str = "session_end") -> StepResult:
        """Close any open position without consulting the rules."""
        result = StepResult(position=self.position, equity=self.equity)
        if self.position == 0:
            return result
        return self._force_close(result, bar, reason)

    def undo_last_close(self) -> bool:
        """Restore the position after a close the broker did not accept.

        A failed close leaves the position open at the broker. Without this,
        the engine would believe it flat, open a second position on the next
        signal, and the journal would stop describing the account. The caller
        is whoever sent the order and saw it fail.

        Returns:
            True when a close was undone, False when there was none to undo.
        """
        snapshot = self._last_close
        if snapshot is None:
            return False

        self.position = snapshot.position
        self.entry_time = snapshot.entry_time
        self.entry_step = snapshot.entry_step
        self.entry_price = snapshot.entry_price
        self.max_profit = snapshot.max_profit
        self._returns = list(snapshot.returns)
        self.equity = snapshot.equity

        if self.trades:
            self.trades.pop()
        del self.equity_curve[snapshot.equity_points:]

        self._last_close = None
        logger.warning("Close was refused; the position is open again at %s",
                       snapshot.entry_time)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _force_close(self, result: StepResult, bar: Bar, reason: str) -> StepResult:
        """Close on a guard, booking returns up to but excluding this bar."""
        if self.position == 0:
            result.blocked = reason
            self.equity_curve.append(self.equity)
            result.equity = self.equity
            return result

        cum_return = float(np.sum(self._returns)) * self.position
        trade = self._record(bar, cum_return, reason, exit_time=self._prev_time)
        result.action = "close"
        result.reason = reason
        result.trade = trade
        result.cum_return = cum_return
        result.position = 0
        result.equity = self.equity
        return result

    def _maybe_open(self, result: StepResult, bar: Bar) -> StepResult:
        """Open when the signal clears the threshold.

        No equity point is recorded on a flat step, matching simulate_trading,
        where the flat branch appends nothing.
        """
        if abs(result.signal_z) <= self.params["entry_z"]:
            return result

        self.position = int(np.sign(result.signal_z))
        self.entry_time = bar.time
        self.entry_step = self.step
        self.entry_price = bar.price
        self.max_profit = 0.0
        # With book_entry_bar the triggering bar is part of the trade, as in
        # the backtest. Without it, the trade starts at this bar's close.
        self._returns = [bar.y] if self.book_entry_bar else []

        result.action = "open"
        result.reason = "entry"
        result.position = self.position
        logger.info("Opened %s at %s, price %.5f, signal_z %.2f",
                    "long" if self.position == 1 else "short",
                    bar.time, bar.price, result.signal_z)
        return result

    def _manage(self, result: StepResult, bar: Bar, current_b: float) -> StepResult:
        """Apply the exit rules to an open position, in the original order."""
        self._returns.append(bar.y)
        cum_return = float(np.sum(self._returns)) * self.position
        current_z_score = cum_return / current_b if current_b else 0.0

        # Signed: only favourable movement counts, so a drawdown cannot arm
        # the trailing stop. Original: the absolute excursion, which can.
        excursion = cum_return if self.signed_exits else abs(cum_return)
        if excursion > self.max_profit:
            self.max_profit = excursion

        steps_in_pos = self.step - self.entry_step

        if self.signed_exits:
            exit_signal, exit_reason = self._exits_signed(
                cum_return, current_z_score, current_b, steps_in_pos
            )
        else:
            exit_signal, exit_reason = self._exits_original(
                cum_return, current_z_score, current_b, steps_in_pos
            )

        if exit_signal and exit_reason == "max_hold":
            cum_return -= self.params["time_decay"] * steps_in_pos

        result.cum_return = cum_return
        result.position = self.position

        if exit_signal:
            trade = self._record(bar, cum_return, exit_reason, exit_time=bar.time)
            result.action = "close"
            result.reason = exit_reason
            result.trade = trade
            result.position = 0
        else:
            self.equity_curve.append(self.equity)

        result.equity = self.equity
        return result

    def _exits_signed(
        self, cum_return: float, current_z_score: float,
        current_b: float, steps_in_pos: int,
    ) -> tuple[bool, str]:
        """Exit rules with cum_return read as signed profit, both directions.

        A profit is positive and a loss negative for a long and for a short
        alike, so one branch serves both and the thresholds mean what their
        names say.
        """
        p = self.params

        # Timeout on an early loss. The original multiplies by position on
        # top of an already-signed quantity, which made this fire on the
        # first bar of every short.
        if steps_in_pos <= p["entry_timeout"]:
            if cum_return <= -p["stop_loss"] * current_b / 2:
                return True, "timeout_loss"

        # Trailing stop, armed only by favourable movement.
        if p["trailing_activation"] > 0 and self.max_profit > p["trailing_activation"] * current_b:
            if cum_return < self.max_profit - p["trailing_distance"] * current_b:
                return True, "trailing_stop"

        if cum_return > p["take_profit"] * current_b:
            return True, "take_profit"

        if cum_return < -p["stop_loss"] * current_b:
            return True, "stop_loss"

        # Give back to the tunnel, once the trade has earned enough.
        if self.max_profit > p["trend_filter"] * current_b:
            if abs(current_z_score) < p["exit_z"]:
                return True, "exit_z"

        # Reverse draw: the move has turned against the position.
        if p["reverse_sensitivity"] > 0 and current_z_score < -0.5:
            if np.random.random() < p["reverse_sensitivity"]:
                return True, "reverse"

        if steps_in_pos > p["max_hold"]:
            return True, "max_hold"

        return False, ""

    def _exits_original(
        self, cum_return: float, current_z_score: float,
        current_b: float, steps_in_pos: int,
    ) -> tuple[bool, str]:
        """Exit rules transcribed from 8_backtest.py, defects included."""
        p = self.params

        if steps_in_pos <= p["entry_timeout"]:
            if self.position * cum_return <= -p["stop_loss"] * current_b / 2:
                return True, "timeout_loss"

        if p["trailing_activation"] > 0 and self.max_profit > p["trailing_activation"] * current_b:
            stop_z = self.max_profit - p["trailing_distance"] * current_b
            if self.position == 1 and cum_return < stop_z:
                return True, "trailing_stop"
            if self.position == -1 and -cum_return < stop_z:
                return True, "trailing_stop"

        if self.position == 1 and cum_return > p["take_profit"] * current_b:
            return True, "take_profit"
        if self.position == -1 and -cum_return > p["take_profit"] * current_b:
            return True, "take_profit"

        if self.position == 1 and cum_return < -p["stop_loss"] * current_b:
            return True, "stop_loss"
        if self.position == -1 and -cum_return < -p["stop_loss"] * current_b:
            return True, "stop_loss"

        if self.max_profit > p["trend_filter"] * current_b:
            if abs(current_z_score) < p["exit_z"]:
                return True, "exit_z"

        if p["reverse_sensitivity"] > 0:
            if self.position == 1 and current_z_score < -0.5:
                if np.random.random() < p["reverse_sensitivity"]:
                    return True, "reverse"
            if self.position == -1 and -current_z_score < -0.5:
                if np.random.random() < p["reverse_sensitivity"]:
                    return True, "reverse"

        if steps_in_pos > p["max_hold"]:
            return True, "max_hold"

        return False, ""

    def _record(
        self, bar: Bar, cum_return: float, reason: str, exit_time: datetime | None,
    ) -> dict[str, Any]:
        """Book a completed trade and move equity."""
        self._last_close = _Snapshot(
            position=self.position,
            entry_time=self.entry_time,
            entry_step=self.entry_step,
            entry_price=self.entry_price,
            max_profit=self.max_profit,
            returns=list(self._returns),
            equity=self.equity,
            equity_points=len(self.equity_curve),
        )

        steps_in_pos = self.step - self.entry_step
        trade = {
            "entry_time": self.entry_time,
            "exit_time": exit_time,
            "position": self.position,
            "entry_step": self.entry_step,
            "exit_step": self.step,
            "duration": steps_in_pos,
            "return": cum_return,
            "return_pct": cum_return * 100,
            "reason": reason,
            "entry_price": self.entry_price,
            "exit_price": bar.price,
        }
        self.trades.append(trade)

        self.equity += cum_return
        self.equity_curve.append(self.equity)

        logger.info("Closed %s after %d bar(s): %s, return %.6f",
                    "long" if self.position == 1 else "short",
                    steps_in_pos, reason, cum_return)

        self.position = 0
        self.entry_time = None
        self.entry_price = 0.0
        self.max_profit = 0.0
        self._returns = []
        return trade
