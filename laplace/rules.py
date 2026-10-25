"""Position state and the journal, driven by a pluggable strategy.

simulate_trading in 8_backtest.py is one long function doing three jobs at
once: walking an array, holding the position state, and applying the rules.
Live, the walking disappears — bars arrive one at a time, five minutes apart.
Only the middle job and the rules are wanted.

So the body of that loop lives here, fed one bar at a time, and the walking
stays outside it. scripts/backtest.py drives it over stored data and
laplace.live drives it over the socket, and both execute THE SAME rule code.

What is here and what is not
----------------------------
Here: the position, the trade journal, the equity curve, the time-gap and
market-closed guards, and the undo for a close the broker refused. All of it
is common to every strategy, so every strategy is measured the same way.

Not here: the decision. Whether to open and in which direction, and whether
to close and for what reason, come from laplace.strategy. Adding a rule means
adding a class there; this file does not change.

book_entry_bar — where the trade's return starts
------------------------------------------------
On (the original), profit is summed from y[entry_step] inclusive: the bar
whose move produced the signal. That bar closes before anyone knows a signal
exists. Measured on the research test set, it was 97.8 per cent of everything
the backtest booked. 7_cem.py does NOT do this — it measures from the entry
bar's close — so the parameters were optimised under one accounting and
applied under another. Off, the trade starts at the entry bar's close, which
is what a live run can actually capture.

It stays an engine switch rather than a strategy one because it is
accounting, not a decision: it changes what a trade is credited with, not
what the rule asks for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .core.gaps import detect_time_gap, MARKET_CLOSED_COL_INDEX, is_market_closed
from .strategy import EntryContext, ExitContext, Strategy

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
    """Position state, and the strategy that decides what changes it.

    Args:
        strategy: Supplies the entry and exit decisions. Everything about
            position, journal, equity and the guards stays here.
        gap_threshold_min: Minutes above which a break force-closes the
            position and blocks entry.
        book_entry_bar: Include the triggering bar's return in the trade.
            Accounting, not a decision, so it belongs to the engine.
    """

    strategy: Strategy
    gap_threshold_min: float = 10.0
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



    @staticmethod
    def _has_closed_column(window) -> bool:
        """True when the window is wide enough to carry market_closed."""
        return window.ndim == 2 and window.shape[1] > MARKET_CLOSED_COL_INDEX
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

        # The column only exists when the model was trained on the full
        # feature set. Narrower windows come out of create_windows already
        # day-bounded, every bar inside a block of market_closed == 0, so
        # there is nothing left for this guard to catch.
        if self._has_closed_column(bar.window) and is_market_closed(bar.window):
            self._prev_time = bar.time
            return self._force_close(result, bar, "market_closed")

        self._prev_time = bar.time

        current_b = bar.b
        result.signal_z = bar.y / current_b if current_b else 0.0

        # Parameters are read through .get so a strategy that does not use
        # a key simply leaves the engine's behaviour at its neutral value,
        # instead of raising on a rule that was never meant to have it.
        p = self.strategy.params
        if p.get("volatility_adapt"):
            current_b = current_b * p["risk_multiplier"]

        # Where the entry threshold sits in price terms, for anyone placing
        # the order by hand. Costs nothing and changes nothing.
        entry_z = p.get("entry_z", 0.0)
        result.long_trigger = bar.price * float(np.exp(entry_z * bar.b))
        result.short_trigger = bar.price * float(np.exp(-entry_z * bar.b))

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
        """Open when the strategy asks for a direction.

        No equity point is recorded on a flat step, matching simulate_trading,
        where the flat branch appends nothing.
        """
        direction = self.strategy.entry(EntryContext(bar=bar, signal_z=result.signal_z))
        if direction == 0:
            return result

        self.position = direction
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
        """Update position state, then ask the strategy whether to close."""
        self._returns.append(bar.y)
        cum_return = float(np.sum(self._returns)) * self.position
        current_z_score = cum_return / current_b if current_b else 0.0

        # Signed: only favourable movement counts, so a drawdown cannot arm
        # the trailing stop. Original: the absolute excursion, which can.
        # The engine holds max_profit because it is position state, but the
        # strategy says which convention its branches expect.
        excursion = cum_return if self.strategy.signed_excursion else abs(cum_return)
        if excursion > self.max_profit:
            self.max_profit = excursion

        steps_in_pos = self.step - self.entry_step

        exit_signal, exit_reason = self.strategy.exit(
            ExitContext(
                bar=bar,
                position=self.position,
                cum_return=cum_return,
                current_z_score=current_z_score,
                current_b=current_b,
                steps_in_pos=steps_in_pos,
                max_profit=self.max_profit,
            )
        )

        if exit_signal and exit_reason == "max_hold":
            cum_return -= self.strategy.params["time_decay"] * steps_in_pos

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