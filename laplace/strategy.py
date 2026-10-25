"""The decision half of the rule engine, behind one interface.

Engine owns the position, the journal and the equity curve. A strategy
owns the decision: whether to open and in which direction, and whether
to close and for what reason. Nothing in this module books a return,
moves equity or knows what a trade costs.

The rule moved here is the thirteen-parameter z-score rule from
8_backtest.py, transcribed without changing a sign. Both exit bodies are
kept side by side behind signed_exits, exactly as rules.py held them, so
the four measured variants stay reproducible until the triggering-bar
question is settled.

Adding a strategy means adding a class here. Engine does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # import only for type checking, so rules.py can import us
    from .rules import Bar


@dataclass(frozen=True)
class EntryContext:
    """What a strategy sees when the engine is flat."""

    bar: "Bar"
    signal_z: float


@dataclass(frozen=True)
class ExitContext:
    """What a strategy sees while a position is open.

    Every field is computed by the engine before the call, in the order
    simulate_trading computed it, so a strategy never walks the history
    itself and two strategies are always asked the same question.
    """

    bar: "Bar"
    position: int
    cum_return: float
    current_z_score: float
    current_b: float
    steps_in_pos: int
    max_profit: float


class Strategy(Protocol):
    """The two questions the engine asks, and nothing else."""

    name: str
    params: dict[str, float]

    @property
    def signed_excursion(self) -> bool:
        """Whether max_profit should track favourable movement only.

        The engine maintains max_profit because it is position state, but
        only the rule knows which convention its exit branches expect.
        """

    def entry(self, context: EntryContext) -> int:
        """Return +1 to go long, -1 to go short, 0 to stay flat."""

    def exit(self, context: ExitContext) -> tuple[bool, str]:
        """Return (True, reason) to close, (False, '') to hold."""


@dataclass
class ZScoreRule:
    """The original rule: enter on a z-score, leave on one of seven branches.

    Args:
        params: The thirteen CEM parameters.
        signed_exits: Treat cum_return as signed profit in every branch.
            Off reproduces 8_backtest.py including its sign defects.
    """

    params: dict[str, float] = field(default_factory=dict)
    signed_exits: bool = False
    name: str = "zscore"

    @property
    def signed_excursion(self) -> bool:
        return self.signed_exits

    def entry(self, context: EntryContext) -> int:
        """Open once the signal clears entry_z, in the signal's direction."""
        if abs(context.signal_z) <= self.params["entry_z"]:
            return 0
        return int(np.sign(context.signal_z))

    def exit(self, context: ExitContext) -> tuple[bool, str]:
        if self.signed_exits:
            return self._exits_signed(context)
        return self._exits_original(context)

    def _exits_signed(self, c: ExitContext) -> tuple[bool, str]:
        """Exit rules with cum_return read as signed profit, both directions.

        A profit is positive and a loss negative for a long and for a short
        alike, so one branch serves both and the thresholds mean what their
        names say.
        """
        p = self.params

        # Timeout on an early loss. The original multiplies by position on
        # top of an already-signed quantity, which made this fire on the
        # first bar of every short.
        if c.steps_in_pos <= p["entry_timeout"]:
            if c.cum_return <= -p["stop_loss"] * c.current_b / 2:
                return True, "timeout_loss"

        # Trailing stop, armed only by favourable movement.
        if p["trailing_activation"] > 0 and c.max_profit > p["trailing_activation"] * c.current_b:
            if c.cum_return < c.max_profit - p["trailing_distance"] * c.current_b:
                return True, "trailing_stop"

        if c.cum_return > p["take_profit"] * c.current_b:
            return True, "take_profit"

        if c.cum_return < -p["stop_loss"] * c.current_b:
            return True, "stop_loss"

        # Give back to the tunnel, once the trade has earned enough.
        if c.max_profit > p["trend_filter"] * c.current_b:
            if abs(c.current_z_score) < p["exit_z"]:
                return True, "exit_z"

        # Reverse draw: the move has turned against the position.
        if p["reverse_sensitivity"] > 0 and c.current_z_score < -0.5:
            if np.random.random() < p["reverse_sensitivity"]:
                return True, "reverse"

        if c.steps_in_pos > p["max_hold"]:
            return True, "max_hold"

        return False, ""

    def _exits_original(self, c: ExitContext) -> tuple[bool, str]:
        """Exit rules transcribed from 8_backtest.py, defects included."""
        p = self.params

        if c.steps_in_pos <= p["entry_timeout"]:
            if c.position * c.cum_return <= -p["stop_loss"] * c.current_b / 2:
                return True, "timeout_loss"

        if p["trailing_activation"] > 0 and c.max_profit > p["trailing_activation"] * c.current_b:
            stop_z = c.max_profit - p["trailing_distance"] * c.current_b
            if c.position == 1 and c.cum_return < stop_z:
                return True, "trailing_stop"
            if c.position == -1 and -c.cum_return < stop_z:
                return True, "trailing_stop"

        if c.position == 1 and c.cum_return > p["take_profit"] * c.current_b:
            return True, "take_profit"
        if c.position == -1 and -c.cum_return > p["take_profit"] * c.current_b:
            return True, "take_profit"

        if c.position == 1 and c.cum_return < -p["stop_loss"] * c.current_b:
            return True, "stop_loss"
        if c.position == -1 and -c.cum_return < -p["stop_loss"] * c.current_b:
            return True, "stop_loss"

        if c.max_profit > p["trend_filter"] * c.current_b:
            if abs(c.current_z_score) < p["exit_z"]:
                return True, "exit_z"

        if p["reverse_sensitivity"] > 0:
            if c.position == 1 and c.current_z_score < -0.5:
                if np.random.random() < p["reverse_sensitivity"]:
                    return True, "reverse"
            if c.position == -1 and -c.current_z_score < -0.5:
                if np.random.random() < p["reverse_sensitivity"]:
                    return True, "reverse"

        if c.steps_in_pos > p["max_hold"]:
            return True, "max_hold"

        return False, ""