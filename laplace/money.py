"""
Position size and profit in account currency.

The research backtest reports returns as sums of log returns multiplied by a
leverage constant:

    total_return = np.sum(returns) * leverage * 100
    max_drawdown = np.nanmax(drawdown) * 100 * leverage

with initial_capital in the signature and unused. That is a percentage of an
unstated capital, at a fixed position size, with no compounding, and with
leverage applied after the ratio was taken — which is why a drawdown there
can exceed one hundred per cent. It is a comparative number, not money.

This module sits beside the rules and never inside them. It takes a decision
and the account balance and returns a lot size; separately it converts a
price move into the account currency. The rules do not know it exists, so
turning it on cannot change which trades are taken.

Compounding arrives on its own once sizing is on, because the size is
computed from the balance at the moment of entry. Kelly, when it comes, is a
different risk_fraction and nothing else.

Until config/instruments.json carries a tick value for the instrument, sizing
stays off and lots are fixed. Those numbers come from the terminal: Market
Watch, right-click the symbol, Specification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class MoneyError(ValueError):
    """The instrument specification is missing a value sizing needs."""


@dataclass(frozen=True)
class Instrument:
    """Contract specification, as the terminal reports it.

    Attributes:
        symbol: Name as the broker spells it.
        point: Smallest price increment, e.g. 0.01 for US.100+ and 0.00001
            for EURUSD+. Measured, not assumed: tools/check_bridge.py in the
            bridge sets a stop of N points and reads back the applied
            distance.
        contract_size: Units of the instrument in one lot.
        tick_value_per_lot: Account currency gained by one point of move on
            one lot. This is what converts a price move into money.
        min_lot: Smallest tradeable size.
        lot_step: Granularity above the minimum.
        currency: Account currency the tick value is quoted in.
    """

    symbol: str
    point: float
    min_lot: float
    lot_step: float
    currency: str
    contract_size: float | None = None
    tick_value_per_lot: float | None = None

    @classmethod
    def from_mapping(cls, symbol: str, data: dict[str, Any]) -> "Instrument":
        return cls(
            symbol=symbol,
            point=float(data["point"]),
            min_lot=float(data.get("min_lot", 0.01)),
            lot_step=float(data.get("lot_step", 0.01)),
            currency=str(data.get("currency", "USD")),
            contract_size=(float(data["contract_size"])
                           if data.get("contract_size") is not None else None),
            tick_value_per_lot=(float(data["tick_value_per_lot"])
                                if data.get("tick_value_per_lot") is not None else None),
        )

    @property
    def is_priceable(self) -> bool:
        """Whether a price move can be converted into account currency."""
        return self.tick_value_per_lot is not None

    def money_per_price_unit(self, lots: float) -> float:
        """Account currency per one unit of price, for a given size.

        Raises:
            MoneyError: The tick value has not been filled in.
        """
        if self.tick_value_per_lot is None:
            raise MoneyError(
                f"{self.symbol} has no tick_value_per_lot in instruments.json; "
                f"read it from the terminal's symbol specification"
            )
        return (self.tick_value_per_lot / self.point) * lots


@dataclass(frozen=True)
class Sizing:
    """A sizing decision and why it came out that way."""

    lots: float
    reason: str


class PositionSizer:
    """Turns a balance and a stop distance into a number of lots.

    Args:
        instrument: Contract specification.
        enabled: When false, every call returns fixed_lots.
        risk_fraction: Fraction of equity to put behind one trade.
        fixed_lots: Size used while sizing is off.
    """

    def __init__(
        self,
        instrument: Instrument,
        enabled: bool,
        risk_fraction: float,
        fixed_lots: float,
    ) -> None:
        self._instrument = instrument
        self._enabled = enabled
        self._risk_fraction = risk_fraction
        self._fixed_lots = fixed_lots

    def size(self, balance: float, stop_distance_price: float) -> Sizing:
        """Lots for one entry.

        Args:
            balance: Account balance or equity, read from the bridge. Never
                from configuration: a demo account that gets reset arrives
                with a different number and nothing should carry over.
            stop_distance_price: Distance from entry to stop, in price units.
                With the strategy's stop_loss parameter expressed in units of
                b, that is stop_loss * b * price.

        Returns:
            A size, rounded down to the broker's lot step.
        """
        if not self._enabled:
            return Sizing(self._fixed_lots, "sizing off, fixed size")

        if not self._instrument.is_priceable:
            return Sizing(
                self._fixed_lots,
                f"{self._instrument.symbol} has no tick value, fixed size",
            )

        if stop_distance_price <= 0:
            return Sizing(self._fixed_lots, "stop distance not positive, fixed size")

        risk_money = balance * self._risk_fraction
        money_per_unit_one_lot = self._instrument.money_per_price_unit(1.0)
        raw_lots = risk_money / (stop_distance_price * money_per_unit_one_lot)

        lots = self._round_down(raw_lots)

        if lots < self._instrument.min_lot:
            return Sizing(
                self._instrument.min_lot,
                f"risk of {risk_money:.2f} implies {raw_lots:.4f} lots, "
                f"below the minimum",
            )

        return Sizing(lots, f"risking {risk_money:.2f} over {stop_distance_price:.5f}")

    def _round_down(self, lots: float) -> float:
        """Round to the broker's lot step, downward."""
        step = self._instrument.lot_step
        return math.floor(lots / step) * step


def profit_in_currency(
    instrument: Instrument,
    entry_price: float,
    exit_price: float,
    position: int,
    lots: float,
) -> float:
    """Convert a closed position into account currency.

    Args:
        instrument: Contract specification.
        entry_price: Fill price.
        exit_price: Closing price.
        position: 1 for long, -1 for short.
        lots: Size traded.

    Raises:
        MoneyError: The tick value has not been filled in.
    """
    move = (exit_price - entry_price) * position
    return move * instrument.money_per_price_unit(lots)


def log_return_to_price(price: float, log_return: float) -> float:
    """Price a log return corresponds to, starting from a given price."""
    return price * math.exp(log_return)
