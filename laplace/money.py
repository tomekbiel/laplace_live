"""
Profit the way the terminal computes it, in index points and account currency.

Everything upstream of this module works in log returns, which is right for
statistics and wrong for money. A log return is a property of a price
series; a profit is a property of a position in a contract, and the two
differ by the contract specification, the lot size and the margin the
broker takes.

The terminal's arithmetic, which this reproduces:

    points  = exit - entry                      signed by the direction
    profit  = points / tick_size * tick_value * lots
    margin  = price * contract_size * lots / leverage

The second line is the universal form and the reason this module asks for
tick_size and tick_value rather than a value per point: those two fields
are what MT4 publishes for every symbol, and dividing one by the other
gives the value per point without anyone having to know whether an index
CFD is quoted per unit or per twenty.

The specification is read from config/instruments.json and never guessed.
Find it in the terminal under Market Watch, right click the symbol,
Specification: Contract size, Tick size, Tick value, Minimum volume,
Volume step, Leverage. A guessed contract size is off by whatever factor
the broker chose, and every profit figure downstream is off by the same
factor without anything looking wrong.

The drawdown here is computed on an equity curve denominated in the
account currency and starting from a real balance. The inherited metric
divided by a running peak that could pass through zero, which is why a
backtest in this project once reported a drawdown of sixty-two thousand
per cent. A balance cannot pass through zero without the account being
closed, so this ratio stays finite and means what it says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


class SpecError(ValueError):
    """The instrument specification is missing a field that has no default."""


@dataclass(frozen=True)
class ContractSpec:
    """What the broker says this instrument is.

    Attributes:
        symbol: As the broker spells it.
        tick_size: Smallest price change, the terminal's Point.
        tick_value: Account currency gained per tick, per one lot.
        contract_size: Units per lot. Used for notional and margin only;
            profit comes from tick_value, which already folds it in.
        min_lot: Smallest tradeable volume.
        lot_step: Volume granularity.
        leverage: Account leverage for this instrument, e.g. 20 for 5%.
        currency: Account currency, for labelling.
    """

    symbol: str
    tick_size: float
    tick_value: float
    contract_size: float
    min_lot: float
    lot_step: float
    leverage: float
    currency: str = "USD"

    @property
    def value_per_point(self) -> float:
        """Account currency per one index point, per lot."""
        return self.tick_value / self.tick_size

    @classmethod
    def from_mapping(cls, symbol: str, data: dict[str, Any]) -> "ContractSpec":
        """Build from one entry of config/instruments.json.

        The field names are the terminal's, as instruments.json spells them:
        `point` is the tick size and `tick_value_per_lot` is what one tick
        of it is worth on one lot.

        tick_value_per_lot may be left null when the instrument is quoted
        in the account currency, which covers a USD index on a USD account
        and a USD-quoted currency pair. It is then contract_size * point,
        which is the same identity that makes a standard lot of EURUSD
        worth ten dollars a pip. An instrument quoted in anything else
        needs the real figure, because the conversion rate moves.
        """
        if data.get("point") in (None, ""):
            raise SpecError(f"{symbol}: instruments.json is missing `point`.")
        tick_size = float(data["point"])

        contract_size = data.get("contract_size")
        tick_value = data.get("tick_value_per_lot")

        if tick_value in (None, ""):
            if contract_size in (None, ""):
                raise SpecError(
                    f"{symbol}: give either `tick_value_per_lot` or "
                    "`contract_size`. Market Watch, right click the symbol, "
                    "Specification, Contract size."
                )
            tick_value = float(contract_size) * tick_size

        if contract_size in (None, ""):
            # Only notional and margin need it; profit does not.
            contract_size = float(tick_value) / tick_size

        if data.get("leverage") in (None, ""):
            raise SpecError(
                f"{symbol}: instruments.json is missing `leverage`. Open a "
                "position and divide price * contract_size * lots by the "
                "margin the terminal reserved."
            )

        return cls(
            symbol=symbol,
            tick_size=tick_size,
            tick_value=float(tick_value),
            contract_size=float(contract_size),
            min_lot=float(data.get("min_lot", 0.01)),
            lot_step=float(data.get("lot_step", 0.01)),
            leverage=float(data["leverage"]),
            currency=str(data.get("currency", "USD")),
        )

# ----------------------------------------------------------------------
# Conversions
# ----------------------------------------------------------------------

def points_from_log(log_return, price):
    """Index points moved, from a log return at a given price.

    price * (exp(r) - 1), not price * r. The difference is second order
    and invisible at one basis point; at the size of a macro bar it is
    not, and the terminal does not approximate.
    """
    return np.asarray(price, dtype=float) * np.expm1(np.asarray(log_return, dtype=float))


def profit(points, lots: float, spec: ContractSpec):
    """Account currency from a signed move in index points."""
    return np.asarray(points, dtype=float) * spec.value_per_point * lots


def notional(price, lots: float, spec: ContractSpec):
    """Face value of the position."""
    return np.asarray(price, dtype=float) * spec.contract_size * lots


def margin(price, lots: float, spec: ContractSpec):
    """What the broker locks up for the position."""
    return notional(price, lots, spec) / spec.leverage


# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------

def round_lots(lots: float, spec: ContractSpec) -> float:
    """Snap a volume down to the broker's grid; zero when below the minimum.

    Rounded down rather than to nearest, because a volume rounded up can
    be refused for margin at the moment of the order, which in a live run
    turns a trade into an error rather than a smaller trade.
    """
    if lots < spec.min_lot:
        return 0.0
    steps = math.floor((lots - spec.min_lot) / spec.lot_step + 1e-9)
    return round(spec.min_lot + steps * spec.lot_step, 8)


def lots_for_risk(balance: float, risk_fraction: float, stop_points: float,
                  spec: ContractSpec) -> float:
    """Volume such that the stop costs `risk_fraction` of the balance."""
    if stop_points <= 0 or risk_fraction <= 0:
        return 0.0
    loss_per_lot = abs(stop_points) * spec.value_per_point
    if loss_per_lot <= 0:
        return 0.0
    return round_lots(balance * risk_fraction / loss_per_lot, spec)


def max_lots_for_margin(balance: float, price: float, spec: ContractSpec,
                        usage: float = 0.5) -> float:
    """The largest volume whose margin stays within `usage` of the balance.

    The demo account in this project holds a little over ten thousand, and
    a third of a lot on an index quoted near thirty thousand can already
    exceed it, depending on the contract size. Sizing that ignores margin
    produces a backtest the account could not have taken.
    """
    per_lot = margin(price, 1.0, spec)
    if per_lot <= 0:
        return 0.0
    return round_lots(balance * usage / per_lot, spec)


# ----------------------------------------------------------------------
# Account simulation
# ----------------------------------------------------------------------

def run_account(points: Iterable[float], prices: Iterable[float],
                spec: ContractSpec, balance: float = 10000.0,
                lots: float = 0.30, risk_fraction: float | None = None,
                stops: Iterable[float] | None = None,
                margin_usage: float = 0.5) -> dict:
    """Walk a sequence of trades through an account and report what happened.

    Args:
        points: Signed move per trade, in index points.
        prices: Entry price per trade, for margin and sizing.
        lots: Fixed volume, used when risk_fraction is None. The terminal's
            order box in this project sits at 0.30.
        risk_fraction: When set, each trade is sized so its stop costs this
            fraction of the running balance, and `stops` supplies the stop
            distance in points.
        margin_usage: Volume is capped so margin stays within this fraction
            of the balance; a trade that cannot be afforded is skipped and
            counted, because a backtest that takes positions the account
            could not open is measuring a different account.

    Returns per-trade currency results, the equity curve and the summary.
    """
    points = np.asarray(list(points), dtype=float)
    prices = np.asarray(list(prices), dtype=float)
    stop_array = (np.asarray(list(stops), dtype=float)
                  if stops is not None else None)

    equity = float(balance)
    curve = [equity]
    results: list[float] = []
    volumes: list[float] = []
    skipped = 0

    for i, move in enumerate(points):
        price = prices[i] if i < len(prices) else prices[-1]
        if risk_fraction is not None and stop_array is not None:
            size = lots_for_risk(equity, risk_fraction, stop_array[i], spec)
        else:
            size = round_lots(lots, spec)
        affordable = max_lots_for_margin(equity, price, spec, margin_usage)
        size = min(size, affordable)
        if size <= 0:
            skipped += 1
            results.append(0.0)
            volumes.append(0.0)
            curve.append(equity)
            continue
        gain = float(profit(move, size, spec))
        equity += gain
        results.append(gain)
        volumes.append(size)
        curve.append(equity)
        if equity <= 0:
            break

    return {
        "profit": np.array(results),
        "lots": np.array(volumes),
        "equity": np.array(curve),
        "skipped": skipped,
        "metrics": account_metrics(np.array(results), np.array(curve),
                                   balance, spec),
    }


def account_metrics(results: np.ndarray, curve: np.ndarray,
                    balance: float, spec: ContractSpec) -> dict:
    """Summary in the account currency, with a drawdown that cannot explode."""
    traded = results[results != 0.0]
    if traded.size == 0:
        return {"trades": 0}

    wins = traded[traded > 0]
    losses = traded[traded < 0]
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak

    return {
        "trades": int(traded.size),
        "currency": spec.currency,
        "start_balance": float(balance),
        "end_balance": float(curve[-1]),
        "total_profit": float(curve[-1] - balance),
        "return_pct": float((curve[-1] / balance - 1.0) * 100.0),
        "win_rate": float(100.0 * wins.size / traded.size),
        "avg_trade": float(traded.mean()),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if losses.size and losses.sum() != 0 else float("inf")),
        "max_drawdown_pct": float(drawdown.max() * 100.0),
        "max_drawdown_abs": float((peak - curve).max()),
        "largest_win": float(traded.max()),
        "largest_loss": float(traded.min()),
    }