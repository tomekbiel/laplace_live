"""Order execution for the live signal loop.

The rule decides direction; this module decides nothing. It brings the
account in line with the rule's state once per closed bar, holding at most
one position and treating the terminal as the only source of truth about
exposure, since the bridge emits no events.

Exits go out as market closes rather than as broker-side stops: the
bridge's modify path expresses a stop as a point distance from the open
price, which cannot describe a stop sitting in profit, and the point unit
has not been verified against this broker. A fixed disaster stop is
available for unattended runs and is never moved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Everything the executor is allowed to decide about size and safety.

    Attributes:
        lots: Volume per position.
        symbol: Instrument; None falls back to the bridge configuration.
        comment: Free text carried with the order, for the account history.
        min_free_margin: Opening is skipped below this free margin, so a
            drawdown narrows exposure rather than ending in a margin call.
        disaster_stop_points: Broker-side stop distance sent once at open
            and never modified; 0 sends none. The unit is whatever the
            bridge's modify path uses, so verify it by hand first.
        dry_run: When set, every decision is logged and nothing is sent.
    """

    lots: float = 0.10
    symbol: str | None = None
    comment: str = "laplace-k20"
    min_free_margin: float = 2000.0
    disaster_stop_points: float = 0.0
    dry_run: bool = True


class Executor:
    """Keeps the account aligned with the rule's state, one bar at a time."""

    def __init__(self, client, config: ExecutionConfig) -> None:
        self._client = client
        self._cfg = config

    def step(self, state, price: float, stamp) -> str:
        """Act on the rule's state for the bar that has just closed.

        Args:
            state: None when the rule is flat, otherwise the open position's
                record, carrying 'side' at least.
            price: Close of the bar, used only for logging.
            stamp: Timestamp of the bar, used only for logging.

        Returns:
            A short description of what was done, for the live log.
        """
        positions = self._client.own_positions()

        if len(positions) > 1:
            tickets = sorted(positions)
            logger.error("More than one own position: %s, closing all", tickets)
            return self._close_all(positions, "duplicate")

        wanted = 0 if state is None else int(state["side"])
        held = 0
        ticket = None
        if positions:
            ticket, position = next(iter(positions.items()))
            held = 1 if position.is_long else -1 if position.is_short else 0

        if wanted == held:
            return "hold" if held else "flat"

        if held and wanted == 0:
            return self._close(ticket, "signal")

        if held and wanted != held:
            self._close(ticket, "reverse")
            return self._open(wanted, price, stamp)

        return self._open(wanted, price, stamp)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open(self, side: int, price: float, stamp) -> str:
        """Open one position in the given direction, margin permitting."""
        label = "BUY" if side > 0 else "SELL"

        account = self._client.account_info()
        if account.free_margin < self._cfg.min_free_margin:
            logger.warning(
                "Free margin %.2f below the floor %.2f, skipping %s",
                account.free_margin, self._cfg.min_free_margin, label,
            )
            return "skipped_margin"

        if self._cfg.dry_run:
            logger.info("DRY RUN would %s %.2f lots at %.2f (%s)",
                        label, self._cfg.lots, price, stamp)
            return f"dry_{label.lower()}"

        send = self._client.buy if side > 0 else self._client.sell
        result = send(
            symbol=self._cfg.symbol,
            lots=self._cfg.lots,
            stop_loss=0.0,
            take_profit=0.0,
            comment=self._cfg.comment,
        )
        logger.info("Opened %s ticket %s at %s",
                    label, result.ticket, result.open_price)

        if self._cfg.disaster_stop_points > 0:
            self._client.modify(result.ticket,
                                stop_loss=self._cfg.disaster_stop_points,
                                take_profit=0.0)
            logger.info("Disaster stop set at %.0f points from open",
                        self._cfg.disaster_stop_points)

        return f"opened_{label.lower()}_{result.ticket}"

    def _close(self, ticket: int, why: str) -> str:
        """Close one position at market."""
        if self._cfg.dry_run:
            logger.info("DRY RUN would close %s (%s)", ticket, why)
            return f"dry_close_{why}"
        self._client.close(ticket)
        logger.info("Closed %s (%s)", ticket, why)
        return f"closed_{why}_{ticket}"

    def _close_all(self, positions, why: str) -> str:
        """Close every position this bridge is responsible for."""
        if self._cfg.dry_run:
            logger.info("DRY RUN would close all own positions (%s)", why)
            return f"dry_close_all_{why}"
        self._client.close_own()
        logger.warning("Closed all own positions (%s)", why)
        return f"closed_all_{why}"

    def flatten(self, why: str = "shutdown") -> str:
        """Close everything; called when the loop stops."""
        positions = self._client.own_positions()
        if not positions:
            return "already_flat"
        return self._close_all(positions, why)