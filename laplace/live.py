"""
The bar-close loop.

Every five minutes, a few seconds after a bar closes, this asks the bridge
for the recent candles, rebuilds the features, predicts, and hands the bar to
the rule engine. Whether the engine's decision reaches the broker is the one
switch between a paper run and a live one; everything above that switch is
identical, so a paper run is a true rehearsal rather than a separate code
path.

Alignment with the backtest
---------------------------
simulate_trading walks arrays where X[i] is a window and y[i] the return it
predicts, so b_pred[i] comes from the twenty bars BEFORE y[i], and the
market-closed guard reads the last row of that window — the bar before the
target. The same pairing is reproduced here by running the research window
builder over the fetched frame and taking its final sample. The engine
therefore sees exactly what it would have seen in the backtest, which is the
point of the whole arrangement.

A second, forward prediction is made from the window ending on the last
closed bar. It has no target yet; it exists to state where the entry
threshold will sit during the bar now forming, which is the warning a person
placing orders by hand needs before the move happens.

Why candles and not the tick stream
-----------------------------------
The model was trained on the terminal's own five-minute bars, built from bid
prices. Bars assembled here from the tick feed would differ: the stream
begins when the process connects, loses ticks across any disconnection, and
knows nothing of what arrived beforehand. The feed is used to confirm the
market is moving and to record the quote at decision time, never to build
features.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from . import features as feature_builder
from . import model as model_io
from . import reader, windows
from .config import Config, add_bridge_to_path, load_instruments
from .journal import Journal
from .money import Instrument, PositionSizer
from .rules import Bar, Engine

logger = logging.getLogger(__name__)

TUNNEL_K = (1, 3)

# Minutes per timeframe name, for scheduling and for discarding the bar that
# is still forming.
TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


@dataclass
class Cycle:
    """One pass of the loop, kept for the journal and for testing."""

    bar_time: datetime
    price: float
    mu: float
    b: float
    forward_mu: float
    forward_b: float
    long_trigger: float
    short_trigger: float


class LiveRunner:
    """Drives the rule engine over live candles.

    Args:
        cfg: Validated configuration.
        execute: Send orders to the broker. False runs the whole pipeline and
            writes the journal without touching the account.
        max_cycles: Stop after this many bars; None runs until interrupted.
    """

    def __init__(self, cfg: Config, execute: bool, max_cycles: int | None = None) -> None:
        self._cfg = cfg
        self._execute = execute
        self._max_cycles = max_cycles

        self._step_minutes = TIMEFRAME_MINUTES.get(cfg.data.timeframe.upper(), 5)
        self._params = cfg.strategy.load()

        self._model = None
        self._engine: Engine | None = None
        self._journal: Journal | None = None
        self._transport = None
        self._fetcher = None
        self._client = None
        self._sizer: PositionSizer | None = None

        self._last_target: datetime | None = None
        self._ticket: int | None = None
        self._lots: float = 0.0
        self._clock_offset = timedelta(0)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Open the bridge and load everything that does not change."""
        add_bridge_to_path(self._cfg)

        from mt4bridge.config import load_config as load_bridge_config
        from mt4bridge.history import HistoryFetcher
        from mt4bridge.trading import TradingClient
        from mt4bridge.transport import Transport

        bridge_cfg = load_bridge_config(str(self._cfg.data.bridge_config_path))
        self._transport = Transport(bridge_cfg, commands=True)
        self._fetcher = HistoryFetcher(bridge_cfg, self._transport)
        self._client = TradingClient(bridge_cfg, self._transport)

        info = self._client.account_info()
        if info.server_time is not None:
            self._clock_offset = info.server_time - datetime.now()
            logger.info("Broker clock runs %+.1f h from this machine",
                        self._clock_offset.total_seconds() / 3600)

        self._model = model_io.load_model(self._cfg.experiment.resolved_model_path)

        specs = load_instruments()
        spec = specs.get(self._cfg.data.symbol)
        if spec is None:
            raise RuntimeError(
                f"{self._cfg.data.symbol} has no entry in config/instruments.json"
            )
        instrument = Instrument.from_mapping(self._cfg.data.symbol, spec)
        self._sizer = PositionSizer(
            instrument,
            enabled=self._cfg.money.enabled,
            risk_fraction=self._cfg.money.risk_fraction,
            fixed_lots=self._cfg.money.fixed_lots,
        )

        self._engine = Engine(
            params=self._params,
            gap_threshold_min=2 * self._step_minutes,
            signed_exits=self._cfg.rules.signed_exits,
            book_entry_bar=self._cfg.rules.book_entry_bar,
        )

        return info

    def _open_journal(self, account_info) -> None:
        mode = "live" if self._execute else "paper"
        self._journal = Journal(
            self._cfg.journal.runs_dir,
            f"{self._cfg.experiment.run_name}_{self._cfg.rules.label}",
            mode,
        )
        self._journal.write_manifest({
            "model": str(self._cfg.experiment.resolved_model_path),
            "model_parameters": model_io.describe_parameters(self._model),
            "experiment": {
                "dataset": self._cfg.experiment.dataset,
                "architecture": self._cfg.experiment.architecture,
                "activation": self._cfg.experiment.activation,
                "seed": self._cfg.experiment.seed,
            },
            "instrument": self._cfg.data.symbol,
            "timeframe": self._cfg.data.timeframe,
            "cem_params": self._params,
            "rules": {
                "signed_exits": self._cfg.rules.signed_exits,
                "book_entry_bar": self._cfg.rules.book_entry_bar,
            },
            "account": {
                "number": account_info.number,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "leverage": account_info.leverage,
                "server_time": account_info.server_time,
            },
            "money": {
                "enabled": self._cfg.money.enabled,
                "risk_fraction": self._cfg.money.risk_fraction,
                "fixed_lots": self._cfg.money.fixed_lots,
            },
            "config_source": str(self._cfg.source),
        })

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _broker_now(self) -> datetime:
        return datetime.now() + self._clock_offset

    def _sleep_to_next_bar(self) -> None:
        """Wait until the next bar has closed, plus the configured delay."""
        now = self._broker_now()
        minutes = self._step_minutes
        minute_block = (now.minute // minutes + 1) * minutes
        next_close = now.replace(second=0, microsecond=0, minute=0) + timedelta(minutes=minute_block)
        target = next_close + timedelta(seconds=self._cfg.runtime.bar_close_delay_sec)

        wait = (target - now).total_seconds()
        if wait > 0:
            logger.info("Next bar closes at %s; waiting %.0f s",
                        next_close.strftime("%H:%M"), wait)
            time.sleep(wait)

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def _fetch_frame(self):
        """Ask the bridge for recent candles and build the prepared frame."""
        end = self._broker_now()
        span_minutes = self._cfg.runtime.fetch_bars * self._step_minutes
        # Sessions close, so a span must be far longer than the bar count
        # suggests to come back with that many bars.
        start = end - timedelta(minutes=span_minutes * 3)

        candles = self._fetcher.fetch(
            self._cfg.data.symbol,
            self._cfg.data.timeframe,
            start.strftime("%Y.%m.%d %H:%M"),
            end.strftime("%Y.%m.%d %H:%M"),
            timeout=self._cfg.runtime.request_timeout_sec,
        )
        if not candles:
            raise RuntimeError("The terminal returned no candles")

        frame = reader.frame_from_candles(
            candles, self._cfg.data.symbol, self._cfg.data.price_column
        )
        frame = reader.drop_forming_bar(frame, end, self._step_minutes)

        return feature_builder.prepare(
            frame,
            window_size=self._cfg.windows.rolling_window,
            gap_threshold_days=self._cfg.data.gap_threshold_days,
            interval_minutes=self._step_minutes,
        )

    def _cycle(self) -> bool:
        """Process one bar. Returns False when there was nothing new."""
        prepared = self._fetch_frame()

        X, y, ts, mids, _, _ = windows.create_windows(
            prepared,
            lookback=self._cfg.windows.lookback,
            predict_step=self._cfg.windows.predict_step,
            zscore_clip=self._cfg.windows.zscore_clip,
        )
        if len(X) == 0:
            logger.warning("No complete window in the fetched data")
            return False

        target_time = _as_datetime(ts[-1])
        if self._last_target is not None and target_time <= self._last_target:
            logger.debug("No new bar since %s", self._last_target)
            return False
        self._last_target = target_time

        # The pairing simulate_trading uses: the window before the target.
        prediction = model_io.predict_one(self._model, X[-1])

        # The forward view: a window ending on the last closed bar, which has
        # no target yet, giving the threshold for the bar now forming.
        try:
            ahead = windows.last_window(
                prepared,
                lookback=self._cfg.windows.lookback,
                zscore_clip=self._cfg.windows.zscore_clip,
            )
            forward = model_io.predict_one(self._model, ahead.x)
            long_trigger, short_trigger = forward.trigger_prices(
                ahead.price, self._params["entry_z"]
            )
        except windows.WindowError as exc:
            logger.info("No forward window: %s", exc)
            forward = prediction
            long_trigger = short_trigger = 0.0

        bar = Bar(
            time=target_time,
            y=float(y[-1]),
            mu=prediction.mu,
            b=prediction.b,
            price=float(mids[-1]),
            window=X[-1],
        )

        result = self._engine.observe(bar)
        self._act(result, bar)
        self._record(bar, prediction, forward, long_trigger, short_trigger, result)
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _act(self, result, bar: Bar) -> None:
        """Send the engine's decision to the broker, when executing."""
        if result.action == "open":
            balance = self._balance()
            stop_distance = self._params["stop_loss"] * bar.b * bar.price
            sizing = self._sizer.size(balance, stop_distance)
            self._lots = sizing.lots
            logger.info("Size: %.2f lots (%s)", sizing.lots, sizing.reason)

            if not self._execute:
                return
            execution = (self._client.buy if result.position == 1 else self._client.sell)(
                symbol=self._cfg.data.symbol, lots=sizing.lots
            )
            self._ticket = execution.ticket
            logger.info("Order filled: ticket %s at %s",
                        execution.ticket, execution.open_price)

        elif result.action == "close":
            if not self._execute or self._ticket is None:
                return
            try:
                self._client.close(self._ticket)
            except Exception as exc:
                # The position is still open at the broker. Letting the
                # engine believe otherwise would have it open a second one
                # on the next signal, and the journal would stop describing
                # the account.
                logger.error("Close refused for ticket %s: %s", self._ticket, exc)
                self._engine.undo_last_close()
                if self._journal is not None:
                    self._journal.note(
                        f"close refused for ticket {self._ticket}: {exc}; "
                        f"the engine still holds the position"
                    )
                return
            logger.info("Closed ticket %s", self._ticket)
            self._ticket = None

    def _balance(self) -> float:
        """Current balance, read from the account every time it is needed."""
        try:
            return float(self._client.account_info().balance)
        except Exception as exc:
            logger.warning("Could not read the balance: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def _record(self, bar, prediction, forward, long_trigger, short_trigger, result) -> None:
        k1 = prediction.tunnel(1)
        k3 = prediction.tunnel(3)
        self._journal.decision({
            "bar_time": bar.time,
            "price": bar.price,
            "mu": prediction.mu,
            "b": prediction.b,
            "signal_z": result.signal_z,
            "k1_lower": k1[0], "k1_upper": k1[1],
            "k3_lower": k3[0], "k3_upper": k3[1],
            "long_trigger": long_trigger,
            "short_trigger": short_trigger,
            "action": result.action,
            "reason": result.reason,
            "position": result.position,
            "cum_return": result.cum_return,
            "equity": result.equity,
            "lots": self._lots if result.position else "",
            "blocked": result.blocked,
            "note": f"forward b={forward.b:.8f}",
        })
        self._journal.equity(bar.time, result.equity)

        if result.trade is not None:
            row = dict(result.trade)
            row["lots"] = self._lots
            row["profit_currency"] = ""
            self._journal.trade(row)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run until interrupted or until max_cycles bars have passed."""
        account = self._connect()
        self._open_journal(account)

        print(self._cfg.describe())
        print(f"mode       : {'LIVE — orders will be sent' if self._execute else 'paper'}")
        print(f"journal    : {self._journal.dir}")

        cycles = 0
        try:
            while self._max_cycles is None or cycles < self._max_cycles:
                self._sleep_to_next_bar()
                try:
                    if self._cycle():
                        cycles += 1
                except Exception as exc:
                    logger.exception("Cycle failed: %s", exc)
                    self._journal.note(f"cycle failed: {exc}")
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self._shutdown()
        return cycles

    def _shutdown(self) -> None:
        """Report what is still open and close the journal."""
        if self._engine is not None and self._engine.position != 0:
            message = (
                f"Stopped holding a {'long' if self._engine.position == 1 else 'short'} "
                f"position opened {self._engine.entry_time}"
                + (f", ticket {self._ticket}" if self._ticket else "")
            )
            logger.warning(message)
            if self._journal is not None:
                self._journal.note(message)

        if self._journal is not None:
            self._journal.summarise({
                "bars_processed": self._engine.step if self._engine else 0,
                "trades": len(self._engine.trades) if self._engine else 0,
                "equity": self._engine.equity if self._engine else 0.0,
                "open_position": self._engine.position if self._engine else 0,
                "open_ticket": self._ticket,
            })
            self._journal.close()

        if self._transport is not None:
            self._transport.close()


def _as_datetime(value: Any) -> datetime:
    """Convert a numpy datetime64 or a datetime to a plain datetime."""
    if isinstance(value, datetime):
        return value
    return np.datetime64(value, "s").astype("datetime64[s]").astype(datetime)
