"""
Live signals, for copying by hand. Orders are sent only with --live.

The rule is not reimplemented here. Every cycle this rebuilds the bars of
the current session and runs the same `simulate` the backtest runs, which
is possible because a trade never crosses a session boundary: replaying
the session from its first bar reproduces exactly the decisions the
backtest would have made, including the one in force right now.

That removes the failure this project has already had once, where a live
loop and a backtest carried two copies of the same rule and drifted apart.
There is one copy.

What it prints: one line per five-minute bar, and a marked line whenever
the state changes, so an open or a close is visible at a glance while the
terminal is on another screen. Everything is written to a CSV in the same
shape as the backtest's, so the two files can be compared directly after
a session.

The entry threshold is frozen. It was fitted as a quantile on the training
slice and recomputing it from live data would make the rule drift toward
whatever the last few days did.

Usage:
    python scripts/run_signal.py
    python scripts/run_signal.py --window 20 --level 0.938 --max-hold 48
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from laplace import features as feature_builder
from laplace import model as model_io
from laplace import oscillators as osc
from laplace import reader, windows
from laplace.config import load_config, setup_logging
from laplace.signals import simulate

# The settled configuration. Changing any of these makes it a different
# rule, and the figures measured on history no longer describe it.
RULE = {
    "oscillator": "stochastic_k",
    "window": 20,
    "level": 0.900,
    "momentum": False,
    "exit_rule": "opposite",
    "max_hold": 48,
    "trail_after_signal": True,
    "trail_start": 88.0,
    "trail_min": 20.0,
    "trail_tau": 24,
    "hold_after_arm": 48,
}


def channel_edges(closes, window: int, level: float) -> tuple[float, float]:
    """Prices at which %K would reach -level and +level.

    Inverting K = 2(p - L)/(H - L) - 1 gives the two prices that trigger
    the rule, so the live display can name them before the bar closes.
    """
    recent = closes[-window:]
    low, high = float(recent.min()), float(recent.max())
    span = high - low
    if span <= 0.0:
        return float("nan"), float("nan")
    lower = low + (1.0 - level) * span / 2.0
    upper = low + (1.0 + level) * span / 2.0
    return lower, upper


def stop_price(side: int, entry_price: float, best_points: float,
               width: float) -> float:
    """Price at which the armed trailing stop closes the position."""
    if not np.isfinite(width):
        return float("nan")
    return entry_price + side * (best_points - width)


def ma_slope_points(closes, window: int = 20) -> float:
    """Slope of the moving average over the last two bars, in points."""
    if len(closes) < window + 1:
        return float("nan")
    now = float(closes[-window:].mean())
    before = float(closes[-window - 1:-1].mean())
    return now - before


def load_session(cfg, model, tail: int):
    """Bars of the most recent open-market block, as the backtest sees them."""
    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=tail
    )
    prepared = feature_builder.prepare(
        frame,
        window_size=model.rolling_window or cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )
    X, y, ts, mids, block_ids, _ = windows.create_windows(
        prepared,
        lookback=model.lookback or cfg.windows.lookback,
        predict_step=model.predict_step or cfg.windows.predict_step,
        feature_cols=model.feature_columns or None,
        zscore_clip=model.zscore_clip,
    )
    block_ids = np.asarray(block_ids)
    if len(block_ids) == 0:
        return None
    return {
        "y": np.asarray(y, dtype=float),
        "ts": pd.DatetimeIndex(pd.to_datetime(ts)),
        "mids": np.asarray(mids, dtype=float),
        "blocks": block_ids,
        # The current session, as a mask rather than a slice. Slicing here
        # was the defect: the oscillator then starts each session from a
        # cold window, while the backtest carries the previous session's
        # closes into it, and the two stop being the same rule.
        "current": block_ids == block_ids[-1],
    }


def state_of(session, level: float, window: int, max_hold: int) -> dict:
    """Run the rule over the whole tail, then report the current session."""
    reading = osc.stochastic_k(price=session["mids"], window=window)
    trades = simulate(
        reading, session["y"], session["blocks"], level,
        price=session["mids"], contrarian=not RULE["momentum"],
        exit_rule=RULE["exit_rule"], max_hold=max_hold,
        trail_after_signal=RULE["trail_after_signal"],
        trail_start=RULE["trail_start"],
        trail_min=RULE["trail_min"],
        trail_tau=RULE["trail_tau"],
        hold_after_arm=RULE["hold_after_arm"],
    )
    current = session["current"]
    keep = [k for k in range(len(trades["ret"]))
            if current[int(trades["entry"][k])]]
    # The threshold is a price, known as soon as the previous bar closes.
    # %K measures position inside the extremes of `window` bars, so with
    # the earlier window - 1 closes fixed, the price that first reaches
    # the level is arithmetic, and it holds for the whole next bar. A
    # pending order at it enters at the moment the condition is met
    # instead of up to five minutes later.
    recent = session["mids"][-(window - 1):]
    recent = recent[np.isfinite(recent)]
    triggers = {"long": float("nan"), "short": float("nan")}
    if recent.size:
        low, high = float(recent.min()), float(recent.max())
        span = high - low
        triggers = {
            "long": low + 0.5 * (1.0 + level) * span,
            "short": low + 0.5 * (1.0 - level) * span,
        }
    return {
        "reading": float(reading[-1]) if np.isfinite(reading[-1]) else float("nan"),
        "price": float(session["mids"][-1]),
        "time": session["ts"][-1],
        "closed": len(keep),
        "triggers": triggers,
        "trades": [
            {
                "entry_bar": int(trades["entry"][k]),
                "entry_time": session["ts"][int(trades["entry"][k])],
                "side": int(trades["side"][k]),
                "entry_price": float(trades["entry_price"][k]),
                "exit_price": float(trades["exit_price"][k]),
                "points": float(trades["side"][k]
                                * (trades["exit_price"][k]
                                   - trades["entry_price"][k])),
                "bars": int(trades["bars"][k]),
                "reason": str(trades["reason"][k]),
            }
            for k in keep
        ],
        "open": trades.get("open_position"),
    }


LOG_PATH = Path("logs") / "live_signal.csv"


def append_log(row: dict) -> None:
    """Append one bar to the live log, writing the header on first use."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def report(ts, price, k_value, ma_slope, state, closes) -> dict:
    """One line per bar, plus the row that goes into the live log."""
    lower, upper = channel_edges(closes, RULE["window"], RULE["level"])
    row = {
        "time": ts, "price": round(price, 2), "k": round(k_value, 3),
        "ma_slope": round(ma_slope, 3), "state": "FLAT",
        "side": 0, "entry": "", "best": "", "width": "", "stop": "",
        "points": "", "armed": 0,
    }
    if state is None:
        print(f"{ts}  {price:9.2f}  K {k_value:+.3f}  FLAT   "
              f"buy<{lower:.2f}  sell>{upper:.2f}")
        return row

    side = int(state["side"])
    entry = float(state["entry_price"])
    best = float(state["best_points"])
    width = float(state["trail_width"])
    points = side * (price - entry)
    armed = bool(state["armed"])
    stop = stop_price(side, entry, best, width)
    row.update({
        "state": "LONG" if side > 0 else "SHORT", "side": side,
        "entry": round(entry, 2), "best": round(best, 1),
        "points": round(points, 1), "armed": int(armed),
    })
    label = "LONG " if side > 0 else "SHORT"
    if armed:
        row["width"] = round(width, 1)
        row["stop"] = round(stop, 2)
        print(f"{ts}  {price:9.2f}  K {k_value:+.3f}  {label} "
              f"{points:+7.1f} pts  ARMED  stop {stop:.2f} "
              f"(width {width:.0f}, best {best:+.0f})")
    else:
        arm_at = upper if side > 0 else lower
        print(f"{ts}  {price:9.2f}  K {k_value:+.3f}  {label} "
              f"{points:+7.1f} pts  arms at {arm_at:.2f}")
    return row

def wait_for_next_bar(interval_minutes: int = 5, offset_seconds: int = 10) -> None:
    """Sleep until shortly after the next bar closes.

    The rule reads closed bars, so one pass per bar is all the work there
    is; the offset leaves the collector time to write the new row before
    the loop reads it.
    """
    now = datetime.now()
    step = interval_minutes * 60
    into = now.minute * 60 + now.second + now.microsecond / 1e6
    time.sleep(step - (into % step) + offset_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live signals and orders.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tail", type=int, default=3000,
                        help="bars to read; only the current session is used")
    parser.add_argument("--window", type=int, default=RULE["window"])
    parser.add_argument("--level", type=float, default=RULE["level"],
                        help="frozen entry threshold; do not refit this live")
    parser.add_argument("--max-hold", type=int, default=RULE["max_hold"])
    parser.add_argument("--poll", type=float, default=30.0,
                        help="seconds to wait when the bar has not advanced")
    parser.add_argument("--out", default=None,
                        help="CSV of state changes; runs/signals_<date>.csv")
    parser.add_argument("--live", action="store_true",
                        help="send orders; without it nothing is sent")
    parser.add_argument("--lots", type=float, default=0.10,
                        help="volume per position")
    parser.add_argument("--min-free-margin", type=float, default=2000.0,
                        help="skip opening below this free margin")
    parser.add_argument("--disaster-stop", type=float, default=0.0,
                        help="fixed broker-side stop distance, 0 for none")
    parser.add_argument("--flatten-on-exit", action="store_true",
                        help="close the open position when the loop stops")
    parser.add_argument("--bridge-path", default=r"C:\python\mt4_bridge",
                        help="directory holding the mt4bridge package")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    model_path = (Path(args.model) if args.model
                  else cfg.experiment.resolved_model_path)
    model = model_io.load_scaled(model_path)

    out = Path(args.out) if args.out else (
        cfg.journal.runs_dir / f"signals_{datetime.now():%Y-%m-%d}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not out.exists()
    handle = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=[
        "wall_clock", "bar_time", "price", "reading", "event",
        "side", "entry_price", "points", "bars", "reason",
    ])
    if fresh:
        writer.writeheader()

    side_name = "momentum" if RULE["momentum"] else "contrarian"
    print("=" * 78)
    print(f"rule   : stochastic_k w{args.window}, level {args.level}, "
          f"{side_name}, exit '{RULE['exit_rule']}', max hold {args.max_hold}")
    print(f"trail  : start {RULE['trail_start']:.0f} -> min "
          f"{RULE['trail_min']:.0f} over {RULE['trail_tau']} bars, "
          f"hold after arm {RULE['hold_after_arm']}")
    print(f"series : {cfg.data.series_path}")
    print(f"log    : {out}")
    print("=" * 78)

    executor = None
    transport = None
    if args.lots > 0:
        sys.path.insert(0, args.bridge_path)
        from mt4bridge.config import load_config as load_bridge_config
        from mt4bridge.trading import TradingClient
        from mt4bridge.transport import Transport
        from laplace.execution import ExecutionConfig, Executor

        bridge_cfg = load_bridge_config(None)
        transport = Transport(bridge_cfg, commands=True)
        transport.__enter__()
        client = TradingClient(bridge_cfg, transport)
        if not client.ping():
            raise SystemExit("bridge did not answer a heartbeat")
        executor = Executor(client, ExecutionConfig(
            lots=args.lots,
            min_free_margin=args.min_free_margin,
            disaster_stop_points=args.disaster_stop,
            dry_run=not args.live,
        ))
        account = client.account_info()
        print(f"account {account.number}  balance {account.balance:.2f}  "
              f"free margin {account.free_margin:.2f}  "
              f"{'LIVE, orders are sent' if args.live else 'DRY RUN, nothing is sent'}")
    else:
        print("signal only, no bridge client")

    previous_bar = None
    previous_open = None
    try:
        while True:
            try:
                session = load_session(cfg, model, args.tail)
            except Exception as exc:          # the bridge may be mid-write
                print(f"  read failed: {exc}")
                time.sleep(args.poll)
                continue

            if session is None or len(session["mids"]) < args.window + 2:
                print("  waiting for enough bars in this session")
                time.sleep(args.poll)
                continue

            state = state_of(session, args.level, args.window, args.max_hold)
            bar_time = state["time"]

            # The bar has not advanced: wait briefly rather than recompute,
            # which is what flooded the bridge with position queries.
            if bar_time == previous_bar:
                time.sleep(args.poll)
                continue

            age = datetime.now() - bar_time.to_pydatetime()
            if age > timedelta(minutes=12):
                print("  (stale, is the bridge fetching?)")

            price = float(session["mids"][-1])
            k_value = float(state["reading"])
            ma_slope = ma_slope_points(session["mids"], RULE["window"])
            row = report(bar_time, price, k_value, ma_slope, state["open"],
                         session["mids"])

            if executor is not None:
                try:
                    row["action"] = executor.step(state["open"], price, bar_time)
                except Exception as exc:
                    # One failed round trip must not end an unattended run;
                    # the next bar reconciles against the terminal anyway.
                    row["action"] = f"error_{type(exc).__name__}"
                    print(f"  bridge error: {exc}")
            append_log(row)

            if previous_bar is None:
                # Everything the rule has already done in this session, printed
                # once at startup. A console that only shows the current state
                # cannot distinguish "the rule found nothing" from "the rule is
                # not reporting", and those need different fixes.
                done = state["trades"]
                print(f"\nthis session so far: {len(done)} closed trade(s), "
                      f"{len(session['mids'])} bars")
                total = 0.0
                for k, trade in enumerate(done, 1):
                    side = "BUY " if trade["side"] > 0 else "SELL"
                    total += trade["points"]
                    print(f"  {k:>3}. {trade['entry_time']:%m-%d %H:%M}  {side}"
                          f" {trade['entry_price']:.2f} -> {trade['exit_price']:.2f}"
                          f"  {trade['points']:+8.2f} pts"
                          f"  {trade['bars']:>3} bars  {trade['reason']}")
                print(f"  total {total:+.2f} pts")
                if state["open"]:
                    position = state["open"]
                    side = "BUY " if position["side"] > 0 else "SELL"
                    print(f"  open: {side} from {position['entry_price']:.2f}, "
                          f"{position['points']:+.2f} pts, "
                          f"{position['bars']} bars")
                print()

            now_open = state["open"]
            event = None
            if now_open is not None and previous_open is None:
                event = "OPEN"
            elif now_open is None and previous_open is not None:
                event = "CLOSE"

            if event == "OPEN":
                side = "BUY" if now_open["side"] > 0 else "SELL"
                print(f"  >>> {side} at {now_open['entry_price']:.2f}")
            elif event == "CLOSE" and state["trades"]:
                done = state["trades"][-1]
                print(f"  >>> CLOSE  {done['points']:+.1f} pts  "
                      f"{done['bars']} bars  ({done['reason']})")

            writer.writerow({
                "wall_clock": datetime.now().isoformat(timespec="seconds"),
                "bar_time": bar_time.isoformat(),
                "price": state["price"],
                "reading": state["reading"],
                "event": event or "",
                "side": (now_open or (state["trades"][-1] if state["trades"] else {})).get("side", ""),
                "entry_price": (now_open or {}).get("entry_price", ""),
                "points": (now_open or (state["trades"][-1] if state["trades"] else {})).get("points", ""),
                "bars": (now_open or {}).get("bars", ""),
                "reason": (state["trades"][-1] if state["trades"] else {}).get("reason", "")
                          if event == "CLOSE" else "",
            })
            handle.flush()

            previous_bar = bar_time
            previous_open = state["open"]

            wait_for_next_bar()
    finally:
        handle.close()
        if executor is not None and args.flatten_on_exit:
            executor.flatten("shutdown")
        if transport is not None:
            transport.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())