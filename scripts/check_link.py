"""
Connectivity check: MT4 -> bridge -> this project.

The first thing to run. It touches no model, no CEM parameters and no orders,
so it can be run before any of those are in place, and it answers one
question only: do candles come out of the terminal and arrive here in the
shape the feature pipeline expects.

Five stages, each reporting on its own, so a failure names its own cause:

  1. config      paths resolve, the bridge is where it was said to be
  2. transport   sockets open, the advisor answers a heartbeat
  3. account     the account replies, and the broker's clock is read
  4. candles     a history request returns bars for this instrument
  5. pipeline    those bars pass through reader, features and windows

Stage 5 is the one worth watching. It proves that the frame coming out of
the socket is the same shape the model was trained on — 20 bars by 7
features — without needing the model to say so.

Usage:
    python scripts/check_link.py
    python scripts/check_link.py --bars 600
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laplace import features as feature_builder
from laplace import reader, windows
from laplace.config import add_bridge_to_path, load_config, setup_logging

TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                     "H1": 60, "H4": 240, "D1": 1440}


def stage(name: str) -> None:
    print(f"\n--- {name} " + "-" * max(0, 56 - len(name)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the path from MT4 to this project.")
    parser.add_argument("--bars", type=int, default=400,
                        help="roughly how many bars to ask for, default 400")
    parser.add_argument("--config", default=None, help="path to strategy.json")
    args = parser.parse_args()

    # --- 1. configuration --------------------------------------------
    stage("config")
    cfg = load_config(args.config)
    setup_logging(cfg)

    print(f"  project    : {Path(__file__).resolve().parent.parent}")
    print(f"  bridge     : {cfg.data.bridge_root}")
    print(f"  instrument : {cfg.data.symbol} {cfg.data.timeframe}")
    print(f"  warm-up    : {cfg.windows.warmup_bars} bars needed for one window")

    if not cfg.data.bridge_config_path.is_file():
        print(f"  FAIL: no bridge configuration at {cfg.data.bridge_config_path}")
        return 1
    print(f"  bridge cfg : {cfg.data.bridge_config_path}")

    step_minutes = TIMEFRAME_MINUTES.get(cfg.data.timeframe.upper(), 5)

    # --- 2. transport -------------------------------------------------
    stage("transport")
    add_bridge_to_path(cfg)
    try:
        from mt4bridge.config import load_config as load_bridge_config
        from mt4bridge.history import HistoryFetcher
        from mt4bridge.protocol import heartbeat
        from mt4bridge.trading import TradingClient
        from mt4bridge.transport import Transport, TransportTimeout
    except ImportError as exc:
        print(f"  FAIL: cannot import the bridge package: {exc}")
        print(f"  Is {cfg.data.bridge_root} the project root, the folder "
              f"holding the mt4bridge package?")
        return 1

    bridge_cfg = load_bridge_config(str(cfg.data.bridge_config_path))
    print(f"  commands   -> {bridge_cfg.connection.commands_endpoint}")
    print(f"  responses  <- {bridge_cfg.connection.responses_endpoint}")

    with Transport(bridge_cfg, commands=True) as transport:
        answered = 0
        for _ in range(3):
            try:
                transport.request(heartbeat(), timeout=5.0)
                answered += 1
            except TransportTimeout:
                pass
        print(f"  heartbeat  : {answered}/3 answered")
        print(f"  sockets    : {transport.status}")

        if answered == 0:
            print("\n  FAIL: the advisor is not answering. Check that the terminal")
            print("  is running with DWX_ZeroMQ_Server attached to a chart.")
            return 1

        # --- 3. account -----------------------------------------------
        stage("account")
        client = TradingClient(bridge_cfg, transport)
        info = client.account_info()
        print(f"  number     : {info.number}")
        print(f"  balance    : {info.balance:.2f}")
        print(f"  equity     : {info.equity:.2f}")

        broker_now = datetime.now()
        if info.server_time is not None:
            offset = (info.server_time - datetime.now()).total_seconds() / 3600
            broker_now = info.server_time
            print(f"  server time: {info.server_time:%Y-%m-%d %H:%M:%S} "
                  f"({offset:+.1f} h from this machine)")
        else:
            print("  server time: not reported; using local time")

        positions = client.own_positions()
        print(f"  our open   : {len(positions)}")

        # --- 4. candles -----------------------------------------------
        stage("candles")
        fetcher = HistoryFetcher(bridge_cfg, transport)
        span = timedelta(minutes=args.bars * step_minutes * 3)
        start, end = broker_now - span, broker_now
        print(f"  requesting : {start:%Y.%m.%d %H:%M} .. {end:%Y.%m.%d %H:%M}")

        candles = fetcher.fetch(
            cfg.data.symbol, cfg.data.timeframe,
            start.strftime("%Y.%m.%d %H:%M"), end.strftime("%Y.%m.%d %H:%M"),
            timeout=cfg.runtime.request_timeout_sec,
        )
        if not candles:
            print("  FAIL: the terminal returned no candles. Open the chart "
                  "for this symbol and timeframe.")
            return 1
        print(f"  received   : {len(candles)} bars "
              f"({candles[0].time} .. {candles[-1].time})")

    # --- 5. pipeline ------------------------------------------------
    stage("pipeline")
    frame = reader.frame_from_candles(candles, cfg.data.symbol, cfg.data.price_column)
    print(f"  frame      : {frame.height} rows, columns {frame.columns}")

    trimmed = reader.drop_forming_bar(frame, broker_now, step_minutes)
    if trimmed.height != frame.height:
        print(f"  forming bar: dropped {frame['timestamp'][-1]} (still open)")
    frame = trimmed

    prepared = feature_builder.prepare(
        frame,
        window_size=cfg.windows.rolling_window,
        gap_threshold_days=cfg.data.gap_threshold_days,
        interval_minutes=step_minutes,
    )
    print(f"  prepared   : {prepared.height} rows")
    print(f"  interpolated: {int(prepared['market_closed'].sum())} closed-market rows")
    print(f"  features   : {[c for c in prepared.columns if c not in ('timestamp','instrument','mid')]}")

    X, y, ts, mids, blocks, _ = windows.create_windows(
        prepared,
        lookback=cfg.windows.lookback,
        predict_step=cfg.windows.predict_step,
        zscore_clip=cfg.windows.zscore_clip,
    )
    if len(X) == 0:
        print("  FAIL: no complete window. The session may be too short, or "
              "the fetched span too narrow.")
        return 1

    print(f"  windows    : {X.shape}  (sessions: {len(set(blocks.tolist()))})")
    print(f"  newest pair: window ending before {ts[-1]}, target y={y[-1]:+.6f}")

    if X.shape[1:] != (cfg.windows.lookback, 7):
        print(f"  WARNING: windows are {X.shape[1:]}, the model expects "
              f"({cfg.windows.lookback}, 7)")

    try:
        ahead = windows.last_window(prepared, lookback=cfg.windows.lookback,
                                    zscore_clip=cfg.windows.zscore_clip)
        print(f"  forward    : window ends {ahead.time}, price {ahead.price:.2f}, "
              f"session {ahead.block_length} bars")
    except windows.WindowError as exc:
        print(f"  forward    : none ({exc})")

    print("\n" + "=" * 60)
    print("  The path from MT4 through the bridge into this project works.")
    print("  Next: set experiment.results_root and run predict_last.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
