"""
Run the loop and send orders.

The only difference from run_paper.py is that the engine's decisions reach
the broker. It asks for confirmation once at the start, because a process
that trades should not be startable by a stray arrow-up in a terminal.

Usage:
    python scripts/run_live.py
    python scripts/run_live.py --yes --cycles 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laplace.config import load_config, setup_logging
from laplace.live import LiveRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strategy against the account.")
    parser.add_argument("--cycles", type=int, default=None,
                        help="stop after this many bars; runs until interrupted otherwise")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument("--config", default=None, help="path to strategy.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    problems = cfg.check()
    for problem in problems:
        print(f"  config: {problem}")
    if problems:
        print("\nFix the above before running.")
        return 1

    print(cfg.describe())
    if not args.yes:
        answer = input("\nThis will place real orders on the account. Type 'trade' to go on: ")
        if answer.strip().lower() != "trade":
            print("Stopped.")
            return 1

    runner = LiveRunner(cfg, execute=True, max_cycles=args.cycles)
    bars = runner.run()
    print(f"\nProcessed {bars} bar(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
