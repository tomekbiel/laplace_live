"""
Run the loop without sending orders.

Everything a live run does happens here — fetching, features, windows,
prediction, the rule engine, the journal — except the order itself. A paper
run is therefore a rehearsal of the real thing rather than a different
program, and its journal can be compared with a live one row for row.

Usage:
    python scripts/run_paper.py
    python scripts/run_paper.py --cycles 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laplace.config import load_config, setup_logging
from laplace.live import LiveRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strategy without trading.")
    parser.add_argument("--cycles", type=int, default=None,
                        help="stop after this many bars; runs until interrupted otherwise")
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

    runner = LiveRunner(cfg, execute=False, max_cycles=args.cycles)
    bars = runner.run()
    print(f"\nProcessed {bars} bar(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
