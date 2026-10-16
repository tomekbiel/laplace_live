"""
Split each trade into the part that could be traded and the part that could not.

The backtest sums a position's return from y[entry_step] inclusive — the bar
whose move produced the signal. That bar closes before anyone knows a signal
exists, so its return is in the books but was never available. Everything
after it is.

The split is exact, not estimated. At the entry bar the journal records

    signal_z = y / b        and        b

and the position takes the sign of signal_z, so the triggering bar's
contribution to the trade is

    |signal_z| * b

always positive, always the first term of the sum. Subtracting it leaves the
return earned after the signal was visible.

What the answer means:

* tradeable mean clearly above the spread -> the strategy earns after the
  signal and the backtest only overstates the size.
* tradeable mean near zero or negative -> the whole result is the triggering
  bar, and the rule as written has no edge beyond it.

Longs and shorts are reported apart, because the exit branches behave
differently for the two: the short branches of take_profit and stop_loss
negate an already-signed quantity, so shorts leave through the timeout almost
immediately while longs run on.

Usage:
    python tools/decompose_trades.py runs/2026-10-14_2147_backtest_...
    python tools/decompose_trades.py <run> --spread-bp 4.5
    python tools/decompose_trades.py <run> --by-reason
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Spread on US.100+ measured through the bridge: 1.4 index points at 30830,
# which is one full spread per round turn.
DEFAULT_SPREAD_BP = 4.5


@dataclass
class Split:
    """One trade, taken apart."""

    reason: str
    position: int
    duration: int
    booked: float        # what the backtest recorded
    trigger: float       # the bar that produced the signal
    tradeable: float     # what was left to earn


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_entry_index(decisions: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """Map entry bar time -> (signal_z, b) for every opening decision."""
    index: dict[str, tuple[float, float]] = {}
    for row in decisions:
        if row.get("action") != "open":
            continue
        try:
            index[row["bar_time"]] = (float(row["signal_z"]), float(row["b"]))
        except (KeyError, ValueError):
            continue
    return index


def split_trades(
    trades: list[dict[str, str]], index: dict[str, tuple[float, float]]
) -> tuple[list[Split], int]:
    """Decompose every trade whose entry bar can be found in the journal."""
    out: list[Split] = []
    unmatched = 0

    for row in trades:
        key = row.get("entry_time", "")
        if key not in index:
            unmatched += 1
            continue

        signal_z, b = index[key]
        booked = float(row["return"])
        trigger = abs(signal_z) * b

        out.append(Split(
            reason=row.get("reason", ""),
            position=int(float(row.get("position", 0))),
            duration=int(float(row.get("duration", 0))),
            booked=booked,
            trigger=trigger,
            tradeable=booked - trigger,
        ))

    return out, unmatched


def report(title: str, splits: list[Split], spread_bp: float) -> None:
    """Print one group's figures, everything in basis points."""
    if not splits:
        print(f"\n{title}: none")
        return

    booked = np.array([s.booked for s in splits]) * 10000
    trigger = np.array([s.trigger for s in splits]) * 10000
    tradeable = np.array([s.tradeable for s in splits]) * 10000
    net = tradeable - spread_bp

    print(f"\n{title}  (n = {len(splits)})")
    print(f"  {'':<22}{'mean':>10}{'median':>10}{'win %':>9}{'total':>12}")
    for label, values in (("booked by backtest", booked),
                          ("trigger bar", trigger),
                          ("tradeable", tradeable),
                          (f"tradeable - {spread_bp:.1f} bp", net)):
        print(f"  {label:<22}{values.mean():>10.2f}{np.median(values):>10.2f}"
              f"{(values > 0).mean() * 100:>9.1f}{values.sum():>12.1f}")

    share = trigger.sum() / booked.sum() * 100 if booked.sum() else float("nan")
    print(f"  the trigger bar is {share:.1f}% of everything booked")


def report_by_reason(splits: list[Split], spread_bp: float) -> None:
    """Per exit reason, which is where the two directions separate."""
    reasons = sorted({s.reason for s in splits})
    print(f"\n{'exit reason':<16}{'n':>6}{'booked':>10}{'trigger':>10}"
          f"{'tradeable':>11}{'net':>9}{'bars':>7}")
    for reason in reasons:
        group = [s for s in splits if s.reason == reason]
        booked = np.mean([s.booked for s in group]) * 10000
        trig = np.mean([s.trigger for s in group]) * 10000
        trade = np.mean([s.tradeable for s in group]) * 10000
        bars = np.mean([s.duration for s in group])
        print(f"{reason:<16}{len(group):>6}{booked:>10.2f}{trig:>10.2f}"
              f"{trade:>11.2f}{trade - spread_bp:>9.2f}{bars:>7.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate the triggering bar from the rest of each trade."
    )
    parser.add_argument("run", help="a run directory under runs/")
    parser.add_argument("--spread-bp", type=float, default=DEFAULT_SPREAD_BP,
                        help=f"round-turn cost in basis points, "
                             f"default {DEFAULT_SPREAD_BP}")
    parser.add_argument("--by-reason", action="store_true",
                        help="also break the figures down by exit reason")
    args = parser.parse_args()

    run = Path(args.run)
    trades = read_rows(run / "trades.csv")
    decisions = read_rows(run / "decisions.csv")

    if not trades:
        print("No trades in this run.")
        return 1

    index = build_entry_index(decisions)
    splits, unmatched = split_trades(trades, index)

    print(f"run        : {run.name}")
    print(f"trades     : {len(trades)}")
    print(f"decomposed : {len(splits)}")
    if unmatched:
        print(f"unmatched  : {unmatched} (no opening row found for the entry time)")
    if not splits:
        print("\nNothing could be decomposed; the journal has no opening rows.")
        return 1

    print(f"spread     : {args.spread_bp:.1f} bp per round turn")
    print("\nAll figures are basis points of log return, per trade.")

    report("ALL TRADES", splits, args.spread_bp)
    report("LONGS", [s for s in splits if s.position == 1], args.spread_bp)
    report("SHORTS", [s for s in splits if s.position == -1], args.spread_bp)

    if args.by_reason:
        report_by_reason(splits, args.spread_bp)

    tradeable = np.array([s.tradeable for s in splits]) * 10000
    net = tradeable.mean() - args.spread_bp
    print("\n" + "=" * 60)
    if net > 1.0:
        print(f"  After the trigger bar and costs: {net:+.2f} bp per trade.")
        print("  The rule earns beyond the bar that produced the signal.")
    elif net > -1.0:
        print(f"  After the trigger bar and costs: {net:+.2f} bp per trade.")
        print("  Too close to zero to call either way on this sample.")
    else:
        print(f"  After the trigger bar and costs: {net:+.2f} bp per trade.")
        print("  What the backtest books comes from the triggering bar.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
