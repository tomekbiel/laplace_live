"""Compare finished runs side by side, each read through its own manifest.

A run directory holds manifest.json, summary.json, trades.csv, equity.csv
and decisions.csv. This reads the manifest for what the run WAS and
recomputes the figures from trades.csv, so a run written by older code
still lines up with a new one and no figure is trusted without its
configuration beside it.

Costs are applied here rather than inside the engine: the gross column
comes straight from the journal and the net column subtracts a flat
round-turn cost per trade. Ranking is invariant to cost only when two
runs have the same trade count, which is exactly why both columns are
printed together.

Example:
    python tools/compare_runs.py --cost-bp 4.5
    python tools/compare_runs.py runs/2026-10-18_* --cost-bp 4.5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The backtest reports a levered percentage total; keeping the same
# convention here means a number printed by either tool means one thing.
DEFAULT_LEVERAGE = 20

# Manifest fields that must agree before two runs are comparable at all.
COMPARABLE_FIELDS = ("instrument", "timeframe", "model")


@dataclass(frozen=True)
class RunSummary:
    """One run, reduced to the row it occupies in the table."""

    path: Path
    label: str
    instrument: str
    timeframe: str
    model: str
    trades: int
    avg_bp: float
    net_bp: float
    win_rate: float
    profit_factor: float
    total_pct: float
    avg_duration: float


def read_manifest(run_dir: Path) -> dict:
    """Load manifest.json, tolerating a run that predates a field."""
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def variant_label(manifest: dict, run_dir: Path) -> str:
    """Name the variant from the manifest, falling back to the directory.

    The directory name carries the label too, but only because the run
    was named after it; the manifest is the record, so it wins when both
    are present.
    """
    rules = manifest.get("rules") or {}
    if "signed_exits" in rules and "book_entry_bar" in rules:
        signs = "signed" if rules["signed_exits"] else "orig"
        entry = "entrybar" if rules["book_entry_bar"] else "noentrybar"
        return f"{signs}-{entry}"
    return run_dir.name.split("_")[-1]


def profit_factor(returns: pd.Series) -> float:
    """Gross profit over gross loss; infinite when nothing lost."""
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))


def load_run(run_dir: Path, cost_bp: float, leverage: int) -> RunSummary | None:
    """Build one row, or None when the directory holds no finished run."""
    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        return None

    trades = pd.read_csv(trades_path)
    if trades.empty or "return" not in trades.columns:
        return None

    manifest = read_manifest(run_dir)
    gross = pd.to_numeric(trades["return"])
    # A flat round-turn charge, applied once per trade, in the same
    # fractional units the journal stores.
    net = gross - cost_bp / 1e4

    duration = (
        float(pd.to_numeric(trades["duration"]).mean())
        if "duration" in trades.columns
        else float("nan")
    )

    return RunSummary(
        path=run_dir,
        label=variant_label(manifest, run_dir),
        instrument=str(manifest.get("instrument", "?")),
        timeframe=str(manifest.get("timeframe", "?")),
        model=Path(str(manifest.get("model", "?"))).name,
        trades=len(trades),
        avg_bp=float(gross.mean() * 1e4),
        net_bp=float(net.mean() * 1e4),
        win_rate=float((gross > 0).mean() * 100),
        profit_factor=profit_factor(gross),
        total_pct=float(gross.sum() * 100 * leverage),
        avg_duration=duration,
    )


def collect(paths: list[Path], cost_bp: float, leverage: int) -> list[RunSummary]:
    """Load every run directory given, newest last."""
    runs: list[RunSummary] = []
    for path in sorted(paths):
        summary = load_run(path, cost_bp, leverage)
        if summary is not None:
            runs.append(summary)
    return runs


def warn_incomparable(runs: list[RunSummary]) -> list[str]:
    """Name the fields that differ across runs and so break comparison.

    Two runs on different instruments or different models can be printed
    in one table, but their numbers do not belong in one ranking. Saying
    so is cheaper than discovering it after a decision.
    """
    messages: list[str] = []
    for field in COMPARABLE_FIELDS:
        values = {getattr(run, field) for run in runs}
        if len(values) > 1:
            messages.append(f"{field} differs: {', '.join(sorted(values))}")
    return messages


def print_table(runs: list[RunSummary], cost_bp: float) -> None:
    """Print one row per run, widest column first."""
    name_width = max((len(run.path.name) for run in runs), default=20)
    name_width = min(name_width, 56)

    header = (
        f"{'run':<{name_width}} {'variant':>18} {'trades':>7} "
        f"{'avg bp':>8} {'win%':>7} {'PF':>7} {'total%':>10} {'dur':>6}"
    )
    if cost_bp > 0:
        header += f" {'net bp':>8}"
    print(header)
    print("-" * len(header))

    for run in runs:
        name = run.path.name[-name_width:]
        factor = "inf" if run.profit_factor == float("inf") else f"{run.profit_factor:.2f}"
        line = (
            f"{name:<{name_width}} {run.label:>18} {run.trades:>7} "
            f"{run.avg_bp:>8.2f} {run.win_rate:>7.2f} {factor:>7} "
            f"{run.total_pct:>10.1f} {run.avg_duration:>6.1f}"
        )
        if cost_bp > 0:
            line += f" {run.net_bp:>8.2f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="run directories; default is every directory under runs/",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="where runs live"
    )
    parser.add_argument(
        "--filter", default="", help="keep only runs whose name contains this"
    )
    parser.add_argument(
        "--cost-bp",
        type=float,
        default=0.0,
        help="flat round-turn cost per trade, in basis points",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=DEFAULT_LEVERAGE,
        help="leverage used for the total%% column, as in the backtest",
    )
    parser.add_argument("--out", type=Path, help="write the table as JSON")
    args = parser.parse_args()

    paths = args.runs or [p for p in args.runs_dir.iterdir() if p.is_dir()]
    if args.filter:
        paths = [p for p in paths if args.filter in p.name]
    if not paths:
        raise SystemExit("No run directories matched.")

    runs = collect(paths, args.cost_bp, args.leverage)
    if not runs:
        raise SystemExit("No directory held a usable trades.csv.")

    print_table(runs, args.cost_bp)

    for message in warn_incomparable(runs):
        print(f"\nwarning: {message}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cost_bp": args.cost_bp,
            "leverage": args.leverage,
            "runs": [
                {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(run).items()}
                for run in runs
            ],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()