"""
Per-run output: one directory, a manifest and three CSV files.

There will be many runs. What makes a directory of them readable months
later is that each one records, at the moment it started, exactly which
model and which parameters produced it — a file name alone will not survive
the question "which activation was that".

    manifest.json   model, parameters, data range, account, configuration
    decisions.csv   every bar, including the ones where nothing happened
    trades.csv      the columns the research backtest writes, plus prices
    equity.csv      the running equity curve

decisions.csv is the important one. trades.csv says what happened; the
decision log says what was considered and rejected, which is what answers a
live run diverging from the backtest. It carries mu, b, the tunnel edges and
the two trigger prices, so a disagreement can be traced to the bar where the
two paths parted.

Rows are flushed as they are written. A run that dies at hour three leaves
three hours of data rather than an empty file.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DECISION_COLUMNS = (
    "bar_time", "price", "mu", "b", "signal_z",
    "k1_lower", "k1_upper", "k3_lower", "k3_upper",
    "long_trigger", "short_trigger",
    "action", "reason", "position", "cum_return", "equity",
    "lots", "blocked", "note",
)

TRADE_COLUMNS = (
    "entry_time", "exit_time", "position", "entry_step", "exit_step",
    "duration", "return", "return_pct", "reason",
    "entry_price", "exit_price", "lots", "profit_currency",
)

EQUITY_COLUMNS = ("bar_time", "equity", "equity_currency")


class Journal:
    """Writes one run's output.

    Args:
        root: The runs directory.
        run_name: Identifier from the experiment configuration.
        mode: 'paper' or 'live', recorded in the directory name so the two
            can never be mistaken for each other in a listing.
        started: Start time; now when omitted.
    """

    def __init__(
        self,
        root: Path,
        run_name: str,
        mode: str,
        started: datetime | None = None,
    ) -> None:
        self.started = started or datetime.now()
        stamp = self.started.strftime("%Y-%m-%d_%H%M")
        self.dir = root / f"{stamp}_{mode}_{run_name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        self._decisions = self._open("decisions.csv", DECISION_COLUMNS)
        self._trades = self._open("trades.csv", TRADE_COLUMNS)
        self._equity = self._open("equity.csv", EQUITY_COLUMNS)

        logger.info("Journal at %s", self.dir)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _open(self, name: str, columns: tuple[str, ...]):
        """Open a CSV with its header, returning the handle and writer."""
        path = self.dir / name
        handle = path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        handle.flush()
        return handle, writer

    def write_manifest(self, payload: dict[str, Any]) -> None:
        """Record what this run is, before any bar is processed."""
        payload = dict(payload)
        payload["started"] = self.started.isoformat(timespec="seconds")
        payload["mode"] = self.mode
        payload["git"] = _git_revision()
        (self.dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def decision(self, row: dict[str, Any]) -> None:
        """Append one bar's record, whether or not anything happened."""
        handle, writer = self._decisions
        writer.writerow(row)
        handle.flush()

    def trade(self, row: dict[str, Any]) -> None:
        handle, writer = self._trades
        writer.writerow(row)
        handle.flush()

    def equity(self, bar_time: Any, value: float, currency: float | None = None) -> None:
        handle, writer = self._equity
        writer.writerow({
            "bar_time": bar_time,
            "equity": value,
            "equity_currency": "" if currency is None else currency,
        })
        handle.flush()

    def note(self, text: str) -> None:
        """Record something that is neither a bar nor a trade."""
        with (self.dir / "notes.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat(timespec='seconds')}  {text}\n")

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def summarise(self, payload: dict[str, Any]) -> None:
        """Write the closing figures beside the manifest."""
        payload = dict(payload)
        payload["finished"] = datetime.now().isoformat(timespec="seconds")
        (self.dir / "summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def close(self) -> None:
        for handle, _ in (self._decisions, self._trades, self._equity):
            handle.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _git_revision() -> str:
    """Short commit of the working tree, or an empty string outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""
