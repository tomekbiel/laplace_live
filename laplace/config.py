"""
Configuration: one validated object built from config/strategy.json.

Every path, threshold and identifier the live strategy needs is read once,
checked once, and passed down as a frozen object. Nothing else in the package
reads the JSON, and nothing reads an environment variable at import time —
a value that can change between two calls is a value that will eventually
differ between the backtest and the live run.

The experiment identifiers mirror 5_train.py exactly, so the model path this
builds is the path that training wrote:

    <results_root>/<dataset>/<architecture>/models/<activation>_seed<seed>.keras

A run is named from the same four fields, which is what makes a directory of
dozens of runs readable months later.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "strategy.json"
DEFAULT_INSTRUMENTS = PROJECT_ROOT / "config" / "instruments.json"


class ConfigError(ValueError):
    """The configuration file is missing a value or holds an impossible one."""


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Which trained model to run.

    Attributes:
        dataset: Dataset stem as training used it, e.g. 'US.100+5'.
        architecture: Architecture label, e.g. 'dense128' or 'fcn'.
        activation: Activation name, e.g. 'relu' or 'bielu_ou'.
        seed: Training seed.
        model_path: Explicit path, overriding the composed one. Set this to
            run the pre-versioning 'laplace_minimal.keras', which is the only
            way to reproduce the original backtest figures exactly.
        results_root: Root of the research repository's results tree.
    """

    dataset: str
    architecture: str
    activation: str
    seed: int
    results_root: Path
    model_path: Path | None = None

    @property
    def run_name(self) -> str:
        """Identifier used for the run directory and the manifest."""
        return f"{self.dataset}_{self.architecture}_{self.activation}_seed{self.seed}"

    @property
    def resolved_model_path(self) -> Path:
        """The model file to load, composed the way 5_train.py saves it."""
        if self.model_path is not None:
            return self.model_path
        return (
            self.results_root
            / self.dataset
            / self.architecture
            / "models"
            / f"{self.activation}_seed{self.seed}.keras"
        )


@dataclass(frozen=True)
class DataConfig:
    """Where candles come from and how they are read.

    Attributes:
        bridge_root: Root of the mt4_bridge project.
        symbol: Instrument as the broker spells it, e.g. 'US.100+'.
        timeframe: Timeframe name, e.g. 'M5'.
        price_column: Which OHLC column becomes 'mid'. MT4 builds its bars
            from bid prices, so 'close' is the bid close — the series the
            model was trained on. Changing this changes what the model sees.
        gap_threshold_days: Breaks longer than this cut off earlier history,
            exactly as remove_data_before_long_gap does in the research code.
    """

    bridge_root: Path
    symbol: str
    timeframe: str
    price_column: str
    gap_threshold_days: int

    @property
    def series_path(self) -> Path:
        """The bridge's CSV for this symbol and timeframe."""
        safe = self.symbol.replace("/", "_").replace("\\", "_")
        return (
            self.bridge_root / "data" / "rates" / safe / self.timeframe
            / f"{safe}_{self.timeframe}.csv"
        )

    @property
    def bridge_config_path(self) -> Path:
        """The bridge's own configuration, used when placing orders."""
        return self.bridge_root / "config" / "bridge.json"

    @property
    def timebase_path(self) -> Path:
        """Where the bridge recorded the broker's clock offset."""
        return self.bridge_root / "data" / "meta" / "timebase.json"


@dataclass(frozen=True)
class WindowConfig:
    """Window geometry. These must match what the model was trained with.

    Attributes:
        lookback: Bars per window.
        predict_step: How far ahead the target sits.
        rolling_window: Window for modified_z_score and robust_volatility.
        zscore_clip: Clipping applied to the z-score feature.
    """

    lookback: int
    predict_step: int
    rolling_window: int
    zscore_clip: float

    @property
    def warmup_bars(self) -> int:
        """Bars needed inside one open block before a single prediction.

        The rolling features consume `rolling_window` bars before they stop
        being null, the window itself consumes `lookback` more, and
        log_return and lag1_return cost one each.
        """
        return self.rolling_window + self.lookback + self.predict_step + 1


@dataclass(frozen=True)
class StrategyConfig:
    """The thirteen CEM parameters, from file or inline.

    Attributes:
        cem_params_path: npz written by 7_cem.py.
        params: Inline override; when set, the file is not read.
    """

    cem_params_path: Path | None
    params: dict[str, float] | None

    # Names the rules read, in the order 7_cem.py defines them. ClassVar
    # keeps this out of the dataclass fields; it is a constant, not a
    # value a configuration file may set.
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "entry_z", "entry_timeout", "trailing_activation", "trailing_distance",
        "take_profit", "stop_loss", "exit_z", "reverse_sensitivity",
        "max_hold", "time_decay", "trend_filter", "volatility_adapt",
        "risk_multiplier",
    )

    def load(self) -> dict[str, float]:
        """Return the parameter dictionary, validating that none is missing."""
        if self.params is not None:
            values = dict(self.params)
        elif self.cem_params_path is not None:
            import numpy as np

            if not self.cem_params_path.is_file():
                raise ConfigError(f"CEM results not found: {self.cem_params_path}")
            data = np.load(self.cem_params_path, allow_pickle=True)
            values = dict(data["best_params"].item())
        else:
            raise ConfigError("Neither strategy.params nor strategy.cem_params_path is set")

        missing = [name for name in self.REQUIRED if name not in values]
        if missing:
            raise ConfigError(f"CEM parameters missing: {', '.join(missing)}")
        return {name: float(values[name]) for name in self.REQUIRED}


@dataclass(frozen=True)
class RulesConfig:
    """Which version of the exit rules to run.

    Both default to the original behaviour, so an unchanged configuration
    reproduces the figures that were measured before the defects were found.

    Attributes:
        signed_exits: Read cum_return as signed profit in every exit branch,
            repairing the three sign defects inherited from 7_cem.py.
        book_entry_bar: Include the triggering bar's return in the trade, as
            8_backtest.py does. Off, the trade starts at that bar's close,
            which is what a live run can capture.
    """

    signed_exits: bool
    book_entry_bar: bool

    @property
    def label(self) -> str:
        """Short tag for the run directory, so variants never collide."""
        parts = []
        parts.append("signed" if self.signed_exits else "orig")
        parts.append("entrybar" if self.book_entry_bar else "noentrybar")
        return "-".join(parts)


@dataclass(frozen=True)
class MoneyConfig:
    """Position sizing. Off until the instrument specification is filled in.

    Attributes:
        enabled: When false, every entry uses fixed_lots and the journal
            reports results in price units, as the backtest does.
        risk_fraction: Fraction of equity to put at risk per trade once
            sizing is on. Kelly, when it arrives, computes this.
        fixed_lots: Size used while sizing is off.
    """

    enabled: bool
    risk_fraction: float
    fixed_lots: float


@dataclass(frozen=True)
class RuntimeConfig:
    """Loop timing.

    Attributes:
        bar_close_delay_sec: How long after a bar closes to ask for it. MT4
            needs a moment to finish the bar, and CopyRates returns the bar
            still forming, which this package discards.
        fetch_bars: How many bars to request each cycle. Must exceed
            WindowConfig.warmup_bars by a wide margin, because a request
            spanning a weekend returns far fewer bars than its span suggests.
        request_timeout_sec: Per-request timeout passed to the bridge.
        max_retries: Attempts before a cycle is abandoned and logged.
    """

    bar_close_delay_sec: float
    fetch_bars: int
    request_timeout_sec: float
    max_retries: int


@dataclass(frozen=True)
class JournalConfig:
    """Where run output goes."""

    runs_dir: Path


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    to_file: bool


# ----------------------------------------------------------------------
# Root
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """The whole configuration, validated."""

    experiment: ExperimentConfig
    data: DataConfig
    windows: WindowConfig
    strategy: StrategyConfig
    rules: RulesConfig
    money: MoneyConfig
    runtime: RuntimeConfig
    journal: JournalConfig
    logging: LoggingConfig
    source: Path

    def check(self) -> list[str]:
        """Return a list of problems that would stop a run, empty when sound.

        Reported rather than raised, so a caller can print every problem at
        once instead of fixing them one failed start at a time.
        """
        problems: list[str] = []

        model = self.experiment.resolved_model_path
        if not model.is_file():
            problems.append(f"model file not found: {model}")

        series = self.data.series_path
        if not series.is_file():
            problems.append(f"candle series not found: {series}")

        if self.runtime.fetch_bars < self.windows.warmup_bars * 2:
            problems.append(
                f"runtime.fetch_bars ({self.runtime.fetch_bars}) is small for a "
                f"warm-up of {self.windows.warmup_bars} bars; use at least twice it"
            )

        if self.money.enabled and self.money.risk_fraction <= 0:
            problems.append("money.enabled is true but risk_fraction is not positive")

        try:
            self.strategy.load()
        except ConfigError as exc:
            problems.append(str(exc))

        return problems

    def describe(self) -> str:
        """One block of text for the console and the manifest."""
        return (
            f"model      : {self.experiment.resolved_model_path}\n"
            f"series     : {self.data.series_path}\n"
            f"instrument : {self.data.symbol} {self.data.timeframe}\n"
            f"window     : lookback {self.windows.lookback}, "
            f"warm-up {self.windows.warmup_bars} bars\n"
            f"exits      : {self.rules.label}\n"
            f"sizing     : {'on' if self.money.enabled else f'fixed {self.money.fixed_lots}'}"
        )


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _path(value: Any, base: Path) -> Path:
    """Resolve a configured path, relative ones against the project root."""
    p = Path(str(value))
    return p if p.is_absolute() else (base / p)


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate strategy.json.

    Args:
        path: Configuration file; the project's own is used when omitted.

    Raises:
        ConfigError: The file is missing, unparseable, or lacks a value.
    """
    source = Path(path) if path else DEFAULT_CONFIG
    if not source.is_file():
        raise ConfigError(f"Configuration not found: {source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source} is not valid JSON: {exc}") from exc

    try:
        exp = raw["experiment"]
        dat = raw["data"]
        win = raw["windows"]
        stg = raw["strategy"]
        rul = raw.get("rules", {})
        mon = raw["money"]
        run = raw["runtime"]
        jor = raw["journal"]
        log = raw.get("logging", {})
    except KeyError as exc:
        raise ConfigError(f"{source} is missing section {exc}") from exc

    experiment = ExperimentConfig(
        dataset=str(exp["dataset"]),
        architecture=str(exp["architecture"]),
        activation=str(exp["activation"]).lower(),
        seed=int(exp["seed"]),
        results_root=_path(exp["results_root"], PROJECT_ROOT),
        model_path=(_path(exp["model_path"], PROJECT_ROOT)
                    if exp.get("model_path") else None),
    )

    data = DataConfig(
        bridge_root=_path(dat["bridge_root"], PROJECT_ROOT),
        symbol=str(dat["symbol"]),
        timeframe=str(dat["timeframe"]).upper(),
        price_column=str(dat.get("price_column", "close")).lower(),
        gap_threshold_days=int(dat.get("gap_threshold_days", 7)),
    )

    windows = WindowConfig(
        lookback=int(win["lookback"]),
        predict_step=int(win["predict_step"]),
        rolling_window=int(win["rolling_window"]),
        zscore_clip=float(win["zscore_clip"]),
    )

    strategy = StrategyConfig(
        cem_params_path=(_path(stg["cem_params_path"], PROJECT_ROOT)
                         if stg.get("cem_params_path") else None),
        params=stg.get("params"),
    )

    rules = RulesConfig(
        signed_exits=bool(rul.get("signed_exits", False)),
        book_entry_bar=bool(rul.get("book_entry_bar", True)),
    )

    money = MoneyConfig(
        enabled=bool(mon.get("enabled", False)),
        risk_fraction=float(mon.get("risk_fraction", 0.0)),
        fixed_lots=float(mon.get("fixed_lots", 0.01)),
    )

    runtime = RuntimeConfig(
        bar_close_delay_sec=float(run.get("bar_close_delay_sec", 8)),
        fetch_bars=int(run.get("fetch_bars", 400)),
        request_timeout_sec=float(run.get("request_timeout_sec", 60)),
        max_retries=int(run.get("max_retries", 3)),
    )

    journal = JournalConfig(runs_dir=_path(jor.get("runs_dir", "runs"), PROJECT_ROOT))

    logging_cfg = LoggingConfig(
        level=str(log.get("level", "INFO")).upper(),
        to_file=bool(log.get("to_file", True)),
    )

    return Config(
        experiment=experiment,
        data=data,
        windows=windows,
        strategy=strategy,
        rules=rules,
        money=money,
        runtime=runtime,
        journal=journal,
        logging=logging_cfg,
        source=source,
    )


def load_instruments(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read contract specifications, ignoring the leading note key."""
    source = Path(path) if path else DEFAULT_INSTRUMENTS
    if not source.is_file():
        raise ConfigError(f"Instrument specifications not found: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def setup_logging(cfg: Config, run_dir: Path | None = None) -> None:
    """Send logs to the console, and to the run directory when one exists."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if cfg.logging.to_file and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "run.log", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def add_bridge_to_path(cfg: Config) -> None:
    """Make the mt4_bridge package importable.

    The bridge is a separate project with its own repository, so it is not a
    dependency that pip installed; it is a sibling directory whose location
    the configuration names.
    """
    root = str(cfg.data.bridge_root)
    if root not in sys.path:
        sys.path.insert(0, root)
