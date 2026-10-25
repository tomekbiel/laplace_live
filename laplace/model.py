"""Loading a trained model and turning its output into mu and b.

A model file alone does not say what it was trained on, and that cost an
evening: a network whose b was flat by design was traded for weeks because
nothing recorded its feature set. So a model is loaded together with the
card written beside it, and the pair — not the weights alone — is what the
rest of the project consumes. The card carries the feature columns, the
rolling window, the clipping and the intraday profile, so a caller builds
windows the model can actually read instead of building them from a
configuration that may have moved on.

The second output of the network is a raw value, not a scale. Softplus and
MIN_B are applied here so no caller has to remember.

Deseasonalisation lives here for the same reason. A model trained on
y / s(minute) predicts b_norm, and the scale a trading rule needs is
b_norm * s(minute). Keeping that multiplication in one place is what stops
a backtest from running thresholds that mean one thing at 03:00 and
another at the open.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .core.activations import ACTIVATION_CUSTOM_OBJECTS
from .core.losses import CUSTOM_OBJECTS, MIN_B

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """The model could not be loaded, or does not fit the data."""


@dataclass(frozen=True)
class Prediction:
    """One step's predictive distribution.

    Attributes:
        mu: Conditional median of the next log return. The model is trained
            with a Laplace likelihood, whose data term is an absolute
            deviation, so the estimator is the median.
        b: Laplace scale, already seasonal where the model is deseasonalised.
    """

    mu: float
    b: float

    def tunnel(self, k: float) -> tuple[float, float]:
        """Lower and upper edge of the k*b band."""
        return self.mu - k * self.b, self.mu + k * self.b

    def trigger_prices(self, price: float, entry_z: float) -> tuple[float, float]:
        """Prices at which the entry rule would fire during the next bar.

        The rule compares the realised return against entry_z * b, and b is
        known once the window closes, so the threshold can be stated as a
        price before the bar begins:

            P_long  = P * exp( entry_z * b)
            P_short = P * exp(-entry_z * b)

        Args:
            price: Close of the bar the window ends on.
            entry_z: The entry threshold, in units of b.

        Returns:
            (long trigger, short trigger).
        """
        return (
            price * float(np.exp(entry_z * self.b)),
            price * float(np.exp(-entry_z * self.b)),
        )


def read_card(path: Path) -> dict:
    """The model's own record of how it was trained.

    Returns an empty mapping for a model that predates cards; those were
    all seven-feature, un-deseasonalised runs, and the defaults below match
    that case.
    """
    card_path = path.with_suffix(".json")
    if not card_path.is_file():
        logger.warning("No card beside %s; assuming legacy settings", path.name)
        return {}
    return json.loads(card_path.read_text(encoding="utf-8"))


def eastern_minutes(timestamps, broker_offset_hours: float) -> np.ndarray:
    """Minute of day on the New York clock, from broker timestamps.

    Broker stamps carry no zone and sit on a fixed offset from UTC, so they
    are localised to that offset and then converted. Done on the raw stamps
    rather than on an hour encoding, because this has to stay exact across
    the daylight-saving boundary, where the same broker hour is the cash
    open in one season and the macro slot in the other.
    """
    series = pd.Series(pd.to_datetime(np.atleast_1d(timestamps)))
    offset = timezone(timedelta(hours=broker_offset_hours))
    local = series.dt.tz_localize(offset).dt.tz_convert("America/New_York")
    return (local.dt.hour * 60 + local.dt.minute).to_numpy()


def seasonal_scale(card: dict, timestamps) -> np.ndarray:
    """The multiplier this model's b has to be scaled by.

    Interpolated across bucket centres rather than held as steps: the
    profile is a smooth daily shape, and a step function puts cliffs into b
    at bucket edges that no market event explains. Ones for a model that
    was not deseasonalised, so every caller can multiply unconditionally.
    """
    length = len(np.atleast_1d(timestamps))
    if not card.get("deseasonalised"):
        return np.ones(length)

    table = {int(k): float(v) for k, v in card["intraday_profile"].items()}
    if not table:
        return np.ones(length)

    bucket_minutes = int(card.get("profile_bucket_minutes", 15))
    minutes = eastern_minutes(timestamps, float(card.get("broker_utc_offset", 2.0)))

    centres = np.array(sorted(table), dtype=float) + bucket_minutes / 2.0
    values = np.array([table[int(c - bucket_minutes / 2.0)] for c in centres])

    # Wrap at midnight so the interpolation closes the circle.
    extended_x = np.concatenate(([centres[-1] - 1440.0], centres, [centres[0] + 1440.0]))
    extended_y = np.concatenate(([values[-1]], values, [values[0]]))

    return np.maximum(np.interp(minutes.astype(float), extended_x, extended_y), 1e-6)


@dataclass
class ScaledModel:
    """A trained network together with everything needed to use it.

    The window settings are read from the card rather than from the
    configuration, so a model and the features it is fed cannot drift
    apart. Pass these to windows.create_windows instead of the config
    values.
    """

    model: object
    card: dict
    path: Path

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """The columns this model was trained on, in order."""
        return tuple(self.card.get("features", ()))

    @property
    def rolling_window(self) -> int | None:
        """Window for the robust statistics; None when the card is silent."""
        value = self.card.get("rolling_window")
        return int(value) if value is not None else None

    @property
    def lookback(self) -> int | None:
        value = self.card.get("lookback")
        return int(value) if value is not None else None

    @property
    def predict_step(self) -> int | None:
        value = self.card.get("predict_step")
        return int(value) if value is not None else None

    @property
    def zscore_clip(self) -> float:
        """Infinity where the model was trained without clipping."""
        value = self.card.get("zscore_clip")
        return float("inf") if value is None else float(value)

    @property
    def input_shape(self):
        return self.model.input_shape

    def describe_parameters(self) -> str:
        return describe_parameters(self.model)

    def predict_batch(self, x: np.ndarray, timestamps=None):
        """Predict for many windows at once, on the reconstructed scale.

        Args:
            x: Shape (n, lookback, n_features).
            timestamps: One per window, for the seasonal multiplier. Only
                optional for a model that was not deseasonalised; omitting
                them for one that was raises, because silently returning
                b_norm as b is the failure this class exists to prevent.

        Returns:
            (mu, b), both already seasonal.
        """
        mu, b = predict_batch(self.model, x)

        if not self.card.get("deseasonalised"):
            return mu, b

        if timestamps is None:
            raise ModelError(
                f"{self.path.name} was trained deseasonalised, so b has to be "
                f"multiplied by the intraday profile; pass the timestamps."
            )

        scale = seasonal_scale(self.card, timestamps)
        if len(scale) != len(mu):
            raise ModelError(
                f"{len(timestamps)} timestamps for {len(mu)} windows."
            )
        return mu * scale, b * scale

    def predict_one(self, x: np.ndarray, timestamp=None) -> Prediction:
        """Predict for a single window of shape (lookback, n_features)."""
        stamps = None if timestamp is None else [timestamp]
        mu, b = self.predict_batch(x[np.newaxis, ...], stamps)
        return Prediction(mu=float(mu[0]), b=float(b[0]))


def load_model(path: Path):
    """Load a trained Keras model with every custom object it may need.

    compile=False: inference needs neither the optimiser nor the loss, and
    the loss function is not registered for deserialisation.

    Args:
        path: The .keras file.

    Raises:
        ModelError: The file is absent or Keras refused to rebuild it.
    """
    if not path.is_file():
        raise ModelError(f"Model file not found: {path}")

    import tensorflow as tf

    custom = {**CUSTOM_OBJECTS, **ACTIVATION_CUSTOM_OBJECTS}

    try:
        model = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    except Exception as exc:  # Keras raises a wide range of types here
        raise ModelError(f"Could not load {path}: {exc}") from exc

    logger.info("Loaded %s, input %s", path.name, model.input_shape)
    return model


def load_scaled(path: Path) -> ScaledModel:
    """Load a model together with its card. This is what callers want."""
    card = read_card(path)
    model = load_model(path)
    if card:
        logger.info(
            "%s: features %s, rolling %s, clip %s, deseasonalised %s",
            path.name, card.get("features"), card.get("rolling_window"),
            card.get("zscore_clip"), card.get("deseasonalised"),
        )
    return ScaledModel(model=model, card=card, path=path)


def describe_parameters(model) -> str:
    """Report the learned BieLU parameters, empty for a standard activation.

    Written into the run manifest, because a model file alone does not say
    what its kappa settled at and that number has been worth comparing
    across runs.
    """
    import tensorflow as tf

    found = []
    for layer in model.layers:
        if layer.__class__.__name__.startswith("BieLU"):
            for weight in layer.weights:
                effective = tf.nn.softplus(weight).numpy()[()]
                found.append(f"{layer.name}:{effective:.6f}")
    return "; ".join(found)


def predict_batch(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict for many windows at once, without the seasonal multiplier.

    Kept for callers that hold a bare Keras model. Anything that trades
    should use ScaledModel instead.

    Raises:
        ModelError: The window shape does not match the model's input.
    """
    expected = tuple(model.input_shape[1:])
    actual = tuple(x.shape[1:])
    if expected != actual:
        raise ModelError(
            f"The model expects windows of {expected} but was given {actual}. "
            f"A model trained on a different feature set cannot read these windows."
        )

    import tensorflow as tf

    raw = model.predict(x, verbose=0)
    mu = raw[:, 0]
    b = tf.nn.softplus(raw[:, 1]).numpy() + MIN_B
    return mu, b


def predict_one(model, x: np.ndarray) -> Prediction:
    """Predict for a single window, without the seasonal multiplier."""
    mu, b = predict_batch(model, x[np.newaxis, ...])
    return Prediction(mu=float(mu[0]), b=float(b[0]))