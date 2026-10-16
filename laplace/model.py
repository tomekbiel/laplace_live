"""
Loading the trained model and turning its output into mu and b.

Two things here are fixes rather than copies, and both are the difference
between running and not running:

* 8_backtest.py passed only core.losses.CUSTOM_OBJECTS to load_model. A model
  whose layers include BieLU_OU or BieLU_Laplace cannot be rebuilt from that
  alone, so loading one raises. core.activations says in a comment that the
  two dictionaries must be merged at load time; this does the merging.
* The model path is composed the way 5_train.py writes it. 8_backtest.py
  still pointed at models/saved/laplace_minimal.keras, the layout from before
  runs were versioned.

The second output of the network is a raw value, not a scale. Every consumer
in the research code applies softplus and adds MIN_B before using it, and
getting that wrong gives a plausible-looking number that is not b. It is done
in one place here so no caller has to remember.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
        b: Laplace scale. The tunnel mu +/- k*b covers 1 - exp(-k) of the
            distribution.
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

        This is the same inequality in price terms, nothing new. It is what
        lets a decision be announced in advance.

        Args:
            price: Close of the bar the window ends on.
            entry_z: The CEM entry threshold.

        Returns:
            (long trigger, short trigger).
        """
        return (
            price * float(np.exp(entry_z * self.b)),
            price * float(np.exp(-entry_z * self.b)),
        )


def load_model(path: Path):
    """Load a trained Keras model with every custom object it may need.

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
        model = tf.keras.models.load_model(path, custom_objects=custom)
    except Exception as exc:  # Keras raises a wide range of types here
        raise ModelError(f"Could not load {path}: {exc}") from exc

    logger.info("Loaded %s, input %s", path.name, model.input_shape)
    return model


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
    """Predict for many windows at once.

    Args:
        model: A loaded model.
        x: Shape (n, lookback, n_features).

    Returns:
        (mu, b) as arrays of length n, with b already passed through softplus.

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
    """Predict for a single window of shape (lookback, n_features)."""
    mu, b = predict_batch(model, x[np.newaxis, ...])
    return Prediction(mu=float(mu[0]), b=float(b[0]))
