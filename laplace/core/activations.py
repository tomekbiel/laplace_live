"""
core/activations.py

Single source for activation functions used in the calibration experiments.

Two tiers:
  1. BASELINE — standard activations, selected by string name, swapped in a loop.
     (ReLU, GELU, ELU, SELU, plus the closest family members from the survey.)
  2. BieLU FAMILY — the project's own learnable activations, each a Layer with a
     trainable parameter. These are Tomasz's contribution and are kept separate,
     fully commented, deterministic (NO stochastic noise), and smooth at zero.

----------------------------------------------------------------------------
BieLU FAMILY — mathematical basis (verified numerically before implementation)
----------------------------------------------------------------------------
The family has the shape  f(x) = x * g(|x|)  where g is a decaying kernel:

    BieLU-OU:       f(x) = x * exp(-kappa * |x|)     kappa learnable (> 0)
    BieLU-Laplace:  f(x) = x * exp(-|x| / b)         b     learnable (> 0)

Behaviour (mean-reversion analogue of the Ornstein-Uhlenbeck process):
  - small |x|  -> exp(...) ~ 1  -> f(x) ~ x   (signal passes through the centre)
  - large |x|  -> exp(...) -> 0 -> f(x) -> 0  (extremes are damped, "pulled back")
  This mirrors OU mean-reversion: the further from the centre, the stronger the
  pull toward it. The heavy-tailed exp(-|x|) kernel is the Laplace kernel.

Smoothness at the origin (the key advantage over ReLU):
  The kink of |x| is multiplied by x, which cancels it. Verified numerically:
      f'(0+) = f'(0-) = 1   for all kappa
  So BieLU is CONTINUOUSLY DIFFERENTIABLE at 0 (no subgradient needed), unlike
  ReLU. This also makes it suitable for ODE/continuous-depth architectures,
  where smooth vector fields are required and ReLU fails.

Honesty of the claim: this form is INSPIRED BY the mean-reverting structure of
the OU process with a Laplace (heavy-tailed) kernel. It is NOT a closed-form
solution of the OU SDE (that solution is Gaussian). "Inspired by", not
"derived from".

The learnable parameter is constrained positive via softplus, and is shared
per layer (one scalar) by default — a per-feature variant is possible later.
----------------------------------------------------------------------------
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


# ==============================================================================
# TIER 1 — BASELINE ACTIVATIONS (selected by string, swapped in a loop)
# ==============================================================================
# Standard, well-known activations plus the three closest family members from
# the activation survey (rectified sech, Suish, LaLU). The family members are
# the meaningful baselines for the BieLU claim: if BieLU beats them, it shows
# the LEARNABLE parameter + process-derived form matter, not just the x*g(|x|)
# shape which they already have.

BASELINE_NAMES = [
    'relu',       # reference; kink at 0, discards negative pre-activations
    'gelu',       # strongest standard baseline (transformer default)
    'elu',        # exponential negative branch; mitigates dead neurons
    'selu',       # self-normalising variant
]

FAMILY_BASELINE_NAMES = [
    'rsech',      # rectified sech: x*sech(x). Symmetric kernel, NO parameter.
    'suish',      # max(x, x*exp(-|x|)). Contains the BieLU term but with max, no param.
    'lalu',       # x * Laplace-CDF(x). "Laplace" but a CDF-GATE (rising), not a kernel.
]


# --- family baselines implemented as plain functions (no learnable params) ---

@register_keras_serializable(package="bielu")
def _rsech(x):
    """Rectified hyperbolic secant: f(x) = x * sech(x) = x / cosh(x).
    Closest symmetric kernel to BieLU but WITHOUT a learnable parameter."""
    return x / tf.math.cosh(x)


@register_keras_serializable(package="bielu")
def _suish(x):
    """Suish: f(x) = max(x, x * exp(-|x|)).
    Contains the term x*exp(-|x|) (BieLU-OU at kappa=1) but gated by max and
    with no learnable parameter -> asymmetric, fixed."""
    return tf.maximum(x, x * tf.exp(-tf.abs(x)))


@register_keras_serializable(package="bielu")
def _lalu(x):
    """LaLU: f(x) = x * CDF_Laplace(x).
    Uses the Laplace CDF (a rising 0->1 gate, like GELU) — fundamentally
    different from BieLU's decaying kernel. Included to show 'Laplace as a gate'
    is not the same as 'Laplace as a damping kernel'."""
    cdf = tf.where(x < 0, 0.5 * tf.exp(x), 1.0 - 0.5 * tf.exp(-x))
    return x * cdf


def get_activation(name):
    """
    Return an activation usable inside a Dense layer's `activation=` argument,
    OR a callable for the family baselines. Standard names resolve to Keras
    built-ins; family names resolve to the functions above.

    Usage in the model:
        x = layers.Dense(128, activation=get_activation(ACTIVATION))(x)

    For the LEARNABLE BieLU family, do NOT use this — use the Layer classes
    below (they hold trainable parameters and cannot be a plain `activation=`).
    """
    name = name.lower()

    # Standard Keras built-ins (strings Keras understands directly)
    if name in ('relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'linear'):
        return name

    # Family baselines (functions, no learnable parameters)
    if name == 'rsech':
        return _rsech
    if name == 'suish':
        return _suish
    if name == 'lalu':
        return _lalu

    raise ValueError(
        f"Unknown activation '{name}'. "
        f"Use one of {BASELINE_NAMES + FAMILY_BASELINE_NAMES}, "
        f"or the BieLU Layer classes for the learnable family."
    )


# ==============================================================================
# TIER 2 — BieLU FAMILY (learnable Layers, the project's contribution)
# ==============================================================================
# These are Layers (not plain functions) because they carry a trainable
# parameter. Use them explicitly in the model instead of `activation=`:
#
#     x = layers.Dense(128)(x)      # NO activation here
#     x = BieLU_OU()(x)             # BieLU applied as its own layer
#
# Each is deterministic (no noise), smooth at 0, and constrains its parameter
# positive via softplus.

@register_keras_serializable(package="bielu")
class BieLU_OU(layers.Layer):
    """
    BieLU-OU:  f(x) = x * exp(-kappa * |x|),  kappa learnable, kappa > 0.

    OU-inspired mean-reversion: the further x is from 0, the more it is damped;
    near 0 the signal passes through almost unchanged. Peak magnitude at
    |x| = 1/kappa. Smooth at 0 (f'(0) = 1).

    kappa is stored as a raw variable and passed through softplus so it stays
    strictly positive. One scalar shared across the layer by default.
    """

    def __init__(self, kappa_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kappa_init = float(kappa_init)

    def build(self, input_shape):
        # Raw parameter; softplus(raw) gives the positive kappa actually used.
        # Initialise raw so that softplus(raw) ~ kappa_init.
        # softplus^-1(y) = log(exp(y) - 1)
        import math
        raw_init = math.log(math.exp(self.kappa_init) - 1.0)
        self.kappa_raw = self.add_weight(
            name='kappa_raw',
            shape=(),                      # single scalar shared over the layer
            initializer=tf.constant_initializer(raw_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        kappa = tf.nn.softplus(self.kappa_raw)      # ensure kappa > 0
        return x * tf.exp(-kappa * tf.abs(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'kappa_init': self.kappa_init})
        return cfg


@register_keras_serializable(package="bielu")
class BieLU_Laplace(layers.Layer):
    """
    BieLU-Laplace:  f(x) = x * exp(-|x| / b),  b learnable, b > 0.

    Same x*g(|x|) shape as BieLU-OU, but parameterised by a Laplace SCALE b
    (the width of the "pass-through" region) instead of a rate kappa.
    Equivalent up to reparameterisation (b = 1/kappa), but exposes the scale
    directly, which is the natural quantity in the Laplace framing. Peak
    magnitude at |x| = b. Smooth at 0 (f'(0) = 1).
    """

    def __init__(self, b_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.b_init = float(b_init)

    def build(self, input_shape):
        import math
        raw_init = math.log(math.exp(self.b_init) - 1.0)
        self.b_raw = self.add_weight(
            name='b_raw',
            shape=(),
            initializer=tf.constant_initializer(raw_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        b = tf.nn.softplus(self.b_raw) + 1e-6       # ensure b > 0
        return x * tf.exp(-tf.abs(x) / b)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'b_init': self.b_init})
        return cfg


# ==============================================================================
# CUSTOM OBJECTS — for loading models that used a BieLU layer
# ==============================================================================
# When a saved model contains a BieLU layer, load_model needs these classes.
# Merge with core.losses.CUSTOM_OBJECTS at load time.

ACTIVATION_CUSTOM_OBJECTS = {
    'BieLU_OU': BieLU_OU,
    'BieLU_Laplace': BieLU_Laplace,
}