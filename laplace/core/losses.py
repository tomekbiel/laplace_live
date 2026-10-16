"""
core/losses.py
Single source for the Laplace loss function and helper metrics.
Moved 1:1 from 5_train.py (Phase 1 — zero change in behaviour).
The name 'laplace_loss' is kept deliberately: previously saved models
store this name internally, so load_model resolves them via CUSTOM_OBJECTS.
"""

import tensorflow as tf


# ==============================================================================
# CONSTANTS
# ==============================================================================

MIN_B = 1e-9  # minimum scale parameter to prevent nervousness in flat market


# ==============================================================================
# LAPLACE LOSS FUNCTION (manual implementation)
# ==============================================================================

def laplace_loss(y_true, y_pred):
    """
    y_true: actual log_returns (batch_size, 1)
    y_pred: [mu, b] (batch_size, 2)
    """
    mu = y_pred[:, 0:1]
    b = y_pred[:, 1:2]

    # Safeguard: b must be > 0
    b = tf.math.softplus(b) + 1e-6

    # Floor on b to prevent nervousness in flat market (Zjawisko B)
    # Minimum b prevents model from becoming too confident when volatility is low
    b = tf.maximum(b, MIN_B)

    # Negative log-likelihood for the Laplace distribution
    loss = tf.math.log(2 * b) + tf.abs(y_true - mu) / b

    return tf.reduce_mean(loss)


# ==============================================================================
# HELPER METRICS
# ==============================================================================

def median_absolute_error(y_true, y_pred):
    """Median of absolute errors (robust to outliers)"""
    e = tf.abs(y_true - y_pred[:, 0])
    n = tf.shape(e)[0]
    s = tf.sort(e)
    return (s[(n - 1) // 2] + s[n // 2]) / 2.0


# ==============================================================================
# CUSTOM OBJECTS FOR MODEL LOADING
# ==============================================================================
# Every file that calls load_model passes this as custom_objects.
# Keys MUST match the names used when the model was saved.

CUSTOM_OBJECTS = {
    'laplace_loss': laplace_loss,
    'median_absolute_error': median_absolute_error,
    'MIN_B': MIN_B,
}