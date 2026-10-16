"""
core/metrics.py

Single source for the calibration metrics of the Laplace density network.

These assess the MODEL (probabilistic quality of the predictive distribution),
NOT the trading strategy. Trading metrics (Sharpe, drawdown, profit factor)
stay in the agent (8_backtest). The line is deliberate: this file is the
"paper" side — publishable calibration evidence — and never imports agent code.

Covers Section 6 of the proposal:
  6.1 Primary calibration quality:
        - tunnel coverage (k*b bands vs theoretical Laplace)
        - negative log-likelihood (NLL)
        - Probability Integral Transform (PIT)
        - Continuous Ranked Probability Score (CRPS)
        - Kupiec / Christoffersen VaR tests
  6.2 Secondary local adaptivity (the central claim, H1):
        - correlation of predicted scale b with realised local volatility

----------------------------------------------------------------------------
KEY STATISTICAL NOTE — why MEDIAN, not MEAN
----------------------------------------------------------------------------
The model is trained with the Laplace negative log-likelihood, whose data term
is |y - mu| / b. This is an L1 objective. The estimator that minimises the sum
of ABSOLUTE deviations is the MEDIAN, not the mean:

        argmin_mu  sum |y - mu|   =   median(y)          (Laplace / L1)
        argmin_mu  sum (y - mu)^2 =   mean(y)            (Gauss  / L2)

Therefore the predicted mu is the CONDITIONAL MEDIAN of the return, not the
mean. Error metrics are reported with a MEDIAN-based version (MedAE, median
bias) as the primary figure, because:
  (1) it is consistent with what the model actually estimates (the median),
  (2) it is robust to the heavy tails of the data — a few extreme returns
      inflate the MEAN error but not the MEDIAN error,
  (3) for heavy-tailed (near alpha-stable) returns the mean can be unstable
      or, in the limit, undefined, while the median is always well-defined.
Mean-based versions (MAE, mean bias) are kept alongside only for comparison
with the wider literature, which defaults to the mean.
----------------------------------------------------------------------------
"""

import numpy as np
from scipy.stats import laplace, chi2


# ==============================================================================
# 6.1 — TUNNEL COVERAGE
# ==============================================================================
# The tunnel mu +/- k*b is symmetric around mu = the conditional MEDIAN.
# The Laplace distribution is symmetric about its median, so a symmetric
# band is the natural calibration target here.
# Theoretical coverage of a k*b band under Laplace: P(|y-mu| < k*b) = 1 - e^{-k}.

def tunnel_coverage(y_true, mu, b, k_values=(1, 2, 3, 4)):
    """
    For each k: empirical fraction of y_true inside [mu - k*b, mu + k*b],
    compared to the theoretical Laplace coverage 1 - exp(-k).

    Returns: dict {k: {'coverage': %, 'theoretical': %, 'lower': array, 'upper': array}}
    'lower'/'upper' are returned so the caller can reuse them for plotting.
    """
    stats = {}
    for k in k_values:
        lower = mu - k * b
        upper = mu + k * b
        in_tunnel = (y_true >= lower) & (y_true <= upper)
        coverage = in_tunnel.mean() * 100.0
        theoretical = (1.0 - np.exp(-k)) * 100.0
        stats[k] = {
            'coverage': coverage,
            'theoretical': theoretical,
            'lower': lower,
            'upper': upper,
        }
    return stats


# ==============================================================================
# 6.1 — PREDICTION ERROR (median-first, mean for reference)
# ==============================================================================
# Primary: median absolute error and median bias — consistent with the
# Laplace median estimator and robust to heavy tails.
# Secondary: mean absolute error and mean bias — for literature comparison only.

def prediction_error(y_true, mu):
    """
    Returns a dict of location-accuracy metrics.

    Primary (median-based, consistent with Laplace/L1 and robust to tails):
        medae        = median(|mu - y|)
        median_bias  = median(mu - y)
    Secondary (mean-based, for comparison with the wider literature):
        mae          = mean(|mu - y|)
        mean_bias    = mean(mu - y)
    """
    err = mu - y_true
    return {
        # --- primary: median (matches the estimator, robust to heavy tails) ---
        'medae': float(np.median(np.abs(err))),
        'median_bias': float(np.median(err)),
        # --- secondary: mean (literature default, tail-sensitive) ---
        'mae': float(np.mean(np.abs(err))),
        'mean_bias': float(np.mean(err)),
    }


# ==============================================================================
# 6.1 — NEGATIVE LOG-LIKELIHOOD (NLL)
# ==============================================================================
# The same quantity the model minimises, reported per-sample on held-out data.
# Lower is better. This is the standard MLE quality measure.

def nll(y_true, mu, b):
    """
    Mean Laplace negative log-likelihood on held-out data:
        NLL = log(2b) + |y - mu| / b
    Uses the L1 residual |y - mu|, consistent with the Laplace median.
    """
    return float(np.mean(np.log(2.0 * b) + np.abs(y_true - mu) / b))


# ==============================================================================
# 6.1 — PROBABILITY INTEGRAL TRANSFORM (PIT)
# ==============================================================================
# PIT = CDF_Laplace(y; mu, b). If the predictive distribution is well
# calibrated, the PIT values are Uniform(0,1): mean ~0.5, and a flat histogram.
# Systematic deviation reveals miscalibration (too narrow / too wide / biased).
# The Laplace CDF is centred on mu (the median), so PIT respects the median
# nature of the model automatically.

def pit(y_true, mu, b):
    """
    Probability Integral Transform values, one per observation.
    Return the array so the caller can histogram it or run a uniformity test.
    Well-calibrated -> approximately Uniform(0,1).
    """
    return laplace.cdf(y_true, loc=mu, scale=b)


def pit_summary(y_true, mu, b):
    """
    Convenience summary of PIT calibration:
        pit_mean  -> should be ~0.5
        pit_std   -> should be ~0.2887 (= 1/sqrt(12), std of Uniform(0,1))
    Deviation of pit_mean from 0.5 indicates a location (median) bias;
    deviation of pit_std indicates the bands are too wide or too narrow.
    """
    p = pit(y_true, mu, b)
    return {
        'pit_mean': float(p.mean()),          # target 0.5
        'pit_std': float(p.std()),            # target 0.2887
        'pit_ideal_std': float(1.0 / np.sqrt(12.0)),
    }


# ==============================================================================
# 6.1 — CONTINUOUS RANKED PROBABILITY SCORE (CRPS)
# ==============================================================================
# CRPS rewards a forecast distribution that is both accurate AND sharp
# (narrow when it can be). It generalises MAE to full distributions.
# Closed form for a Laplace(mu, b) forecast, with z = |y - mu| / b:
#       CRPS = b * ( z + exp(-z) - 3/3 )
# Lower is better. Verified numerically: CRPS increases when b is mis-scaled.

def crps_laplace(y_true, mu, b):
    """
    Mean CRPS for a Laplace predictive distribution (closed form).
    z = |y - mu| / b ;  CRPS = b * (z + exp(-z) - 3/3).
    Lower is better; always >= 0.
    """
    z = np.abs(y_true - mu) / b
    crps = b * (z + np.exp(-z) - 0.75)
    return float(np.mean(crps))


# ==============================================================================
# 6.1 — KUPIEC / CHRISTOFFERSEN VaR TESTS
# ==============================================================================
# The k*b tunnels are, in risk terms, Value-at-Risk intervals: a point
# OUTSIDE the tunnel is a "failure" (exception). For a Laplace tunnel the
# expected failure probability is exp(-k).
#
# Kupiec POF (Proportion Of Failures): tests whether the OBSERVED failure
#   rate matches the expected one (unconditional coverage). LR ~ chi2(1).
#
# Christoffersen independence: tests whether failures are INDEPENDENT in
#   time (no clustering). Clustered exceptions mean the model misses regimes
#   of high volatility even if the average rate is right. LR ~ chi2(1).
#
# For both: a HIGH p-value = the model passes (calibrated / independent).

def kupiec_pof(y_true, mu, b, k):
    """
    Kupiec Proportion-Of-Failures test for the k*b tunnel.
    Expected failure prob p = exp(-k). Tests unconditional coverage.
    Returns dict with failures, observed vs expected rate, LR stat, p-value.
    """
    lower = mu - k * b
    upper = mu + k * b
    inside = (y_true >= lower) & (y_true <= upper)
    failures = int((~inside).sum())
    N = len(y_true)
    p_exp = np.exp(-k)
    pi_obs = failures / N

    # Guard: LR undefined if no or all failures
    if failures == 0 or failures == N:
        return {'k': k, 'failures': failures, 'obs_rate': pi_obs,
                'exp_rate': float(p_exp), 'LR': float('nan'), 'p_value': float('nan')}

    lr = -2.0 * (
        (N - failures) * np.log((1.0 - p_exp) / (1.0 - pi_obs))
        + failures * np.log(p_exp / pi_obs)
    )
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {'k': k, 'failures': failures, 'obs_rate': float(pi_obs),
            'exp_rate': float(p_exp), 'LR': float(lr), 'p_value': p_value}


def christoffersen_independence(y_true, mu, b, k):
    """
    Christoffersen independence test for the k*b tunnel.
    Tests whether tunnel breaches cluster in time (regime failure) rather
    than occurring independently. LR ~ chi2(1); HIGH p-value = independent.

    Builds the 2x2 transition counts of the breach indicator:
        n_ij = number of transitions from state i to state j
        (state 1 = breach, 0 = inside)
    """
    lower = mu - k * b
    upper = mu + k * b
    breach = ((y_true < lower) | (y_true > upper)).astype(int)

    # Transition counts between consecutive steps
    prev = breach[:-1]
    curr = breach[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # Conditional breach probabilities
    denom0 = n00 + n01
    denom1 = n10 + n11
    if denom0 == 0 or denom1 == 0:
        return {'k': k, 'LR': float('nan'), 'p_value': float('nan')}

    pi01 = n01 / denom0                     # P(breach | previous inside)
    pi11 = n11 / denom1                     # P(breach | previous breach)
    pi = (n01 + n11) / (denom0 + denom1)    # overall breach prob

    # Degenerate guards (log of 0)
    if pi in (0.0, 1.0) or pi01 in (0.0, 1.0) or pi11 in (0.0, 1.0):
        return {'k': k, 'LR': float('nan'), 'p_value': float('nan')}

    # Likelihood ratio: independent vs first-order Markov
    log_l_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    log_l_alt = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
                 + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    lr = -2.0 * (log_l_null - log_l_alt)
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {'k': k, 'LR': float(lr), 'p_value': p_value}


# ==============================================================================
# 6.2 — LOCAL ADAPTIVITY: correlation of b with realised volatility (H1)
# ==============================================================================
# THIS IS THE CENTRAL CLAIM OF THE PROJECT.
#
# A generic activation (ReLU) tends to collapse the predicted scale b to a
# near-constant value -> the tunnel has the same width everywhere -> it cannot
# track intraday volatility regimes. A process-derived activation should let b
# widen in volatile stretches and tighten in calm ones.
#
# We quantify this as the correlation between the predicted scale b and a
# rolling estimate of realised local volatility. Realised volatility uses the
# MEDIAN-based MAD (median absolute deviation) rather than the standard
# deviation, staying consistent with the Laplace/median framing and robust to
# the heavy tails:
#       realised_vol_t = median(|y - median(y)|) over a trailing window
#
#   corr ~ 0      -> constant b, fixed-width bands  (expected for ReLU)
#   corr  > 0     -> b tracks the local regime      (expected for BieLU)
#
# If b is constant (std = 0) the correlation is mathematically undefined;
# we return 0.0 (not NaN) — a constant scale has, by definition, zero
# co-variation with anything.

def realised_volatility_mad(y_true, window=20):
    """
    Trailing realised local volatility via median absolute deviation (MAD).
    MAD is the median-based, tail-robust analogue of the rolling std, and is
    the natural volatility proxy under a Laplace/median model.
    Returns an array aligned with y_true; first `window` entries are NaN.
    """
    n = len(y_true)
    rv = np.full(n, np.nan)
    for i in range(window, n):
        w = y_true[i - window:i]
        med = np.median(w)
        rv[i] = np.median(np.abs(w - med))   # MAD
    return rv


# A scale is treated as "constant" if its variation is below this tolerance.
# NOTE: a collapsed model does not give b_std of exactly 0.0 — it gives float
# noise (e.g. ~1e-11 or ~1e-20). Testing `== 0.0` would miss this and produce
# a meaningless ~1e-16 correlation from numerical dust. The tolerance treats
# such a scale as constant, which is the correct interpretation.
_B_CONSTANT_TOL = 1e-5

# Correlation threshold for judging whether b tracks local volatility
_CORR_THRESHOLD = 0.15


def scale_volatility_corr(b, y_true, window=20):
    """
    Correlation between predicted scale b and realised local volatility (MAD).
    This is the quantitative test of H1 (local adaptivity).

    Returns dict:
        corr           -> Pearson correlation (0.0 if b is effectively constant)
        b_is_constant  -> True if predicted b has (near) zero variation
        b_std          -> std of b, so the caller can see the collapse directly
        verdict        -> interpretation of the relationship
    """
    rv = realised_volatility_mad(y_true, window)
    mask = ~np.isnan(rv)
    b_valid = np.asarray(b)[mask]
    rv_valid = rv[mask]

    b_std = float(np.std(b_valid))
    rv_std = float(np.std(rv_valid))

    # An (effectively) constant scale has no co-variation -> correlation
    # undefined -> return 0.0. This is the expected ReLU baseline: b collapses
    # to one value (up to float noise), so the tunnels are fixed-width.
    if b_std < _B_CONSTANT_TOL or rv_std < _B_CONSTANT_TOL:
        return {'corr': 0.0, 'b_is_constant': True, 'b_std': b_std,
                'verdict': 'b is effectively constant (fixed-width tunnels)'}

    corr = float(np.corrcoef(b_valid, rv_valid)[0, 1])

    # Three-way verdict based on correlation sign and magnitude
    if corr >= _CORR_THRESHOLD:
        verdict = 'b tracks the local regime (adaptive tunnels)'
    elif corr <= -_CORR_THRESHOLD:
        verdict = 'b moves inversely to local volatility'
    else:
        verdict = 'b varies but is uncorrelated with local volatility'

    return {'corr': corr, 'b_is_constant': False, 'b_std': b_std, 'verdict': verdict}


# ==============================================================================
# CONVENIENCE — safe Pearson correlation (guards constant inputs)
# ==============================================================================

def safe_correlation(a, b):
    """
    Pearson correlation with a guard: if either input is (effectively) constant
    the correlation is undefined, so return 0.0 instead of NaN or numerical dust.
    Uses a tolerance because a collapsed predictor gives float noise, not 0.0.
    """
    if np.std(a) < _B_CONSTANT_TOL or np.std(b) < _B_CONSTANT_TOL:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])