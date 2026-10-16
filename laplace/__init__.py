"""
laplace — live execution layer for the Laplace density network.

The research project trains the model and measures its calibration. This
package takes a trained model and runs it forward on candles the MT4 bridge
stores, one five-minute bar at a time.

Layout:

    core/       copied verbatim from the research repository, never edited
    reader      bridge CSV -> the frame the research pipeline expects
    features    the seven model features, same arithmetic as 2_build_features
    windows     day-bounded windows, same construction as 3_prepare_windows
    model       loading and prediction
    rules       the entry and exit rules, lifted from 8_backtest
    money       position size and profit in account currency
    journal     per-run directory, manifest and CSV output
    live        the bar-close loop

The rules run identically in the backtest and live: one engine, two drivers.
"""

__version__ = "0.1.0"
