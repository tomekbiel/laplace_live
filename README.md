# laplace_live

Runs a trained Laplace density network forward on candles the MT4 bridge
stores, one five-minute bar at a time.

The research project trains the model and measures its calibration. This one
takes a finished model and trades it. The two are kept apart on purpose: the
research pipeline writes `results/<dataset>/<architecture>/models/*.keras`,
and this project only ever reads from there.

## Layout

    config/strategy.json      which model, which instrument, which thresholds
    config/instruments.json   contract specifications, for the money layer
    laplace/core/             copied verbatim from the research repo, never edited
    laplace/reader.py         bridge CSV -> the frame the pipeline expects
    laplace/features.py       the seven features, same arithmetic as 2_build_features
    laplace/windows.py        day-bounded windows, same as 3_prepare_windows
    laplace/model.py          loading and prediction
    laplace/rules.py          the entry and exit rules, as a state machine
    laplace/money.py          position size and profit in account currency
    laplace/journal.py        per-run directory, manifest and CSVs
    laplace/live.py           the bar-close loop
    scripts/                  the four entry points
    runs/                     one directory per run

## One rule engine, two drivers

`laplace/rules.py` holds the rules. `scripts/backtest.py` drives it over
stored data and `laplace/live.py` drives it over the socket. Both execute the
same code, so a live run and the reference run can only differ through the
market or through execution, never through two copies of the rules drifting
apart.

## Order of work

    python scripts/predict_last.py          one prediction, nothing written
    python scripts/backtest.py              the reference figures
    python scripts/run_paper.py             the loop, no orders
    python scripts/run_live.py              the loop, orders

`predict_last.py` needs only the bridge's stored candles and a model file.
`backtest.py` additionally needs the CEM parameters. The two runners need a
running terminal with the expert advisor attached.

## Keeping core/ honest

    diff -r laplace/core <research-repo>/core

Must print nothing. A difference means the live strategy has moved away from
the model that was trained and measured.

## Behaviour carried over unchanged

These come from `8_backtest.py` and are reproduced rather than corrected, so
that the first live comparison is against the figures that are already known.
Each is a candidate for the improvement stage.

1. **Entry books the bar that triggered it.** The position opens at step *s*
   and profit is summed from `y[s]` inclusive — the return that produced the
   signal. Live, that move is already over when the bar closes, so a live run
   gives up about `entry_z * b` per trade against the backtest.

2. **The short branches of take_profit and stop_loss are inverted.**
   `cum_return` already carries the position sign and the short branches
   negate it again. Measured on a synthetic series: of 122 shorts, one exited
   as `take_profit` booking a **loss** of -0.0016, while 114 exited as
   `timeout_loss` booking an average **profit** of +0.0009. The exits happen;
   their labels mean the opposite, and the thresholds that govern them are
   not the ones the names suggest.

3. **Equity is recorded only while a position is open.** Flat bars add no
   point, so drawdown is measured over trading time rather than wall-clock
   time.

4. **Leverage is applied after the drawdown ratio.** `max_drawdown` is a
   ratio of equity multiplied by 20 afterwards, which is why it can exceed
   one hundred per cent. It is not a fraction of an account.

5. **No transaction costs.** Returns come from the bid close; the spread is
   never subtracted. On US.100+ the spread measured 1.4 index points at
   30830, about 4.5 basis points per side.

6. **`max_hold` and `time_decay` were inert during CEM.** The Monte Carlo
   paths are 24 steps long and `max_hold` ranges over 48 to 144, so the
   condition could never be met while the parameters were being optimised.
   Both are live here, where a position can last that long.

## Prices and clocks

Timestamps are broker server time, UTC+2, as the bridge recorded in
`data/meta/timebase.json`. They are never converted: `hour_sin` and
`hour_cos` encode the clock the model learned.

MetaTrader builds bars from **bid** prices, so the close the model trained on
is a bid close. Anything feeding live quotes into this pipeline must use the
bid for the same reason; a mid quote sits half a spread away from everything
the model has seen. The broker reports a spread of zero in the candles, so
the ask cannot be recovered from stored history.
