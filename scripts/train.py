"""Train the Laplace density network, with the daily cycle taken out first.

The intraday volatility profile is deterministic and already measured: it
is a lookup table over the minute of day, estimated from far more bars than
a network sees in any one window. Making the network learn it wastes
capacity and, as measured here, lets the clock account for almost all of
b's variation — a calendar wearing the name of a tunnel.

So the scale is decomposed:

    y_norm = y / s(minute)          what the network is trained on
    b      = b_norm * s(minute)     the scale handed back to the caller
    mu     = mu_norm * s(minute)

s is the seasonal part, measured; b_norm is the reactive part, learned.
This is the usual intraday deseasonalisation, in the manner of Andersen and
Bollerslev, and it removes the clock features from the input entirely.

The profile is built on the US Eastern clock, because the events that move
this instrument are tied to New York and the broker clock sits on a fixed
offset: the same broker hour is the cash open in summer and the macro slot
in winter. It is estimated from the TRAINING slice only.

Every figure reported is on the reconstructed scale, so a deseasonalised
model and a plain one are compared on the same quantity.

Usage:
    python scripts/train.py --features level --deseasonalise
    python scripts/train.py --features level --deseasonalise --zscore-clip 0 ^
                            --rolling-window 60 --activations bielu_ou
    python scripts/train.py --features full4          # legacy, clock inside
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from laplace import features as feature_builder
from laplace import reader, windows
from laplace.config import load_config, setup_logging
from laplace.core.activations import BieLU_Laplace, BieLU_OU, get_activation

try:
    from laplace.core.losses import MIN_B
except ImportError:
    MIN_B = 1e-9

# Deseasonalised sets carry no clock. full4 and full7 are kept only as the
# comparators that have to be beaten before the clock features are deleted
# from features.py for good.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "min1": ("log_return",),
    "level": ("log_return", "modified_z_score"),
    "full4": ("log_return", "modified_z_score", "hour_sin", "hour_cos"),
    "full7": feature_builder.FEATURE_COLUMNS,
}

LEARNABLE = {"bielu_ou": BieLU_OU, "bielu_laplace": BieLU_Laplace}

DEFAULT_ACTIVATIONS = ("relu", "gelu", "bielu_ou")


def eastern_minutes(timestamps: np.ndarray, broker_offset_hours: float) -> np.ndarray:
    """Minute of day on the New York clock, from broker timestamps.

    Broker stamps carry no zone and sit on a fixed offset from UTC, so they
    are localised to that offset and then converted. Done on the raw stamps
    rather than on an hour encoding, because this has to be exact across the
    daylight-saving boundary.
    """
    series = pd.Series(pd.to_datetime(timestamps))
    offset = timezone(timedelta(hours=broker_offset_hours))
    local = series.dt.tz_localize(offset).dt.tz_convert("America/New_York")
    return (local.dt.hour * 60 + local.dt.minute).to_numpy()


def intraday_profile(
    minutes: np.ndarray,
    y: np.ndarray,
    fit: slice,
    bucket_minutes: int = 15,
    smooth: int = 3,
) -> dict[int, float]:
    """Typical absolute return per minute-of-day bucket, normalised to mean 1.

    The median, not the mean: one macro print must not define the level the
    whole bucket is divided by. Estimated on `fit` alone, so nothing from the
    validation or test period reaches the training target.

    Returns a bucket -> scale map whose values average one, which keeps
    y_norm in the same order of magnitude as y.
    """
    buckets = (minutes // bucket_minutes) * bucket_minutes
    absolute = np.abs(y)

    fit_buckets = buckets[fit]
    fit_absolute = absolute[fit]

    table: dict[int, float] = {}
    for bucket in np.unique(fit_buckets):
        values = fit_absolute[fit_buckets == bucket]
        if values.size >= 20:
            table[int(bucket)] = float(np.median(values))

    if not table:
        raise SystemExit("The training slice is too thin to build a profile.")

    fallback = float(np.median(list(table.values())))

    # A neighbour median smooths the thinnest buckets without flattening the
    # open, which is a real step rather than noise.
    keys = sorted(table)
    if smooth > 1:
        raw = [table[k] for k in keys]
        half = smooth // 2
        smoothed = [
            float(np.median(raw[max(0, i - half):i + half + 1]))
            for i in range(len(raw))
        ]
        table = dict(zip(keys, smoothed))

    mean_level = float(np.mean(list(table.values()))) or fallback
    return {k: v / mean_level for k, v in table.items()}


def profile_scale(
    minutes: np.ndarray, table: dict[int, float], bucket_minutes: int
) -> np.ndarray:
    """Look the profile up for every bar, falling back to one where unseen."""
    buckets = (minutes // bucket_minutes) * bucket_minutes
    scale = np.array([table.get(int(b), 1.0) for b in buckets], dtype=np.float64)
    return np.maximum(scale, 1e-6)


def build_model(input_shape: tuple[int, int], activation: str, width: int, depth: int):
    """A dense head over the flattened window: two outputs, mu and raw b."""
    from tensorflow import keras

    inputs = keras.Input(shape=input_shape, name="window")
    x = keras.layers.Flatten()(inputs)

    for index in range(depth):
        if activation in LEARNABLE:
            x = keras.layers.Dense(width, name=f"dense_{index}")(x)
            x = LEARNABLE[activation](name=f"{activation}_{index}")(x)
        else:
            x = keras.layers.Dense(
                width, activation=get_activation(activation), name=f"dense_{index}"
            )(x)

    outputs = keras.layers.Dense(2, name="laplace_head")(x)
    return keras.Model(inputs, outputs, name=f"{activation}_dense{width}")


def laplace_nll(y_true, y_pred):
    """Laplace negative log-likelihood on whatever target is supplied."""
    import tensorflow as tf

    y = tf.reshape(tf.cast(y_true, tf.float32), (-1,))
    mu = y_pred[:, 0]
    b = tf.nn.softplus(y_pred[:, 1]) + MIN_B
    return tf.reduce_mean(tf.math.log(2.0 * b) + tf.abs(y - mu) / b)


def chronological_split(
    count: int, val_fraction: float, test_fraction: float
) -> tuple[slice, slice, slice]:
    """Split by time. A random split puts tomorrow in the training set."""
    test_size = int(count * test_fraction)
    val_size = int(count * val_fraction)
    train_end = count - val_size - test_size
    return (
        slice(0, train_end),
        slice(train_end, train_end + val_size),
        slice(train_end + val_size, count),
    )


def evaluate(
    model, X: np.ndarray, y: np.ndarray, scale: np.ndarray, minutes: np.ndarray
) -> dict:
    """Report on the reconstructed scale, so every run is comparable.

    b varies on two time scales and they are not interchangeable. The slow
    part follows the clock; the fast part reacts to what the returns just
    did. Only the fast part makes a tunnel widen on a spike, so the two are
    reported separately rather than summed into one spread.
    """
    import tensorflow as tf

    raw = model.predict(X, verbose=0)
    mu = raw[:, 0] * scale
    b = (tf.nn.softplus(raw[:, 1]).numpy() + MIN_B) * scale

    deviation = np.abs(y - mu)
    nll = float(np.mean(np.log(2.0 * b) + deviation / b))
    coverage = {f"k{k}": float(np.mean(deviation <= k * b)) for k in (1, 2, 3)}

    hours = minutes // 60
    hour_means = {h: float(b[hours == h].mean()) for h in np.unique(hours)}
    clock_curve = np.array([hour_means[h] for h in hours])
    residual = b - clock_curve

    clock_r2 = float(1.0 - residual.var() / b.var()) if b.var() > 0 else float("nan")

    # The empirical multiplier: no distribution is assumed, so this is the
    # tunnel that actually covers what it claims to cover.
    normalised = deviation / b
    k_empirical = {
        f"p{int(p * 100)}": float(np.quantile(normalised, p)) for p in (0.63, 0.95, 0.99)
    }

    return {
        "nll": nll,
        "b_mean": float(b.mean()),
        "b_spread": float(b.std() / b.mean()) if b.mean() else float("nan"),
        "b_clock_r2": clock_r2,
        "b_reactive": float(residual.std() / b.mean()) if b.mean() else float("nan"),
        "corr_b_abs_dev": float(np.corrcoef(b, deviation)[0, 1]),
        "coverage": coverage,
        "k_empirical": k_empirical,
        "b_by_hour": {int(h): v for h, v in hour_means.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=tuple(FEATURE_SETS), default="level")
    parser.add_argument("--activations", nargs="+", default=list(DEFAULT_ACTIVATIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--deseasonalise", action="store_true",
                        help="divide the target by the measured intraday profile")
    parser.add_argument("--profile-bucket", type=int, default=15,
                        help="minutes per profile bucket")
    parser.add_argument("--broker-utc-offset", type=float, default=2.0)
    parser.add_argument("--zscore-clip", type=float, default=None,
                        help="override the clip on modified_z_score; 0 disables it")
    # Longer windows make the robust statistics steadier: a 20-bar MAD can
    # collapse in a quiet stretch and turn one ordinary move into z = 160.
    # A better-estimated denominator shortens the tail without censoring it,
    # and once the window exceeds the lookback the z-score stops being
    # derivable from the window's own returns and starts carrying context.
    parser.add_argument("--rolling-window", type=int, default=None,
                        help="window for the robust statistics; the config "
                             "value when omitted")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--tail", type=int, default=200000)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    columns = FEATURE_SETS[args.features]
    clip = cfg.windows.zscore_clip if args.zscore_clip is None else args.zscore_clip
    clip = float("inf") if clip <= 0 else clip
    # Resolved here beside the clip, because both shape the feature rather
    # than the model, and a run is only reproducible if both are recorded.
    rolling = (cfg.windows.rolling_window if args.rolling_window is None
               else args.rolling_window)

    frame = reader.read_series(
        cfg.data.series_path, cfg.data.symbol, cfg.data.price_column, tail=args.tail
    )
    prepared = feature_builder.prepare(
        frame,
        window_size=rolling,
        gap_threshold_days=cfg.data.gap_threshold_days,
    )

    X, y, ts, _, _, _ = windows.create_windows(
        prepared,
        lookback=cfg.windows.lookback,
        predict_step=cfg.windows.predict_step,
        feature_cols=columns,
        zscore_clip=clip,
    )
    if len(X) == 0:
        print("No windows could be built.")
        return 1

    train, val, test = chronological_split(len(X), args.val_fraction, args.test_fraction)
    minutes = eastern_minutes(ts, args.broker_utc_offset)

    if args.deseasonalise:
        table = intraday_profile(minutes, y, train, args.profile_bucket)
        scale = profile_scale(minutes, table, args.profile_bucket)
        target = (y / scale).astype(np.float32)
    else:
        table = {}
        scale = np.ones(len(y))
        target = y

    clip_tag = "noclip" if clip == float("inf") else f"clip{clip:g}"
    season_tag = "deseason" if args.deseasonalise else "raw"

    print(f"feature set : {args.features} {columns}")
    print(f"windows     : {len(X)}  shape {X.shape[1:]}")
    print(f"split       : train {train.stop - train.start} · "
          f"val {val.stop - val.start} · test {test.stop - test.start}")
    print(f"target      : {season_tag}, z-score {clip_tag}")
    print(f"rolling     : {rolling} bars for the robust statistics")
    if args.deseasonalise:
        lo, hi = min(table.values()), max(table.values())
        print(f"profile     : {len(table)} buckets, scale {lo:.2f} .. {hi:.2f} "
              f"(Eastern clock, fitted on train only)")

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, dict]] = []

    for activation in args.activations:
        for seed in args.seeds:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)

            model = build_model(X.shape[1:], activation, args.width, args.depth)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=laplace_nll)

            stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            )
            model.fit(
                X[train], target[train],
                validation_data=(X[val], target[val]),
                epochs=args.epochs, batch_size=args.batch,
                callbacks=[stop], verbose=0,
            )

            name = (f"{activation}_seed{seed}_{args.features}_"
                    f"{season_tag}_{clip_tag}_r{rolling}")
            path = args.out / f"{name}.keras"
            model.save(path)

            report = evaluate(model, X[test], y[test], scale[test], minutes[test])
            results.append((name, report))

            card = {
                "activation": activation,
                "seed": seed,
                "features": list(columns),
                "feature_set": args.features,
                "deseasonalised": args.deseasonalise,
                "profile_bucket_minutes": args.profile_bucket,
                "profile_clock": "America/New_York",
                "broker_utc_offset": args.broker_utc_offset,
                "intraday_profile": {str(k): v for k, v in table.items()},
                "zscore_clip": None if clip == float("inf") else clip,
                "lookback": cfg.windows.lookback,
                "predict_step": cfg.windows.predict_step,
                "rolling_window": rolling,
                "architecture": f"dense{args.width}x{args.depth}",
                "instrument": cfg.data.symbol,
                "timeframe": cfg.data.timeframe,
                "source": str(cfg.data.series_path),
                "windows": int(len(X)),
                "split": {"val_fraction": args.val_fraction,
                          "test_fraction": args.test_fraction,
                          "chronological": True},
                "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "test": report,
            }
            path.with_suffix(".json").write_text(
                json.dumps(card, indent=2), encoding="utf-8"
            )

            print(f"  {name:<52} NLL {report['nll']:>8.4f}  "
                  f"reactive {report['b_reactive']:>7.4f}  "
                  f"clock R2 {report['b_clock_r2']:>6.3f}")

    width = 52
    print("\n" + "=" * 102)
    print(f"{'model':<{width}}{'NLL':>10}{'reactive':>10}{'clock R2':>10}"
          f"{'corr':>8}{'k1 cov':>9}{'k@63%':>8}")
    print("-" * 102)
    for name, report in sorted(results, key=lambda item: -item[1]["b_reactive"]):
        print(f"{name:<{width}}{report['nll']:>10.4f}{report['b_reactive']:>10.4f}"
              f"{report['b_clock_r2']:>10.3f}{report['corr_b_abs_dev']:>+8.3f}"
              f"{report['coverage']['k1'] * 100:>8.1f}%"
              f"{report['k_empirical']['p63']:>8.2f}")
    print("=" * 102)
    print("reactive = spread of b after the hourly curve is removed; this is what")
    print("makes a tunnel widen on a spike. clock R2 near 1 means b is a calendar.")
    print("k@63% is the measured multiplier for 63% coverage; Laplace theory says 1.00.")
    return 0


if __name__ == "__main__":
    sys.exit(main())