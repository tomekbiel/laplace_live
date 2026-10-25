"""Is b constant in the model itself, or only in our pipeline?

Runs the trained network over the split it was trained on, so the answer
does not depend on anything laplace_live does. The hour is recovered from
the window's own clock features, which makes the check self-contained.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from laplace import model as model_io
from laplace.config import load_config

SPLIT = Path(
    r"C:\python\intraday_activation_time_series_research\data\splits\US.100+5_split.npz"
)

cfg = load_config()
model_path = cfg.experiment.resolved_model_path
print("model path        :", model_path)

model = model_io.load_model(model_path)
print("model input shape :", model.input_shape)

data = np.load(SPLIT, allow_pickle=True)
X = data["X_test"] if "X_test" in data else data["X_train"]
print("split shape       :", X.shape)

raw = model.predict(X, verbose=0)
mu = raw[..., 0].ravel()
b = np.log1p(np.exp(raw[..., 1])).ravel() + 1e-9      # softplus + MIN_B

print(f"\nb: min {b.min():.3e}  max {b.max():.3e}  "
      f"mean {b.mean():.3e}  std {b.std():.3e}")
print(f"relative spread   : {b.std() / b.mean():.4f}")
print(f"mu: min {mu.min():.3e}  max {mu.max():.3e}  std {mu.std():.3e}")

# Clock features sit at columns 4 and 5 of the seven-feature set; the last
# row of a window is the bar before the target, so its angle is the hour
# the prediction was made at.
if X.shape[2] >= 6:
    hour = (np.arctan2(X[:, -1, 4], X[:, -1, 5]) / (2 * np.pi) * 24) % 24
    print(f"\ncorr(b, hour-of-day sin) : {np.corrcoef(b, X[:, -1, 4])[0, 1]:+.3f}")
    print(f"{'hour':>6} {'n':>7} {'mean b':>12}")
    for h in range(0, 24, 2):
        mask = (hour >= h) & (hour < h + 2)
        if mask.sum():
            print(f"{h:>6} {mask.sum():>7} {b[mask].mean():>12.3e}")