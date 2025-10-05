from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    @classmethod
    def fit(cls, probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        x = probs.reshape(-1, 1)
        y = labels.astype(int)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(x, y)
        return cls(model=lr)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        x = probs.reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


@dataclass
class TemperatureScaler:
    temperature: float

    @classmethod
    def fit(cls, logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> "TemperatureScaler":
        # Simple 1D optimization over temperature using grid search
        import numpy as np
        from sklearn.metrics import log_loss

        t_space = np.linspace(0.5, 5.0, 46)
        best_t = 1.0
        best_ll = float("inf")
        for t in t_space:
            p = 1.0 / (1.0 + np.exp(-logits / t))
            ll = log_loss(labels, p, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_t = t
        return cls(temperature=float(best_t))

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


