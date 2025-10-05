from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_metrics(probs: Iterable[float], labels: Iterable[int], target_spec: float = 0.9) -> Dict[str, float]:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    metrics: Dict[str, float] = {}
    if len(np.unique(y)) < 2:
        # Avoid exceptions if only one class present in a tiny val set
        return {"auroc": float("nan"), "auprc": float("nan"), "sens_at_spec": float("nan"), "threshold": float("nan")}

    metrics["auroc"] = float(roc_auc_score(y, p))
    metrics["auprc"] = float(average_precision_score(y, p))

    fpr, tpr, thr = roc_curve(y, p)
    spec = 1.0 - fpr
    # Find threshold closest to target specificity
    idx = int(np.argmin(np.abs(spec - target_spec)))
    metrics["sens_at_spec"] = float(tpr[idx])
    metrics["threshold"] = float(thr[idx])
    return metrics


