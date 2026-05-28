"""Local CRFE ranking tracer for lambda sensitivity experiments.

This module mirrors the multiclass elimination loop in ``Library/CRFE/_crfe.py``
but records the full feature elimination path. It intentionally lives outside
the package so the original library remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier


def binary_change(y_train: np.ndarray, y_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train = np.where(y_train == 0, -1, y_train)
    y_cal = np.where(y_cal == 0, -1, y_cal)
    classes, y_train = np.unique(y_train, return_inverse=True)
    return y_train, y_cal, classes


def compute_beta_binary(w: np.ndarray, y_cal: np.ndarray, X_cal: np.ndarray) -> np.ndarray:
    return -np.sum(w[0] * y_cal[:, np.newaxis] * X_cal, axis=0).astype(float)


def compute_beta_multiclass(
    w: np.ndarray,
    y_cal: np.ndarray,
    X_cal: np.ndarray,
    lambda_value: float,
    lambda_p: float,
) -> np.ndarray:
    w_sum = np.sum(w, axis=0)
    w_y = w[y_cal]
    lambda_terms = lambda_value * w_y * X_cal
    rest_terms = lambda_p * (w_sum[None, :] - w_y) * X_cal
    return -np.sum(lambda_terms - rest_terms, axis=0).astype(float)


@dataclass(frozen=True)
class RankingTrace:
    """Feature ranking and diagnostic values from one CRFE run."""

    lambda_value: float
    lambda_p: float
    classes: list
    ranked_features: list[int]
    eliminated_features: list[int]
    eliminated_betas: list[float]


def trace_crfe_ranking(estimator, X_tr, y_tr, X_cal, y_cal, lambda_value: float) -> RankingTrace:
    """Run CRFE elimination and return a complete best-to-worst feature ranking.

    Parameters
    ----------
    estimator:
        Linear estimator with ``coef_`` after fitting, matching the CRFE
        estimator expectation. Multiclass estimators are wrapped in
        ``OneVsRestClassifier`` exactly as in the library.
    X_tr, y_tr, X_cal, y_cal:
        Training and calibration split used by CRFE.
    lambda_value:
        Multiclass CRFE lambda. The companion lambda prime is computed as in
        ``CRFE.recursive_elimination``.
    """

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must be in [0, 1]")

    X_tr = np.asarray(X_tr, dtype=float)
    X_cal = np.asarray(X_cal, dtype=float)
    y_tr = np.asarray(y_tr)
    y_cal = np.asarray(y_cal)

    if X_tr.shape[1] != X_cal.shape[1]:
        raise ValueError("Training and calibration matrices must have the same number of features")

    classes = np.unique(y_tr)
    cal_classes = np.unique(y_cal)
    if not np.array_equal(np.sort(classes), np.sort(cal_classes)):
        raise ValueError("All training classes must also be present in calibration data")

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_tr_encoded = np.array([class_to_idx[y] for y in y_tr])
    y_cal_encoded = np.array([class_to_idx[y] for y in y_cal])

    if len(classes) == 2:
        y_tr_encoded, y_cal_encoded, classes = binary_change(y_tr_encoded, y_cal_encoded)

    lambda_p = (1.0 - lambda_value) / (len(classes) - 1) if len(classes) > 1 else 0.0

    feature_indices = np.arange(X_tr.shape[1])
    X_tr_work = np.vstack([feature_indices, X_tr])
    X_cal_work = np.vstack([feature_indices, X_cal])

    eliminated_features: list[int] = []
    eliminated_betas: list[float] = []

    while X_tr_work.shape[1] > 1:
        current_indices = X_tr_work[0].astype(int)
        X_tr_data = X_tr_work[1:]
        X_cal_data = X_cal_work[1:]

        model = clone(estimator)

        if len(classes) == 2:
            model.fit(X_tr_data, y_tr_encoded)
            beta = compute_beta_binary(model.coef_, y_cal_encoded, X_cal_data)
        else:
            if isinstance(model, OneVsRestClassifier):
                model.fit(X_tr_data, y_tr_encoded)
            else:
                model = OneVsRestClassifier(model, n_jobs=1)
                model.fit(X_tr_data, y_tr_encoded)
            weights = np.array([est.coef_[0] for est in model.estimators_])
            beta = compute_beta_multiclass(
                weights,
                y_cal_encoded,
                X_cal_data,
                lambda_value,
                lambda_p,
            )

        delete_pos = int(np.argmax(beta))
        eliminated_features.append(int(current_indices[delete_pos]))
        eliminated_betas.append(float(beta[delete_pos]))

        keep_mask = np.ones(X_tr_work.shape[1], dtype=bool)
        keep_mask[delete_pos] = False
        X_tr_work = X_tr_work[:, keep_mask]
        X_cal_work = X_cal_work[:, keep_mask]

    survivor = [int(X_tr_work[0, 0])]
    ranked_features = survivor + list(reversed(eliminated_features))

    return RankingTrace(
        lambda_value=float(lambda_value),
        lambda_p=float(lambda_p),
        classes=np.asarray(classes).tolist(),
        ranked_features=ranked_features,
        eliminated_features=eliminated_features,
        eliminated_betas=eliminated_betas,
    )
