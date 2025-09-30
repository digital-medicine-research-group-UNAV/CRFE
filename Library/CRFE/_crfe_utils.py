"""Utility helpers for the streamlined CRFE implementation."""

from __future__ import annotations

import numpy as np


def to_list(array_like):
    """Return a Python list while tolerating numpy inputs."""
    if isinstance(array_like, list):
        return array_like
    if isinstance(array_like, np.ndarray):
        return array_like.tolist()
    return list(array_like)


def binary_change(y_tr, y_cal):
    """Convert labels from {0, 1} to {-1, 1} keeping class names."""
    y_tr = np.asarray(y_tr, dtype=int)
    y_cal = np.asarray(y_cal, dtype=int)

    y_tr = np.where(y_tr == 0, -1, y_tr)
    y_cal = np.where(y_cal == 0, -1, y_cal)

    class_names = np.unique(y_tr)
    mapping = {value: idx for idx, value in enumerate(class_names)}

    y_tr_encoded = np.array([mapping[val] for val in y_tr], dtype=int)
    try:
        y_cal_encoded = np.array([mapping[val] for val in y_cal], dtype=int)
    except KeyError as exc:
        raise ValueError("Calibration labels contain unseen classes.") from exc

    return y_tr_encoded, y_cal_encoded, class_names


def compute_beta_binary(weights, y_cal, X_cal):
    """Compute beta scores for the binary case using vector operations."""
    weights = np.asarray(weights)
    y_cal = np.asarray(y_cal, dtype=float)
    X_cal = np.asarray(X_cal, dtype=float)

    return -np.sum(weights[0] * y_cal[:, np.newaxis] * X_cal, axis=0)


def compute_beta_multiclass(weights, y_cal, X_cal, lam, lam_p):
    """Compute beta scores for the multiclass case following the new CRFE formulation."""
    weights = np.asarray(weights, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    X_cal = np.asarray(X_cal, dtype=float)

    n_features = X_cal.shape[1]
    n_samples = X_cal.shape[0]

    beta = np.zeros(n_features, dtype=float)
    weight_sum = np.sum(weights, axis=0)

    for j in range(n_features):
        column = X_cal[:, j]
        accum = 0.0
        for i in range(n_samples):
            y_i = y_cal[i]
            lambda_term = lam * weights[y_i, j] * column[i]
            sum_term = lam_p * (weight_sum[j] - weights[y_i, j]) * column[i]
            accum -= (lambda_term - sum_term)
        beta[j] = accum

    return beta
