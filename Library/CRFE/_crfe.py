"""Conformal Recursive Feature Elimination mirroring the CRFE_pro implementation."""

from __future__ import annotations

import numpy as np
from numbers import Integral

from sklearn.base import BaseEstimator, clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils._param_validation import HasMethods, Interval, Real

from CRFE._crfe_utils import (
    binary_change,
    compute_beta_binary,
    compute_beta_multiclass,
)
from CRFE.stopping import ParamParada, StoppingCriteria


class CRFE(BaseEstimator):
    """Conformal Recursive Feature Elimination with proximity-based stopping."""

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "features_to_select": [Interval(Integral, 0, None, closed="neither")],
        "Lambda": [Interval(Real, 0, 1, closed="both")],
        "epsilon": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        estimator=None,
        features_to_select: int = 1,
        Lambda: float = 0.5,
        stopping_activated: bool = False,
        epsilon: float = 0.4,
        stopping_params: ParamParada | None = None,
    ) -> None:
        self.estimator = clone(estimator) if estimator is not None else None
        self.estimator_ = clone(estimator) if estimator is not None else None
        self.features_to_select = features_to_select
        self.features: list[int] = []
        self.betas: list[float] = []
        self.Lambda = Lambda
        self.stopping_activated = stopping_activated
        self.epsilon = epsilon
        self.starting_flag = True
        self.results_dicc: dict = {}

        self.stopping_params = stopping_params if stopping_params is not None else ParamParada()
        self.new_stopping_criteria = StoppingCriteria(self.stopping_params)
        self.stopping_reason_: str | None = None

    def _ensure_estimator(self) -> None:
        if self.estimator is None:
            raise ValueError("An estimator with a 'coef_' attribute must be provided before fitting.")

    def _proximity_based_stopping_criteria(self, X_tr_data, Y_tr):
        if not self.stopping_activated:
            return False, -1, None
        return self.new_stopping_criteria.update(X_tr_data, Y_tr)

    def recursive_elimination(self, X_tr, Y_tr, X_cal, Y_cal):
        self._ensure_estimator()

        n_features = X_tr.shape[1]
        feature_indices = np.arange(n_features)

        if len(self.classes_) > 1:
            self.Lambda_p = (1 - self.Lambda) / (len(self.classes_) - 1)
        else:
            self.Lambda_p = 0.0

        self.stopping_boost = int(n_features * self.epsilon)
        self.stopping_list = []
        self.starting_flag = True
        self.stopping_list_grad_2 = [0, 0]
        self.std_dev = [0, 0]

        self.new_stopping_criteria.reset()
        self.stopping_reason_ = None

        X_tr_work = np.vstack([feature_indices, X_tr])
        X_cal_work = np.vstack([feature_indices, X_cal])

        n = n_features

        while n != self.features_to_select:
            current_indices = X_tr_work[0].astype(int)

            X_tr_data = X_tr_work[1:]
            X_cal_data = X_cal_work[1:]

            model = clone(self.estimator)

            if len(self.classes_) == 2:
                model.fit(X_tr_data, Y_tr)
                beta = compute_beta_binary(model.coef_, Y_cal, X_cal_data)
            else:
                if isinstance(model, OneVsRestClassifier):
                    model.fit(X_tr_data, Y_tr)
                else:
                    model = OneVsRestClassifier(model, n_jobs=-1)
                    model.fit(X_tr_data, Y_tr)
                weights = np.array([est.coef_[0] for est in model.estimators_])
                beta = compute_beta_multiclass(weights, Y_cal, X_cal_data, self.Lambda, self.Lambda_p)

            deleted_index = int(np.argmax(beta))

            debe_parar, mejor_idx, motivo = self._proximity_based_stopping_criteria(X_tr_data, Y_tr)
            if debe_parar:
                self.stopping_reason_ = motivo
                break

            keep_mask = np.ones(n, dtype=bool)
            keep_mask[deleted_index] = False

            X_tr_work = X_tr_work[:, keep_mask]
            X_cal_work = X_cal_work[:, keep_mask]

            X_tr_work[0]  = current_indices[keep_mask]
            X_cal_work[0] = current_indices[keep_mask]

            n = X_tr_work.shape[1]

        self.features = X_tr_work[0].astype(int)
        if "beta" in locals():
            self.betas = np.delete(beta, deleted_index).tolist()
        else:
            self.betas = []

    def fit(self, X_tr, Y_tr, X_cal, Y_cal):
        self._validate_params()

        X_tr = np.asarray(X_tr, dtype=float)
        Y_tr = np.asarray(Y_tr)
        X_cal = np.asarray(X_cal, dtype=float)
        Y_cal = np.asarray(Y_cal)

        if X_tr.shape[1] != X_cal.shape[1]:
            raise ValueError("X training and X calibration must have the same number of features")

        train_classes = np.unique(Y_tr)
        cal_classes = np.unique(Y_cal)
        if not np.array_equal(np.sort(train_classes), np.sort(cal_classes)):
            raise ValueError("All classes in training must be present in calibration")

        self.classes_, Y_tr = np.unique(Y_tr, return_inverse=True)

        if len(self.classes_) == 2:
            Y_tr, Y_cal, self.classes_ = binary_change(Y_tr, Y_cal)

        self.recursive_elimination(X_tr, Y_tr, X_cal, Y_cal)

        self.idx_features_ = self.features
        self.idx_betas_ = self.betas
        self.estimator_ = clone(self.estimator)

        return self
