"""
Optimized Stepwise Recursive Feature Elimination module

Performance improvements:
- Vectorized operations
- Efficient array handling
- Reduced memory allocations
- Better data structure management
- Parallel processing where applicable
"""

import numpy as np
import sys 
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import itertools
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from conformal_module import CP
from _crfe_utils import to_list, binary_change, find_max, NC_OvsA_SVMl_dev
from stopping_module import StoppingCriteria, ParamParada
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Stepwise_RFE:
    """
    Optimized Stepwise Recursive Feature Elimination.
    
    This class provides an efficient implementation of RFE with conformal prediction
    evaluation at each step.
    
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a fit method and coef_ attribute.
    features_to_select : int, default=1
        Number of features to select.
    Lambda : float, default=0.5
        Multi-class parameter for conformal prediction.
    """
    
    def __init__(self, estimator,
                features_to_select=1,
                Lambda=0.5,
                stopping_activated=False,
                epsilon=0.4,
                stopping_params=None,
                strict_conformal=False,
                selection_fraction=0.5,
                random_state=None):
        
        self.estimator_ = clone(estimator)
        self.estimator = clone(estimator)
        self.features_to_select: int = features_to_select
        self.stopping_activated = stopping_activated
        self.results_dicc: dict = {}
        self.features = []
        self.Lambda = Lambda
        self.strict_conformal = strict_conformal
        self.selection_fraction = selection_fraction
        self.random_state = random_state

        # Initialize stopping criteria
        self.stopping_params = stopping_params if stopping_params is not None else ParamParada()
        self.new_stopping_criteria = StoppingCriteria(self.stopping_params)
    
    def _proximity_based_stopping_criteria(self, X_tr_data, Y_tr):
        
        if not self.stopping_activated:
            return False, -1, None
            
        # Use the new stopping criteria
        debe_parar, mejor_idx, motivo = self.new_stopping_criteria.update(X_tr_data, Y_tr)
        
        return debe_parar, mejor_idx, motivo




    def _predict_scores_hinge(self, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
        """Optimized hinge loss scoring with efficient array operations."""
        # Use array views instead of delete operations
        X_tr_view = X_tr[1:]
        X_cal_view = X_cal[1:]
        X_test_view = X_test[1:]

        if len(self.classes_) == 2:
            estimator = self.estimator.fit(X_tr_view, Y_tr)
            estimator = CalibratedClassifierCV(estimator).fit(X_tr_view, Y_tr)
        else:
            if isinstance(self.estimator, OneVsRestClassifier):
                estimator = self.estimator.fit(X_tr_view, Y_tr)
                estimator = CalibratedClassifierCV(estimator).fit(X_tr_view, Y_tr)
            else:
                estimator = OneVsRestClassifier(self.estimator, n_jobs=-1)
                estimator = CalibratedClassifierCV(estimator).fit(X_tr_view, Y_tr)

        # Vectorized probability computations
        probabilities_cal = estimator.predict_proba(X_cal_view)
        NCM_cal = 1.0 - probabilities_cal[np.arange(len(Y_cal)), Y_cal]

        probabilities_test = estimator.predict_proba(X_test_view)
        NCM_test = 1.0 - probabilities_test

        scores = CP(0.1).Conformal_prediction_scores(Y_test, NCM_test, NCM_cal, self.classes_)
        
        return scores

    def _predict_scores_SVM(self, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
        """Optimized SVM scoring with efficient operations."""
        X_tr_view = X_tr[1:]
        X_cal_view = X_cal[1:]
        X_test_view = X_test[1:]

        # Initialize Lambda_p if not already set
        if not hasattr(self, 'Lambda_p'):
            self.Lambda_p = (1 - self.Lambda) / (len(self.classes_) - 1) if len(self.classes_) > 2 else 0

        if len(self.classes_) == 2:
            estimator = self.estimator_.fit(X_tr_view, Y_tr)
            w = estimator.coef_
            bias = estimator.intercept_

            multiclass = False
            NCM_cal = NC_OvsA_SVMl_dev(X_cal_view, Y_cal, w, bias,None,None, multiclass)

            # Vectorized NCM computation for test set
            NCM_test = [NC_OvsA_SVMl_dev(
                        np.tile(sample, (len(self.classes_), 1)),
                        self.classes_,
                        w, bias,
                        None,
                        None,
                        multiclass) for sample in X_test_view]
        else:
            if isinstance(self.estimator_, OneVsRestClassifier):
                estimator = self.estimator_.fit(X_tr_view, Y_tr)
                w = estimator.coef_
                bias = estimator.intercept_
            else:
                estimator = OneVsRestClassifier(self.estimator_, n_jobs=-1)
                estimator.fit(X_tr_view, Y_tr)

                # Efficient coefficient extraction
                w = np.array([est.coef_[0] for est in estimator.estimators_])
                bias = np.array([est.intercept_[0] for est in estimator.estimators_])

            multiclass = True
            NCM_cal = NC_OvsA_SVMl_dev(X_cal_view, Y_cal, w, bias, self.Lambda, self.Lambda_p, multiclass)

            NCM_test = [NC_OvsA_SVMl_dev(
                        np.tile(sample, (len(self.classes_), 1)),
                        self.classes_,
                        w, bias,
                        self.Lambda,
                        self.Lambda_p,
                        multiclass) for sample in X_test_view]

        scores = CP(0.1).Conformal_prediction_scores(Y_test, NCM_test, NCM_cal, self.classes_)
        print("Scores: ", scores[0], scores[1], scores[2])
        sys.stdout.flush()

        return scores

    def _get_conformal_work_arrays(self, current_indices, X_cal_work, Y_cal):
        """Return the arrays used for pure conformal calibration in strict mode."""
        if not getattr(self, "strict_conformal", False):
            return X_cal_work, Y_cal

        X_cp_subset = self._strict_X_cp[:, current_indices]
        X_cp_work = np.vstack([current_indices, X_cp_subset])
        Y_cp = self._strict_Y_cp
        return X_cp_work, Y_cp

    def _fit(self, X_train, y_train, X_cal, Y_cal, X_test, y_test):
        """Optimized fitting with efficient RFE implementation."""
        n_features = X_train.shape[1]
        feature_indices = np.arange(n_features)
        
        self.Lambda_p = (1 - self.Lambda) / (len(self.classes_) - 1)

        # Reset the new stopping criteria for this run
        self.new_stopping_criteria.reset()

        # Pre-allocate output dictionary
        list_of_names = ["Index", "coverage", "inefficiency", 
                        "certainty", "uncertainty", "mistrust",
                        "S_score", "F_score", "Creditibily"]
        OUT = {name: [] for name in list_of_names}

        # Add header row efficiently
        X_train_work = np.vstack([feature_indices, X_train])
        X_cal_work = np.vstack([feature_indices, X_cal])
        X_test_work = np.vstack([feature_indices, X_test])

        n = n_features

        while n != self.features_to_select:
            current_indices = X_train_work[0].astype(int)
            
            # Get data views
            X_train_data = X_train_work[1:]
            
            # Clone estimator for clean RFE
            model = clone(self.estimator)

            # Run traditional RFE to select n-1 features
            rfe = RFE(estimator=model, n_features_to_select=n - 1, step=1)
            rfe.fit(X_train_data, y_train)

            # Find which feature was eliminated
            deleted_index = np.where(~rfe.support_)[0][0]

            debe_parar, mejor_idx, motivo = self._proximity_based_stopping_criteria(X_train_data, y_train)
            if debe_parar:
                print(f"Stopping due to: {motivo} at iteration {n-self.features_to_select}")
                break

            # Efficient column deletion using boolean indexing
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[deleted_index] = False
            
            X_train_work = X_train_work[:, keep_mask]
            X_cal_work = X_cal_work[:, keep_mask]
            X_test_work = X_test_work[:, keep_mask]

            # Update feature indices
            X_train_work[0] = current_indices[keep_mask]
            X_cal_work[0]   = current_indices[keep_mask]
            X_test_work[0]  = current_indices[keep_mask]



            n = X_train_work.shape[1]
            
            print("Remaining features: ", n)
            sys.stdout.flush()

            # Calculate scores
            X_cp_work, Y_cp = self._get_conformal_work_arrays(current_indices[keep_mask], X_cal_work, Y_cal)
            scores = self._predict_scores_SVM(X_train_work, y_train, X_cp_work, Y_cp, X_test_work, y_test)

            OUT["Index"].append(current_indices[keep_mask].astype(int))
            for i, name in enumerate(list_of_names[1:]):
                OUT[name].append(scores[i])
        
        self.features = X_train_work[0].astype(int)

        return OUT

    def _strict_split_calibration(self, X_cal, Y_cal):
        """Split calibration data into selection and pure CP-calibration subsets."""
        if not 0 < self.selection_fraction < 1:
            raise ValueError("selection_fraction must be strictly between 0 and 1 when strict_conformal=True")

        try:
            X_sel, X_cp, Y_sel, Y_cp = train_test_split(
                X_cal,
                Y_cal,
                train_size=self.selection_fraction,
                random_state=self.random_state,
                stratify=Y_cal,
            )
        except ValueError as e:
            raise ValueError(
                "Could not split calibration set in strict_conformal mode. "
                "Ensure the calibration set is large enough and each class has enough samples for both splits."
            ) from e

        sel_classes = set(np.unique(Y_sel))
        cp_classes = set(np.unique(Y_cp))
        all_classes = set(np.unique(Y_cal))
        if sel_classes != all_classes or cp_classes != all_classes:
            raise ValueError(
                "strict_conformal=True requires both internal calibration splits to contain all classes."
            )

        return X_sel, Y_sel, X_cp, Y_cp

    def fit(self, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
        """
        Optimized fit method with efficient validation and processing.
        
        Parameters
        ----------
        X_tr : array-like of shape (n_samples, n_features)
            Training data
        Y_tr : array-like of shape (n_samples,)
            Training labels
        X_cal : array-like of shape (n_cal_samples, n_features)
            Calibration data
        Y_cal : array-like of shape (n_cal_samples,)
            Calibration labels
        X_test : array-like of shape (n_test_samples, n_features)
            Test data
        Y_test : array-like of shape (n_test_samples,)
            Test labels
            
        Returns
        -------
        self : object
        """
        
        # Efficient array conversion
        arrays_to_convert = [
            (X_tr, 'X_tr'), (Y_tr, 'Y_tr'), (X_cal, 'X_cal'), 
            (Y_cal, 'Y_cal'), (X_test, 'X_test'), (Y_test, 'Y_test')
        ]
        
        converted_arrays = {}
        for arr, name in arrays_to_convert:
            if not isinstance(arr, np.ndarray):
                converted_arrays[name] = np.asarray(arr)
            else:
                converted_arrays[name] = arr

        X_tr, Y_tr = converted_arrays['X_tr'], converted_arrays['Y_tr']
        X_cal, Y_cal = converted_arrays['X_cal'], converted_arrays['Y_cal']
        X_test, Y_test = converted_arrays['X_test'], converted_arrays['Y_test']

        # Validation checks
        if X_tr.shape[1] != X_cal.shape[1]:
            raise ValueError("X training and X calibration must have the same number of features")

        # More efficient class validation
        tr_classes = set(Y_tr)
        cal_classes = set(Y_cal)
        if not tr_classes.issubset(cal_classes) or not cal_classes.issubset(tr_classes):
            raise ValueError("All classes in training must be present in calibration")

        if self.strict_conformal:
            X_sel, Y_sel_raw, X_cp, Y_cp_raw = self._strict_split_calibration(X_cal, Y_cal)
        else:
            X_sel, Y_sel_raw = X_cal, Y_cal
            X_cp, Y_cp_raw = X_cal, Y_cal

        # Fix: Handle class assignment properly
        self.classes_ = np.unique(Y_tr)
        # Create label encoder mapping for consistent encoding
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        Y_tr_encoded = np.array([class_to_idx[y] for y in Y_tr])
        Y_sel_encoded = np.array([class_to_idx[y] for y in Y_sel_raw])
        Y_cp_encoded = np.array([class_to_idx[y] for y in Y_cp_raw])

        if len(self.classes_) == 2:
            Y_tr_encoded, Y_sel_encoded, self.classes_ = binary_change(Y_tr_encoded, Y_sel_encoded)
            _, Y_cp_encoded, _ = binary_change(Y_tr_encoded.copy(), Y_cp_encoded)

        self._strict_X_cp = X_cp
        self._strict_Y_cp = Y_cp_encoded

        self.results_dicc = self._fit(X_tr, Y_tr_encoded, X_sel, Y_sel_encoded, X_test, Y_test)

        # Set final attributes  
        self.idx_features_ = self.features
        self.estimator_ = clone(self.estimator)
       
        return self


class Stepwise_LASSO(Stepwise_RFE):
    """
    Stepwise feature elimination based on L1 (lasso-like) importance.

    Architecture and outputs mirror Stepwise_RFE, but the feature removed at each
    iteration is selected using LogisticRegression with L1 regularization.
    """

    def __init__(
        self,
        estimator,
        features_to_select=1,
        Lambda=0.5,
        stopping_activated=False,
        epsilon=0.4,
        stopping_params=None,
        strict_conformal=False,
        selection_fraction=0.5,
        lasso_c=0.1,
        lasso_max_iter=4000,
        random_state=0,
        n_jobs=-1,
    ):
        super().__init__(
            estimator=estimator,
            features_to_select=features_to_select,
            Lambda=Lambda,
            stopping_activated=stopping_activated,
            epsilon=epsilon,
            stopping_params=stopping_params,
            strict_conformal=strict_conformal,
            selection_fraction=selection_fraction,
            random_state=random_state,
        )
        self.lasso_c = lasso_c
        self.lasso_max_iter = lasso_max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _select_feature_to_remove_lasso(self, X_train_data, y_train):
        """Select one feature to remove using mean absolute L1 coefficient."""
        if X_train_data.shape[1] <= 1:
            return 0

        lasso_base = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.lasso_c,
            max_iter=self.lasso_max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        unique_classes = np.unique(y_train)
        if len(unique_classes) > 2:
            lasso_selector = OneVsRestClassifier(lasso_base, n_jobs=self.n_jobs)
        else:
            lasso_selector = lasso_base

        lasso_selector.fit(X_train_data, y_train)

        if hasattr(lasso_selector, "estimators_"):
            coef_matrix = np.vstack(
                [np.ravel(est.coef_) for est in lasso_selector.estimators_]
            )
        else:
            coef_matrix = np.atleast_2d(np.ravel(lasso_selector.coef_))

        importances = np.mean(np.abs(coef_matrix), axis=0)
        if not np.all(np.isfinite(importances)):
            importances = np.nan_to_num(importances, nan=np.inf, posinf=np.inf, neginf=np.inf)

        return int(np.argmin(importances))

    def _fit(self, X_train, y_train, X_cal, Y_cal, X_test, y_test):
        """Fit using lasso-based elimination while preserving Stepwise_RFE flow."""
        n_features = X_train.shape[1]
        feature_indices = np.arange(n_features)

        self.Lambda_p = (1 - self.Lambda) / (len(self.classes_) - 1)
        self.new_stopping_criteria.reset()

        list_of_names = [
            "Index",
            "coverage",
            "inefficiency",
            "certainty",
            "uncertainty",
            "mistrust",
            "S_score",
            "F_score",
            "Creditibily",
        ]
        OUT = {name: [] for name in list_of_names}

        X_train_work = np.vstack([feature_indices, X_train])
        X_cal_work = np.vstack([feature_indices, X_cal])
        X_test_work = np.vstack([feature_indices, X_test])

        n = n_features

        while n != self.features_to_select:
            current_indices = X_train_work[0].astype(int)
            X_train_data = X_train_work[1:]

            deleted_index = self._select_feature_to_remove_lasso(X_train_data, y_train)

            debe_parar, mejor_idx, motivo = self._proximity_based_stopping_criteria(X_train_data, y_train)
            if debe_parar:
                print(f"Stopping due to: {motivo} at iteration {n-self.features_to_select}")
                break

            keep_mask = np.ones(n, dtype=bool)
            keep_mask[deleted_index] = False

            X_train_work = X_train_work[:, keep_mask]
            X_cal_work = X_cal_work[:, keep_mask]
            X_test_work = X_test_work[:, keep_mask]

            X_train_work[0] = current_indices[keep_mask]
            X_cal_work[0] = current_indices[keep_mask]
            X_test_work[0] = current_indices[keep_mask]

            n = X_train_work.shape[1]

            print("Remaining features: ", n)
            sys.stdout.flush()

            X_cp_work, Y_cp = self._get_conformal_work_arrays(current_indices[keep_mask], X_cal_work, Y_cal)
            scores = self._predict_scores_SVM(
                X_train_work, y_train, X_cp_work, Y_cp, X_test_work, y_test
            )

            OUT["Index"].append(current_indices[keep_mask].astype(int))
            for i, name in enumerate(list_of_names[1:]):
                OUT[name].append(scores[i])

        self.features = X_train_work[0].astype(int)

        return OUT


class Stepwise_ELASTICNET(Stepwise_RFE):
    """
    Stepwise feature elimination based on Elastic Net importance.

    Architecture and outputs mirror Stepwise_RFE, but the feature removed at each
    iteration is selected using LogisticRegression with elasticnet regularization.
    """

    def __init__(
        self,
        estimator,
        features_to_select=1,
        Lambda=0.5,
        stopping_activated=False,
        epsilon=0.4,
        stopping_params=None,
        strict_conformal=False,
        selection_fraction=0.5,
        elasticnet_c=0.1,
        elasticnet_l1_ratio=0.5,
        elasticnet_max_iter=4000,
        random_state=0,
        n_jobs=-1,
    ):
        super().__init__(
            estimator=estimator,
            features_to_select=features_to_select,
            Lambda=Lambda,
            stopping_activated=stopping_activated,
            epsilon=epsilon,
            stopping_params=stopping_params,
            strict_conformal=strict_conformal,
            selection_fraction=selection_fraction,
            random_state=random_state,
        )
        self.elasticnet_c = elasticnet_c
        self.elasticnet_l1_ratio = elasticnet_l1_ratio
        self.elasticnet_max_iter = elasticnet_max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _select_feature_to_remove_elasticnet(self, X_train_data, y_train):
        """Select one feature to remove using mean absolute Elastic Net coefficient."""
        if X_train_data.shape[1] <= 1:
            return 0

        elasticnet_base = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=self.elasticnet_c,
            l1_ratio=self.elasticnet_l1_ratio,
            max_iter=self.elasticnet_max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        unique_classes = np.unique(y_train)
        if len(unique_classes) > 2:
            elasticnet_selector = OneVsRestClassifier(elasticnet_base, n_jobs=self.n_jobs)
        else:
            elasticnet_selector = elasticnet_base

        elasticnet_selector.fit(X_train_data, y_train)

        if hasattr(elasticnet_selector, "estimators_"):
            coef_matrix = np.vstack(
                [np.ravel(est.coef_) for est in elasticnet_selector.estimators_]
            )
        else:
            coef_matrix = np.atleast_2d(np.ravel(elasticnet_selector.coef_))

        importances = np.mean(np.abs(coef_matrix), axis=0)
        if not np.all(np.isfinite(importances)):
            importances = np.nan_to_num(importances, nan=np.inf, posinf=np.inf, neginf=np.inf)

        return int(np.argmin(importances))

    def _fit(self, X_train, y_train, X_cal, Y_cal, X_test, y_test):
        """Fit using Elastic Net-based elimination while preserving Stepwise_RFE flow."""
        n_features = X_train.shape[1]
        feature_indices = np.arange(n_features)

        self.Lambda_p = (1 - self.Lambda) / (len(self.classes_) - 1)
        self.new_stopping_criteria.reset()

        list_of_names = [
            "Index",
            "coverage",
            "inefficiency",
            "certainty",
            "uncertainty",
            "mistrust",
            "S_score",
            "F_score",
            "Creditibily",
        ]
        OUT = {name: [] for name in list_of_names}

        X_train_work = np.vstack([feature_indices, X_train])
        X_cal_work = np.vstack([feature_indices, X_cal])
        X_test_work = np.vstack([feature_indices, X_test])

        n = n_features

        while n != self.features_to_select:
            current_indices = X_train_work[0].astype(int)
            X_train_data = X_train_work[1:]

            deleted_index = self._select_feature_to_remove_elasticnet(X_train_data, y_train)

            debe_parar, mejor_idx, motivo = self._proximity_based_stopping_criteria(X_train_data, y_train)
            if debe_parar:
                print(f"Stopping due to: {motivo} at iteration {n-self.features_to_select}")
                break

            keep_mask = np.ones(n, dtype=bool)
            keep_mask[deleted_index] = False

            X_train_work = X_train_work[:, keep_mask]
            X_cal_work = X_cal_work[:, keep_mask]
            X_test_work = X_test_work[:, keep_mask]

            X_train_work[0] = current_indices[keep_mask]
            X_cal_work[0] = current_indices[keep_mask]
            X_test_work[0] = current_indices[keep_mask]

            n = X_train_work.shape[1]

            print("Remaining features: ", n)
            sys.stdout.flush()

            X_cp_work, Y_cp = self._get_conformal_work_arrays(current_indices[keep_mask], X_cal_work, Y_cal)
            scores = self._predict_scores_SVM(
                X_train_work, y_train, X_cp_work, Y_cp, X_test_work, y_test
            )

            OUT["Index"].append(current_indices[keep_mask].astype(int))
            for i, name in enumerate(list_of_names[1:]):
                OUT[name].append(scores[i])

        self.features = X_train_work[0].astype(int)

        return OUT




class Stepwise_BORUTA(Stepwise_RFE):
    """
    Stepwise feature elimination using Boruta as the ranking signal.

    Two modes:
      - recompute_each_step=False (default): run Boruta once on full X_train,
        build a global removal order, then remove one feature per iteration.
        (Fast, stable, good for comparable stepwise curves.)
      - recompute_each_step=True: rerun Boruta at every step on the remaining features
        and remove the currently "worst" feature. (More adaptive, much slower.)
    """

    def __init__(
        self,
        estimator,
        features_to_select=1,
        Lambda=0.5,
        stopping_activated=False,
        epsilon=0.4,
        stopping_params=None,
        strict_conformal=False,
        selection_fraction=0.5,
        # Boruta / RF parameters:
        n_estimators="auto",
        max_iter=50,
        perc=100,
        alpha=0.05,
        random_state=0,
        n_jobs=-1,
        recompute_each_step=False,  # Performance warning risk if = True
    ):
        super().__init__(
            estimator=estimator,
            features_to_select=features_to_select,
            Lambda=Lambda,
            stopping_activated=stopping_activated,
            epsilon=epsilon,
            stopping_params=stopping_params,
            strict_conformal=strict_conformal,
            selection_fraction=selection_fraction,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.recompute_each_step = recompute_each_step 

        # Cached global removal order for Option B
        self._boruta_removal_order_ = None

    def _run_boruta(self, X, y):
        """
        Runs Boruta on the given data and returns:
          ranking: array[int] where 1 is best, larger is worse
          support: accepted mask
          support_weak: tentative mask (if available)
        """
        # Local import so your module can still import even if boruta isn't installed
        from boruta import BorutaPy

        rf = RandomForestClassifier(
            n_estimators=200 if self.n_estimators == "auto" else self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced",
        )

        boruta = BorutaPy(
            estimator=rf,
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            perc=self.perc,
            alpha=self.alpha,
            random_state=self.random_state,
            verbose=0,
        )
        boruta.fit(X, y)

        ranking = np.array(boruta.ranking_, dtype=int)
        support = np.array(boruta.support_, dtype=bool)
        support_weak = np.array(
            getattr(boruta, "support_weak_", np.zeros_like(support)),
            dtype=bool,
        )
        return ranking, support, support_weak

    def _build_global_removal_order(self, X_train, y_train):
        """
        Option B: Run Boruta once and build a deterministic removal order:
          rejected (worst->best), then tentative (worst->best), then accepted (worst->best).
        """
        ranking, support, support_weak = self._run_boruta(X_train, y_train)

        rejected = ~(support | support_weak)
        tentative = support_weak & ~support
        accepted = support

        def group_order(mask):
            idx = np.where(mask)[0]
            if idx.size == 0:
                return []
            # worst first => descending ranking
            return list(idx[np.argsort(-ranking[idx])])

        removal_order = group_order(rejected) + group_order(tentative) + group_order(accepted)
        return removal_order

    def _select_feature_to_remove_boruta_step(self, X_train_data, y_train):
        """
        Option A helper: recompute Boruta on remaining features and remove the worst one.
        """
        ranking, support, support_weak = self._run_boruta(X_train_data, y_train)

        rejected = ~(support | support_weak)
        tentative = support_weak & ~support
        accepted = support

        if np.any(rejected):
            candidates = np.where(rejected)[0]
        elif np.any(tentative):
            candidates = np.where(tentative)[0]
        else:
            candidates = np.where(accepted)[0]

        # worst = highest ranking value
        return int(candidates[np.argmax(ranking[candidates])])

    def _fit(self, X_train, y_train, X_cal, Y_cal, X_test, y_test):
        n_features = X_train.shape[1]
        feature_ids = np.arange(n_features)

        # These are used by your conformal scoring logic
        self.Lambda_p = (1 - self.Lambda) / (len(self.classes_) - 1)
        self.new_stopping_criteria.reset()

        list_of_names = [
            "Index",
            "coverage",
            "inefficiency",
            "certainty",
            "uncertainty",
            "mistrust",
            "S_score",
            "F_score",
            "Creditibily",
        ]
        OUT = {name: [] for name in list_of_names}

        # Your code uses "index row" stacked on top of X
        X_train_work = np.vstack([feature_ids, X_train])
        X_cal_work = np.vstack([feature_ids, X_cal])
        X_test_work = np.vstack([feature_ids, X_test])

        # Option B: compute global removal order once
        if not self.recompute_each_step:
            self._boruta_removal_order_ = self._build_global_removal_order(X_train, y_train)
            removal_ptr = 0

        n = n_features
        while n != self.features_to_select:
            current_ids = X_train_work[0].astype(int)
            X_train_data = X_train_work[1:]

            # Choose which feature to remove (column index in current matrices)
            if self.recompute_each_step:
                deleted_col = self._select_feature_to_remove_boruta_step(X_train_data, y_train)
            else:
                # Find next original feature id still present, then map -> current column index
                while (
                    removal_ptr < len(self._boruta_removal_order_)
                    and self._boruta_removal_order_[removal_ptr] not in current_ids
                ):
                    removal_ptr += 1
                if removal_ptr >= len(self._boruta_removal_order_):
                    # Fallback: if something goes odd, remove last column
                    deleted_col = n - 1
                else:
                    target_id = self._boruta_removal_order_[removal_ptr]
                    deleted_col = int(np.where(current_ids == target_id)[0][0])
                    removal_ptr += 1

            # Your existing stopping criteria hook
            debe_parar, mejor_idx, motivo = self._proximity_based_stopping_criteria(X_train_data, y_train)
            if debe_parar:
                print(f"Stopping due to: {motivo} at iteration {n - self.features_to_select}")
                break

            keep_mask = np.ones(n, dtype=bool)
            keep_mask[deleted_col] = False

            # Apply deletion to all stacks
            X_train_work = X_train_work[:, keep_mask]
            X_cal_work = X_cal_work[:, keep_mask]
            X_test_work = X_test_work[:, keep_mask]

            # Ensure index rows stay aligned
            X_train_work[0] = current_ids[keep_mask]
            X_cal_work[0] = current_ids[keep_mask]
            X_test_work[0] = current_ids[keep_mask]

            n = X_train_work.shape[1]
            print("Remaining features: ", n)
            sys.stdout.flush()

            X_cp_work, Y_cp = self._get_conformal_work_arrays(current_ids[keep_mask], X_cal_work, Y_cal)
            scores = self._predict_scores_SVM(
                X_train_work, y_train,
                X_cp_work, Y_cp,
                X_test_work, y_test
            )

            OUT["Index"].append(current_ids[keep_mask].astype(int))
            for i, name in enumerate(list_of_names[1:]):
                OUT[name].append(scores[i])

        self.features = X_train_work[0].astype(int)
        return OUT