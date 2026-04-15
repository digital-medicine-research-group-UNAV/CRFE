"""
Optimized Conformal Prediction module

Performance improvements:
- Vectorized operations for p-value computations
- Efficient array operations
- Reduced memory allocations
- JIT compilation for critical sections
"""

import os
import numpy as np
import itertools
from numba import jit, njit

from sklearn.utils._param_validation import HasMethods, Interval, Real


@njit
def compute_p_values_fast(NCM_test_sample, NCM_cal, random_seed=None):
    """Optimized p-value computation using numba with improved randomization."""
    n_cal = len(NCM_cal)
    p_values = np.zeros(len(NCM_test_sample), dtype=np.float64)
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    for i, ncm_test in enumerate(NCM_test_sample):
        # Count how many calibration scores are >= test score
        count_geq = np.sum(NCM_cal >= ncm_test)
        count_eq = np.sum(NCM_cal == ncm_test)
        
        # Compute p-value with randomization
        p_val = (count_geq + 1) / (n_cal + 1)
        random_term = np.random.uniform(0, 1) * (count_eq + 1) / (n_cal + 1)
        p_values[i] = p_val + random_term
    
    return p_values


@njit
def compute_metrics_fast(set_predictions, y_test, n_classes):
    """Optimized metrics computation using numba."""
    n_samples = len(set_predictions)
    coverage = 0
    inefficiency = 0
    certainty = 0
    uncertainty = 0
    mistrust = 0
    
    for i in range(n_samples):
        #print("\n", y_test[i], set_predictions[i])
        prediction_set = set_predictions[i]
        set_size = np.sum(prediction_set)
        
        # Check if true label is in prediction set
        
        if prediction_set[y_test[i]]:
            coverage += 1
            
            if set_size == 1:
                certainty += 1
            elif set_size == n_classes:
                uncertainty += 1
        
        if set_size == 0:
            mistrust += 1
            
        inefficiency += set_size
    
    return coverage, inefficiency, certainty, uncertainty, mistrust


class CP:
    """
    Optimized Conformal Prediction class.

    Parameters
    -------------
    alpha : float, default=0.1
        Significance level for conformal prediction (1 - confidence level).

    Attributes
    ------------
    Coverage_ : float
        Empirical coverage rate
    Inefficiency_ : float  
        Average size of prediction sets
    Certainty_ : float
        Rate of singleton prediction sets
    Uncertainty_ : float
        Rate of prediction sets containing all classes
    Desconfianza_ : float
        Rate of empty prediction sets (mistrust)
    S_score_ : float
        Average sum of p-values
    Creditibily_ : float
        Average maximum p-value
    F_score_ : float
        Average F-score (S_score - Creditibility)
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, 1, closed="both")]
    }

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def Split_conformal_prediction(self, NCM_test, NCM_cal, random_seed=None):
        """
        Optimized split conformal prediction with vectorized operations and validation.
        
        Parameters
        ----------
        NCM_test : array-like of shape (n_test_samples, n_classes)
            Non-conformity measures for test samples
        NCM_cal : array-like of shape (n_cal_samples,)
            Non-conformity measures for calibration samples
            
        Returns
        -------
        Set_prediction : list of arrays
            Boolean arrays indicating which classes are in each prediction set
        p_values_Y : list of arrays
            P-values for each class for each test sample
        """
        NCM_cal = np.asarray(NCM_cal, dtype=np.float32)
        Set_prediction = []
        p_values_Y = []
        
        for ncm_test_sample in NCM_test:
            ncm_test_sample = np.asarray(ncm_test_sample, dtype=np.float32)
            
            # Use optimized numba function for p-value computation
            p_values = compute_p_values_fast(ncm_test_sample, NCM_cal)
            #print("\n p_values:", p_values)
            # Vectorized prediction set computation
            set_prediction = p_values >= self.alpha
            #print(" set_prediction:", set_prediction)
            p_values_Y.append(p_values.tolist())
            Set_prediction.append(set_prediction.tolist())

        return Set_prediction, p_values_Y

    def Conformal_prediction_scores(self, y_test, NCM_test, NCM_cal, y_classes_name, 
                                  output_path=None, Flag=False):
        """
        Optimized conformal prediction scoring with efficient computations.
        
        Parameters
        ----------
        y_test : array-like
            True labels for test samples
        NCM_test : array-like
            Non-conformity measures for test samples
        NCM_cal : array-like
            Non-conformity measures for calibration samples
        y_classes_name : array-like
            Class names/labels
        output_path : str, optional
            Path to save detailed results
        Flag : bool, default=False
            Whether to save detailed results to file
            
        Returns
        -------
        list
            [Coverage, Inefficiency, Certainty, Uncertainty, Mistrust, S_score, Creditibility, F_score]
        """
        
        # Convert inputs to numpy arrays for efficiency
        y_test = np.asarray(y_test, dtype=np.int32)
        
        # Compute prediction sets and p-values
        self.set_prediction, self.p_y_list = self.Split_conformal_prediction(NCM_test, NCM_cal)
        
        # Vectorized score computations
        p_y_array = np.array(self.p_y_list, dtype=np.float32)
        Creditibily = np.max(p_y_array, axis=1)
        S_score = np.sum(p_y_array, axis=1)
        F_score = S_score - Creditibily
        
        # Convert prediction sets to boolean array for efficient processing
        set_pred_array = np.array(self.set_prediction, dtype=bool)

        #for i in range(len(self.set_prediction)):
        #    print("\n", y_test[i], set_pred_array[i],  p_y_array[i])


        # Use optimized numba function for metrics computation
        coverage, inefficiency, certainty, uncertainty, mistrust = compute_metrics_fast(
            set_pred_array, y_test, len(y_classes_name)
        )
        
        n_samples = len(self.p_y_list)
        
        # Compute final metrics
        self.Coverage_ = coverage / n_samples
        self.Inefficiency_ = inefficiency / n_samples
        self.Certainty_ = certainty / n_samples
        self.Uncertainty_ = uncertainty / n_samples
        self.Desconfianza_ = mistrust / n_samples
        self.S_score_ = np.mean(S_score)
        self.Creditibily_ = np.mean(Creditibily)
        self.F_score_ = np.mean(F_score)

        # Optional file output (kept for compatibility)
        if Flag and output_path:
            self._write_results_to_file(output_path)

        return [
            self.Coverage_, self.Inefficiency_, self.Certainty_, 
            self.Uncertainty_, self.Desconfianza_, self.S_score_, 
            self.Creditibily_, self.F_score_
        ]

    def _write_results_to_file(self, output_path):
        """Write detailed results to file."""
        with open(output_path, 'a') as out:
            out.write(f"\n ---------------------------------------------\n")
            out.write(f"METRICS: \n\n")
            out.write(f"Set prediction:  {self.set_prediction}\n\n")
            out.write(f"Empirical coverage:  {self.Coverage_}\n")
            out.write(f"N score (Efficiency):  {self.Inefficiency_}\n")
            out.write(f"Certainty:  {self.Certainty_}\n")
            out.write(f"Uncertainty:  {self.Uncertainty_}\n")
            out.write(f"Mistrust:  {self.Desconfianza_}\n")
            out.write(f"S score:  {self.S_score_}\n")
            out.write(f"F score:  {self.F_score_}\n")
            out.write(f"Creditibily (Max p_y):  {self.Creditibily_}\n\n\n")
            out.write("#####################################\n\n")
