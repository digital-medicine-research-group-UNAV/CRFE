

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import itertools
import random
import os
from math import comb
from scipy.stats import norm
import math
import pickle
import argparse
from sklearn.neighbors import KernelDensity
from pathlib import Path

RESULTS_PARENT = os.environ.get("RESULTS_PARENT", "RESULTS")


class StabilityCalculator:
    """
    A class to compute the stability of feature selections using the Nogueira stability
    metric and to derive a conformal interval via resampling.
    """
    def __init__(self, alpha=0.1, lin_spaces=600, max_samples=200):
        """
        Initialize the StabilityCalculator.
        
        Parameters:
            alpha (float): Significance level for the conformal interval.
            lin_spaces (int): Number of points in the linspace for evaluating stability.
        """
        self.alpha = alpha
        self.lin_spaces = lin_spaces
        self.max_samples = max_samples

    @staticmethod
    def check_input_type(Z):
        """
        Ensure that Z is a 2D numpy array.
        
        Parameters:
            Z (list or numpy.ndarray): The input binary matrix.
            
        Returns:
            numpy.ndarray: The input converted to a numpy array (if necessary).
            
        Raises:
            ValueError: If Z is not 2-dimensional.
        """
        if isinstance(Z, list):
            Z = np.asarray(Z)
        elif not isinstance(Z, np.ndarray):
            raise ValueError('The input matrix Z should be of type list or numpy.ndarray')
        if Z.ndim != 2:
            raise ValueError('The input matrix Z should be of dimension 2')
        return Z

    @staticmethod
    def get_nogueira_stability(Z):
        """
        THE FOLLOWING FUNCTIONS WERE ORIGINALLY DEVELOPED BY [1] On the Stability of Feature Selection. Sarah Nogueira, Konstantinos Sechidis, Gavin Brown. Journal of Machine Learning Reasearch (JMLR). 2017
        Compute the Nogueira stability estimate for a binary matrix.
        
        Parameters:
            Z (list or numpy.ndarray): A binary matrix (M x d) where each row is a feature set.
            
        Returns:
            float: The computed stability measure.
        """
        Z = StabilityCalculator.check_input_type(Z)
        M, d = Z.shape
        hatPF = np.mean(Z, axis=0)
        kbar = np.sum(hatPF)
        denom = (kbar/d) * (1 - kbar/d)
        return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom

    def folded_samples(self, data_cal, k):
        """
        Generate folded samples from data_cal using combinations of indices.
        If the total number of combinations is greater than max_samples, a random sample is used.
        
        Parameters:
            data_cal (list): List of data rows.
            k (int): Number of samples to choose in each combination.
            
        Returns:
            list: A list of folded samples (each is a list of rows from data_cal).
        """
        n = len(data_cal)
        total = comb(n, k)
        samples = []
        if total > self.max_samples:
            # Sample max_samples random combinations without generating all combinations.
            for _ in range(self.max_samples):
                indices = sorted(np.random.choice(n, k, replace=False))
                samples.append([data_cal[i] for i in indices])
        else:
            for comb_indices in itertools.combinations(range(n), k):
                samples.append([data_cal[i] for i in comb_indices])
        return samples


    def get_stability(self, data_train, M=-1):
        """
        Compute the conformal interval for the stability measure and the overall Nogueira stability.
        
        Parameters:
            data_train (np.ndarray): A binary matrix of shape (number of feature sets, n_features).
            M (int): If -1, set M = max(2, number of feature sets // 2).
            
        Returns:
            tuple: (conformal_interval, overall_stability)
                   where conformal_interval is a list [low, high] and overall_stability is the stability of data_train.
        """
        m_tr = data_train.shape[0]
        d = data_train.shape[1]
        
        # Instead of an expensive loop, use a heuristic:
        if M == -1:
            M = max(2, m_tr // 2)

        # Get folded samples (with random sampling if needed)
        data_train_sep = self.folded_samples(data_train, M)
        
        # Compute stability for each folded sample
        stability_measures = [self.get_nogueira_stability(np.array(data)) for data in data_train_sep]
        
        # Create a grid for y_trial values
        Y_trial = np.linspace(-1/(M-1), 1, num=self.lin_spaces)
        P_values = []
        for y_trial in Y_trial:
            stability_array = np.array(stability_measures + [y_trial])
            mean_val = stability_array.mean()
            std_val = stability_array.std()
            ncm = np.abs((stability_array - mean_val) / std_val)
            ncm_last = ncm[-1]
            # Vectorized computation of count of elements less than or equal to ncm_last
            count_less = np.sum(ncm[:-1] < ncm_last)
            count_equal = np.sum(ncm[:-1] == ncm_last)
            tau = random.random()
            p_value = (count_less + tau * count_equal) / len(ncm)
            P_values.append(p_value)
            
        # Determine the conformal interval based on the p-values
        intervalo = [y for y, p in zip(Y_trial, P_values) 
                     if (len(stability_measures) + 1) * p <= np.ceil((1. - self.alpha) * (len(stability_measures) + 1))]
        try:
            set_prediction = [intervalo[0], intervalo[-1]]
        except IndexError:
            set_prediction = intervalo

        overall_stability = self.get_nogueira_stability(data_train)
        return set_prediction, overall_stability

    def compute_stability(self, feat_subsets, n_features, M=-1):
        """
        Compute the stability measures for each index position from the given feature subsets.
        
        Parameters:
            feat_subsets (list): A list (for each repetition) of feature selection results.
                                 Each element should be a list of selections for each index position.
                                 For example, if you load your merged pickle file and extract the
                                 "Index" field, then feat_subsets = [ele["Index"] for ele in OUT].
            n_features (int): Total number of features.
            M (int): Parameter to pass to get_stability; if -1, it is computed automatically.
            
        Returns:
            pandas.DataFrame: A dataframe with columns ['Low_int', 'High_int', 'Nogue'] for each index.
        """
        df = pd.DataFrame(columns=['Low_int', 'High_int', 'Nogue'])
        # Assume each feat_subsets element is a list (of length n) of feature indices for that run.
        num_indices = len(feat_subsets[0])
        for i in range(num_indices):
            data_train = []
            for subset in feat_subsets:
                # Create a binary vector of length n_features: 1 if the feature is selected, 0 otherwise.
                ele_matrix = [1 if j in subset[i] else 0 for j in range(n_features)]
                data_train.append(ele_matrix)
            set_pre, nogue = self.get_stability(np.array(data_train), M=M)
            if not set_pre:
                set_pre = [-.5, 1.] # these are the theoretical maximum and minimum values
            df.loc[len(df)] = [set_pre[0], set_pre[1], nogue]
        return df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)

    args = parser.parse_args()

    dataset = args.data_path
    method = args.method

    # Instantiate the calculator with desired parameters.
    stability_calc = StabilityCalculator(alpha=0.1, lin_spaces=500)

    project_root = Path(__file__).resolve().parent
    results_root = project_root / RESULTS_PARENT / f"results_{dataset}"

    # Load the merged results.
    with open(results_root / f"merged_{method}_results.pickle", 'rb') as f:
        OUT = pickle.load(f)

    # Extract feature subsets. (Assumes each dictionary in OUT has an "Index" key.)
    feat_subsets = [ele["Index"] for ele in OUT]
    print(feat_subsets)

    # Determine n_features (this depends on your data; for example, you might compute it as):
    n_features = max(s.max() if s.size > 0 else 0 for subset in feat_subsets for s in subset) + 1
    print(f"Number of features: {n_features}")

    df_stability = stability_calc.compute_stability(feat_subsets, n_features, M=-1)

    #Remember that the first row are the stability results for the n-1 features.
    results_root.mkdir(parents=True, exist_ok=True)
    df_stability.to_csv(results_root / f"stability_{method}.csv", index=False)
