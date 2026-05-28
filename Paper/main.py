"""

"""

import numpy as np
import sys
import os
import json
import argparse
import pickle
from functools import lru_cache

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from _crfe_utils import READER
from _crfe_strict_ import CRFE, Stepwise_SMFS
from rfe_module_strict import Stepwise_RFE, Stepwise_LASSO, Stepwise_ELASTICNET, Stepwise_BORUTA

RESULTS_PARENT = os.environ.get("RESULTS_PARENT", "RESULTS")


# Cache random number generator for efficiency
@lru_cache(maxsize=128)
def get_random_state(seed):
    """Cached random state generator to avoid recreation."""
    return np.random.default_rng(seed)


def random_integer(seed):
    """Optimized random integer generation."""
    rng = get_random_state(seed)
    return rng.integers(low=0, high=100000)


def create_estimator():
    """Factory function for creating optimized estimator."""
    return LinearSVC(
        tol=1e-4, 
        loss='squared_hinge',
        max_iter=14000,
        dual="auto"
    )


def split_data_efficiently(X, Y, run_id, test_size=0.15, cal_size=0.45):

    """

    Parameters
    ----------
    X : array-like
        Feature matrix
    Y : array-like  
        Labels
    run_id : int
        Run identifier for seeding
    test_size : float, default=0.20
        Proportion of data for testing
    cal_size : float, default=0.5
        Proportion of remaining data for calibration
        
    Returns
    -------
    tuple
        (X_tr, X_cal, X_test, Y_tr, Y_cal, Y_test)
    """

    seed = random_integer(42 + run_id)
    
    # First split: separate test set
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, 
        test_size=test_size, 
        shuffle=True, 
        stratify=Y, 
        random_state=seed
    )
    
    # Second split: separate training and calibration
    X_tr, X_cal, Y_tr, Y_cal = train_test_split(
        X_temp, Y_temp, 
        test_size=cal_size, 
        shuffle=True, 
        stratify=Y_temp, 
        random_state=seed
    )
    
    return X_tr, X_cal, X_test, Y_tr, Y_cal, Y_test

def standardize_split_data(X_tr, X_cal, X_test, scaler=None):

    # 1. Definir el Scaler (Por defecto StandardScaler para SVM)
    if scaler is None:
        scaler = StandardScaler()
        
    # 2. FIT
    scaler.fit(X_tr)
    
    # 3. TRANSFORM: 
    X_tr_scaled = scaler.transform(X_tr)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Retornar los datos escalados y las etiquetas originales
    return X_tr_scaled, X_cal_scaled, X_test_scaled

 
def run_crfe_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    
    crfe = CRFE(estimator, 
                features_to_select=1,
                stopping_activated=False,
                strict_conformal=True,
                selection_fraction=0.5)
    
    crfe.fit(X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
    return crfe.results_dicc


def run_smfs_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    
    smfs = Stepwise_SMFS(
                estimator, 
                features_to_select=1,
                stopping_activated=False,
                strict_conformal=True,
                selection_fraction=0.5)
    
    smfs.fit(X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
    return smfs.results_dicc


def run_rfe_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    
    rfe = Stepwise_RFE(estimator,
                       features_to_select=1,
                       stopping_activated=False,
                       strict_conformal=True,
                       selection_fraction=0.5)
    
    rfe.fit(X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
    return rfe.results_dicc


def run_lasso_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    
    lasso = Stepwise_LASSO(estimator,
                       features_to_select=1,
                       stopping_activated=False,
                       strict_conformal=True,
                       selection_fraction=0.5)
    
    lasso.fit(X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)

    return lasso.results_dicc


def run_elasticnet_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    
    elasticnet = Stepwise_ELASTICNET(estimator,
                       features_to_select=1,
                       stopping_activated=False,
                       strict_conformal=True,
                       selection_fraction=0.5)
    
    elasticnet.fit(X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)

    return elasticnet.results_dicc


def run_boruta_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    
    boruta = Stepwise_BORUTA(estimator,
                       features_to_select=1,
                       stopping_activated=False,
                       strict_conformal=True,
                       selection_fraction=0.5)
    
    boruta.fit(X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)

    return boruta.results_dicc


def save_results_efficiently(results, data_path, model, run_id):
    
    results_dir = f"{RESULTS_PARENT}/results_{data_path}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_subdir = os.path.join(results_dir, "results")
    os.makedirs(results_subdir, exist_ok=True)
    
    filepath = os.path.join(results_subdir, f"result_{model}_{run_id}.pickle")
    
    # Use protocol 4 for better performance and compatibility
    with open(filepath, 'wb') as f:
        pickle.dump([results], f, protocol=4)
    
    print(f"Results saved to: {filepath}")
    sys.stdout.flush()


def main_(run_id, model, data_path):

    """
    
    Parameters
    ----------
    run_id : int
        Experiment run identifier
    model : str
        Model type ('crfe' or 'rfe')
    data_path : str
        Path to dataset
    """
    
    print(f"Starting experiment: run_id={run_id}, model={model}, data_path={data_path}")
    sys.stdout.flush()
    
    # Load data efficiently
    reader = READER()
    X, Y, y_classes = reader.get_data(data_path)
    
    print(f"Dataset loaded: {X.shape} samples, {len(y_classes)} classes: {y_classes}")
    sys.stdout.flush()

    # Efficient data splitting
    X_tr, X_cal, X_test, Y_tr, Y_cal, Y_test = split_data_efficiently(X, Y, run_id)
    X_tr, X_cal, X_test = standardize_split_data(X_tr, X_cal, X_test)
    
    print(f"Data split - Train: {X_tr.shape}, Cal: {X_cal.shape}, Test: {X_test.shape}")
    sys.stdout.flush()

    # Create estimator
    estimator = create_estimator()

    # Run experiment based on model type
    if model == "crfe":
        print("Running CRFE experiment...")
        sys.stdout.flush()
        results = run_crfe_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
        
    elif model == "rfe":
        print("Running RFE experiment...")
        sys.stdout.flush()
        results = run_rfe_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)

    elif model == "lasso":
        print("Running LASSO experiment...")
        sys.stdout.flush()
        results = run_lasso_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
    
    elif model == "elasticnet":
        print("Running ELASTICNET experiment...")
        sys.stdout.flush()
        results = run_elasticnet_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)

    elif model == "boruta":
        print("Running BORUTA experiment...")
        sys.stdout.flush()
        results = run_boruta_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
    
    elif model == "smfs":
        print("Running SMFS experiment...")
        sys.stdout.flush()
        results = run_smfs_experiment(estimator, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)
        
    else:
        raise ValueError(f"Model '{model}' not found.")

    # Save results efficiently
    save_results_efficiently(results, data_path, model, run_id)
    
    print(f"Experiment completed successfully for run_id={run_id}")
    sys.stdout.flush()

    return results


def parse_arguments():
    """Parse command line arguments with validation."""
    parser = argparse.ArgumentParser(
        description="Optimized feature selection experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--run_id', 
        type=int, 
        required=True,
        help='Experiment run identifier (integer)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['crfe', 'rfe', 'smfs', 'lasso', 'elasticnet', 'boruta'],
        help='Model type to run'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to dataset'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if args.run_id < 0:
            raise ValueError("run_id must be non-negative")
            
        # Run main experiment
        main_(args.run_id, args.model, args.data_path)
        
        print("All experiments completed successfully.")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
