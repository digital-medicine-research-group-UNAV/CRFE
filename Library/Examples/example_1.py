import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

# Allow importing the local CRFE package from the Library folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CRFE._crfe import CRFE
from CRFE.stopping import ParamParada

# Multiclass synthetic example with noisy features
X, Y = make_classification(
    n_samples=800,
    n_features=20,
    n_informative=10,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=3.0,
    flip_y=0.05
)

# Adding additional random noise features
rng = np.random.RandomState()
rand_var = rng.randint(10, size=(800, 35))
X = np.hstack([X, rand_var])

X = MinMaxScaler().fit_transform(X)

X_tr, X_cal, Y_tr, Y_cal = train_test_split(
    X, Y, test_size=0.4, stratify=Y, random_state=7
)

estimator = LinearSVC(tol=1e-4, loss="squared_hinge", max_iter=50000)

crfe = CRFE(
    estimator,
    features_to_select=1,
    stopping_activated=True,
    stopping_params=ParamParada(alpha=0.05, eps=0.02, eta=0.08, paciencia=15),
)
crfe.fit(X_tr, Y_tr, X_cal, Y_cal)

print("Selected features:", crfe.idx_features_)
print("Betas:", np.round(crfe.idx_betas_, 4))
print("Stopping reason:", crfe.stopping_reason_)
