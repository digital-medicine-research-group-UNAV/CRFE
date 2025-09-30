import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CRFE._crfe import CRFE
from CRFE.stopping import ParamParada

rng = check_random_state(0)
iris = load_iris()

# Add synthetic noise to stress the feature elimination
X = np.c_[iris.data, rng.normal(size=(len(iris.data), 6))]
Y = iris.target

X_tr, X_test, Y_tr, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=17
)
X_tr, X_cal, Y_tr, Y_cal = train_test_split(
    X_tr, Y_tr, test_size=0.5, stratify=Y_tr, random_state=17
)

svc = LinearSVC(tol=1e-4, loss="squared_hinge", max_iter=300000)
crfe = CRFE(
    svc,
    features_to_select=1,
    stopping_activated=True,
    stopping_params=ParamParada(alpha=0.05, eps=0.02, eta=0.1, paciencia=25),
)
crfe.fit(X_tr, Y_tr, X_cal, Y_cal)

print("Selected features:", crfe.idx_features_)
print("Stopping reason:", crfe.stopping_reason_)

X_tr_sel = X_tr[:, crfe.idx_features_]
X_test_sel = X_test[:, crfe.idx_features_]
svm_selected = crfe.estimator_.fit(X_tr_sel, Y_tr)

print("Score with selected features:", svm_selected.score(X_test_sel, Y_test))
