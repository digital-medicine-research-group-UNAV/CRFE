

from CRFE._crfe import CRFE
from sklearn.svm import SVR, LinearSVC
from sklearn.datasets import make_classification
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd



from sklearn.datasets import load_breast_cancer


# load dataset
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names).to_numpy()
y = breast_cancer.target



X_tr , X_cal , Y_tr, Y_cal = train_test_split( X, y, test_size=0.5, stratify=y)


estimator = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge',
                      max_iter= 300000,
                      multi_class = "ovr")


crfe = CRFE(estimator , features_to_select = 5)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)
print("Features selected: ", crfe.features_idx_())

