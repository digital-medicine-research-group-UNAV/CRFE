



import sys
import numpy as np

from sklearn.datasets import load_iris, make_classification
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../") 
from CRFE._crfe import CRFE 



generator = check_random_state(0)
iris = load_iris()

# Add some irrelevant features. Random seed is set to make sure that
# irrelevant features are always irrelevant.

X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
Y = iris.target

X_tr , X_test , Y_tr, Y_test = train_test_split( X, Y, test_size=0.2, stratify=Y)
X_tr , X_cal , Y_tr, Y_cal = train_test_split( X_tr, Y_tr, test_size=0.5, stratify=Y_tr)

estimator = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge',
                      max_iter= 300000)

crfe = CRFE(estimator , features_to_select = 3)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)

print("Selected features: ", crfe.idx_features_)

## Delete the dismissed features

X_tr_ = list(np.array(X_tr)[:, crfe.idx_features_]) 
X_test_ = list(np.array(X_test)[:, crfe.idx_features_]) 


SVM_fit = crfe.estimator_.fit(X_tr_, Y_tr)
print(SVM_fit.score(X_test_, Y_test))


SVM_est_2 = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge',
                      max_iter= 300000)

SVM_fit_2 = SVM_est_2.fit(X_tr, Y_tr)
print(SVM_fit_2.score(X_test, Y_test))




