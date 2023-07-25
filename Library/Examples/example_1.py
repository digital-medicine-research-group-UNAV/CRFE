


import numpy as np
import sys

from sklearn.svm import  LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, "../") 
from CRFE._crfe import CRFE

 ## binary classification  # 3 informative feature and 10  random features.
 
X,Y = make_classification(
        n_samples=1000, n_features=3, n_redundant=0, n_classes = 3,
        n_informative=3, n_clusters_per_class=1, class_sep=2.5, flip_y=0.01, scale = None,  
        weights=[0.33,0.33,0.33], shuffle=True)
        
# Add random variables
rand_var = np.random.randint(10, size=(1000, 10) )
X = list(np.hstack([np.array(X), np.array(rand_var) ]))

# Normalize
X = MinMaxScaler().fit_transform(X)

# Split dataset 

X_tr , X_cal , Y_tr, Y_cal = train_test_split( X, Y, test_size=0.5, shuffle = True, stratify=Y)


#### select your linear model  (uncomment code or code your classifier)


# Linear SVM:

from sklearn.svm import  LinearSVC
estimator = LinearSVC(tol = 1.e-3, 
                      loss='squared_hinge',
                      max_iter= 10000)


# Logistic:

#from sklearn.linear_model import LogisticRegression
#estimator = LogisticRegression()

# GLM:

#from sklearn.linear_model import RidgeClassifier
#estimator = RidgeClassifier()




###############  Library use example   ###################


crfe = CRFE(estimator , features_to_select = 3)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)

print("Features selected: " , crfe.idx_features_)
print("Betas: " ,crfe.idx_betas_)

###### We can import the estimator(not fitted yet) or code one by ourself ####

estimator_fit = crfe.estimator_.fit(X_tr_, Y_tr)

X_test_, Y_test = #define test samples

print(estimator_fit.score(X_test_, Y_test))






