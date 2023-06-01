
import numpy as np

from sklearn.svm import SVR, LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../") 
from CRFE._crfe import CRFE

 ## binary classification  # 3 informative feature and 10  random features.
 
X,Y = make_classification(
        n_samples=1000, n_features=3, n_redundant=0, n_classes = 2,
        n_informative=3, n_clusters_per_class=1, class_sep=2.5, flip_y=0.01, scale = None,  weights=[0.5,0.5], shuffle=True)
        
# Add random variables
rand_var = np.random.randint(10, size=(1000, 5) )
X = list(np.hstack([np.array(X), np.array(rand_var) ]))




### Load the dataset  ##

X_tr , X_cal , Y_tr, Y_cal = train_test_split( X, Y, test_size=0.5, stratify=Y)


estimator = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge') 
                      #max_iter= 300000,
                      #multi_class = "ovr")


###############  Library use example   ###################

crfe = CRFE(estimator , features_to_select = 4)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)
print(crfe.idx_features_)
print(crfe.classes_)







