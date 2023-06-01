

from sklearn.svm import SVR, LinearSVC
from sklearn.datasets import make_classification
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.insert(0, "../") 
from CRFE._crfe import CRFE

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]




X_tr , X_cal , Y_tr, Y_cal = train_test_split( X, Y, test_size=0.5, stratify=Y)


estimator = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge') 
                      #max_iter= 300000,
                      #multi_class = "ovr")


###############  Library


crfe = CRFE(estimator , features_to_select = 5)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)
print("Features selected: ", crfe.features_idx_())


