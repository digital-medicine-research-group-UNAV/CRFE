

# CRFE - Conformal Recursive Feature Selection

*CRFE*  is the first feature selection method based on a recursive backward elimintation policy that takes advantage from the Conformal Prediction framework [1]. CRFE´s objective is minimizing the non-conformity of the features, i.e., the same as SMFS do [2], but instead of creating a feature ranking, we apply the RFE policy. Multiclass classficcation is available. An automatic stopping criteria is available. We have developed and implemented an scikit-learn dependent open source library that implements CRFE.


## Requirements

Python 3.7 +

Scikit-learnt 1.2.2+



## Quickstart

Let start with a basic example.

After having downloaded the folder "CRFE" into the directory "Library," we proceed to create a new Python script within the subfolder named "Examples.
Firstly, we will import the required modules.

```python
import sys
import numpy as np

from sklearn.datasets import load_iris, make_classification
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

#from CRFE._crfe import CRFE   
sys.path.insert(0, "../") 
from CRFE._crfe import CRFE 

```

Let´s built a test.

```python

### Load the dataset  ## 
 
generator = check_random_state(0)
iris = load_iris()

# Add random features

X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
Y = iris.target

# Split the dataset in training and calibration 

X_tr , X_test , Y_tr, Y_test = train_test_split( X, Y, test_size=0.2, stratify=Y)
X_tr , X_cal , Y_tr, Y_cal = train_test_split( X_tr, Y_tr, test_size=0.5, stratify=Y_tr)


```

CRFE libray is scikit-learn API dependent. It follows the same scheme than the RFE method in scikit-learn.


```python

estimator = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge',
                      max_iter= 300000)

crfe = CRFE(estimator , features_to_select = 3)
crfe.fit(X_tr, Y_tr, X_cal , Y_cal)

```
Let´s call the atribute `idx_features_` to get the list with the features selected.  

```python
print("Selected features: ", crfe.idx_features_)

## Delete the dismissed features

X_tr_ = list(np.array(X_tr)[:, crfe.idx_features_]) 
X_test_ = list(np.array(X_test)[:, crfe.idx_features_]) 
```
The fitted Estimator is imported, but we can fit any other estimator once the features selected are known. We compare the results against the dataset with all the features. 

```python
SVM_fit = crfe.estimator_.fit(X_tr_, Y_tr)
print(SVM_fit.score(X_test_, Y_test))


SVM_est_2 = LinearSVC(tol = 1.e-4, 
                      loss='squared_hinge',
                      max_iter= 300000)

SVM_fit_2 = SVM_est_2.fit(X_tr, Y_tr)
print(SVM_fit_2.score(X_test, Y_test))
```

At this point, you can use your favourite library for uncertainty quantification such as MAPIE [3].

## References 

[1] V. Balasubramanian, S.-S. Ho, and V. Vovk, Conformal Prediction
for Reliable Machine Learning: Theory, Adaptations and Applications,
1st ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.,
2014.

[2] T. Bellotti, Z. Luo, and A. Gammerman, “Strangeness Minimisation
Feature Selection with Confidence Machines,” in Intelligent Data Engineering
and Automated Learning IDEAL 2006, ser. Lecture Notes in
Computer Science, E. Corchado, H. Yin, V. Botti, and C. Fyfe, Eds.
Berlin, Heidelberg: Springer, 2006, pp. 978–985.

[3] V. Taquet, V. Blot, T. Morzadec, L. Lacombe, and N. Brunel,
“Mapie: an open-source library for distribution-free uncertainty
quantification,” arXiv preprint arXiv:2207.12274, 2022



