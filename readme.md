# CRFE - Conformal Recursive Feature Selection

*CRFE*  is the first feature selection method based on a recursive backward elimintation policy that takes advantage from the Conformal Prediction framework [1]. CRFE´s objective is minimizing the non-conformity of the features [2] using a recursive elimination policy. We also present an automatic stopping criteria for recursive methods. 
 
## Requirements

- Python 3.9+
- scikit-learn 1.2.2+
- numpy

## Quickstart

You can install this repository locally via pip using


```bash
pip install .
```

For editable/development mode:

```bash
pip install -e .
```

Using pixi (see `pixi.toml` in this repository):

```bash
pixi install
pixi run install-local
```




Let start with a basic example. This example is coded in *Examples/example_2.py*.



```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from CRFE import CRFE, ParamParada

rng = check_random_state(0)
iris = load_iris()

# Append irrelevant noise features under a fixed seed
X = np.c_[iris.data, rng.normal(size=(len(iris.data), 6))]
Y = iris.target

X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
X_tr, X_cal, Y_tr, Y_cal = train_test_split(X_tr, Y_tr, test_size=0.5, stratify=Y_tr)

estimator = LinearSVC(tol=1e-4, loss="squared_hinge", max_iter=300000)
crfe = CRFE(
    estimator,
    features_to_select=3,
    stopping_activated=True,
    stopping_params=ParamParada(alpha=0.05, eps=0.02, eta=0.1, paciencia=20),
)
crfe.fit(X_tr, Y_tr, X_cal, Y_cal)

print("Selected features:", crfe.idx_features_)
print("Stopping reason:", crfe.stopping_reason_)
```

The selected features can be reused with any estimator. The cloned estimator is available through `crfe.estimator_`.

```python
X_tr_sel = X_tr[:, crfe.idx_features_]
X_test_sel = X_test[:, crfe.idx_features_]

svm_selected = crfe.estimator_.fit(X_tr_sel, Y_tr)
print("Score with selected features:", svm_selected.score(X_test_sel, Y_test))
```

## Folder layout

```
CRFE_library_pro/
├── Library/
│   ├── CRFE/
│   │   ├── _crfe.py
│   │   ├── _crfe_utils.py
│   │   ├── documentation.txt
│   │   └── stopping.py
│   └── Examples/
│       ├── example_1.py
│       └── example_2.py
├── LICENSE
└── readme.md
```





## References 

[1] V. Balasubramanian, S.-S. Ho, and V. Vovk, Conformal Prediction
for Reliable Machine Learning: Theory, Adaptations and Applications,
1st ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.,
2014.

[2] T. Bellotti, Z. Luo, and A. Gammerman, “Strangeness Minimisation
Feature Selection with Confidence Machines,” in Intelligent Data Engineering
and Automated Learning IDEAL 2006, ser. Lecture Notes in
Computer Science, E. Corchado, H. Yin, V. Botti, and C. Fyfe, Eds.
Berlin, Heidelberg: Springer, pp. 978–985, 2006
