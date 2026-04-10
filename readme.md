# CRFE - Conformal Recursive Feature Selection

*CRFE* is a feature selection method based on recursive backward elimination, grounded in the Conformal Prediction framework [1]. The method minimizes feature non-conformity [2] through an iterative elimination policy and incorporates an automatic stopping criterion for recursive procedures.
 
## Requirements

- Python 3.9+
- scikit-learn 1.2.2+
- numpy

## Quickstart

Install from PyPI with:

```bash
pip install CRFE
```

Install from TestPyPI with:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple CRFE
```

Use with pixi (installs `CRFE` from PyPI via `pixi.toml`):

```bash
pixi install
pixi run check-crfe
```

## Examples

For complete usage scripts, see:

- `Library/Examples/example_1.py` (multiclass synthetic benchmark with noisy variables)
- `Library/Examples/example_2.py` (Iris dataset with synthetic noise and downstream evaluation)

Both examples use the public package interface:

```python
from CRFE import CRFE, ParamParada
```

## Folder layout

```
CRFE/
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
