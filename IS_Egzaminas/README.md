# Wine quality prediction — exam implementation

Master's course *Intelektualiosios sistemos*. Implements the colloquium plan
on the UCI Wine Quality data: **mean** and **linear regression** baselines,
and the four plan methods **SVR-RBF** (main), **MLP**, **RBF network**, and **random forest**.

Hyperparameters are chosen by group-aware CV **on the training block only**.
The frozen 20 % test set is scored once.

## Setup

Python 3.11+. From this directory:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Data are downloaded on first run (UCI, SHA-256 checked). Do not commit CSVs.

## One command for the main experiment

```
python -m winequality.run_experiment
```

Fixed seeds: `SPLIT_SEED = 0` (train/test cut), `DEFAULT_SEED = 0` (model + HPO).
Outputs:

- `results/metrics.json` — all numbers used for comparison
- `results/comparison.csv` — MAE, Acc_T, κ_w, macro-F1
- `results/figures/` — MAE bars, confusion matrices, sensitivity
- `results/error_examples_{red,white}.csv` — largest |residual| samples
- `artifacts/svr_{red,white}.joblib` — fitted main model

## Jupyter užrašinės (naršyklėje)

Lietuviškos užrašinės kataloge `notebooks/` — po vieną modeliui ir `07_modeliu_palyginimas.ipynb`.

```
python -m winequality.paleisti_uzrasines
```

Atsidaro JupyterLab adresu http://127.0.0.1:8888 . Plačiau: `notebooks/README.md`.

## Tests (no test-set peeking)

```
python -m pytest tests -q
```

`test_svr_formula_matches_sklearn` checks that

    f(z) = Σ_i (α_i − α_i*) exp(−γ ‖z_i − z‖²) + b

in `winequality/svr_model.py` (`predict_from_dual`, used by `formula_predict`)
matches `sklearn.svm.SVR.predict`.

`test_rbf_predict_matches_formula` checks that

    φ_k(z) = exp(−‖z − c_k‖² / (2 σ_k²)),   f(z) = Σ_k w_k φ_k(z) + b

in `winequality/rbf.py` is the path used by `RBFNetwork.predict`.

## Single-sample CLI

```
python -m winequality.predict --kind red --json sample.json
```

## Ablation and robustness (included in the one command)

- Ablation A1: linear-kernel SVR vs RBF SVR, same split and metrics.
- Sensitivity: alcohol and volatile acidity swept ±1 training SD.
- Robustness: (1) Gaussian feature noise at 0.1·std; (2) group-aware vs random split (duplicate leakage).
