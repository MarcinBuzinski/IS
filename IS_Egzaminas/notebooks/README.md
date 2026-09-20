# Jupyter užrašinės (naršyklėje)

Lietuviškos užrašinės — po vieną kiekvienam įgyvendintam modeliui ir viena palyginimui.

| Failas | Turinys |
|---|---|
| `01_vidurkio_prognoze.ipynb` | Baseline: \(\hat y = \bar y\) |
| `02_tiesine_regresija.ipynb` | Baseline: \(f(z)=w^\top z+b\) |
| `03_svr.ipynb` | Pagrindinis metodas SVR–RBF (dualinė formulė kode) |
| `04_mlp.ipynb` | Daugiasluoksnis perceptronas |
| `05_rbf_tinklas.ipynb` | RBF tinklas (centrai + ridge) |
| `06_atsitiktinis_miskas.ipynb` | Atsitiktinis miškas (H2 atskaita) |
| `07_modeliu_palyginimas.ipynb` | Tas pats skaidymas, visos metrikos, H1/H2, abliacija |

## Paleidimas naršyklėje

Projekto šaknyje (su aktyviu `.venv`):

```
python -m pip install -r requirements.txt
python -m winequality.paleisti_uzrasines
```

Atsidarys JupyterLab adresu http://127.0.0.1:8888 . Kairėje pasirinkite užrašinę ir spauskite **Run → Run All Cells**.

Kintamasis `KIND = "red"` pakeičiamas į `"white"`, jei norite balto vyno. Skaidymo sėkla ta pati kaip eksperimente (`SPLIT_SEED = 0`).

Palyginimo užrašinė skaito `results/comparison.csv`. Jei failo nėra:

```
python -m winequality.run_experiment
```
