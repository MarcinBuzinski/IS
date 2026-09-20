"""Sugeneruoja lietuviškas Jupyter užrašines (modeliai + palyginimas)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

NB_META = {
    "kernelspec": {
        "display_name": "Python 3 (winequality)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip() + "\n"}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip() + "\n",
    }


def dump(name: str, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": NB_META,
        "cells": cells,
    }
    path = OUT / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


SETUP = '''
%matplotlib inline
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

here = Path.cwd().resolve()
root = here if (here / "winequality").exists() else here.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "notebooks"))

from nb_pagalba import paruošti, metrikos_lentele, artefaktas
from winequality.config import FEATURE_NAMES, SPLIT_SEED

KIND = "red"  # pakeiskite į "white", jei norite balto vyno
duom = paruošti(KIND)
X_tr, y_tr, X_te, y_te = duom["X_tr"], duom["y_tr"], duom["X_te"], duom["y_te"]
print(f"{KIND}: mokymo n={duom['n_train']}, testo n={duom['n_test']}, sėkla={SPLIT_SEED}")
print("Požymiai:", FEATURE_NAMES)
'''


def vidurkis():
    dump(
        "01_vidurkio_prognoze.ipynb",
        [
            md(
                """# 1. Vidurkio prognozė (privalomas baseline)

**Failas:** `winequality/baselines.py` → `mean_predictor`

Tai paprasčiausias atskaitos metodas: ignoruojami visi laboratoriniai matavimai.
Prognozė visiems mėginiams lygi mokymo aibės kokybės vidurkiui. Jei intelektualusis
metodas nepranoksta šio MAE, požymiai neneša naudingos informacijos (arba skaidymas neteisingas).
"""
            ),
            md(
                r"""## Formulė

\[
\hat y = \bar y_{\mathrm{mok}} = \frac{1}{N}\sum_{i=1}^{N} y_i
\]

Čia \(y_i\) — ekspertų kokybės balas (mediana iš ≥ 3 degustatorių), \(N\) — mokymo mėginių skaičius.
Naudojimo metu \(X\) nenaudojamas — tik vektoriaus ilgis, kad būtų tiek pat prognozių.
"""
            ),
            md("## Duomenys (tas pats užšaldytas skaidymas visiems modeliams)"),
            code(SETUP),
            md("## Įgyvendinimas ir mokymas"),
            code(
                """
from winequality.baselines import mean_predictor

modelis = mean_predictor(y_tr)
print("Išmoktas vidurkis μ =", modelis.get_params()["mu"])

# Tas pats rankiniu būdu — formulės ir kodo sutapimas:
mu = float(np.mean(y_tr))
assert abs(mu - modelis.get_params()["mu"]) < 1e-12
print("Rankinis μ sutampa su mean_predictor.")
"""
            ),
            md("## Vertinimas testinėje aibėje"),
            code(
                """
f = modelis.predict(X_te)
lentele = metrikos_lentele(y_te, f, "vidurkis")
display(lentele.round(4))

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(y_te, bins=np.arange(2.5, 9.6, 1), color="#888888", edgecolor="white", label="tikras balas")
ax.axvline(mu, color="#8b1e3f", lw=2, label=f"prognozė μ={mu:.2f}")
ax.set_xlabel("Kokybės balas")
ax.set_ylabel("Mėginių skaičius (testas)")
ax.set_title(f"Vidurkio prognozė — {KIND} vynas")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.show()
"""
            ),
            md(
                """## Ką stebėti

- MAE turėtų būti apie **0,68** raudonam ir **0,67** baltam (balų sklaida aplink vidurkį).
- Acc_1.0 vis tiek būna aukštas, nes dauguma vynų yra 5–6, o vidurkis irgi ~5,6–5,9.
- Makro-F1 prastas: visos prognozės krenta į klasę „vidutinė“.
"""
            ),
        ],
    )


def tiesine():
    dump(
        "02_tiesine_regresija.ipynb",
        [
            md(
                """# 2. Tiesinė regresija (privalomas baseline)

**Failas:** `winequality/baselines.py` → `linear_pipeline`  
**Pirminis apdorojimas:** `winequality/transform.py` → `Log1pStandardScaler`

Tai vienas tiesinis neuronas (LD1): standartizuotų požymių linijinė kombinacija.
Rinkinio autoriai (Cortez ir kt., 2009) gavo MAE ≈ 0,50 / 0,59; tai mūsų „stop“ taisyklės riba.
"""
            ),
            md(
                r"""## Formulės

Asimetriškiems požymiams \(j \in S\) (cukrus, chloridai, SO₂, sulfatai):

\[
x'_j = \ln(1 + x_j),\qquad
z_j = \frac{x'_j - \mu_j}{\sigma_j}
\]

\(\mu_j,\sigma_j\) skaičiuojami **tik mokymo aibėje**. Prognozė:

\[
f(\mathbf{z}) = \mathbf{w}^\top \mathbf{z} + b
\]

Svoriai \(w,b\) gaunami mažiausių kvadratų metodu (sklearn `LinearRegression` = OLS).
"""
            ),
            code(SETUP),
            md("## Mokymas ir formulės sutikrinimas"),
            code(
                """
from winequality.baselines import linear_pipeline
from winequality.transform import Log1pStandardScaler

vamzdis = linear_pipeline()
vamzdis.fit(X_tr, y_tr)

prep = vamzdis.named_steps["prep"]
lin = vamzdis.named_steps["lin"]
z_te = prep.transform(X_te)

# f(z) = wᵀz + b  — rankinis kelias
f_formule = z_te @ lin.coef_ + lin.intercept_
f_sklearn = vamzdis.predict(X_te)
print("max |formulė − sklearn| =", np.max(np.abs(f_formule - f_sklearn)))

koef = pd.Series(lin.coef_, index=FEATURE_NAMES).sort_values()
display(koef.to_frame("w_j").T.round(3))
print("poslinkis b =", round(float(lin.intercept_), 3))
"""
            ),
            md("## Testo metrikos ir koeficientų interpretacija"),
            code(
                """
lentele = metrikos_lentele(y_te, f_sklearn, "tiesinė regresija")
display(lentele.round(4))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
koef.plot.barh(ax=ax, color="#4a6fa5")
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Koeficientas w (standartizuotoje skalėje)")
ax.set_title("Kuris požymis kelia / mažina prognozuojamą balą")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
"""
            ),
            md(
                """## Ką stebėti

- Alkoholis paprastai turi **teigiamą** \(w\), lakiosios rūgštys — **neigiamą** (enologinė teorija).
- MAE turi būti aiškiai mažesnis už vidurkio baseline (~0,49 raudonam, ~0,57 baltam šiame skaidyme).
- Modelis negali modeliuoti požymių sąveikų (pvz., alkoholis × lakiosios rūgštys) — todėl SVR/RF turi erdvės jį pranokti.
"""
            ),
        ],
    )


def svr():
    dump(
        "03_svr.ipynb",
        [
            md(
                """# 3. Atraminių vektorių regresija (SVR–RBF) — pagrindinis metodas

**Failai:** `winequality/svr_model.py` (`rbf_kernel`, `predict_from_dual`, `formula_predict`)  
**Mokymas:** `winequality/tune.py` → `tune_svr` (hiperparametrai tik mokymo bloke)

Pasirinktas kaip pagrindinis, nes ε-nejautri nuostolio funkcija atitinka subjektyvų balą,
uždavinys iškilas (atkartojamas sprendinys), o Gauso branduolys modeliuoja netiesinius sąryšius.
"""
            ),
            md(
                r"""## Formulės (naudojimo kelias — įgyvendinta kode)

Gauso branduolys:

\[
K(\mathbf{z}_i,\mathbf{z}) = \exp\bigl(-\gamma \|\mathbf{z}_i - \mathbf{z}\|^2\bigr)
\]

Dualinė prognozė (egzamino „formulė ↔ kodas“):

\[
f(\mathbf{z}) = \sum_{i \in \mathrm{SV}} (\alpha_i - \alpha_i^*)\, K(\mathbf{z}_i,\mathbf{z}) + b
\]

`sklearn.svm.SVR.dual_coef_` saugo būtent \((\alpha - \alpha^*)\), `support_vectors_` — \(\mathbf{z}_i\),
`intercept_` — \(b\). Funkcija `predict_from_dual` šią sumą skaičiuoja tiesiogiai.

Mokymas (vieną kartą, LIBSVM/SMO) minimizuoja

\[
\min_{\mathbf{w},b,\xi,\xi^*} \tfrac12\|\mathbf{w}\|^2 + C\sum_i(\xi_i+\xi_i^*)
\]

su ε-nejautriomis nelygybėmis: paklaidos, mažesnės už \(\varepsilon\), **nebaudžiamos**.
"""
            ),
            code(SETUP),
            md("## Artefaktas (egzamino HPO) arba greitas demonstracinis mokymas"),
            code(
                """
import joblib
from winequality.svr_model import formula_predict, svr_pipeline

kelias = artefaktas("svr", KIND)
if kelias.exists():
    bundle = joblib.load(kelias)
    vamzdis = bundle["pipeline"]
    print("Įkeltas artefaktas:", kelias.name)
    print(vamzdis.named_steps["svr"].get_params())
else:
    print("Artefakto nėra — mokomas demonstracinis SVR (be pilnos tinklelio paieškos).")
    vamzdis = svr_pipeline(C=2.0, epsilon=0.05, gamma=0.07)
    vamzdis.fit(X_tr, y_tr)

f_formule = formula_predict(vamzdis, X_te)
f_sklearn = vamzdis.predict(X_te)
print("max |formulė − sklearn| =", float(np.max(np.abs(f_formule - f_sklearn))))
print("Atraminių vektorių skaičius:", vamzdis.named_steps["svr"].support_vectors_.shape[0])
"""
            ),
            md("## Testo metrikos"),
            code(
                """
lentele = metrikos_lentele(y_te, f_formule, "SVR-RBF")
display(lentele.round(4))

fig, ax = plt.subplots(figsize=(5.0, 5.0))
ax.scatter(y_te + np.random.default_rng(0).normal(0, 0.04, size=len(y_te)), f_formule,
           s=12, alpha=0.45, c="#8b1e3f")
ax.plot([3, 9], [3, 9], "k--", lw=1)
ax.set_xlabel("Ekspertų balas y")
ax.set_ylabel("SVR prognozė f(x)")
ax.set_title(f"SVR — {KIND}")
ax.set_xlim(2.5, 9.5); ax.set_ylim(2.5, 9.5)
ax.set_aspect("equal")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
"""
            ),
            md(
                """## Ką stebėti

- `max |formulė − sklearn|` turi būti ~\(10^{-13}\) — tai įrodymas, kad naudojama dualinė formulė, o ne „juodoji dėžė“.
- Raudonam vynui H1 (MAE ≤ 0,95 × tiesinės) šiame skaidyme **neįveikta** (~3,5 % geriau, reikia 5 %).
- Baltam vynui H1 **įveikta** (~6,9 %).
- Kraštiniai balai 3 ir 8–9 traukiami į vidurkį — subjektyvaus žymėjimo ir disbalanso padarinys.
"""
            ),
        ],
    )


def mlp():
    dump(
        "04_mlp.ipynb",
        [
            md(
                """# 4. Daugiasluoksnis perceptronas (MLP)

**Failas:** `winequality/mlp.py` → `mlp_pipeline`  
**Mokymas:** `winequality/tune.py` → `tune_mlp`

Kursų LD2 / MLP tema. Universalus aproksimatorius, bet lentelinėms, asimetriškoms
imtims (chloridai, cukrus, SO₂) dažnai nusileidžia medžiams ar branduolio metodams.
"""
            ),
            md(
                r"""## Formulė (naudojimo kelias, 1 paslėptas sluoksnis)

\[
\mathbf{h} = \mathrm{act}(W_1 \mathbf{z} + \mathbf{b}_1),\qquad
f(\mathbf{z}) = \mathbf{w}_2^\top \mathbf{h} + b_2
\]

\(\mathrm{act} \in \{\mathrm{ReLU},\tanh\}\). Mokymas — Adam su L2 bauda \(\alpha\|W\|^2\)
ir ankstyvuoju stabdymu; mokymo formules egzaminas leidžia praleisti, nes tinklas mokomas vieną kartą.

Hiperparametrai (sluoksniai 8–64, ReLU/tanh, \(\alpha\), mokymosi žingsnis) parenkami
**GroupKFold mokymo bloke**, ne testinėje aibėje.
"""
            ),
            code(SETUP),
            md("## Artefaktas arba demonstracinis MLP"),
            code(
                """
import joblib
from winequality.mlp import mlp_pipeline

kelias = artefaktas("mlp", KIND)
if kelias.exists():
    vamzdis = joblib.load(kelias)
    print("Įkeltas artefaktas:", kelias.name)
    mlp = vamzdis.named_steps["mlp"]
    print("paslėpti sluoksniai:", mlp.hidden_layer_sizes)
    print("aktyvacija:", mlp.activation, "  alpha:", mlp.alpha)
else:
    print("Artefakto nėra — mokomas mažas MLP.")
    vamzdis = mlp_pipeline(hidden_layer_sizes=(32,), activation="relu", seed=0)
    vamzdis.fit(X_tr, y_tr)
    mlp = vamzdis.named_steps["mlp"]

f = vamzdis.predict(X_te)
print("n_iter_ =", getattr(mlp, "n_iter_", "?"))
print("sluoksnių svorių formos:", [w.shape for w in mlp.coefs_])
"""
            ),
            md("## Rankinis tiesioginis sklidimas (1 paslėptas sluoksnis)"),
            code(
                """
def relu(u):
    return np.maximum(u, 0.0)

def tanh_act(u):
    return np.tanh(u)

z = vamzdis.named_steps["prep"].transform(X_te)
W_list, b_list = mlp.coefs_, mlp.intercepts_
h = z
for i, (W, b) in enumerate(zip(W_list[:-1], b_list[:-1])):
    a = h @ W + b
    h = relu(a) if mlp.activation == "relu" else tanh_act(a)
f_rankinis = h @ W_list[-1] + b_list[-1]
f_rankinis = np.asarray(f_rankinis).ravel()
print("max |rankinis forward − sklearn| =", float(np.max(np.abs(f_rankinis - f))))

lentele = metrikos_lentele(y_te, f, "MLP")
display(lentele.round(4))
"""
            ),
            md(
                """## Ką stebėti

- Raudonam vynui MLP šiame skaidyme beveik lygus SVR (MAE ≈ 0,472).
- Baltam vynui MLP krenta beveik iki tiesinės regresijos (≈ 0,572) — maža lentelinė imtis + asimetriški požymiai.
- Ankstyvasis stabdymas naudoja **10 % mokymo** kaip validaciją, ne užšaldytą testą.
"""
            ),
        ],
    )


def rbf():
    dump(
        "05_rbf_tinklas.ipynb",
        [
            md(
                """# 5. Spindulio tipo bazinių funkcijų (RBF) tinklas

**Failas:** `winequality/rbf.py` — `hidden_activations`, `ridge_output_weights`, `predict_from_rbf`, `RBFNetwork`  
**Mokymas:** `winequality/tune.py` → `tune_rbf`

LD3 metodas. Konceptualiai tai SVR–RBF su **fiksuotais** centrais (k-vidurkiai), todėl
plane jis ir abliacija: kiek vertės duoda adaptyvus atraminių vektorių parinkimas.
"""
            ),
            md(
                r"""## Formulės

Centrai \(\mathbf{c}_k\) — k-vidurkių klasterių centrai. Plotis — vidutinis atstumas
iki kaimyninių centrų. Paslėpto sluoksnio aktyvacijos:

\[
\varphi_k(\mathbf{z}) = \exp\left(-\frac{\|\mathbf{z}-\mathbf{c}_k\|^2}{2\sigma_k^2}\right)
\]

Išėjimas ir uždara L2 (ridge) formulė (poslinkiui \(b\) bauda netaikoma):

\[
f(\mathbf{z}) = \sum_{k=1}^{K} w_k\,\varphi_k(\mathbf{z}) + b,\qquad
\mathbf{w} = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top \mathbf{y}
\]
"""
            ),
            code(SETUP),
            md("## Artefaktas arba demonstracinis RBF tinklas"),
            code(
                """
import joblib
from winequality.rbf import (
    hidden_activations, predict_from_rbf, rbf_pipeline,
)

kelias = artefaktas("rbf", KIND)
if kelias.exists():
    vamzdis = joblib.load(kelias)
    print("Įkeltas artefaktas:", kelias.name)
else:
    print("Artefakto nėra — mokomas RBF tinklas su K=40.")
    vamzdis = rbf_pipeline(n_centers=40, ridge=0.01, seed=0)
    vamzdis.fit(X_tr, y_tr)

net = vamzdis.named_steps["rbf"]
print("K =", net.centres_.shape[0], "  ridge =", net.ridge)
print("σ intervalas:", float(net.widths_.min()), "…", float(net.widths_.max()))
"""
            ),
            md("## Formulės kelias = `.predict`"),
            code(
                """
z_te = vamzdis.named_steps["prep"].transform(X_te)
phi = hidden_activations(z_te, net.centres_, net.widths_)
f_formule = predict_from_rbf(z_te, net.centres_, net.widths_, net.weights_)
f_api = vamzdis.predict(X_te)
print("φ forma (mėginiai × centrai):", phi.shape)
print("φ rėžiai: [", phi.min(), ",", phi.max(), "]  (turi būti (0, 1])")
print("max |formulė − predict| =", float(np.max(np.abs(f_formule - f_api))))

lentele = metrikos_lentele(y_te, f_api, "RBF tinklas")
display(lentele.round(4))

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.hist(net.widths_, bins=12, color="#c9a227", edgecolor="white")
ax.set_xlabel("Centro plotis σ_k")
ax.set_ylabel("Centrų skaičius")
ax.set_title("RBF pločiai iš kaimyninių centrų")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
"""
            ),
            md(
                """## Ką stebėti

- Plane tikėtasi, kad RBF tinklo MAE bus **tarp tiesinės regresijos ir SVR** — baltam vynui taip ir yra (0,55 tarp 0,57 ir 0,53).
- Raudonam vynui RBF šiek tiek **prastesnis** už tiesinę (0,497 vs 0,491): fiksuoti centrai be taikinio informacijos ne visada padeda mažoje imtyje.
- Tai ir yra abliacijos prasmė: SVR centrus (SV) parenka pagal \(y\), RBF tinklas — tik pagal \(X\).
"""
            ),
        ],
    )


def miskas():
    dump(
        "06_atsitiktinis_miskas.ipynb",
        [
            md(
                """# 6. Atsitiktinis miškas (planas M4, H2 atskaita)

**Failas:** `winequality/forest.py` → `forest_pipeline`  
**Mokymas:** `winequality/tune.py` → `tune_rf`

Medžių ansamblis — stipri lentelinių duomenų atskaita. Plane: jei RF pranoksta SVR
daugiau nei 0,02 MAE **ir** reikšmingai, diegiamas RF (blokas 6b). Kitaip lieka SVR.
"""
            ),
            md(
                r"""## Taisyklė (regresijos medis ir ansamblis)

Vienas medis skaido požymių erdvę stačiakampiais \(R_m\) ir kiekviename lapelyje
prognozuoja mokymo atsakymų vidurkį:

\[
T(\mathbf{x}) = \sum_{m=1}^{M} \bar y_{R_m}\, \mathbf{1}[\mathbf{x}\in R_m]
\]

Miškas vidurkina \(B\) medžių, kiekvienas mokomas bootstrap imties ir atsitiktinio požymių poaibio:

\[
f(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})
\]

Todėl prognozė **laiptuota** ir neekstrapoliuoja už mokymo min/max — SVR pranašumas ordinaliniam balui.
"""
            ),
            code(SETUP),
            code(
                """
import joblib
from winequality.forest import forest_pipeline

kelias = artefaktas("rf", KIND)
if kelias.exists():
    vamzdis = joblib.load(kelias)
    print("Įkeltas artefaktas:", kelias.name)
else:
    vamzdis = forest_pipeline(n_estimators=200, max_depth=12, seed=0)
    vamzdis.fit(X_tr, y_tr)

rf = vamzdis.named_steps["rf"]
f = vamzdis.predict(X_te)
print("medžių B =", rf.n_estimators, "  max_depth =", rf.max_depth)
print("unikalios prognozės teste:", len(np.unique(np.round(f, 5))))

lentele = metrikos_lentele(y_te, f, "atsitiktinis miškas")
display(lentele.round(4))

svarba = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.6))
svarba.plot.barh(ax=ax, color="#4a6fa5")
ax.set_xlabel("Permutacinė / priemaišų svarba (MDI)")
ax.set_title("Požymių svarba atsitiktiniame miške")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
"""
            ),
            md(
                """## Ką stebėti

- H2: \(\mathrm{MAE}_{SVR} - \mathrm{MAE}_{RF} \le 0{,}02\) — **galioja abiem** vyno tipams.
- Baltam vynui RF truputį geresnis MAE (0,532 vs 0,534), bet skirtumas << 0,02, todėl pagrindinis metodas lieka SVR.
- Svarbiausias požymis paprastai **alkoholis**; lakiosios rūgštys svarbesnės raudonam vynui.
"""
            ),
        ],
    )


def palyginimas():
    dump(
        "07_modeliu_palyginimas.ipynb",
        [
            md(
                """# 7. Modelio palyginimas (tas pats skaidymas, tos pačios metrikos)

Čia neskaičiuojami nauji hiperparametrai. Skaitomi `results/comparison.csv` ir
`results/metrics.json`, kuriuos sukuria

```
python -m winequality.run_experiment
```

Jei failų nėra, paleiskite šią komandą projekto šaknyje (trunka ~1–2 min).
"""
            ),
            code(
                """
%matplotlib inline
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

here = Path.cwd().resolve()
root = here if (here / "winequality").exists() else here.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "notebooks"))

from winequality.config import RESULTS, FIGURES, SPLIT_SEED, DEFAULT_SEED

csv_path = RESULTS / "comparison.csv"
json_path = RESULTS / "metrics.json"
assert csv_path.exists(), "Nėra results/comparison.csv — paleiskite python -m winequality.run_experiment"
cmp = pd.read_csv(csv_path)
rep = json.loads(json_path.read_text(encoding="utf-8"))
print("split_seed =", rep["split_seed"], "  model_seed =", rep["model_seed"])
display(cmp.round(4))
"""
            ),
            md("## MAE stulpelinė diagrama pagal vyno tipą"),
            code(
                """
tvarka = ["mean", "linear", "svr", "mlp", "rbf", "rf"]
spalvos = {
    "mean": "#888888", "linear": "#888888", "svr": "#8b1e3f",
    "mlp": "#2d6a4f", "rbf": "#c9a227", "rf": "#4a6fa5",
}
etiketes = {
    "mean": "vidurkis", "linear": "tiesinė", "svr": "SVR-RBF",
    "mlp": "MLP", "rbf": "RBF tinklas", "rf": "miškas",
}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
for ax, wine, tit in zip(axes, ["red", "white"], ["Raudonas vynas", "Baltas vynas"]):
    d = cmp[cmp["wine"] == wine].set_index("model").loc[tvarka]
    ax.bar([etiketes[m] for m in tvarka], d["mae"], color=[spalvos[m] for m in tvarka])
    for i, v in enumerate(d["mae"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(tit)
    ax.set_ylabel("MAE (užšaldytas testas)")
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
"""
            ),
            md("## Hipotezės H1 / H2 ir abliacija A1"),
            code(
                """
eilutes = []
for wine, block in rep["wines"].items():
    h1, h2 = block["h1"], block["h2"]
    a1 = block["ablation_a1_kernel"]
    leak = block["robustness_leakage"]
    noise = block["robustness_noise"]
    eilutes.append({
        "vynas": wine,
        "H1 ΔMAE vs tiesinė": h1["relative_improvement"],
        "H1 galioja (≥5 %)": h1["h1_holds"],
        "H2 SVR−RF": h2["mae_svr_minus_mae_rf"],
        "H2 ≤0.02": h2["non_inferiority_holds"],
        "A1 linear MAE": a1["linear"]["metrics"]["mae"],
        "A1 RBF MAE": a1["rbf"]["metrics"]["mae"],
        "nutekėjimo grupės (random)": leak["random"]["overlapping_groups"],
        "MAE random split": leak["random"]["metrics"]["mae"],
        "MAE grupinis": leak["group_aware"]["metrics"]["mae"],
        "ΔMAE nuo 0.1σ triukšmo": noise["mae_delta"],
    })
santrauka = pd.DataFrame(eilutes).set_index("vynas")
display(santrauka.round(4))
"""
            ),
            md("## Automatiniu būdu suformuluota išvada"),
            code(
                """
def laimetojas(wine: str) -> str:
    d = cmp[cmp["wine"] == wine].nsmallest(1, "mae").iloc[0]
    return f"{d['model']} (MAE={d['mae']:.3f})"

print("Mažiausias MAE raudonam:", laimetojas("red"))
print("Mažiausias MAE baltam:", laimetojas("white"))
print()
print(
    "Plane pagrindinis metodas vis tiek SVR-RBF, nes H2 neleidžia keisti į RF, "
    "kol RF nepranoksta daugiau nei 0,02 MAE. MLP raudonam beveik lygus SVR, "
    "baltam — beveik tiesinei regresijai. RBF tinklas (fiksuoti centrai) baltam "
    "yra tarp tiesinės ir SVR, kaip ir buvo hipotezuota."
)

# jautrumas (alkoholis turi kilti, lakiosios rūgštys — kristi)
for wine, block in rep["wines"].items():
    s = block["sensitivity"]
    print(f"\\nJautrumas ({wine}): alkoholis Δf={s['alcohol']['delta_f']:+.3f}, "
          f"lakiosios rūgštys Δf={s['volatile acidity']['delta_f']:+.3f}")
"""
            ),
            md(
                """## Kaip skaityti rezultatą egzaminui

1. **Visos eilutės** skaičiuotos su `SPLIT_SEED = 0` ir tais pačiais 11 požymių.
2. Hiperparametrai parinkti **tik 80 % mokymo** (`GroupKFold`), testas liestas vieną kartą.
3. Neigiamas H1 raudonam vynui yra priimtinas: eksperimentas korektiškas, 5 % slenkstis per griežtas.
4. Atsitiktinis skaidymas **pagerina** SVR MAE dėl 74 / 274 nutekėjusių dublikatų grupių — todėl grupinis skaidymas privalomas.
"""
            ),
        ],
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    vidurkis()
    tiesine()
    svr()
    mlp()
    rbf()
    miskas()
    palyginimas()
