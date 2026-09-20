# Handoff brief: implement the wine-quality prediction plan

Purpose of this file: give a coding agent (or developer) working in a **different repository** all
context needed to implement the plan described in `Vyno_kokybes_vertinimas_igyvendinimo_planas.docx`
(Lithuanian, 12 pages, master's course *Intelektualiosios sistemos*). Everything below was decided and
verified in the planning phase; treat it as the specification. Where the plan leaves a choice open, it
says so explicitly (hypotheses H1–H4, ablations A1–A9).

Source repository of the plan: `marcin-bu-inski/IS_Kolis_Egz` on Cursor Origin
(`https://origin.cursor.com/git/marcin-bu-inski/IS_Kolis_Egz.git`). Files worth copying from it:
`doc/planas.md` (full plan, Markdown + LaTeX), `scripts/explore_data.py` (stats/figures/baselines),
`scripts/fetch_data.sh` (data download with SHA-256), `data/raw/*.csv`.

---

## 1. Task in one paragraph

Predict the expert sensory quality score (integer 3–9, median of ≥3 blind tasters) of Portuguese
*Vinho Verde* wines from 11 physicochemical measurements, using the UCI **Wine Quality** dataset
(1599 red, 4898 white). Formulated as **ordinal regression**: a continuous prediction `f(x)` is
rounded to an integer score and mapped to three classes (low ≤4, medium 5–6, high ≥7). Mandatory
baselines: mean prediction and linear regression. Main method: **SVR with Gaussian (RBF) kernel**,
trained separately for red and white. Alternatives to implement and compare: MLP, RBF network,
random forest (RF is the "strong reference" with a pre-registered non-inferiority rule).
Primary metric: **MAE**; secondary: Acc_T (|error| ≤ T, T = 0.5 and 1.0), quadratic weighted kappa,
macro-F1 + confusion matrices on the 3 classes.

## 2. Data facts (computed from the raw CSVs, use for sanity checks)

Source: https://archive.ics.uci.edu/dataset/186/wine+quality (CC BY 4.0). Cite Cortez et al. 2009,
DOI 10.1016/j.dss.2009.05.016. Files `winequality-red.csv`, `winequality-white.csv`, separator `;`.

SHA-256:
```
4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e  winequality-red.csv
76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836  winequality-white.csv
```

Columns (11 features + `quality`): fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. No missing values.

| | red | white |
|---|---|---|
| n | 1599 | 4898 |
| exact duplicate rows (all 12 columns) | 240 (15.0 %) | 937 (19.1 %) |
| conflicting duplicates (same features, different score) | 0 | 0 |
| quality mean ± sd | 5.64 ± 0.81 | 5.88 ± 0.89 |
| score distribution (3/4/5/6/7/8/9) | 10/53/681/638/199/18/– | 20/163/1457/2198/880/175/5 |
| strongest correlations with quality | alcohol +0.48, volatile acidity −0.39, sulphates +0.25 | alcohol +0.44, density −0.31, chlorides −0.21 |
| skewed features (skew > 1) | residual sugar 4.5, chlorides 5.7, free SO₂ 1.3, total SO₂ 1.5, sulphates 2.4 | volatile acidity 1.6, citric acid 1.3, residual sugar 1.1, chlorides 5.0, free SO₂ 1.4 |

Baseline MAE already measured (10-fold × 3 repeats; group-aware CV in parentheses):

| model | red | white |
|---|---|---|
| mean | 0.684 (0.684) | 0.671 (0.671) |
| median | 0.658 (0.658) | 0.630 (0.630) |
| linear regression (standardised) | 0.505 (0.505) | 0.586 (0.587) |
| SVR, sklearn defaults (orientation only) | 0.455 (0.477) | 0.516 (0.538) |

The SVR gap between plain and group-aware CV (~0.02) is the duplicate-leakage effect; **always use
group-aware splitting** (see §4).

## 3. Input / output contract of the final system

Input: one row per sample, 11 floats in original units + `type` ∈ {0 red, 1 white}. Reject rows with
missing values or values outside physical ranges (e.g. pH < 2.5 or > 4.5) with an explicit error.

Output (JSON):
```json
{
  "score_cont": 6.31,            // f(x), continuous
  "score_int": 6,                // clip(round(f(x)), 3, 9)
  "class": "medium",             // low (<=4) / medium (5-6) / high (>=7)
  "confidence": 0.38,            // 1 - 2*|f(x) - score_int|  (0 = on rounding boundary)
  "flags": ["extrapolation"]     // any of: extrapolation, disagreement, extreme
}
```
Flags: `extrapolation` = a feature outside the training min/max; `disagreement` = when an expert
score `y` is supplied and |f(x) − y| > 1 (triggers re-tasting); `extreme` = score_int ≤ 4 or ≥ 8.

## 4. Experimental protocol (pre-registered — do not change after seeing test results)

1. **Duplicate groups**: group id = hash of the 11 feature values; identical rows never cross a split.
2. **Split**: stratified by score (merge 3 with 4 and 9 with 8 for stratification), group-aware
   (`StratifiedGroupKFold`). Hold out 20 % as the **frozen test set**, touched once at the end.
3. **Nested CV on the remaining 80 %**: outer 10-fold × 3 repeats (30 estimates per model, seeds 0/1/2);
   inner 5-fold randomized search: 60 configurations for SVR and MLP, 30 for RF.
4. **Preprocessing inside the pipeline only** (fit on training fold): `log1p` on residual sugar,
   chlorides, free SO₂, total SO₂, sulphates; then z-score standardisation. Target stays in score units
   (so ε is interpretable). Extreme but genuine values (white: density 1.039, sugar 65.8) are kept and flagged.
5. **Models**
   - Mean / median dummy; linear regression (OLS, optionally ridge λ).
   - **SVR-RBF** (main): C ∈ {2⁻², …, 2⁶}, ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5} (plus a data-driven initial
     estimate), γ ∈ {2⁻⁷, …, 2¹} with `scale` = 1/(d·Var) as starting point.
   - MLP: 1–2 hidden layers, 8–64 units, ReLU/tanh, L2, early stopping, Adam.
   - RBF network: K centres by k-means, widths from neighbour distances, output weights by ridge
     least squares (closed form). Conceptually SVR with fixed centres → also serves as an ablation.
   - Random forest / gradient boosting: n_estimators, max_depth, learning rate.
   - Course requirement: also implement the linear neuron, RBF network and a small MLP **from scratch
     in numpy**, and solve the SVR dual with `cvxopt` on ≤500 samples; compare against scikit-learn
     (max abs difference of predictions < 1e-3) as a correctness test.
6. **Statistics**: paired Wilcoxon signed-rank over the 30 outer folds, Holm correction; 95 % bootstrap
   CIs on the frozen test set. One pre-declared primary comparison (H1).
7. **Metrics**: MAE (on f and on rounded score), RMSE, Acc_0.5, Acc_1.0, quadratic weighted kappa,
   MAE per true score; for classes: macro-F1, per-class precision/recall, confusion matrices 7×7
   (scores) and 3×3 (classes). Use `class_weight='balanced'` in classification variants; SMOTE only
   as an ablation.

### Hypotheses (decide with the protocol above)
- **H1 (primary)**: SVR-RBF MAE is ≥ 5 % lower than linear regression for both wine types
  (orientation thresholds: ≤ 0.480 red, ≤ 0.558 white), Wilcoxon p < 0.05 after Holm.
- **H2 (non-inferiority)**: MAE_SVR − MAE_RF ≤ 0.02 for both types. If RF beats SVR by more than 0.02
  **and** significantly, switch the deployed model to RF and document it (decision block 6b in the plan).
- **H3**: the SVR gain over linear regression is larger for white (n = 4898) than red (n = 1599); on the
  learning curve it is no longer significant at n ≤ 300.
- **H4**: gains come from fine ordering: Acc_0.5 improves ≥ 3 percentage points while Acc_1.0 improves
  < 2 pp; a linear-kernel SVR (A1) does not achieve H1.
- If H1 fails: deploy linear regression (the "stop" rule).

### Ablations
A1 linear vs RBF kernel · A2 ε = 0 vs tuned ε · A3/A4 without log1p / without standardisation ·
A5 separate models vs one model with `type` feature · A6 group-aware vs random splitting (leakage size) ·
A7 dropping feature groups (acids; SO₂; alcohol+density; sulphates+chlorides) · A8 learning curve
10/25/50/100 % · A9 regression+rounding vs 3-class classification vs ordinal (CORN-style).

### Error analysis
Residuals by true score (expect regression to the mean: 3–4 over-predicted, 8–9 under-predicted);
list of samples with |error| ≥ 2 for an oenologist; error vs alcohol, volatile acidity, confidence;
sensitivity analysis (vary one feature −1…+1 sd, others at mean) and permutation importance, compared
with domain expectations (alcohol ↑ → score ↑, volatile acidity ↑ → score ↓).

### Compute budget
Laptop, 8 cores, no GPU. SVR fit on ~3.9k rows ≈ 0.5–1 s → ~9 000 fits ≈ 1.5–2.5 CPU-h; MLP ≈ 5–8 CPU-h;
RF ≈ 2 CPU-h; total ≤ 15 CPU-h (2–3 h wall time in parallel), RAM < 4 GB.

## 5. Formulas to implement (inference path)

- Preprocess: `x'_j = ln(1 + x_j)` for j in the skewed set; `z_j = (x'_j − μ_j) / σ_j` with μ, σ from the training fold.
- SVR prediction: `f(z) = Σ_{i∈SV} (α_i − α_i*) K(z_i, z) + b`, `K(a, b) = exp(−γ‖a − b‖²)`.
- SVR training (once): primal ½‖w‖² + C Σ(ξ_i + ξ_i*) with ε-insensitive constraints; dual with
  0 ≤ α_i, α_i* ≤ C and Σ(α_i − α_i*) = 0 (LIBSVM/SMO in sklearn, `cvxopt` for the from-scratch check).
- Post-process: `q̂ = clip(round(f), 3, 9)`; class by thresholds; `conf = 1 − 2|f − q̂|`.
- Metrics: MAE, Acc_T = mean(1[|y − f| ≤ T]), macro-F1 = mean over classes of 2PR/(P+R),
  κ_w = 1 − Σ w_kl O_kl / Σ w_kl E_kl with w_kl = (k − l)² / (K − 1)².

## 6. Suggested repository layout (from the plan)

```
data/fetch.py            download + SHA-256 check
features/transform.py    log1p + standardisation (sklearn Transformer)
data/split.py            duplicate groups, StratifiedGroupKFold, frozen test set
models/baseline.py       mean, median, linear (OLS/ridge), numpy linear neuron
models/svr.py            sklearn SVR wrapper + cvxopt dual solver (≤500 rows)
models/mlp.py            sklearn MLPRegressor + small numpy backprop MLP
models/rbf.py            k-means centres + ridge output weights (numpy)
models/forest.py         RandomForest / GradientBoosting
training/tune.py         nested CV, randomized search, seeds, logging to CSV/JSON
evaluation/metrics.py    MAE, Acc_T, kappa_w, macro-F1, confusion matrices
evaluation/stats.py      Wilcoxon + Holm, bootstrap CIs
evaluation/ablation.py   A1–A9 runners
evaluation/errors.py     residual analysis, sensitivity, permutation importance
serve/predict.py         CLI / REST: JSON in → JSON out (contract in §3)
serve/monitor.py         feature drift (PSI), monthly MAE, retrain trigger (MAE ↑ > 10 %)
tests/                   metrics vs sklearn.metrics, split leakage = 0 rows, baseline reproduction
```

Acceptance tests: reproduce linear-regression MAE ≈ 0.505 / 0.587 (group CV); assert zero shared
duplicate groups between train and test; from-scratch vs library predictions agree < 1e-3.

## 7. AI-usage protocol from the plan (applies to the implementation phase too)

All numbers reported must come from experiment logs, never from an AI response; the frozen test set
is evaluated once; from-scratch implementations are cross-checked against library ones; every
literature claim needs a DOI resolved via Crossref (LLM-generated citations are often fabricated).

## 8. Key references already verified (all 2022+ except the dataset paper)

Dataset: Cortez et al. 2009, DOI 10.1016/j.dss.2009.05.016; UCI DOI 10.24432/C56S3T.
SVR/kernels: Du et al. 2024 (10.3390/math12243935); Karal 2023 (10.1016/j.engappai.2023.105841);
Wu & Wang 2022 data-driven ε (10.1007/s13042-022-01672-x). Tabular benchmarks: Grinsztajn et al. 2022
(NeurIPS D&B); Shwartz-Ziv & Armon 2022 (10.1016/j.inffus.2021.11.011); McElfresh et al. 2023
(arXiv 2305.02997). Wine-quality ML 2022+: Yavas et al. 2025 (10.3390/bdcc9030055); Jain et al. 2023
(10.1038/s41598-023-44111-9); Tigga et al. 2024 (10.52756/ijerr.2024.v45spl.003); Ye 2023
(10.54097/hset.v49i.8505). Methodology: Kapoor & Narayanan 2023 leakage (10.1016/j.patter.2023.100804);
Bischl et al. 2023 HPO (10.1002/widm.1484); Bates et al. 2024 CV (10.1080/01621459.2023.2197686);
Shi, Cao & Raschka 2023 CORN ordinal (10.1007/s10044-023-01181-9); Rezvani & Wang 2023 imbalance
(10.1016/j.asoc.2023.110415). Rating noise: Bodington 2022 (10.1017/jwe.2022.53, 10.1017/jwe.2022.55).
