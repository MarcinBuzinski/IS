# AI usage log (exam)

Session: implementation of the colloquium plan as the exam solution.
Model: Cursor Grok 4.6. Date: 2026-09-20.

## Main prompts (summary)

1. Handoff review: does the colloquium PDF + HANDOFF.md contain everything needed?
2. Clarification that the exam (not only the plan) is the deliverable; request to implement baselines, the plan's main method with formulas, and one other plan method.
3. Confirmation to install local tools; do not write the prose report yet.
4. Add the remaining two plan methods (MLP and RBF network) to the same comparison.

## Accepted suggestions

- Exam-sized protocol: frozen 80/20 group-aware split; inner GroupKFold HPO; not the plan's full 10×3×60 nested CV.
- Main method SVR-RBF; also RF, MLP, and numpy RBF network so all four plan methods are compared.
- Formula-to-code link via an explicit dual prediction function, cross-checked against sklearn; RBF net uses the closed-form φ_k / ridge-weight path.
- Ablation A1 (kernel) + sensitivity; robustness via noise and group vs random split.

## Rejected suggestions

- CORN, SMOTE, Wilcoxon/Holm, and a serving/monitor stack for the exam — still out of scope.
- Using the frozen test set inside RandomizedSearchCV — rejected; inner CV is training-block only.
- Reporting MAE numbers from the colloquium exploratory script as exam results — rejected; exam numbers must come from `results/metrics.json` after `python -m winequality.run_experiment`.
- Skipping MLP/RBF after the first exam-sized subset — rejected once the user asked for the remaining plan methods.

## AI errors / unchecked assumptions and how they were checked

1. **Assumption (unchecked until tests run):** sklearn's `SVR.dual_coef_` is exactly (α − α*) so `predict_from_dual` can replace `SVR.predict`.  
   **Check:** `tests/test_core.py::test_svr_formula_matches_sklearn` and a max-abs assertion in `run_experiment.py` (threshold 1e-6).

2. **Assumption:** `sklearn.utils.fixes.loguniform` exists on current scikit-learn.  
   **Check:** not trusted; code uses `scipy.stats.loguniform` instead.

3. **Risk of fabricated citations / copied exploratory MAE:** the colloquium plan quoted SVR default MAE 0.455/0.516. Those must not be presented as this run's result.  
   **Check:** this implementation writes a new `results/metrics.json`; README states that file is the only numeric source for a later report.

4. **Handoff vs exam compute budget:** blindly running 9 000 SVR fits would overspend a laptop session.  
   **Check:** `SVR_N_ITER=12`, `RF_N_ITER=8`, `INNER_CV_SPLITS=3` in `winequality/config.py` (still train-only HPO, smaller grid).
