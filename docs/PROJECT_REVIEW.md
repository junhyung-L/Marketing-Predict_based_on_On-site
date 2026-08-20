# Project Review: Marketing Fatigue Analysis

## Assessment

| Dimension | Score | Evidence-based assessment |
|---|---:|---|
| Problem definition | 9/10 | A concrete high-imbalance CRM conversion problem, target, and operational decision are specified. |
| Data complexity | 7/10 | Code is structured for multichannel, longitudinal CRM inputs; the reported data scale is not independently verifiable in this repository. |
| Feature engineering | 9/10 | Four theory-derived dynamic feature families are implemented in the preprocessing and model code. |
| Modeling | 8/10 | Nine model families and an ablation structure are present. |
| Experimental rigor | 7/10 | Stratified holdout, three-fold validation, seed control, a sensitivity sweep, and a run manifest are present; source data and the original environment are absent. |
| Interpretability | 8/10 | SHAP analysis and class-specific rank-shift interpretation are documented. |
| Business/research impact | 8/10 | Top-K targeting analysis offers a credible operational framing, but deployment impact is not verified. |
| Reproducibility | 5/10 | Shared paths, an import-derived requirements file, and a result-regression check now exist; raw data access and pinned versions remain absent. |
| **Overall** | **7.9/10** | Strong research depth and feature engineering; provenance and runnable-data packaging remain the main weaknesses. |

Scores are an assessment of the repository, not measured facts.

## Strengths

- The project frames class imbalance as an operational targeting problem, not
  merely a classification exercise.
- Feature groups map explicit marketing concepts to computable signals, making
  the modeling choices interpretable.
- The evaluation is broader than a single metric: F1, PR-AUC, recall, Top-K
  hits, sensitivity runs, and SHAP are all represented.
- The public README distinguishes study-level reported values from retained
  executable outputs and does not present them as causal or realized impact.

## Gaps and priorities

1. **Pin the reproduction environment.** The repository now centralizes paths
   and provides an import-derived dependency list, but the original package
   versions and Python version are not recorded.
2. **Publish a data schema and run manifest.** Link every published metric to
   a source artifact, dataset version, and experiment settings.
3. **Provide a safe demonstration path.** Include a synthetic or small approved
   sample that exercises preprocessing and one model end to end.
4. **Clarify result versions.** The retained experiment notes and JSON output
   report different XGBoost values. Label them as distinct runs or reconcile
   them before presenting a single headline metric.
5. **Separate notebooks from production scripts.** Keep notebooks for analysis,
   but expose one reproducible command per experiment.

## Evidence notes

- Main methodology and headline results: retained code and experiment records.
- XGBoost split and validation setup: `research/xgb.py`.
- Sensitivity reports: `research/predict/tau_sweep_results/`.
- Reproduction utilities: `research/project_config.py`,
  `research/RUN_MANIFEST.md`, and `research/verify_baseline.py`.
- No production deployment, individual ownership, or realized business outcome
  is established by the repository.
