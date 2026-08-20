# Academic Project Summary

## Research question

Can dynamic, theory-derived marketing features improve purchase-conversion prediction and targeting decisions in an extremely imbalanced e-commerce CRM setting?

## Methodological contribution

The research translates consumer and marketing concepts into four dynamic feature families: purchase/repurchase cycles, temporal response, marketing fatigue, and channel effectiveness.
These features are evaluated as inductive biases for models trained on sparse tabular CRM data.

## Experimental design

The submitted research study reports two years of multichannel logs, approximately 172 million events, and an original-distribution holdout rate of 0.1234%.
The repository includes nine ML and deep-learning benchmark scripts.
The XGBoost code uses a stratified 15% test split, seed control, and three-fold stratified validation.
Additional code evaluates fatigue half-life multipliers between 0.50x and 1.50x and Top-K targeting outcomes.

## Findings

The submitted research study reports XGBoost F1 of 0.3806 and PR-AUC of 0.3620, with recall increasing from 30.74% to 52.30% relative to the reported baseline.
SHAP analyses are interpreted as a staged process: fatigue signals screen non-responders, temporal signals adjust timing, and repurchase-cycle/high-engagement variables support final conversion prediction.

## Limitations

The study is observational and does not establish causal treatment effects.
It uses one platform's data, retains fixed initial fatigue parameters, and does not report hardware-level efficiency measurements.
The repository also lacks raw data, a dependency lockfile, and a portable run configuration.

## Follow-up research

- Test targeting policy prospectively through randomized or quasi-experimental designs.
- Learn individualized fatigue half-lives from response histories.
- Evaluate cross-platform and cross-industry generalization.
- Measure compute time, energy use, and cost across model families.
