# Reproduction Manifest

## Baseline contract

The refactor must preserve the historical experiment definition:

- Seed: `1`
- Holdout test fraction: `0.15`, stratified by `is_purchased`
- Validation: three-fold `StratifiedKFold` with shuffling and seeded splits
- Primary dataset: `final_data_100k_64.parquet`
- Primary result report: `predict/result_xgb.json`

`verify_baseline.py` compares the retained XGBoost report with its saved metric contract.
It is a report-regression test, not a model rerun.

## Run with the original data

Set paths explicitly; do not copy data into the repository:

```powershell
$env:MFA_DATASET_PATH = 'D:\path\to\final_data_100k_64.parquet'
$env:MFA_RESULTS_DIR = 'D:\path\to\new-results'
python research\xgb.py
python research\verify_baseline.py --result "$env:MFA_RESULTS_DIR\result_xgb.json"
```

An exact match can require a matched Python/library environment and the same hardware settings.
The original dependency versions are not stored in this repository, so version confirmation remains user input required.
