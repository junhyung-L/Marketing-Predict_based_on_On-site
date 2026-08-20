# Purchase Conversion Prediction with Dynamic CRM Features

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

This project treats purchase conversion prediction as a targeting problem in an e-commerce CRM setting where the positive rate is only 0.1234%. Rather than increasing model complexity alone, it turns marketing theory into customer-level, time-varying features: purchase-cycle readiness, response timing, accumulated message fatigue, and channel/content fit.

The retained manuscript reports that domain-enhanced XGBoost achieved F1 0.3806 and PR-AUC 0.3620 on its original-distribution holdout. Recall increased from 30.74% to 52.30% relative to the baseline feature set. Those are classification results for the documented dataset and split—not verified revenue, campaign savings, or causal uplift.

## Framing the prediction problem

Static customer summaries miss an important part of marketing behaviour: the same customer can be just after a purchase, at a preferred activity hour, or overexposed to a repeated message theme. I organized this context into four feature families.

| Feature family | Operational signal | Purpose |
|---|---|---|
| Purchase and repurchase cycle | customer-specific intervals, repurchase readiness, post-purchase refractory period | represent position in an individual purchase cycle |
| Temporal response | preferred hour/day alignment and proximity to payday or quarter-end | distinguish sending time from customer context |
| Marketing fatigue | recency-weighted exposure with channel-specific half-lives | model repeated exposure and recovery together |
| Channel effectiveness | topic freshness and similarity to prior converting paths | identify a more suitable contact point |

Fatigue is not simply message count. It is a decayed sum in which recent messages carry more weight and the effect subsides over time, with different assumptions by channel. It is therefore an associative prediction signal, not proof that a message caused fatigue.

## Data and preparation boundary

The manuscript uses Kaggle REES46’s *E-commerce multichannel direct messaging 2021–2023*: approximately 172 million messaging/response logs across roughly two years, 1.85 million customers, and 1,907 campaigns. It combines four relational tables: interaction logs, campaign metadata, customer attributes, and temporal context.

The documented preparation keeps marketing campaigns tied to customer profiles or promotions and excludes operational transaction notifications. Customer journeys are rebuilt from history available before each prediction point. The manuscript specifies a stratified 70/15/15 train/validation/test split that preserves the original class distribution, with imputation and scaling statistics fitted on training data only.

The executable artefacts retained in this repository have a narrower boundary. `RUN_MANIFEST.md` describes `final_data_100k_64.parquet`, seed 1, a stratified 15% holdout, and three-fold `StratifiedKFold`; the retained XGBoost JSON records a 1,499,900-row holdout confusion matrix with 64 features. This is not the same scale or split as the manuscript experiment, so the two are kept separate.

## Model comparison and findings

The manuscript compares nine models: Random Forest, XGBoost, MLP, CNN, RNN, LSTM, CNN-LSTM, RNN-LSTM, and TabNet. Because accuracy is misleading at this prevalence, it reports PR-AUC, F1, F2, and recall alongside precision. The manuscript describes PR-AUC-driven Optuna optimisation, five-fold validation, early stopping, and pruning.

| XGBoost holdout metric | Baseline features | Domain features |
|---|---:|---:|
| Precision | 0.4199 | 0.2991 |
| Recall | 0.3074 | 0.5230 |
| F1 | 0.3550 | 0.3806 |
| F2 | 0.3248 | 0.4549 |
| PR-AUC | 0.3177 | 0.3620 |

The precision–recall trade-off matters here: more converters are captured, while precision falls. Choosing a preferred operating point would require campaign contact cost and the cost of unnecessary messages. The 13.1% figure in the manuscript is a TabNet **F2-score** change (0.3659 to 0.4138), not an observed marketing-cost reduction.

The manuscript’s top-K analysis further reports that selecting the top 0.78% by predicted score captured 95.83% of holdout conversions while reducing the selected message volume by 99.22%. This is a simulated ranking result on that holdout. The repository does not contain a live send record, unit costs, or a control group, so it should not be presented as realised savings or incremental conversion.

## What remains useful—and what remains open

SHAP and ablation analyses provide an interpretable view of associations: fatigue and temporal-response features act as early filtering signals, while purchase-cycle and recent high-engagement signals contribute later in the model’s ranking. They do not establish treatment effects.

The repository retains preprocessing, model, and half-life sensitivity scripts plus compact JSON/CSV result files, but not the source Parquet files or trained model binaries. A stronger next iteration would version the data and row counts, fix package versions and split timestamps in one manifest, evaluate time-ordered generalisation, and add a randomized or uplift design before making campaign-impact claims.

## Evidence

- [Run manifest](research/RUN_MANIFEST.md)
- [Retained XGBoost result](research/predict/result_xgb.json)
- [Preprocessing](research/preprocessing_fast.py), [XGBoost experiment](research/xgb.py), and [half-life sweep](research/run_xgb_tau_sweep.py)
