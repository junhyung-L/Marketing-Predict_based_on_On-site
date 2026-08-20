"""Compute Top-K targeting metrics for the five completed XGBoost tau runs.

Uses each tau run's saved three-fold ``all`` XGBoost models. If a model artifact
is absent, it trains only that fold from the already saved best parameters; it
never runs Optuna or rebuilds the preprocessing data.
"""
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split


SEED = 1
TEST_FRAC = 0.15
N_SPLITS = 3
TOP_KS = (1_000, 2_000, 3_000, 5_000, 10_000)
TAUS = ("0.50x", "0.75x", "1.00x", "1.25x", "1.50x")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TAU_DATA_DIR", SCRIPT_DIR / "tau_sweep_data"))
RESULT_DIR = Path(os.environ.get("TAU_RESULT_DIR", SCRIPT_DIR / "tau_sweep_results"))
OUT_JSON = RESULT_DIR / "tau_targeting_metrics.json"
OUT_CSV = RESULT_DIR / "tau_targeting_metrics.csv"

COLS_V0 = [
    "avg_campaign_duration", "avg_time_since_complaint", "avg_time_since_first_purchase",
    "avg_time_since_last_click", "avg_time_since_last_open", "avg_time_since_unsubscribe",
    "camp_campaign_typebulk", "camp_campaign_typetransactional", "camp_campaign_typetrigger",
    "camp_channelemail", "camp_channelmobile_push", "camp_channelmultichannel", "camp_channelsms",
    "camp_topicevent", "camp_topichappy.birthday", "camp_topicleave.review",
    "camp_topicoffer.after.purchase", "camp_topicother", "camp_topicsale.out",
    "channel_email", "channel_mobile_push", "channel_web_push",
    "email_provider_gmail.com", "email_provider_mail.ru", "email_provider_other", "is_holiday",
    "message_type_bulk", "message_type_transactional", "message_type_trigger",
    "platform.", "platform.desktop", "platform.phablet", "platform.smartphone", "platform.tablet",
    "prev_is_clicked", "prev_is_complained", "prev_is_opened", "prev_is_unsubscribed",
    "total_campaigns", "total_messages", "total_purchases",
]
COLS_V1 = ["days_since_last_purchase", "feat_rtb_hazard", "feat_postbuy_refrac"]
COLS_V2 = ["cal_is_weekend", "cal_week_of_month", "feat_dow_shift", "feat_eoq_bump", "feat_hour_shift", "feat_payday_bump"]
COLS_V3 = [
    "ctx_tc_open_rate_30d", "feat_fatigue", "feat_last_any_hours", "feat_last_email_hours",
    "feat_last_mobile_push_hours", "u_cadence_std_30d", "u_click_rate_30d",
    "u_open_cnt_30d", "u_open_rate_30d",
]
COLS_V4 = ["feat_like_last_success", "feat_path_align", "feat_topic_novelty", "topic_N7", "topic_t_since_hours"]
ALL_FEATURES = COLS_V0 + COLS_V1 + COLS_V2 + COLS_V3 + COLS_V4


def slug(label: str) -> str:
    return label.replace(".", "_")


def subsample_train(X, y, seed):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(neg) <= 200_000:
        return X, y
    rng = np.random.RandomState(seed)
    selected = np.concatenate([pos, rng.choice(neg, size=200_000, replace=False)])
    rng.shuffle(selected)
    return X[selected], y[selected]


def device_name():
    try:
        probe = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=1)
        probe.fit(np.array([[0.0]], dtype=np.float32), np.array([0]))
        return "cuda"
    except Exception:
        return "cpu"


def load_params(result_path):
    with result_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if "all" not in report:
        raise KeyError(f"Missing 'all' result in {result_path}")
    return report["all"]["best_params"]


def main():
    device = device_name()
    rows = []
    all_results = {}
    for label in TAUS:
        data_path = DATA_DIR / f"final_data_{slug(label)}.parquet"
        result_path = RESULT_DIR / f"result_xgb_{slug(label)}.json"
        model_dir = RESULT_DIR / "models" / result_path.stem
        if not data_path.exists():
            raise FileNotFoundError(f"Missing tau dataset: {data_path}")
        if not result_path.exists():
            raise FileNotFoundError(f"Missing tau XGBoost result: {result_path}")

        print(f"\n===== {label}: targeting metrics =====")
        df = pd.read_parquet(data_path)
        features = [column for column in ALL_FEATURES if column in df.columns]
        if len(features) != 64:
            raise ValueError(f"{label}: expected 64 all features, found {len(features)}")
        X = df[features].to_numpy(dtype=np.float32)
        y = df["is_purchased"].to_numpy(dtype=np.int32)
        indices = np.arange(len(df))
        train_full, test_idx = train_test_split(
            indices, test_size=TEST_FRAC, stratify=y, random_state=SEED
        )
        train_idx, _ = train_test_split(
            train_full, test_size=0.17647, stratify=y[train_full], random_state=SEED
        )
        params = load_params(result_path)
        params.update({
            "objective": "binary:logistic", "eval_metric": "aucpr", "booster": "gbtree",
            "tree_method": "hist", "device": device, "random_state": SEED, "n_jobs": -1,
        })
        test_probs = np.zeros(len(test_idx), dtype=np.float32)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for fold, (fit_rel, val_rel) in enumerate(splitter.split(train_idx, y[train_idx])):
            model_path = model_dir / f"xgb_all_fold{fold}.pkl"
            if model_path.exists():
                with model_path.open("rb") as handle:
                    model = pickle.load(handle)
                print(f"  [LOAD] fold {fold}: {model_path.name}")
            else:
                print(f"  [TRAIN] fold {fold}: saved model missing; training with saved best params")
                fit_idx = train_idx[fit_rel]
                val_idx = train_idx[val_rel]
                X_fit, y_fit = subsample_train(X[fit_idx], y[fit_idx], SEED + fold)
                model = xgb.XGBClassifier(**params, early_stopping_rounds=40)
                model.fit(X_fit, y_fit, eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            for start in range(0, len(test_idx), 1_000_000):
                stop = min(start + 1_000_000, len(test_idx))
                test_probs[start:stop] += model.predict_proba(X[test_idx[start:stop]])[:, 1] / N_SPLITS
            del model
            gc.collect()

        y_test = y[test_idx]
        order = np.argsort(test_probs)[::-1]
        positives = int(y_test.sum())
        prevalence = positives / len(y_test)
        tau_result = {"test_population": int(len(y_test)), "test_positives": positives, "rows": []}
        for k in TOP_KS:
            hits = int(y_test[order[:k]].sum())
            precision_at_k = hits / k
            recall_at_k = hits / positives
            lift = precision_at_k / prevalence
            item = {
                "tau_multiplier": label, "top_k": k, "population_share": k / len(y_test),
                "tp_hits": hits, "precision_at_k": precision_at_k,
                "cumulative_recall": recall_at_k, "lift_vs_random": lift,
            }
            tau_result["rows"].append(item)
            rows.append(item)
            print(f"  Top {k:,}: TP={hits:,} | P@K={precision_at_k:.4%} | Recall={recall_at_k:.4%} | Lift={lift:.2f}x")
        all_results[label] = tau_result
        del df, X, y, test_probs
        gc.collect()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {OUT_JSON}")
    print(f"[SAVED] {OUT_CSV}")


if __name__ == "__main__":
    main()
