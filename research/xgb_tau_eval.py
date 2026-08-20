import os, sys, random, json, gc, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)

import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================
# STARTUP MEMORY CLEANUP
# =====================================================
gc.collect()
# =====================================================

# =====================================================
# 1. CONFIG & SEED
# =====================================================
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

from project_config import PROJECT_ROOT, resolve_data_dir, resolve_parquet_path, resolve_results_dir

BASE_DIR = str(PROJECT_ROOT)
NEW_FOLDER = BASE_DIR
DATA_DIR = str(resolve_data_dir())
PATH_PARQUET = os.environ.get("TAU_PARQUET", str(resolve_parquet_path()))

N_TRIALS = 15
TEST_FRAC = 0.15
N_SPLITS = 3


try:
    _dummy = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=1)
    _dummy.fit(np.array([[0.0]], dtype=np.float32), np.array([0]))
    DEVICE = "cuda"
except Exception:
    DEVICE = "cpu"

print(f"[DEVICE] Detected device for XGBoost: {DEVICE}")

# =======================================================
# 2. FEATURE GROUPS (v0 ~ v4) FOR ABLATION STUDY
# =======================================================
cols_v0 = [
    'avg_campaign_duration', 'avg_time_since_complaint', 'avg_time_since_first_purchase',
    'avg_time_since_last_click', 'avg_time_since_last_open', 'avg_time_since_unsubscribe',
    'camp_campaign_typebulk', 'camp_campaign_typetransactional', 'camp_campaign_typetrigger',
    'camp_channelemail', 'camp_channelmobile_push', 'camp_channelmultichannel', 'camp_channelsms',
    'camp_topicevent', 'camp_topichappy.birthday', 'camp_topicleave.review',
    'camp_topicoffer.after.purchase', 'camp_topicother', 'camp_topicsale.out',
    'channel_email', 'channel_mobile_push', 'channel_web_push',
    'email_provider_gmail.com', 'email_provider_mail.ru', 'email_provider_other',
    'is_holiday',
    'message_type_bulk', 'message_type_transactional', 'message_type_trigger',
    'platform.', 'platform.desktop', 'platform.phablet', 'platform.smartphone', 'platform.tablet',
    'prev_is_clicked', 'prev_is_complained', 'prev_is_opened', 'prev_is_unsubscribed',
    'total_campaigns', 'total_messages', 'total_purchases'
]

cols_v1 = ['days_since_last_purchase', 'feat_rtb_hazard', 'feat_postbuy_refrac']
cols_v2 = ['cal_is_weekend', 'cal_week_of_month', 'feat_dow_shift', 'feat_eoq_bump', 'feat_hour_shift', 'feat_payday_bump']
cols_v3 = ['ctx_tc_open_rate_30d', 'feat_fatigue', 'feat_last_any_hours', 'feat_last_email_hours', 'feat_last_mobile_push_hours', 'u_cadence_std_30d', 'u_click_rate_30d', 'u_open_cnt_30d', 'u_open_rate_30d']
cols_v4 = ['feat_like_last_success', 'feat_path_align', 'feat_topic_novelty', 'topic_N7', 'topic_t_since_hours']



# =====================================================
# 3. LOAD DATASET
# =====================================================
if not os.path.exists(PATH_PARQUET):
    print(f"[ERROR] Parquet file does not exist at {PATH_PARQUET}!")
    sys.exit(1)

print(f"[LOAD] Loading dataset from: {PATH_PARQUET}")
df = pd.read_parquet(PATH_PARQUET)
TARGET = "is_purchased"

# Memory Downcasting (Fast Vectorized Cast)
print("[MEMORY OPTIMIZATION] Fast downcasting feature dtypes...")
float_cols = df.select_dtypes(include=['float64']).columns
int_cols = [c for c in df.select_dtypes(include=['int64', 'int32']).columns if c != TARGET]
cast_dict = {**{c: np.float32 for c in float_cols}, **{c: np.int16 for c in int_cols}}
if cast_dict:
    df = df.astype(cast_dict)
gc.collect()

print(f"[DATASET CHECK] Total Rows: {len(df):,d} | Positives: {(df[TARGET]==1).sum():,d} | Target Imbalance Ratio: {df[TARGET].mean()*100:.6f}%")

def get_existing_cols(df, col_list):
    return [c for c in col_list if c in df.columns]

feat_cols_base  = get_existing_cols(df, cols_v0)
feat_cols_v0_v1 = get_existing_cols(df, cols_v0 + cols_v1)
feat_cols_v0_v2 = get_existing_cols(df, cols_v0 + cols_v2)
feat_cols_v0_v3 = get_existing_cols(df, cols_v0 + cols_v3)
feat_cols_v0_v4 = get_existing_cols(df, cols_v0 + cols_v4)
feat_cols_all   = get_existing_cols(df, cols_v0 + cols_v1 + cols_v2 + cols_v3 + cols_v4)

# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================
def find_best_f1_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


# =====================================================
# 5. INDIVIDUAL OPTUNA & ABLATION EVALUATION
# =====================================================
ablation_results = {}

experiments = [
    # ("base",  feat_cols_base),  # all 실험만 실행
    # ("v0_v1", feat_cols_v0_v1),
    # ("v0_v2", feat_cols_v0_v2),
    # ("v0_v3", feat_cols_v0_v3),
    # ("v0_v4", feat_cols_v0_v4),
    ("all",   feat_cols_all),
]

all_indices = np.arange(len(df))
y_all = df[TARGET].values.astype(np.int32)

idx_tr_full, idx_te = train_test_split(all_indices, test_size=TEST_FRAC, stratify=y_all, random_state=SEED)
idx_tr, idx_va = train_test_split(idx_tr_full, test_size=0.17647, stratify=y_all[idx_tr_full], random_state=SEED)

raw_ratio = (y_all[idx_tr] == 0).sum() / float(max(1, (y_all[idx_tr] == 1).sum()))
print(f"[COST-SENSITIVE] Full Train Set Negative/Positive Ratio = {raw_ratio:.2f}")

def get_memory_safe_subsample(X_in, Y_in, max_negs=250_000, seed=SEED):
    pos_idx = np.where(Y_in == 1)[0]
    neg_idx = np.where(Y_in == 0)[0]
    if len(neg_idx) <= max_negs:
        return X_in, Y_in
    rng_sub = np.random.RandomState(seed)
    sub_neg_idx = rng_sub.choice(neg_idx, size=max_negs, replace=False)
    comb_idx = np.concatenate([pos_idx, sub_neg_idx])
    rng_sub.shuffle(comb_idx)
    return X_in[comb_idx], Y_in[comb_idx]

OUT_JSON_DIR = os.environ.get("TAU_OUTPUT_DIR") or str(resolve_results_dir(__file__))
RESULT_NAME = os.environ.get("TAU_RESULT_NAME", "result_xgb.json")

# Keep artifacts from each tau run instead of overwriting xgb_all_fold*.pkl.
RUN_NAME = os.path.splitext(RESULT_NAME)[0]
OUT_MODEL_DIR = os.path.join(OUT_JSON_DIR, "models", RUN_NAME)
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

for exp_idx, (exp_name, feat_cols) in enumerate(experiments):
    print(f"\n=======================================================")
    print(f"  [STEP 1: OPTUNA] Individual Tuning for '{exp_name}' ({len(feat_cols)} Features)")
    print(f"=======================================================")

    X_feat_matrix = df[feat_cols].to_numpy(dtype=np.float32)
    TUNING_SEED = SEED + exp_idx * 100

    Xva_t = X_feat_matrix[idx_va]
    Yva_t = y_all[idx_va]

    def objective_exp(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'booster': 'gbtree',
            'tree_method': 'hist',
            'device': DEVICE,
            'random_state': TUNING_SEED,
            'n_jobs': -1,
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(1.0, 0.05 * raw_ratio), min(2000.0, 2.0 * raw_ratio), log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.85, 1.0),
            'gamma': trial.suggest_float('gamma', 0.1, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=TUNING_SEED)
        cv_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(idx_tr, y_all[idx_tr])):
            fold_tr_idx = idx_tr[tr_idx]
            fold_va_idx = idx_tr[val_idx]
            X_tr_fold, Y_tr_fold = X_feat_matrix[fold_tr_idx], y_all[fold_tr_idx]
            X_va_fold, Y_va_fold = X_feat_matrix[fold_va_idx], y_all[fold_va_idx]
            X_tr_sub, Y_tr_sub = get_memory_safe_subsample(X_tr_fold, Y_tr_fold, max_negs=200_000, seed=TUNING_SEED + fold_idx)

            model = xgb.XGBClassifier(**param, early_stopping_rounds=30)
            model.fit(X_tr_sub, Y_tr_sub, eval_set=[(X_va_fold, Y_va_fold)], verbose=False)
            val_probs = model.predict_proba(X_va_fold)[:, 1]
            pr_auc = average_precision_score(Y_va_fold, val_probs)
            cv_scores.append(pr_auc)
            del model, X_tr_fold, Y_tr_fold, X_va_fold, Y_va_fold, X_tr_sub, Y_tr_sub
            gc.collect()
        return float(np.mean(cv_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNING_SEED))
    study.optimize(objective_exp, n_trials=N_TRIALS)

    best_optuna_params = study.best_trial.params.copy()
    exp_params = best_optuna_params.copy()
    exp_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'device': DEVICE,
        'random_state': TUNING_SEED,
        'n_jobs': -1,
    })
    print(f"[{exp_name} OPTUNA DONE] Best params: {json.dumps(best_optuna_params, indent=2)}")

    print(f"\n=======================================================")
    print(f"  [STEP 2: EVALUATION] 3-Fold CV Evaluation for '{exp_name}'")
    print(f"=======================================================")

    val_probs_ensemble = np.zeros(len(Xva_t), dtype=np.float32)
    test_probs_ensemble = np.zeros(len(idx_te), dtype=np.float32)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(idx_tr, y_all[idx_tr])):
        sub_tr_idx = idx_tr[tr_idx]
        sub_va_idx = idx_tr[val_idx]
        X_tr_full_f, Y_tr_full_f = X_feat_matrix[sub_tr_idx], y_all[sub_tr_idx]
        X_va_f, Y_va_f = X_feat_matrix[sub_va_idx], y_all[sub_va_idx]
        X_tr_f, Y_tr_f = get_memory_safe_subsample(X_tr_full_f, Y_tr_full_f, max_negs=200_000, seed=SEED + fold_idx)

        fold_model = xgb.XGBClassifier(**exp_params, early_stopping_rounds=40)
        fold_model.fit(X_tr_f, Y_tr_f, eval_set=[(X_va_f, Y_va_f)], verbose=False)

        model_pkl_path = os.path.join(OUT_MODEL_DIR, f"xgb_{exp_name}_fold{fold_idx}.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(fold_model, f)

        val_probs_ensemble += fold_model.predict_proba(Xva_t)[:, 1] / float(N_SPLITS)

        test_batch_size = 1_000_000
        for b_start in range(0, len(idx_te), test_batch_size):
            b_end = min(b_start + test_batch_size, len(idx_te))
            b_idx = idx_te[b_start:b_end]
            b_Xte = X_feat_matrix[b_idx]
            test_probs_ensemble[b_start:b_end] += fold_model.predict_proba(b_Xte)[:, 1] / float(N_SPLITS)

        del fold_model, X_tr_full_f, Y_tr_full_f, X_tr_f, Y_tr_f, X_va_f, Y_va_f
        gc.collect()

    del X_feat_matrix
    gc.collect()

    Yte = y_all[idx_te]

    eval_thr, _ = find_best_f1_threshold(Yva_t, val_probs_ensemble)
    test_preds = (test_probs_ensemble >= eval_thr).astype(int)

    acc  = accuracy_score(Yte, test_preds)
    prec = precision_score(Yte, test_preds, zero_division=0)
    rec  = recall_score(Yte, test_preds, zero_division=0)
    f1   = f1_score(Yte, test_preds, zero_division=0)
    f2   = fbeta_score(Yte, test_preds, beta=2, zero_division=0)
    auc  = roc_auc_score(Yte, test_probs_ensemble)
    pr_auc = average_precision_score(Yte, test_probs_ensemble)
    cm   = confusion_matrix(Yte, test_preds)
    tn, fp, fn, tp = cm.ravel()

    top1000_idx = np.argsort(test_probs_ensemble)[::-1][:1000]
    top1000_hits = int(Yte[top1000_idx].sum())
    top2000_idx = np.argsort(test_probs_ensemble)[::-1][:2000]
    top2000_hits = int(Yte[top2000_idx].sum())
    top3000_idx = np.argsort(test_probs_ensemble)[::-1][:3000]
    top3000_hits = int(Yte[top3000_idx].sum())

    report = {
        "num_features": len(feat_cols),
        "optimal_threshold": float(eval_thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "top1000_hits": top1000_hits,
        "top2000_hits": top2000_hits,
        "top3000_hits": top3000_hits,
        "best_params": best_optuna_params,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

    ablation_results[exp_name] = report
    print(f"[{exp_name:5s} SUMMARY] F1 = {f1:.6f} | PR-AUC = {pr_auc:.6f} | ROC-AUC = {auc:.6f} | Rec = {rec:.6f} | Thr = {eval_thr:.4f}")

    json_save_path = os.path.join(OUT_JSON_DIR, RESULT_NAME)
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)

print("\n=======================================================")
print("       XGBoost FEATURE ABLATION SUMMARY REPORT         ")
print("=======================================================")
print(json.dumps(ablation_results, indent=2))
print(f"\n[SAVED] JSON Report successfully saved to: {json_save_path}")
