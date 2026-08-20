import os, sys, random, json, gc, warnings, copy, pickle, importlib, shutil, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================
# STARTUP MEMORY CLEANUP
# =====================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect") and sys.platform != "win32":
        torch.cuda.ipc_collect()
# =====================================================

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from project_config import PROJECT_ROOT, resolve_data_dir, resolve_parquet_path, resolve_results_dir

BASE_DIR = str(PROJECT_ROOT)
NEW_FOLDER = BASE_DIR
DATA_DIR = str(resolve_data_dir())
PATH_PARQUET = str(resolve_parquet_path())

N_TRIALS = 15
TEST_FRAC = 0.15
N_SPLITS = 3


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

def find_best_f1_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return float(best_thr), float(best_f1)

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

class PyTorchRNNLSTM(nn.Module):
    def __init__(self, in_features, rnn_units, lstm_units, dropout):
        super(PyTorchRNNLSTM, self).__init__()
        self.rnn  = nn.RNN(in_features, rnn_units, batch_first=True)
        self.lstm = nn.LSTM(rnn_units, lstm_units, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, lstm_units // 2),
            nn.ReLU(),
            nn.Linear(lstm_units // 2, 1)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        if self.training:
            self.rnn.flatten_parameters()
            self.lstm.flatten_parameters()
        rnn_out, _ = self.rnn(x)
        lstm_out, _ = self.lstm(rnn_out)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)

ablation_results = {}

experiments = [
    #("base",  feat_cols_base),
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

OUT_JSON_DIR = str(resolve_results_dir(__file__))
os.makedirs(OUT_JSON_DIR, exist_ok=True)
OUT_MODEL_DIR = os.path.join(OUT_JSON_DIR, "models")
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

for exp_idx, (exp_name, feat_cols) in enumerate(experiments):
    print(f"\n=======================================================")
    print(f"  [STEP 1: OPTUNA] Individual Tuning for '{exp_name}' ({len(feat_cols)} Features)")
    print(f"=======================================================")

    raw_feat_matrix = df[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    X_feat_matrix = scaler.fit_transform(raw_feat_matrix)

    TUNING_SEED = SEED + exp_idx * 100

    Xva_t = X_feat_matrix[idx_va]
    Yva_t = y_all[idx_va]

    def predict_proba_batched(model, X_numpy, batch_size=16384):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_numpy), batch_size):
                batch_x = torch.tensor(X_numpy[i:i+batch_size], dtype=torch.float32).to(DEVICE)
                out = torch.sigmoid(model(batch_x)).cpu().numpy()
                probs.append(out)
        return np.concatenate(probs, axis=0)

    def get_batch_iterator(X_np, Y_np, batch_size=4096, shuffle=True):
        num_samples = len(X_np)
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_idx = indices[start_idx:end_idx]
            bx = torch.tensor(X_np[batch_idx], dtype=torch.float32, device=DEVICE)
            by = torch.tensor(Y_np[batch_idx], dtype=torch.float32, device=DEVICE)
            yield bx, by

    def objective_exp(trial):
        pos_weight = trial.suggest_float('pos_weight', 1.0, min(100.0, raw_ratio), log=True)
        rnn_units = trial.suggest_int('rnn_units', 32, 128, step=32)
        lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=TUNING_SEED)
        cv_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(idx_tr, y_all[idx_tr])):
            fold_tr_idx = idx_tr[tr_idx]
            fold_va_idx = idx_tr[val_idx]
            X_tr_fold, Y_tr_fold = X_feat_matrix[fold_tr_idx], y_all[fold_tr_idx]
            X_va_fold, Y_va_fold = X_feat_matrix[fold_va_idx], y_all[fold_va_idx]
            X_tr_sub, Y_tr_sub = get_memory_safe_subsample(X_tr_fold, Y_tr_fold, max_negs=200_000, seed=TUNING_SEED + fold_idx)

            model = PyTorchRNNLSTM(X_feat_matrix.shape[1], rnn_units, lstm_units, dropout).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_pr_auc = -1.0
            best_state = None
            patience, patience_cnt = 6, 0

            for epoch in range(25):
                model.train()
                for bx, by in get_batch_iterator(X_tr_sub, Y_tr_sub, batch_size=4096, shuffle=True):
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                val_probs = predict_proba_batched(model, X_va_fold)
                pr_auc_epoch = average_precision_score(Y_va_fold, val_probs)

                if pr_auc_epoch > best_pr_auc:
                    best_pr_auc = pr_auc_epoch
                    best_state = copy.deepcopy(model.state_dict())
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            val_probs = predict_proba_batched(model, X_va_fold)
            pr_auc = average_precision_score(Y_va_fold, val_probs)
            cv_scores.append(pr_auc)

            del model, criterion, optimizer, X_tr_fold, Y_tr_fold, X_va_fold, Y_va_fold, X_tr_sub, Y_tr_sub
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        return float(np.mean(cv_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNING_SEED))
    study.optimize(objective_exp, n_trials=N_TRIALS, catch=(Exception,))

    best_p = study.best_trial.params
    print(f"[{exp_name} OPTUNA DONE] Best params: {json.dumps(best_p, indent=2)}")

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

        fold_model = PyTorchRNNLSTM(X_feat_matrix.shape[1], best_p['rnn_units'], best_p['lstm_units'], best_p['dropout']).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([best_p['pos_weight']]).to(DEVICE))
        optimizer = optim.Adam(fold_model.parameters(), lr=best_p['lr'])

        best_pr_auc = -1.0
        best_state = None
        patience, patience_cnt = 8, 0

        for epoch in range(35):
            fold_model.train()
            for bx, by in get_batch_iterator(X_tr_f, Y_tr_f, batch_size=4096, shuffle=True):
                optimizer.zero_grad()
                out = fold_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            val_probs_epoch = predict_proba_batched(fold_model, X_va_f)
            pr_auc_epoch = average_precision_score(Y_va_f, val_probs_epoch)

            if pr_auc_epoch > best_pr_auc:
                best_pr_auc = pr_auc_epoch
                best_state = copy.deepcopy(fold_model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        if best_state is not None:
            fold_model.load_state_dict(best_state)

        model_pkl_path = os.path.join(OUT_MODEL_DIR, f"RNNLSTM_{exp_name}_fold{fold_idx}.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(fold_model.state_dict(), f)

        val_probs_ensemble += predict_proba_batched(fold_model, Xva_t) / float(N_SPLITS)
        test_probs_ensemble += predict_proba_batched(fold_model, X_feat_matrix[idx_te]) / float(N_SPLITS)

        del fold_model, criterion, optimizer, X_tr_full_f, Y_tr_full_f, X_tr_f, Y_tr_f, X_va_f, Y_va_f
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    del X_feat_matrix, raw_feat_matrix
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

    # top1000_idx = np.argsort(test_probs_ensemble)[::-1][:1000]
    # top1000_hits = int(Yte[top1000_idx].sum())
    # top2000_idx = np.argsort(test_probs_ensemble)[::-1][:2000]
    # top2000_hits = int(Yte[top2000_idx].sum())
    # top3000_idx = np.argsort(test_probs_ensemble)[::-1][:3000]
    # top3000_hits = int(Yte[top3000_idx].sum())

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
        # "top1000_hits": top1000_hits,
        # "top2000_hits": top2000_hits,
        # "top3000_hits": top3000_hits,
        "best_params": best_p,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

    ablation_results[exp_name] = report
    print(f"[{exp_name:5s} SUMMARY] F1 = {f1:.6f} | PR-AUC = {pr_auc:.6f} | ROC-AUC = {auc:.6f} | Rec = {rec:.6f} | Thr = {eval_thr:.4f}")

    json_save_path = os.path.join(OUT_JSON_DIR, "result_RNNLSTM.json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)

print("\n=======================================================")
print("     RNN-LSTM FEATURE ABLATION SUMMARY REPORT          ")
print("=======================================================")
print(json.dumps(ablation_results, indent=2))
print(f"\n[SAVED] JSON Report successfully saved to: {json_save_path}")
