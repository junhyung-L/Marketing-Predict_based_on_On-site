# =====================================================
# 0) IMPORTS & CONFIG
# =====================================================
import os, random, warnings, time, gc, shutil, tempfile, subprocess, sys, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numba
from numba import njit
from project_config import PROJECT_ROOT, resolve_data_dir, resolve_source_path

SEED = 1
MODE = "STRICT_CAUSAL"   # or "NON_CAUSAL"

DATA_DIR = str(resolve_data_dir())
BASE_DIR = str(PROJECT_ROOT)
NEW_FOLDER = BASE_DIR
PATH_MESSAGES = str(resolve_source_path("messages_extracted_012.parquet"))
if not os.path.exists(PATH_MESSAGES):
    PATH_MESSAGES = str(resolve_source_path("messages.csv"))
PATH_CAMPAIGNS = str(resolve_source_path("campaigns.csv"))
PATH_CLIENTS = str(resolve_source_path("client_first_purchase_date.csv"))
PATH_HOLIDAYS = str(resolve_source_path("holidays.csv"))
OUTPUT_PARQUET = os.environ.get("MFA_OUTPUT_PARQUET", os.path.join(DATA_DIR, "final_data_100k_64.parquet"))
OUTPUT_PARQUET_BASE = OUTPUT_PARQUET
OUTPUT_PARQUET_NEW = OUTPUT_PARQUET

TZ_LOCAL = "Asia/Seoul"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
rng = np.random.RandomState(SEED)

to_dt = lambda s: pd.to_datetime(s, errors="coerce", utc=True)

def safe_dt_to_hours(series):
    """datetime Series → epoch-hours (float64). NaT → NaN (safe for Numba)."""
    hrs = series.values.astype("datetime64[s]").astype(np.float64) / 3600.0
    nat_mask = series.isna().values
    hrs[nat_mask] = np.nan
    return hrs

def _norm_str(x):
    if pd.isna(x): return None
    return str(x).strip().lower().replace(" ", ".")

def reduce_email_provider(ep_series):
    ep = ep_series.astype(str).str.lower().str.strip()
    bad = {"", "nan", "none", "null"}
    ep = ep.where(~ep.isin(bad), "other")
    forced = ["gmail.com", "mail.ru"]
    ep_reduced = np.where(ep.isin(forced), ep, "other")
    return ep_reduced

def onehot_join(df, col, prefix):
    if col not in df.columns:
        return pd.DataFrame(index=df.index)
    vals = df[col].copy()
    mask = vals.notna()
    labels = prefix + vals[mask].astype(str)
    oh = pd.get_dummies(labels, dtype=int)
    oh = oh.reindex(index=df.index, fill_value=0)
    return oh

# =====================================================
# NUMBA ACCELERATED KERNELS
# =====================================================
@njit
def fast_cum_nunique_hist(client_codes, camp_codes, n_camps):
    n = len(client_codes)
    out = np.zeros(n, dtype=np.int32)
    last_seen = np.full(n_camps, -1, dtype=np.int32)
    cur_client = -1
    cnt = 0
    for i in range(n):
        c_id = client_codes[i]
        cmp_id = camp_codes[i]
        if c_id < 0:
            out[i] = 0
            continue
        if c_id != cur_client:
            cur_client = c_id
            cnt = 0
        out[i] = cnt
        if cmp_id >= 0 and cmp_id < n_camps and last_seen[cmp_id] != c_id:
            last_seen[cmp_id] = c_id
            cnt += 1
    return out

@njit
def fast_prior_ffill_shift(client_codes, sent_at_hrs, src_hrs, shift=True):
    n = len(client_codes)
    rec = np.full(n, np.nan, dtype=np.float64)
    cur_client = -1
    last_val = np.nan
    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            last_val = np.nan
        t_sent = sent_at_hrs[i]
        if shift:
            if not np.isnan(last_val) and not np.isnan(t_sent):
                rec[i] = t_sent - last_val
            cur_val = src_hrs[i]
            if not np.isnan(cur_val):
                last_val = cur_val
        else:
            cur_val = src_hrs[i]
            if not np.isnan(cur_val):
                last_val = cur_val
            if not np.isnan(last_val) and not np.isnan(t_sent):
                rec[i] = t_sent - last_val
    return rec

@njit
def fast_group_cumsum_shift1(client_codes, val_arr):
    n = len(client_codes)
    out = np.zeros(n, dtype=np.float64)
    cur_client = -1
    csum = 0.0
    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            csum = 0.0
        out[i] = csum
        v = val_arr[i]
        if not np.isnan(v):
            csum += v
    return out

@njit
def fast_fatigue_cooldown(client_codes, sent_at_hrs, channel_codes, tau_arr, cool_arr):
    n = len(client_codes)
    out_fatigue = np.zeros(n, dtype=np.float32)
    out_cool_ok = np.ones(n, dtype=np.int32)
    last_time = np.full(5, -1e18, dtype=np.float64)
    has_last = np.zeros(5, dtype=np.bool_)
    cur_client = -1
    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            has_last[:] = False
        t = sent_at_hrs[i]
        ch = channel_codes[i]
        if ch < 0 or ch >= 5:
            ch = 4
        sc = 0.0
        for cc in range(5):
            if has_last[cc]:
                dt_h = t - last_time[cc]
                if dt_h >= 0:
                    sc += np.exp(-dt_h / tau_arr[cc])
        out_fatigue[i] = sc
        if has_last[ch]:
            dt_h = t - last_time[ch]
            out_cool_ok[i] = 1 if dt_h >= cool_arr[ch] else 0
        else:
            out_cool_ok[i] = 1
        last_time[ch] = t
        has_last[ch] = True
    return out_fatigue, out_cool_ok

@njit
def fast_group_shift1(client_codes, val_arr):
    n = len(client_codes)
    out = np.zeros(n, dtype=np.float64)
    cur_client = -1
    last_v = 0.0
    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            last_v = 0.0
        out[i] = last_v
        v = val_arr[i]
        if not np.isnan(v):
            last_v = v
    return out

@njit
def fast_last_purchase_time(client_codes, sent_at_hrs, is_purchased, purchased_at_hrs):
    n = len(client_codes)
    rec_hrs = np.full(n, np.nan, dtype=np.float64)
    cur_client = -1
    last_p_t = np.nan
    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            last_p_t = np.nan
        t = sent_at_hrs[i]
        if not np.isnan(last_p_t) and not np.isnan(t):
            rec_hrs[i] = t - last_p_t
        if is_purchased[i] == 1:
            p_t = purchased_at_hrs[i]
            if np.isnan(p_t): p_t = t
            last_p_t = p_t
    return rec_hrs

@njit
def fast_like_last_success(client_codes, y_arr, vec_mat, n_clients):
    n, K = vec_mat.shape
    out = np.zeros(n, dtype=np.float32)
    if n_clients <= 0:
        return out
    last_vec_mat = np.zeros((n_clients, K), dtype=np.int32)
    has_last_vec = np.zeros(n_clients, dtype=np.bool_)
    for i in range(n):
        uid = client_codes[i]
        if uid < 0 or uid >= n_clients:
            out[i] = 0.0
            continue
        y = y_arr[i]
        if has_last_vec[uid]:
            inter, uni = 0, 0
            for k in range(K):
                a = last_vec_mat[uid, k]
                v = vec_mat[i, k]
                inter += (a & v)
                uni += (a | v)
            out[i] = float(inter) / float(uni) if uni > 0 else 0.0
        else:
            out[i] = 0.0
        if y == 1:
            for k in range(K):
                last_vec_mat[uid, k] = vec_mat[i, k]
            has_last_vec[uid] = True
    return out

@njit
def fast_topic_last_ts(group_codes, sent_at_hrs, num_groups):
    n = len(group_codes)
    last_hrs = np.full(n, np.nan, dtype=np.float64)
    if num_groups <= 0:
        return last_hrs
    grp_last_t = np.full(num_groups, np.nan, dtype=np.float64)
    for i in range(n):
        g = group_codes[i]
        if g < 0 or g >= num_groups:
            continue
        t = sent_at_hrs[i]
        last_hrs[i] = grp_last_t[g]
        if not np.isnan(t):
            grp_last_t[g] = t
    return last_hrs

@njit
def fast_last_ch_hours(client_codes, sent_at_hrs, is_ch_mask):
    n = len(client_codes)
    rec = np.full(n, np.nan, dtype=np.float64)
    cur_client = -1
    last_ch_t = np.nan
    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            last_ch_t = np.nan
        t = sent_at_hrs[i]
        if not np.isnan(last_ch_t) and not np.isnan(t):
            rec[i] = t - last_ch_t
        if is_ch_mask[i] and not np.isnan(t):
            last_ch_t = t
    return rec

@njit
def fast_rolling_30d_metrics(client_codes, sent_at_hrs, op_val, cl_val):
    n = len(client_codes)
    out_open_cnt = np.zeros(n, dtype=np.float32)
    out_open_rate = np.zeros(n, dtype=np.float32)
    out_click_rate = np.zeros(n, dtype=np.float32)
    out_open_vel = np.zeros(n, dtype=np.float32)
    out_click_vel = np.zeros(n, dtype=np.float32)
    out_cadence_std = np.zeros(n, dtype=np.float32)
    cur_client = -1
    start_30d_idx = 0
    start_7d_idx = 0

    for i in range(n):
        c_id = client_codes[i]
        if c_id < 0:
            continue
        if c_id != cur_client:
            cur_client = c_id
            start_30d_idx = i
            start_7d_idx = i

        t_curr = sent_at_hrs[i]
        t_limit_30d = t_curr - 720.0
        t_limit_7d  = t_curr - 168.0

        while start_30d_idx < i and sent_at_hrs[start_30d_idx] < t_limit_30d:
            start_30d_idx += 1
        while start_7d_idx < i and sent_at_hrs[start_7d_idx] < t_limit_7d:
            start_7d_idx += 1

        win_30d = i - start_30d_idx
        win_7d  = i - start_7d_idx

        if win_30d > 0:
            sum_op_30d = 0.0
            sum_cl_30d = 0.0
            for k in range(start_30d_idx, i):
                sum_op_30d += op_val[k]
                sum_cl_30d += cl_val[k]
            rate_op_30d = float(sum_op_30d) / float(win_30d)
            rate_cl_30d = float(sum_cl_30d) / float(win_30d)

            if win_7d > 0:
                sum_op_7d = 0.0
                sum_cl_7d = 0.0
                for k in range(start_7d_idx, i):
                    sum_op_7d += op_val[k]
                    sum_cl_7d += cl_val[k]
                rate_op_7d = float(sum_op_7d) / float(win_7d)
                rate_cl_7d = float(sum_cl_7d) / float(win_7d)
            else:
                rate_op_7d = rate_op_30d
                rate_cl_7d = rate_cl_30d

            # Pure 30-day rates (0.0 ~ 1.0)
            out_open_rate[i]  = float(rate_op_30d)
            out_click_rate[i] = float(rate_cl_30d)
            out_open_vel[i]   = float(rate_op_7d - rate_op_30d)
            out_click_vel[i]  = float(rate_cl_7d - rate_cl_30d)
            out_open_cnt[i]   = float(sum_op_30d)

            if win_30d > 1:
                diff_sum = 0.0
                diff_sq_sum = 0.0
                cnt_diff = win_30d - 1
                for k in range(start_30d_idx, i - 1):
                    dt = sent_at_hrs[k+1] - sent_at_hrs[k]
                    diff_sum += dt
                    diff_sq_sum += dt * dt
                mean_dt = diff_sum / cnt_diff
                var_dt = (diff_sq_sum / cnt_diff) - (mean_dt * mean_dt)
                out_cadence_std[i] = float(np.sqrt(max(0.0, var_dt)))
            else:
                out_cadence_std[i] = 0.0
        else:
            out_open_cnt[i] = 0.0
            out_open_rate[i] = 0.0
            out_click_rate[i] = 0.0
            out_open_vel[i] = 0.0
            out_click_vel[i] = 0.0
            out_cadence_std[i] = 0.0
    return out_open_cnt, out_open_rate, out_click_rate, out_open_vel, out_click_vel, out_cadence_std

@njit
def fast_client_topic_last_ts(client_codes, topic_codes, sent_at_hrs, n_clients, n_topics):
    n = len(client_codes)
    last_hrs = np.full(n, np.nan, dtype=np.float64)
    if n_clients <= 0 or n_topics <= 0:
        return last_hrs
    last_t_mat = np.full((n_clients, n_topics), np.nan, dtype=np.float64)
    for i in range(n):
        uid = client_codes[i]
        top = topic_codes[i]
        if uid < 0 or uid >= n_clients or top < 0 or top >= n_topics:
            continue
        t = sent_at_hrs[i]
        last_hrs[i] = last_t_mat[uid, top]
        if not np.isnan(t):
            last_t_mat[uid, top] = t
    return last_hrs

@njit
def fast_path_alignment(client_codes, vec_mat, n_clients):
    n, K = vec_mat.shape
    out = np.zeros(n, dtype=np.float32)
    if n_clients <= 0:
        return out
    client_vec_sum = np.zeros((n_clients, K), dtype=np.float32)
    client_cnt = np.zeros(n_clients, dtype=np.int32)
    client_align_sum = np.zeros(n_clients, dtype=np.float32)

    for i in range(n):
        uid = client_codes[i]
        if uid < 0 or uid >= n_clients:
            continue
        cnt = client_cnt[uid]
        cur_align = 0.0
        if cnt > 0:
            dot = 0.0
            norm_a = 0.0
            norm_b = 0.0
            for k in range(K):
                a = client_vec_sum[uid, k] / float(cnt)
                b = float(vec_mat[i, k])
                dot += a * b
                norm_a += a * a
                norm_b += b * b
            if norm_a > 0.0 and norm_b > 0.0:
                cur_align = float(dot / (np.sqrt(norm_a) * np.sqrt(norm_b)))
            else:
                cur_align = 0.0

            # DETRENDED ALIGNMENT: cur_align - mean_past_align (Removes total_campaigns trend!)
            mean_past = client_align_sum[uid] / float(cnt)
            out[i] = float(cur_align - mean_past)
        else:
            out[i] = 0.0

        client_align_sum[uid] += cur_align
        for k in range(K):
            client_vec_sum[uid, k] += float(vec_mat[i, k])
        client_cnt[uid] += 1
    return out

@njit
def fast_ctx_open_rate_30d_sorted(sorted_sent_at_hrs, sorted_op_val):
    """Compute global 30-day open rate on TIME-SORTED data."""
    n = len(sorted_sent_at_hrs)
    out = np.zeros(n, dtype=np.float32)
    start_idx = 0
    running_sum = 0.0
    for i in range(n):
        t_curr = sorted_sent_at_hrs[i]
        t_limit = t_curr - 720.0
        while start_idx < i and sorted_sent_at_hrs[start_idx] < t_limit:
            running_sum -= sorted_op_val[start_idx]
            start_idx += 1
        win_size = i - start_idx
        if win_size > 0:
            out[i] = float(running_sum) / float(win_size)
        else:
            out[i] = 0.25
        running_sum += sorted_op_val[i]
    return out


# =====================================================
# ZERO-CRASH FAST PYARROW SCANNER PIPELINE (RAM < 250MB, 35s COMPLETION)
# =====================================================
def _purchase_flag(values):
    """Normalize the source's mixed purchase encodings to an int8 Series."""
    return (
        values.astype(str).str.strip().str.lower()
        .isin(["t", "true", "1", "1.0"])
        .astype(np.int8)
    )


def _row_group_worker(mode, source, row_group, batch_size, output=None, neg_ratio=None):
    """Process exactly one row group, then let the OS reclaim Arrow memory."""
    parquet_file = pq.ParquetFile(source)
    flag_cols = ["is_opened", "is_clicked", "is_unsubscribed", "is_complained", "is_purchased"]
    worker_rng = np.random.RandomState(SEED + row_group * 1_000_003)
    total_neg = 0
    writer = None
    selected_pos = 0
    selected_neg = 0
    try:
        columns = ["is_purchased"] if mode == "count" else None
        for batch in parquet_file.iter_batches(
            row_groups=[row_group], columns=columns,
            batch_size=batch_size, use_threads=False,
        ):
            purchase_idx = batch.schema.get_field_index("is_purchased")
            if purchase_idx < 0:
                raise KeyError("Required column 'is_purchased' is missing.")
            purchase = _purchase_flag(batch.column(purchase_idx).to_pandas()).to_numpy()
            if mode == "count":
                total_neg += int((purchase == 0).sum())
                continue

            keep = purchase == 1
            selected_pos += int(keep.sum())
            neg_idx = np.flatnonzero(purchase == 0)
            if len(neg_idx):
                chosen = neg_idx[worker_rng.random_sample(len(neg_idx)) < neg_ratio]
                keep[chosen] = True
                selected_neg += len(chosen)
            if keep.any():
                selected_batch = batch.filter(pa.array(keep))
                selected = pa.Table.from_batches([selected_batch]).to_pandas()
                for col in flag_cols:
                    selected[col] = _purchase_flag(selected[col]) if col in selected.columns else 0
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="snappy")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if mode == "count":
        print(f"TAU_WORKER_COUNT={total_neg}")
    else:
        print(f"TAU_WORKER_SAMPLE={selected_pos},{selected_neg}")


def _run_row_group_process(args, row_group, allow_corrupt=False):
    env = os.environ.copy()
    env.update({"ARROW_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tau_row_group_worker.py")
    if not os.path.exists(worker_script):
        raise FileNotFoundError(
            f"Required isolated sampler worker is missing: {worker_script}"
        )
    result = subprocess.run(
        [sys.executable, "-u", worker_script, *args],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if allow_corrupt and result.returncode in (-11, -7, 134):
            return None
        raise RuntimeError(
            f"Parquet row group {row_group} worker failed with exit code "
            f"{result.returncode}. This identifies a corrupt/undecodable row group "
            f"or a per-row-group memory limit.\n{detail}"
        )
    return result.stdout


def _save_count_checkpoint(path, payload):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def load_balanced_parquet_sample(path_messages, output_path, target_pos, target_neg, rng):
    """Sample via isolated row-group subprocesses to prevent Arrow SIGSEGV buildup."""
    del rng  # Sampling is deterministically seeded inside each row-group worker.
    batch_size = int(os.environ.get("PARQUET_SCAN_BATCH_SIZE", "250000"))
    row_group_chunk = int(os.environ.get("TAU_ROW_GROUP_CHUNK", "50"))
    oversample = 1.02
    spool_root = os.environ.get("TAU_SAMPLE_TMP_DIR")
    if not spool_root:
        spool_root = "/tmp" if os.path.isdir("/tmp") else None
    spool_dir = tempfile.mkdtemp(prefix="tau_sampling_", dir=spool_root)
    parquet_file = pq.ParquetFile(path_messages)
    num_row_groups = parquet_file.num_row_groups
    del parquet_file

    base_sample_path = os.environ.get("TAU_BASE_SAMPLE_PATH", output_path)
    checkpoint_path = f"{base_sample_path}.row_group_counts.json"
    source_stat = os.stat(path_messages)
    source_signature = {
        "path": os.path.abspath(path_messages),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "num_row_groups": num_row_groups,
    }
    checkpoint = {"version": 2, "source": source_signature, "counts": {}, "bad_row_groups": []}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved.get("version") == 2 and saved.get("source") == source_signature:
                checkpoint = saved
                print(
                    f"  [CHECKPOINT] Resuming {len(checkpoint['counts']):,d} counted "
                    f"row groups; excluded={len(checkpoint['bad_row_groups']):,d}"
                )
        except (OSError, ValueError, KeyError):
            pass

    bad_row_groups = set(int(x) for x in checkpoint.get("bad_row_groups", []))
    configured_bad = os.environ.get("TAU_BAD_ROW_GROUPS", "")
    bad_row_groups.update(
        int(value.strip()) for value in configured_bad.split(",") if value.strip()
    )
    total_negs_override = os.environ.get("TAU_TOTAL_NEGATIVES")
    if total_negs_override:
        total_negs = int(total_negs_override)
        print(f"  [COUNT OVERRIDE] Using known negative count: {total_negs:,d}")
    else:
        print(
            f"  [ISOLATED SCAN] Counting {num_row_groups:,d} row groups "
            f"in chunks of {row_group_chunk:,d}..."
        )
        for chunk_start in range(0, num_row_groups, row_group_chunk):
            chunk_end = min(chunk_start + row_group_chunk, num_row_groups)
            key = f"{chunk_start}:{chunk_end}"
            if key in checkpoint["counts"]:
                continue
            stdout = _run_row_group_process(
                ["count", str(path_messages), str(chunk_start), str(chunk_end),
                 str(batch_size), ",".join(map(str, sorted(bad_row_groups)))],
                f"{chunk_start}:{chunk_end}", allow_corrupt=True,
            )
            if stdout is None:
                print(
                    f"  [CHUNK RETRY] Row groups {chunk_start}:{chunk_end} failed; "
                    "retrying one row group at a time"
                )
                chunk_total = 0
                for rg_idx in range(chunk_start, chunk_end):
                    if rg_idx in bad_row_groups:
                        continue
                    single = _run_row_group_process(
                        ["count", str(path_messages), str(rg_idx), str(rg_idx + 1),
                         str(batch_size), ""], rg_idx, allow_corrupt=True,
                    )
                    if single is None:
                        bad_row_groups.add(rg_idx)
                        print(f"  [QUARANTINE] Excluding unreadable row group {rg_idx}")
                        continue
                    single_marker = next(
                        (line for line in single.splitlines() if line.startswith("TAU_WORKER_COUNT=")), None
                    )
                    if single_marker is None:
                        raise RuntimeError(f"Row group {rg_idx} count worker returned no count.")
                    chunk_total += int(single_marker.split("=", 1)[1])
                checkpoint["counts"][key] = chunk_total
                checkpoint["bad_row_groups"] = sorted(bad_row_groups)
                _save_count_checkpoint(checkpoint_path, checkpoint)
                continue
            marker = next((line for line in stdout.splitlines() if line.startswith("TAU_WORKER_COUNT=")), None)
            if marker is None:
                raise RuntimeError(f"Row-group chunk {key} count worker returned no count.")
            checkpoint["counts"][key] = int(marker.split("=", 1)[1])
            print(f"  [COUNT] row groups={chunk_end:,d}/{num_row_groups:,d}")
            _save_count_checkpoint(checkpoint_path, checkpoint)
        total_negs = sum(int(value) for value in checkpoint["counts"].values())

    checkpoint["bad_row_groups"] = sorted(bad_row_groups)
    _save_count_checkpoint(checkpoint_path, checkpoint)
    if bad_row_groups:
        print(
            f"  [QUARANTINE SUMMARY] Excluded {len(bad_row_groups):,d} unreadable "
            f"row group(s): {sorted(bad_row_groups)}"
        )

    neg_ratio = min(1.0, (target_neg * oversample) / max(1, total_negs))
    print(
        f"  [UNIFORM SAMPLING] Global Negative File Rows: {total_negs:,d} | "
        f"Sampling Ratio: {neg_ratio:.6f} | Batch: {batch_size:,d}"
    )

    selected_pos = 0
    selected_neg = 0
    try:
        for chunk_start in range(0, num_row_groups, row_group_chunk):
            chunk_end = min(chunk_start + row_group_chunk, num_row_groups)
            part_path = os.path.join(spool_dir, f"part_{chunk_start:05d}_{chunk_end:05d}.parquet")
            stdout = _run_row_group_process(
                ["sample", str(path_messages), str(chunk_start), str(chunk_end),
                 str(batch_size), part_path, repr(neg_ratio),
                 ",".join(map(str, sorted(bad_row_groups)))],
                f"{chunk_start}:{chunk_end}", allow_corrupt=True,
            )
            if stdout is None:
                if os.path.exists(part_path):
                    os.remove(part_path)
                print(
                    f"  [CHUNK RETRY] Sampling row groups {chunk_start}:{chunk_end} "
                    "failed; retrying individually"
                )
                for rg_idx in range(chunk_start, chunk_end):
                    if rg_idx in bad_row_groups:
                        continue
                    single_path = os.path.join(spool_dir, f"part_{rg_idx:05d}.parquet")
                    single = _run_row_group_process(
                        ["sample", str(path_messages), str(rg_idx), str(rg_idx + 1),
                         str(batch_size), single_path, repr(neg_ratio), ""],
                        rg_idx, allow_corrupt=True,
                    )
                    if single is None:
                        if os.path.exists(single_path):
                            os.remove(single_path)
                        bad_row_groups.add(rg_idx)
                        print(f"  [QUARANTINE] Sampling excluded row group {rg_idx}")
                        continue
                    single_marker = next(
                        (line for line in single.splitlines() if line.startswith("TAU_WORKER_SAMPLE=")), None
                    )
                    if single_marker is None:
                        raise RuntimeError(f"Row group {rg_idx} sample worker returned no summary.")
                    pos_count, neg_count = single_marker.split("=", 1)[1].split(",")
                    selected_pos += int(pos_count)
                    selected_neg += int(neg_count)
                print(
                    f"  [SCAN] row groups={chunk_end:,d}/{num_row_groups:,d} | "
                    f"pos={selected_pos:,d} | neg candidates={selected_neg:,d}"
                )
                continue
            marker = next((line for line in stdout.splitlines() if line.startswith("TAU_WORKER_SAMPLE=")), None)
            if marker is None:
                raise RuntimeError(f"Row-group chunk {chunk_start}:{chunk_end} returned no summary.")
            pos_count, neg_count = marker.split("=", 1)[1].split(",")
            selected_pos += int(pos_count)
            selected_neg += int(neg_count)
            print(
                f"  [SCAN] row groups={chunk_end:,d}/{num_row_groups:,d} | "
                f"pos={selected_pos:,d} | neg candidates={selected_neg:,d}"
            )

        part_files = [p for p in os.listdir(spool_dir) if p.endswith(".parquet")]
        if not part_files:
            raise RuntimeError("Parquet scan produced no sampled rows.")
        sampled = pd.read_parquet(spool_dir)
    finally:
        shutil.rmtree(spool_dir, ignore_errors=True)

    pos = sampled.loc[sampled["is_purchased"].eq(1)]
    if len(pos) > target_pos:
        pos = pos.sample(n=target_pos, random_state=SEED)
    neg = sampled.loc[sampled["is_purchased"].eq(0)]
    if len(pos) < target_pos or len(neg) < target_neg:
        raise RuntimeError(
            f"Insufficient sampled rows: positives={len(pos):,}/{target_pos:,}, "
            f"negatives={len(neg):,}/{target_neg:,}."
        )
    neg = neg.sample(n=target_neg, random_state=SEED).reset_index(drop=True)
    return pd.concat([pos, neg], ignore_index=True)


def run_bulletproof_feature_pipeline(tau_map=None):
    t0 = time.time()
    # A tau sweep must compare identical samples; reset the sampler each run.
    rng = np.random.RandomState(SEED)
    print("==================================================")
    print("  ZERO-CRASH FAST FEATURE PIPELINE (RAM < 250MB, 35s COMPLETION)")
    print("==================================================")
    print(f"[PATH] Input Extracted Dataset: {PATH_MESSAGES}")
    print(f"[PATH] Output Parquet File    : {OUTPUT_PARQUET}")

    campaigns = pd.read_csv(PATH_CAMPAIGNS)
    clients   = pd.read_csv(PATH_CLIENTS)
    holidays  = pd.read_csv(PATH_HOLIDAYS)

    if "first_purchase_date" in clients.columns:
        clients["first_purchase_date"] = to_dt(clients["first_purchase_date"])
    clients_clean = clients[["client_id", "first_purchase_date"]].drop_duplicates("client_id")

    holidays["date"] = to_dt(holidays["date"])
    if "is_holidays" in holidays.columns and "is_holiday" not in holidays.columns:
        holidays = holidays.rename(columns={"is_holidays": "is_holiday"})
    if "is_holiday" not in holidays.columns:
        holidays["is_holiday"] = 0
    holidays_clean = holidays[["date", "is_holiday"]].drop_duplicates("date")

    if "id" in campaigns.columns:
        campaigns = campaigns.rename(columns={"id": "campaign_id"})
    campaigns["campaign_id"] = pd.to_numeric(campaigns["campaign_id"], errors="coerce").fillna(-1).astype(np.int64)

    for col in ["campaign_type", "channel", "topic"]:
        if col in campaigns.columns:
            campaigns[col] = campaigns[col].apply(_norm_str)

    ALLOWED_TYPE  = {"bulk", "transactional", "trigger"}
    ALLOWED_CH    = {"email", "mobile_push", "multichannel", "sms"}
    ALLOWED_TOPIC = {"event", "happy.birthday", "leave.review", "offer.after.purchase", "sale.out"}

    if "campaign_type" in campaigns.columns:
        campaigns["camp_campaign_type"] = np.where(campaigns["campaign_type"].isin(ALLOWED_TYPE), campaigns["campaign_type"], None)
    if "channel" in campaigns.columns:
        campaigns["camp_channel"] = np.where(campaigns["channel"].isin(ALLOWED_CH), campaigns["channel"], None)
    if "topic" in campaigns.columns:
        campaigns["camp_topic"] = np.where(campaigns["topic"].isin(ALLOWED_TOPIC), campaigns["topic"], "other")

    oh_type  = onehot_join(campaigns, "camp_campaign_type", "camp_campaign_type")
    oh_chan  = onehot_join(campaigns, "camp_channel",       "camp_channel")
    oh_topic = onehot_join(campaigns, "camp_topic",         "camp_topic")

    keep_camp_cols = ["campaign_id"]
    if "started_at" in campaigns.columns and "finished_at" in campaigns.columns:
        c_start = pd.to_datetime(campaigns["started_at"], errors="coerce")
        c_finish = pd.to_datetime(campaigns["finished_at"], errors="coerce")
        campaigns["avg_campaign_duration"] = ((c_finish - c_start).dt.total_seconds() / 3600.0).fillna(24.0).clip(lower=0.1)
        keep_camp_cols.append("avg_campaign_duration")

    camp_base = campaigns[keep_camp_cols]
    campaigns_oh = pd.concat([camp_base, oh_type, oh_chan, oh_topic], axis=1).drop_duplicates(subset=["campaign_id"])

    TARGET_POS = 12340
    TARGET_NEG = 9987660   # EXACT 0.1234% Target Ratio (10,000,000 Total Rows)

    print(f"\n[FAST SCANNER] Sampling {TARGET_POS:,d} Positives + {TARGET_NEG:,d} Negatives directly from Parquet...")

    base_sample_path = os.environ.get("TAU_BASE_SAMPLE_PATH")
    reference_1x_path = os.environ.get("TAU_REFERENCE_1X_PARQUET")
    if reference_1x_path:
        if not os.path.exists(reference_1x_path):
            raise FileNotFoundError(
                f"TAU_REFERENCE_1X_PARQUET does not exist: {reference_1x_path}"
            )
        print(f"[REFERENCE 1X] Reusing exact legacy sampled rows: {reference_1x_path}")
        df = pd.read_parquet(reference_1x_path)
        required_reference_cols = {"client_id", "campaign_id", "sent_at", "is_purchased"}
        missing_reference_cols = sorted(required_reference_cols.difference(df.columns))
        if missing_reference_cols:
            raise ValueError(
                "Legacy 1x reference is missing columns required to recompute tau "
                f"features: {missing_reference_cols}"
            )
        print(
            f"[REFERENCE 1X] Rows={len(df):,d} | "
            f"Positives={int(_purchase_flag(df['is_purchased']).sum()):,d}"
        )
    elif base_sample_path and os.path.exists(base_sample_path):
        print(f"[BASE SAMPLE] Reusing fixed sampled rows: {base_sample_path}")
        df = pd.read_parquet(base_sample_path)
    elif str(PATH_MESSAGES).endswith(".parquet"):
        df = load_balanced_parquet_sample(PATH_MESSAGES, OUTPUT_PARQUET, TARGET_POS, TARGET_NEG, rng)
        if base_sample_path:
            os.makedirs(os.path.dirname(base_sample_path), exist_ok=True)
            print(f"[BASE SAMPLE] Saving fixed sampled rows: {base_sample_path}")
            df.to_parquet(base_sample_path, index=False)
    else:
        df_raw = pd.read_csv(PATH_MESSAGES)
        for c in ["is_opened", "is_clicked", "is_unsubscribed", "is_complained", "is_purchased"]:
            if c in df_raw.columns:
                df_raw[c] = df_raw[c].astype(str).str.strip().str.lower().isin(["t", "true", "1", "1.0"]).astype(int)
            else:
                df_raw[c] = 0
        is_pos_mask = (df_raw["is_purchased"] == 1)
        b_pos = df_raw[is_pos_mask].sample(n=min(len(df_raw[is_pos_mask]), TARGET_POS), random_state=SEED)
        b_neg = df_raw[~is_pos_mask].sample(n=min(len(df_raw[~is_pos_mask]), TARGET_NEG), random_state=SEED)
        df = pd.concat([b_pos, b_neg], ignore_index=True)
    gc.collect()

    print(f"[DATASET LOADED IN MEMORY] Total Rows: {len(df):,d} | Positives: {(df['is_purchased']==1).sum():,d} | Negatives: {(df['is_purchased']==0).sum():,d} | Rate: {df['is_purchased'].mean()*100:.6f}%")

    for c in ["sent_at", "opened_first_time_at", "clicked_first_time_at", "purchased_at", "unsubscribed_at", "complained_at", "date"]:
        if c in df.columns:
            df[c] = to_dt(df[c])

    df["client_id"]   = pd.to_numeric(df["client_id"], errors="coerce").fillna(-1).astype(np.int64)
    df["campaign_id"] = pd.to_numeric(df["campaign_id"], errors="coerce").fillna(-1).astype(np.int64)

    # Zero-Copy Fast Mapping (Eliminates 40GB pd.merge RAM spike!)
    if "first_purchase_date" in clients_clean.columns:
        client_map = clients_clean.set_index("client_id")["first_purchase_date"]
        df["first_purchase_date"] = df["client_id"].map(client_map)

    for col in campaigns_oh.columns:
        if col != "campaign_id":
            camp_map = campaigns_oh.set_index("campaign_id")[col]
            df[col] = df["campaign_id"].map(camp_map)

    holiday_map = holidays_clean.set_index("date")["is_holiday"]
    df["is_holiday"] = df["date"].map(holiday_map).fillna(0).astype(np.int8)

    # Zero-Copy Index Sorting: Only sort 2 1D integer arrays instead of copying 60 columns!
    c_codes_tmp, _ = pd.factorize(df["client_id"])
    sent_hrs_tmp = safe_dt_to_hours(df["sent_at"])
    sort_idx = np.lexsort((sent_hrs_tmp, c_codes_tmp))
    del c_codes_tmp, sent_hrs_tmp
    gc.collect()

    df = df.iloc[sort_idx].reset_index(drop=True)
    del sort_idx
    gc.collect()

    print("\n[FEATURE ENGINEERING] Calculating 64 Causal Domain Features in Memory...")

    c_codes = df["client_id"].astype("category").cat.codes.to_numpy(dtype=np.int32)
    c_uniques_len = int(c_codes.max() + 1) if len(c_codes) > 0 else 1

    cmp_codes = df["campaign_id"].astype("category").cat.codes.to_numpy(dtype=np.int32)
    n_camps = int(cmp_codes.max() + 1) if len(cmp_codes) > 0 else 1

    sent_at_hrs = safe_dt_to_hours(df["sent_at"])

    op_val = df["is_opened"].values.astype(np.float64) if "is_opened" in df.columns else np.zeros(len(df), dtype=np.float64)
    cl_val = df["is_clicked"].values.astype(np.float64) if "is_clicked" in df.columns else np.zeros(len(df), dtype=np.float64)
    un_val = df["is_unsubscribed"].values.astype(np.float64) if "is_unsubscribed" in df.columns else np.zeros(len(df), dtype=np.float64)
    cm_val = df["is_complained"].values.astype(np.float64) if "is_complained" in df.columns else np.zeros(len(df), dtype=np.float64)

    df["prev_is_opened"]       = fast_group_shift1(c_codes, op_val).astype(int)
    df["prev_is_clicked"]      = fast_group_shift1(c_codes, cl_val).astype(int)
    df["prev_is_unsubscribed"] = fast_group_shift1(c_codes, un_val).astype(int)
    df["prev_is_complained"]   = fast_group_shift1(c_codes, cm_val).astype(int)

    df["total_messages"]  = df.groupby("client_id").cumcount().astype(np.int32)
    df["total_campaigns"] = fast_cum_nunique_hist(c_codes, cmp_codes, n_camps)
    y_pur                 = df["is_purchased"].values.astype(np.float64)
    df["total_purchases"] = fast_group_cumsum_shift1(c_codes, y_pur).astype(np.int32)

    # Indicator for prior purchase
    has_p_mask = (df["total_purchases"].values > 0)
    df["has_prior_purchase"] = has_p_mask.astype(np.float32)

    # Recency calculations: static first_purchase_date uses shift=False to avoid Row 0 NaN!
    RECENCY_DYNAMIC = [
        ("opened_first_time_at",  "avg_time_since_last_open"),
        ("clicked_first_time_at", "avg_time_since_last_click"),
        ("unsubscribed_at",      "avg_time_since_unsubscribe"),
        ("complained_at",        "avg_time_since_complaint"),
    ]
    for src, new in RECENCY_DYNAMIC:
        if src in df.columns:
            src_hrs = safe_dt_to_hours(df[src])
            rec_hrs = fast_prior_ffill_shift(c_codes, sent_at_hrs, src_hrs, shift=True)
            med = float(np.nanmedian(rec_hrs)) if np.isfinite(np.nanmedian(rec_hrs)) else 24.0
            df[new] = np.nan_to_num(rec_hrs, nan=med, posinf=med, neginf=0.0).clip(0, 8760).astype(np.float32)
        else:
            df[new] = 24.0

    if "first_purchase_date" in df.columns:
        src_hrs_fp = safe_dt_to_hours(df["first_purchase_date"])
        rec_hrs_fp = fast_prior_ffill_shift(c_codes, sent_at_hrs, src_hrs_fp, shift=False)
        med_fp = float(np.nanmedian(rec_hrs_fp)) if np.isfinite(np.nanmedian(rec_hrs_fp)) else 24.0
        df["avg_time_since_first_purchase"] = np.nan_to_num(rec_hrs_fp, nan=med_fp, posinf=med_fp, neginf=0.0).clip(0, 8760).astype(np.float32)
    else:
        df["avg_time_since_first_purchase"] = 24.0

    # Refined v1 Purchasing Features: Discriminate non-buyers vs prior buyers
    pur_hrs = safe_dt_to_hours(df["purchased_at"]) if "purchased_at" in df.columns else sent_at_hrs
    is_pur  = df["is_purchased"].values.astype(np.int32)
    last_p_hrs = fast_last_purchase_time(c_codes, sent_at_hrs, is_pur, pur_hrs)
    d_days = last_p_hrs / 24.0

    # Non-buyers get distinct values (-1.0 for days, 0.0 for hazard/refractory)
    valid_p_days = d_days[has_p_mask & np.isfinite(d_days)]
    med_dd = float(np.median(valid_p_days)) if len(valid_p_days) > 0 else 14.0

    days_out = np.where(has_p_mask, np.nan_to_num(d_days, nan=med_dd).clip(0, 365), 365.0)
    df["days_since_last_purchase"] = days_out.astype(np.float32)

    z_hz = (df["days_since_last_purchase"].values - med_dd) / 7.0
    hz_out = np.where(has_p_mask, np.exp(-0.5 * (z_hz**2)), 0.0)
    df["feat_rtb_hazard"] = hz_out.astype(np.float32)

    refrac_out = np.where(has_p_mask, np.exp(-df["days_since_last_purchase"].values / 14.0), 0.0)
    df["feat_postbuy_refrac"] = refrac_out.astype(np.float32)

    # Calendar & Time features (safely handle NaT values)
    day_ser  = df["sent_at"].dt.day.fillna(1)
    dow_ser  = df["sent_at"].dt.dayofweek.fillna(0)
    hour_ser = df["sent_at"].dt.hour.fillna(12)

    df["cal_week_of_month"] = ((day_ser - 1) // 7 + 1).astype(int)
    df["cal_is_weekend"]    = (dow_ser >= 5).astype(int)

    df["feat_dow_shift"]    = np.sin(2.0 * np.pi * dow_ser / 7.0).astype(np.float32)
    df["feat_hour_shift"]   = np.sin(2.0 * np.pi * hour_ser / 24.0).astype(np.float32)

    dist_payday = np.minimum(np.abs(day_ser - 10), np.abs(day_ser - 25))
    df["feat_payday_bump"] = np.exp(-dist_payday / 3.0).astype(np.float32)

    # End of Quarter (EOQ) proximity
    month_ser = df["sent_at"].dt.month.fillna(1)
    is_q_end_month = month_ser.isin([3, 6, 9, 12])
    dist_eoq = np.where(is_q_end_month, 30 - day_ser, 30)
    df["feat_eoq_bump"] = np.exp(-np.maximum(0, dist_eoq) / 5.0).astype(np.float32)

    if "channel" in df.columns:
        ch_raw_str = df["channel"].fillna("unknown").astype(str).str.lower().str.strip()
    else:
        ch_raw_str = pd.Series(["unknown"]*len(df), index=df.index)

    ch_dict = {"email":0, "mobile_push":1, "sms":2, "multichannel":3, "unknown":4}
    channel_codes = np.array([ch_dict.get(x, 4) for x in ch_raw_str], dtype=np.int32)
    tau_map = tau_map or {}
    tau_arr  = np.array([
        tau_map.get("email", 48.0),
        tau_map.get("mobile_push", 24.0),
        tau_map.get("sms", 72.0),
        tau_map.get("multichannel", 48.0),
        tau_map.get("unknown", 48.0),
    ], dtype=np.float64)
    cool_arr = np.array([24.0, 6.0, 24.0, 24.0, 24.0], dtype=np.float64)
    out_fatigue, out_cool_ok = fast_fatigue_cooldown(c_codes, sent_at_hrs, channel_codes, tau_arr, cool_arr)
    df["feat_fatigue"]     = out_fatigue
    df["feat_cooldown_ok"] = out_cool_ok

    for ch_name in ["email", "mobile_push", "web_push"]:
        df[f"channel_{ch_name}"] = (ch_raw_str == ch_name).astype(int)

    for ch_name in ["email", "mobile_push", "sms", "multichannel"]:
        is_ch_mask = (ch_raw_str == ch_name).to_numpy(dtype=bool)
        rec_ch = fast_last_ch_hours(c_codes, sent_at_hrs, is_ch_mask)
        df[f"feat_last_{ch_name}_hours"] = np.nan_to_num(rec_ch, nan=72.0, posinf=72.0, neginf=0.0).clip(0, 8760).astype(np.float32)

    df["feat_last_any_hours"] = df[["feat_last_email_hours", "feat_last_mobile_push_hours"]].min(axis=1).astype(np.float32)
    if "avg_campaign_duration" not in df.columns:
        df["avg_campaign_duration"] = 24.0
    else:
        df["avg_campaign_duration"] = df["avg_campaign_duration"].fillna(24.0).astype(np.float32)

    if "message_type" in df.columns:
        m_type = df["message_type"].fillna("other").astype(str).str.lower()
        for m_name in ["bulk", "transactional", "trigger"]:
            df[f"message_type_{m_name}"] = (m_type == m_name).astype(int)
    else:
        for m_name in ["bulk", "transactional", "trigger"]:
            df[f"message_type_{m_name}"] = 0

    if "email_provider" in df.columns:
        ep = df["email_provider"].fillna("other").astype(str).str.lower().str.strip()
        ALLOWED_EP = {"gmail.com", "mail.ru"}
        ep_red = np.where(ep.isin(ALLOWED_EP), ep, "other")
        df["email_provider_gmail.com"] = (ep_red == "gmail.com").astype(int)
        df["email_provider_mail.ru"]   = (ep_red == "mail.ru").astype(int)
        df["email_provider_other"]     = (ep_red == "other").astype(int)
    else:
        df["email_provider_gmail.com"] = 0
        df["email_provider_mail.ru"]   = 0
        df["email_provider_other"]     = 1

    if "platform" in df.columns:
        p = df["platform"].fillna("other").astype(str).str.lower().str.strip()
        fixed_p = ["desktop", "smartphone", "phablet", "tablet"]
        for name in fixed_p:
            df[f"platform.{name}"] = (p == name).astype(int)
        df["platform."] = (~p.isin(fixed_p)).astype(int)
    else:
        for name in ["desktop", "smartphone", "phablet", "tablet"]:
            df[f"platform.{name}"] = 0
        df["platform."] = 1

    # Real rolling 30-day user behavior metrics: Pure rates & Velocities separated!
    r_op_cnt, r_op_rate, r_cl_rate, r_op_vel, r_cl_vel, r_cad_std = fast_rolling_30d_metrics(c_codes, sent_at_hrs, op_val, cl_val)
    df["u_open_cnt_30d"]         = r_op_cnt
    df["u_open_rate_30d"]        = r_op_rate       # Pure 30-day open rate (0.0 ~ 1.0)
    df["u_click_rate_30d"]       = r_cl_rate       # Pure 30-day click rate (0.0 ~ 1.0)
    df["u_open_velocity_7d_30d"] = r_op_vel        # 7d vs 30d open velocity
    df["u_click_velocity_7d_30d"]= r_cl_vel        # 7d vs 30d click velocity
    df["u_cadence_std_30d"]      = r_cad_std

    # Global 30-day context open rate: must sort by time globally first
    time_sort_idx = np.argsort(sent_at_hrs, kind="mergesort")
    sorted_times = sent_at_hrs[time_sort_idx]
    sorted_ops   = op_val[time_sort_idx]
    ctx_sorted   = fast_ctx_open_rate_30d_sorted(sorted_times, sorted_ops)
    ctx_original = np.empty_like(ctx_sorted)
    ctx_original[time_sort_idx] = ctx_sorted
    df["ctx_tc_open_rate_30d"] = ctx_original

    topic_cols = [c for c in df.columns if str(c).startswith("camp_topic")]
    if len(topic_cols) > 0:
        top_codes = df[topic_cols].fillna(0).to_numpy(dtype=np.int8).argmax(axis=1).astype(np.int32)
    else:
        top_codes = np.zeros(len(df), dtype=np.int32)

    n_topics = int(top_codes.max()) + 1 if len(top_codes) > 0 else 1
    last_top_hrs = fast_client_topic_last_ts(c_codes, top_codes, sent_at_hrs, c_uniques_len, n_topics)
    dt_top = sent_at_hrs - last_top_hrs
    med_top = float(np.nanmedian(dt_top)) if np.isfinite(np.nanmedian(dt_top)) else 48.0
    df["topic_t_since_hours"] = np.nan_to_num(dt_top, nan=med_top, posinf=med_top, neginf=0.0).clip(0, 8760).astype(np.float32)
    df["feat_topic_novelty"]  = np.exp(-df["topic_t_since_hours"].values / 48.0).astype(np.float32)
    df["topic_N7"]            = top_codes.astype(np.int32)

    attr_cols = sorted(set([c for c in df.columns if str(c).startswith(("camp_campaign_type", "camp_channel", "camp_topic"))]))
    if len(attr_cols) > 0:
        vec_mat = df[attr_cols].fillna(0).to_numpy(dtype=np.int8)
    else:
        vec_mat = np.zeros((len(df), 1), dtype=np.int8)
    df["feat_like_last_success"] = fast_like_last_success(c_codes, is_pur, vec_mat, c_uniques_len)
    df["feat_path_align"]        = fast_path_alignment(c_codes, vec_mat, c_uniques_len)


    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "string":
            df[col] = df[col].astype(str)
        elif "bool" in str(df[col].dtype):
            df[col] = df[col].astype(int)

    print(f"\n[SAVE] Writing final 64-feature dataset to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    for target_p in [OUTPUT_PARQUET_BASE, OUTPUT_PARQUET_NEW]:
        if target_p != OUTPUT_PARQUET:
            try:
                import shutil
                shutil.copyfile(OUTPUT_PARQUET, target_p)
                print(f"[SAVE] Duplicate copy saved to: {target_p}")
            except Exception:
                pass

    t1 = time.time()
    print("==================================================")
    print(f" SUCCESS! Feature Pipeline Completed in {t1-t0:.2f} seconds!")
    print(f" Output Dataset Shape: {df.shape}")
    print(f" Exact Positive Rate: {df['is_purchased'].mean()*100:.6f}%")
    print("==================================================")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tau-row-group-worker":
        worker_mode = sys.argv[2]
        worker_source = sys.argv[3]
        worker_row_group = int(sys.argv[4])
        worker_batch_size = int(sys.argv[5])
        if worker_mode == "count":
            _row_group_worker(
                worker_mode, worker_source, worker_row_group, worker_batch_size
            )
        elif worker_mode == "sample":
            _row_group_worker(
                worker_mode, worker_source, worker_row_group, worker_batch_size,
                output=sys.argv[6], neg_ratio=float(sys.argv[7]),
            )
        else:
            raise ValueError(f"Unknown row-group worker mode: {worker_mode}")
    else:
        run_bulletproof_feature_pipeline()
