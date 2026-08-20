"""SMS/이메일/푸시 반감기(τ) 민감도 실험.

표의 배율별로 전처리 데이터를 다시 만들고, total_* 변수를 제외한
XGBoost 평가를 순서대로 실행한다.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# predict 폴더 안/프로젝트 루트 어디에서 실행해도 동작하도록 경로를 찾는다.
if SCRIPT_DIR.name == "predict":
    ROOT = SCRIPT_DIR.parent
    PREDICT_DIR = SCRIPT_DIR
else:
    ROOT = SCRIPT_DIR
    PREDICT_DIR = SCRIPT_DIR / "predict"

# 전처리 파일은 환경변수로 지정할 수 있으며, 기존 프로젝트와 현재 폴더를 순서대로 찾는다.
# Example: set TAU_PREPROCESS_PATH to an explicit preprocessing script when needed.
PREPROCESS_PATH = Path(os.environ.get("TAU_PREPROCESS_PATH", ROOT / "preprocessing_fast.py"))
if not PREPROCESS_PATH.is_absolute():
    PREPROCESS_PATH = (ROOT / PREPROCESS_PATH).resolve()

XGB_EVAL_PATH = PREDICT_DIR / "xgb_tau_eval.py"
if not XGB_EVAL_PATH.exists():
    XGB_EVAL_PATH = ROOT / "xgb_tau_eval.py"
WORK_DIR = PREDICT_DIR / "tau_sweep_data"
RESULT_DIR = PREDICT_DIR / "tau_sweep_results"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 기준: SMS 72h, 이메일 48h, 푸시 24h
SWEEP = {
    "0.50x": {"sms": 36.0, "email": 24.0, "mobile_push": 12.0},
    "0.75x": {"sms": 54.0, "email": 36.0, "mobile_push": 18.0},
    "1.00x": {"sms": 72.0, "email": 48.0, "mobile_push": 24.0},
    "1.25x": {"sms": 90.0, "email": 60.0, "mobile_push": 30.0},
    "1.50x": {"sms": 108.0, "email": 72.0, "mobile_push": 36.0},
}


def load_preprocessor():
    if not PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            f"전처리 파일을 찾을 수 없습니다: {PREPROCESS_PATH}\n"
            "preprocessing_fast(original).py를 프로젝트 폴더에 두세요."
        )
    print(f"[TAU PREPROCESSOR] Using: {PREPROCESS_PATH}", flush=True)
    spec = importlib.util.spec_from_file_location("tau_preprocessor", PREPROCESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(pre, tau_map, output_path):
    # preprocessing_fast.py의 실제 공개 진입점
    if not hasattr(pre, "add_fatigue_and_cooldown"):
        if not hasattr(pre, "run_bulletproof_feature_pipeline"):
            raise AttributeError("preprocessing_fast.py에서 실행 가능한 전처리 함수를 찾지 못했습니다.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_outputs = {
            name: getattr(pre, name)
            for name in ("OUTPUT_PARQUET", "OUTPUT_PARQUET_BASE", "OUTPUT_PARQUET_NEW")
            if hasattr(pre, name)
        }
        base_sample_path = output_path.parent / "base_sample_seed1.parquet"
        original_base_sample_path = os.environ.get("TAU_BASE_SAMPLE_PATH")
        try:
            # The fast preprocessor reads these module globals at write time.
            # Point all aliases at the tau-specific file to avoid a shared file.
            for name in original_outputs:
                setattr(pre, name, str(output_path))
            os.environ["TAU_BASE_SAMPLE_PATH"] = str(base_sample_path)
            print(f"[TAU OUTPUT] Direct write target: {output_path}")
            print(f"[TAU BASE SAMPLE] Shared sample path: {base_sample_path}")
            pre.run_bulletproof_feature_pipeline(tau_map=tau_map)
            if not output_path.exists():
                raise FileNotFoundError(f"전처리 결과 파일이 없습니다: {output_path}")
        finally:
            for name, value in original_outputs.items():
                setattr(pre, name, value)
            if original_base_sample_path is None:
                os.environ.pop("TAU_BASE_SAMPLE_PATH", None)
            else:
                os.environ["TAU_BASE_SAMPLE_PATH"] = original_base_sample_path
        return

    # build_final_data 내부에서 호출되는 두 feature 함수에 같은 τ를 주입한다.
    original_fatigue = pre.add_fatigue_and_cooldown
    original_topic = pre.add_topic_novelty

    def fatigue_with_tau(df, tau_map_unused=None, cooldown_map=None):
        return original_fatigue(df, tau_map=tau_map, cooldown_map=cooldown_map)

    def topic_with_tau(df, window_days=7, kappa_map=None, tau_map_unused=None):
        return original_topic(df, window_days=window_days, kappa_map=kappa_map, tau_map=tau_map)

    pre.add_fatigue_and_cooldown = fatigue_with_tau
    pre.add_topic_novelty = topic_with_tau
    try:
        data, _, _ = pre.build_final_data(mode=pre.MODE)
        data.to_parquet(output_path, index=False)
    finally:
        pre.add_fatigue_and_cooldown = original_fatigue
        pre.add_topic_novelty = original_topic


def main():
    if not XGB_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"xgb_tau_eval.py를 찾을 수 없습니다. 확인한 경로: {XGB_EVAL_PATH}"
        )
    pre = load_preprocessor()
    all_results = {}
    for label, tau in SWEEP.items():
        print(f"\n===== {label}: SMS={tau['sms']}h / Email={tau['email']}h / Push={tau['mobile_push']}h =====")
        parquet_path = WORK_DIR / f"final_data_{label.replace('.', '_')}.parquet"
        result_path = RESULT_DIR / f"result_xgb_{label.replace('.', '_')}.json"
        if parquet_path.exists() and result_path.exists():
            print(f"[SKIP ALL] 전처리 데이터와 XGBoost 결과가 모두 존재: {label}")
            print(f"  [DATA]   {parquet_path}")
            print(f"  [RESULT] {result_path}")
            with result_path.open(encoding="utf-8") as f:
                all_results[label] = json.load(f)
            continue

        if parquet_path.exists():
            print(f"[SKIP] 이미 완료된 전처리 사용: {parquet_path}")
        else:
            make_dataset(pre, tau, parquet_path)

        env = os.environ.copy()
        env["TAU_PARQUET"] = str(parquet_path)
        env["TAU_OUTPUT_DIR"] = str(RESULT_DIR)
        env["TAU_RESULT_NAME"] = result_path.name
        subprocess.run([sys.executable, str(XGB_EVAL_PATH)], check=True, env=env)

        if not result_path.exists():
            raise FileNotFoundError(f"XGBoost 결과 파일이 생성되지 않았습니다: {result_path}")
        with result_path.open(encoding="utf-8") as f:
            all_results[label] = json.load(f)

    with (RESULT_DIR / "tau_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {RESULT_DIR / 'tau_sweep_summary.json'}")


if __name__ == "__main__":
    main()
