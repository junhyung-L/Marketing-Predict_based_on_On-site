"""Compare a newly generated XGBoost report with the saved regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = {
    "num_features": 64,
    "f1_score": 0.3832933653077538,
    "pr_auc": 0.353403606480214,
    "recall": 0.5180983252296056,
    "precision": 0.3041547732318427,
    "top1000_hits": 497,
    "top2000_hits": 729,
    "top3000_hits": 932,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(__file__).resolve().parent / "predict" / "result_xgb.json",
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    report = json.loads(args.result.read_text(encoding="utf-8"))["all"]
    failures = []
    for metric, expected in EXPECTED.items():
        actual = report[metric]
        if isinstance(expected, float):
            if abs(actual - expected) > args.tolerance:
                failures.append(f"{metric}: expected {expected}, got {actual}")
        elif actual != expected:
            failures.append(f"{metric}: expected {expected}, got {actual}")

    if failures:
        raise SystemExit("Regression check failed:\n" + "\n".join(failures))
    print(f"Regression baseline matched: {args.result}")


if __name__ == "__main__":
    main()
