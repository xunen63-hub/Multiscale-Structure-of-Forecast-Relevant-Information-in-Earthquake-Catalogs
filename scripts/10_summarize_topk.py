import os
import argparse
from pathlib import Path

import pandas as pd


def expand_path(p):
    return str(Path(p).expanduser().resolve())


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-out",
        default="outputs/topk_xgboost",
    )
    parser.add_argument("--topk-list", default="20,30,50,75,100,150,200")
    parser.add_argument(
        "--variant-pattern",
        default="top{k}_stable",
        help="Subdirectory pattern under base-out. Must contain {k}.",
    )
    parser.add_argument(
        "--result-filename",
        default="final_test_metrics_binary_m5_ablation_vs_full.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/topk_xgboost/summary_topk_binary_m5_vs_full.csv",
    )
    args = parser.parse_args()

    if "{k}" not in args.variant_pattern:
        raise ValueError("--variant-pattern must contain {k}.")

    base_out = expand_path(args.base_out)
    topk_list = parse_int_list(args.topk_list)

    rows = []
    missing_paths = []

    for k in topk_list:
        variant_name = args.variant_pattern.format(k=k)
        p = os.path.join(base_out, variant_name, args.result_filename)

        if not os.path.exists(p):
            missing_paths.append(p)
            print("Missing:", p, flush=True)
            continue

        df = pd.read_csv(p)
        df["topk"] = int(k)
        df["variant_name_summary"] = variant_name
        rows.append(df)

    if not rows:
        msg = "No topK result files found. Expected paths:\n" + "\n".join(missing_paths)
        raise RuntimeError(msg)

    out = pd.concat(rows, ignore_index=True)

    required = {
        "horizon_month",
        "n_features",
        "delta_auc_minus_full",
        "delta_pr_auc_minus_full",
        "delta_f1_minus_full",
        "delta_brier_minus_full",
    }
    missing_cols = required - set(out.columns)
    if missing_cols:
        raise ValueError(f"Missing required result columns: {sorted(missing_cols)}")

    out["stable_auc"] = out["delta_auc_minus_full"].abs() <= 0.01
    out["stable_pr_auc"] = out["delta_pr_auc_minus_full"].abs() <= 0.02
    out["stable_f1"] = out["delta_f1_minus_full"].abs() <= 0.02
    out["stable_brier"] = out["delta_brier_minus_full"] <= 0.005

    out["stable_all_metrics"] = (
        out["stable_auc"]
        & out["stable_pr_auc"]
        & out["stable_f1"]
        & out["stable_brier"]
    )

    out["feature_reduction_fraction"] = 1.0 - out["n_features"] / 282.0
    out = out.sort_values(["topk", "horizon_month"]).reset_index(drop=True)

    out_path = expand_path(args.out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    show_cols = [
        "topk",
        "variant_name_summary",
        "horizon_month",
        "n_features",
        "feature_reduction_fraction",
        "f1",
        "full_f1",
        "delta_f1_minus_full",
        "auc",
        "full_auc",
        "delta_auc_minus_full",
        "pr_auc",
        "full_pr_auc",
        "delta_pr_auc_minus_full",
        "brier",
        "full_brier",
        "delta_brier_minus_full",
        "stable_all_metrics",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print(out[show_cols].to_string(index=False), flush=True)
    print("\nSaved:", out_path, flush=True)

    summary = (
        out.groupby("topk")
        .agg(
            stable_horizon_count=("stable_all_metrics", "sum"),
            mean_delta_auc=("delta_auc_minus_full", "mean"),
            worst_delta_auc=("delta_auc_minus_full", "min"),
            mean_delta_pr_auc=("delta_pr_auc_minus_full", "mean"),
            mean_delta_f1=("delta_f1_minus_full", "mean"),
            mean_delta_brier=("delta_brier_minus_full", "mean"),
        )
        .reset_index()
        .sort_values("topk")
    )

    print("\nTopK stability summary:", flush=True)
    print(summary.to_string(index=False), flush=True)

    ok4 = summary[summary["stable_horizon_count"] >= 4]
    if len(ok4):
        print(
            "\nSmallest tested K stable for all 4 horizons:",
            int(ok4.iloc[0]["topk"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
