from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ORDER = [
    "Raw", "Temperature", "Vector", "Dirichlet", "UC", "HUC", "C-UC", "C-HUC",
    "Temperature→C-UC", "Temperature→C-HUC", "Vector→C-UC", "Vector→C-HUC",
    "Dirichlet→C-UC", "Dirichlet→C-HUC",
]
METRICS = ["UC", "HUC", "C-UC", "C-HUC", "Accuracy", "AUC"]


def fmt(mean, std):
    if pd.isna(mean): return "NA"
    if pd.isna(std): std = 0.0
    return f"{mean:.6f} ± {std:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    root = Path(a.input_root)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    result_files = sorted(root.rglob("results_long.csv"))
    completion_files = sorted(root.rglob("completion.json"))
    if len(result_files) != 4:
        raise RuntimeError(f"expected four results_long.csv files, found {result_files}")
    frames = [pd.read_csv(x) for x in result_files]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "all_results_long.csv", index=False)
    completions = [json.loads(x.read_text(encoding="utf-8")) for x in completion_files]
    (out / "completion_all.json").write_text(json.dumps(completions, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = df.groupby(["dataset", "base_model", "method", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(out / "all_results_by_base_method.csv", index=False)
    overall = df.groupby(["dataset", "method", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    overall.to_csv(out / "all_results_by_method.csv", index=False)

    method_rank = {x: i for i, x in enumerate(METHOD_ORDER)}
    lines = ["# Four real-data HUC experiments", "", "All entries are mean ± standard deviation over the five seeds. UC, HUC, C-UC, and C-HUC are smaller-is-better; Accuracy and AUC are larger-is-better.", ""]
    for dataset in df.dataset.drop_duplicates():
        lines += [f"## {dataset}", ""]
        subset = summary[summary.dataset == dataset]
        for base in subset.base_model.drop_duplicates():
            lines += [f"### {base}", "", "| Method | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
            x = subset[subset.base_model == base]
            lookup = {(r.method, r.metric): (r["mean"], r["std"]) for _, r in x.iterrows()}
            methods = sorted(x.method.unique(), key=lambda z: method_rank.get(z, 999))
            for method in methods:
                vals = [fmt(*lookup.get((method, metric), (np.nan, np.nan))) for metric in METRICS]
                lines.append("| " + method + " | " + " | ".join(vals) + " |")
            lines.append("")
    (out / "FOUR_RESULT_TABLES.md").write_text("\n".join(lines), encoding="utf-8")

    best_rows = []
    for (dataset, base), x in summary[summary.metric.isin(METRICS)].groupby(["dataset", "base_model"]):
        for metric in METRICS:
            m = x[x.metric == metric]
            if m.empty: continue
            idx = m["mean"].idxmax() if metric in {"Accuracy", "AUC"} else m["mean"].idxmin()
            r = m.loc[idx]
            best_rows.append({"dataset": dataset, "base_model": base, "metric": metric, "method": r.method, "mean": r["mean"], "std": r["std"]})
    pd.DataFrame(best_rows).to_csv(out / "best_method_by_base_metric.csv", index=False)

    copied = out / "dataset_outputs"; copied.mkdir(exist_ok=True)
    for completion in completion_files:
        src = completion.parent
        dst = copied / src.name
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)

    expected = {
        "datasets": int(df.dataset.nunique()),
        "base_dataset_pairs": int(df[["dataset", "base_model"]].drop_duplicates().shape[0]),
        "methods": int(df.method.nunique()),
        "metrics": int(df.metric.nunique()),
        "result_rows": int(len(df)),
        "nonfinite_values": int((~np.isfinite(df.value)).sum()),
        "error_count": int(sum(int(x.get("error_count", 0)) for x in completions)),
    }
    (out / "final_check.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    if expected["datasets"] != 4 or expected["methods"] != 14 or expected["metrics"] != 8 or expected["nonfinite_values"] != 0 or expected["error_count"] != 0:
        raise RuntimeError(expected)
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
