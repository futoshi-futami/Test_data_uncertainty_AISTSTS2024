from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_METHODS = [
    "Raw",
    "Temperature",
    "Vector",
    "Dirichlet",
    "UC",
    "HUC",
    "C-UC",
    "C-HUC",
    "Temperature→C-UC",
    "Temperature→C-HUC",
    "Vector→C-UC",
    "Vector→C-HUC",
    "Dirichlet→C-UC",
    "Dirichlet→C-HUC",
]


def load_module(project: Path):
    sys.path.insert(0, str(project / "src"))
    script = project / "scripts" / "run_four_real.py"
    spec = importlib.util.spec_from_file_location("run_four_real_recovered", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--dataset", choices=["slurp", "nsl_kdd", "nbaiot", "inaturalist"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    output = args.output.resolve()
    cache = args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    module = load_module(project)

    if list(module.METHODS) != EXPECTED_METHODS:
        raise AssertionError(f"method mismatch: {module.METHODS}")

    loader_map = {
        "slurp": ("SLURP", module.load_slurp),
        "nsl_kdd": ("NSL-KDD", module.load_nsl_kdd),
        "nbaiot": ("N-BaIoT", module.load_nbaiot),
        "inaturalist": ("iNaturalist", module.load_inaturalist),
    }
    display_name, loader = loader_map[args.dataset]
    started = time.time()
    status = {
        "dataset_request": args.dataset,
        "dataset_display": display_name,
        "status": "running",
        "started_unix": started,
        "methods": EXPECTED_METHODS,
        "seeds": list(module.SEEDS),
        "metrics": list(module.METRICS),
    }
    module.stable_json_dump(status, output / "execution_status.json")

    try:
        spec = loader(cache)
        results, checks = module.run_dataset(spec, output)
        module.make_summaries(results, checks, output)

        numeric_values = pd.to_numeric(results["value"], errors="coerce").to_numpy(float)
        if not np.isfinite(numeric_values).all():
            raise AssertionError("non-finite result values")
        if not bool(checks["passed"].all()):
            raise AssertionError("validation checks contain failures")
        methods = sorted(results["method"].unique().tolist())
        if set(methods) != set(EXPECTED_METHODS) or len(methods) != 14:
            raise AssertionError(f"expected 14 methods, found {methods}")
        seeds = sorted(int(v) for v in results["seed"].unique())
        if seeds != sorted(module.SEEDS):
            raise AssertionError(f"seed mismatch: {seeds}")
        metrics = sorted(results["metric"].unique().tolist())
        if set(metrics) != set(module.METRICS):
            raise AssertionError(f"metric mismatch: {metrics}")

        status.update(
            {
                "status": "completed",
                "dataset": spec.name,
                "n": int(len(spec.y)),
                "classes": int(len(spec.class_names)),
                "base_models": list(spec.base_models),
                "base_count": int(len(spec.base_models)),
                "method_count": int(len(methods)),
                "metric_count": int(len(metrics)),
                "seed_count": int(len(seeds)),
                "result_rows": int(len(results)),
                "wide_rows": int(len(results) // len(metrics)),
                "check_rows": int(len(checks)),
                "check_failures": int((~checks["passed"]).sum()),
                "nonfinite_values": int((~np.isfinite(numeric_values)).sum()),
                "elapsed_minutes": float((time.time() - started) / 60.0),
                "source_info": spec.source_info,
            }
        )
        module.stable_json_dump(status, output / "completion.json")
        module.stable_json_dump(status, output / "execution_status.json")
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_minutes": float((time.time() - started) / 60.0),
            }
        )
        module.stable_json_dump(status, output / "execution_status.json")
        (output / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
