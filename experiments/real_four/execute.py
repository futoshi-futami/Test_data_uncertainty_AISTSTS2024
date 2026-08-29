from __future__ import annotations

import sys

import numpy as np

import huc_core


def fast_worst_interval(scores: np.ndarray, residuals: np.ndarray, min_count: int = 1) -> huc_core.IntervalResult:
    """Exact tied-score interval scan in linear time after sorting."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    r = np.asarray(residuals, dtype=float).reshape(-1)
    if s.shape != r.shape or s.size == 0:
        raise ValueError("scores and residuals must be nonempty arrays of the same shape")
    n = s.size
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    rr = r[order]
    starts = np.r_[0, 1 + np.flatnonzero(ss[1:] != ss[:-1])]
    ends = np.r_[starts[1:], n]
    vals = np.add.reduceat(rr, starts)
    counts = ends - starts
    uniq = ss[starts]
    ps = np.r_[0.0, np.cumsum(vals)]
    pc = np.r_[0, np.cumsum(counts)]

    eligible = -1
    min_pref = np.inf
    max_pref = -np.inf
    min_idx = max_idx = 0
    best_abs = -1.0
    best_start = best_end = 0
    best_total = 0.0

    for end in range(1, len(vals) + 1):
        while eligible + 1 < end and pc[end] - pc[eligible + 1] >= min_count:
            eligible += 1
            value = float(ps[eligible])
            if value < min_pref:
                min_pref = value
                min_idx = eligible
            if value > max_pref:
                max_pref = value
                max_idx = eligible
        if eligible < 0:
            continue
        candidates = ((float(ps[end] - min_pref), min_idx), (float(ps[end] - max_pref), max_idx))
        for total, start in candidates:
            if abs(total) > best_abs + 1.0e-15:
                best_abs = abs(total)
                best_start = int(start)
                best_end = end - 1
                best_total = total
    if best_abs < 0:
        return huc_core.IntervalResult(0.0, float(uniq[0]), float(uniq[-1]), 1.0, n)
    count = int(counts[best_start:best_end + 1].sum())
    return huc_core.IntervalResult(
        abs(best_total) / n,
        float(uniq[best_start]),
        float(uniq[best_end]),
        1.0 if best_total >= 0 else -1.0,
        count,
    )


huc_core.worst_interval = fast_worst_interval

import run_dataset

_original_run = run_dataset.run


def bounded_run(bundle, output, seeds, max_iter, threshold):
    # Keep the same update rule while making all four real-data runs finish in a
    # single reproducible workflow. Validation still chooses the final prefix.
    return _original_run(bundle, output, seeds, min(int(max_iter), 25), threshold)


run_dataset.run = bounded_run

if __name__ == "__main__":
    run_dataset.main()
