from __future__ import annotations

import math

import numpy as np

import execute  # applies the fast exact-interval implementation
import huc_core as hc


def brute(scores, residuals, min_count=1):
    s = np.asarray(scores, float)
    r = np.asarray(residuals, float)
    order = np.argsort(s, kind="mergesort")
    ss, rr = s[order], r[order]
    starts = np.r_[0, 1 + np.flatnonzero(ss[1:] != ss[:-1])]
    ends = np.r_[starts[1:], len(s)]
    uniq = ss[starts]
    vals = np.add.reduceat(rr, starts)
    counts = ends - starts
    best = (-1.0, None)
    for i in range(len(vals)):
        total = 0.0
        count = 0
        for j in range(i, len(vals)):
            total += vals[j]
            count += counts[j]
            if count >= min_count and abs(total) > best[0] + 1e-15:
                best = (abs(total), (i, j, total, count))
    i, j, total, count = best[1]
    return best[0] / len(s), float(uniq[i]), float(uniq[j]), 1.0 if total >= 0 else -1.0, int(count)


def test_intervals():
    rng = np.random.default_rng(20260827)
    max_error = 0.0
    for n in range(2, 60):
        for _ in range(50):
            scores = rng.integers(0, max(2, n // 4), n).astype(float)
            residuals = rng.normal(size=n)
            min_count = int(rng.integers(1, n + 1))
            got = hc.worst_interval(scores, residuals, min_count)
            ref = brute(scores, residuals, min_count)
            max_error = max(max_error, abs(got.value - ref[0]))
            if abs(got.value - ref[0]) > 1e-12:
                raise AssertionError((n, min_count, got, ref))
    print("interval_max_error", max_error)


def test_projection():
    rng = np.random.default_rng(44)
    z = rng.normal(size=(500, 17))
    q = hc.simplex_projection(z)
    assert q.min() >= -1e-14
    assert np.max(np.abs(q.sum(axis=1) - 1.0)) < 1e-12
    assert np.max(np.abs(hc.simplex_projection(q) - q)) < 1e-12


def test_all_methods():
    rng = np.random.default_rng(7)
    n, k = 260, 5
    labels = np.repeat(np.arange(k), n // k)
    rng.shuffle(labels)
    p = rng.dirichlet(np.ones(k) * 2, size=n)
    p[np.arange(n), labels] += 1.2
    p = hc.normalize_probs(p)
    paths = [["normal"], ["attack", "network", "dos"], ["attack", "network", "probe"], ["attack", "access", "r2l"], ["attack", "access", "u2r"]]
    tree = hc.build_tree(paths)
    parts = {"stage1": np.arange(0, 60), "stage2": np.arange(60, 120), "validation": np.arange(120, 180), "test": np.arange(180, 260)}
    probs = {x: p[i] for x, i in parts.items()}
    ys = {x: labels[i] for x, i in parts.items()}
    base_groups = np.column_stack([np.ones(n), (np.arange(n) % 2 == 0).astype(float), (np.arange(n) % 3 == 0).astype(float)])
    groups = {x: base_groups[i] for x, i in parts.items()}
    for name in hc.METHODS:
        q, state = hc.run_postprocessing(name, tree, probs, ys, groups, max_iter=5, threshold=0.0)
        assert q.shape == (len(parts["test"]), k)
        assert np.all(np.isfinite(q))
        assert q.min() >= -1e-14
        assert np.max(np.abs(q.sum(axis=1) - 1.0)) < 1e-10
        metrics = hc.evaluate(tree, q, ys["test"], groups["test"])
        assert all(math.isfinite(v) for v in metrics.values())
        assert metrics["C-UC"] + 1e-12 >= metrics["UC"]
        assert metrics["C-HUC"] + 1e-12 >= metrics["HUC"]
        print(name, metrics["UC"], metrics["HUC"], metrics["Accuracy"])


if __name__ == "__main__":
    test_intervals()
    test_projection()
    test_all_methods()
    print("ALL_TESTS_PASSED")
