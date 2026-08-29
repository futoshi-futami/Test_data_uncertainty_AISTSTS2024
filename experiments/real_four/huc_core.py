from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

EPS = 1.0e-12


def simplex_projection(z: np.ndarray) -> np.ndarray:
    """Euclidean projection of rows onto the probability simplex."""
    x = np.asarray(z, dtype=float)
    one = x.ndim == 1
    if one:
        x = x[None, :]
    u = np.sort(x, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, x.shape[1] + 1)
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(x.shape[0]), rho] / (rho + 1)
    out = np.maximum(x - theta[:, None], 0.0)
    return out[0] if one else out


def normalize_probs(p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=float)
    q = np.clip(q, 0.0, np.inf)
    s = q.sum(axis=1, keepdims=True)
    bad = s[:, 0] <= 0
    if np.any(bad):
        q[bad] = 1.0 / q.shape[1]
        s = q.sum(axis=1, keepdims=True)
    return simplex_projection(q / s)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    one = np.eye(probs.shape[1])[np.asarray(labels, dtype=int)]
    return float(np.mean(np.sum((probs - one) ** 2, axis=1)))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), np.asarray(labels, dtype=int)], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def utility_values(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs)
    pred = np.argmax(p, axis=1)
    out = np.zeros_like(p, dtype=float)
    out[np.arange(len(p)), pred] = 1.0
    return out


def utility_score(probs: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(probs, dtype=float), axis=1)


def utility_residual(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = np.argmax(probs, axis=1)
    observed = (pred == np.asarray(labels, dtype=int)).astype(float)
    return observed - probs[np.arange(len(probs)), pred]


@dataclass(frozen=True)
class BinaryTree:
    left: tuple[tuple[int, ...], ...]
    right: tuple[tuple[int, ...], ...]
    names: tuple[str, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.left)

    def contribution(self, probs: np.ndarray, labels: np.ndarray, node: int) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=int)
        L = np.asarray(self.left[node], dtype=int)
        R = np.asarray(self.right[node], dtype=int)
        mass_l = p[:, L].sum(axis=1)
        mass_r = p[:, R].sum(axis=1)
        mass_v = mass_l + mass_r
        q_r = np.divide(mass_r, mass_v, out=np.zeros_like(mass_r), where=mass_v > EPS)
        u = utility_values(p)
        mean_l = np.divide((p[:, L] * u[:, L]).sum(axis=1), mass_l, out=np.zeros_like(mass_l), where=mass_l > EPS)
        mean_r = np.divide((p[:, R] * u[:, R]).sum(axis=1), mass_r, out=np.zeros_like(mass_r), where=mass_r > EPS)
        gap = mean_r - mean_l
        in_l = np.isin(y, L)
        in_r = np.isin(y, R)
        in_v = in_l | in_r
        branch = in_r.astype(float) - q_r * in_v.astype(float)
        return gap * branch

    def branch_direction(self, probs: np.ndarray, node: int, toward_right: float) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        L = np.asarray(self.left[node], dtype=int)
        R = np.asarray(self.right[node], dtype=int)
        mass_l = p[:, L].sum(axis=1)
        mass_r = p[:, R].sum(axis=1)
        d = np.zeros_like(p)
        lshare = np.divide(p[:, L], mass_l[:, None], out=np.full((len(p), len(L)), 1.0 / len(L)), where=mass_l[:, None] > EPS)
        rshare = np.divide(p[:, R], mass_r[:, None], out=np.full((len(p), len(R)), 1.0 / len(R)), where=mass_r[:, None] > EPS)
        d[:, L] -= toward_right * lshare
        d[:, R] += toward_right * rshare
        return d


def build_tree(paths: Sequence[Sequence[str]]) -> BinaryTree:
    class Node:
        def __init__(self, name: str):
            self.name = name
            self.children: dict[str, Node] = {}
            self.leaves: list[int] = []

    root = Node("root")
    for leaf, path in enumerate(paths):
        node = root
        for name in path:
            node = node.children.setdefault(str(name), Node(str(name)))
        node.leaves.append(leaf)

    memo: dict[int, tuple[int, ...]] = {}

    def leaves(node: Node) -> tuple[int, ...]:
        key = id(node)
        if key not in memo:
            vals = list(node.leaves)
            for child in node.children.values():
                vals.extend(leaves(child))
            memo[key] = tuple(sorted(set(vals)))
        return memo[key]

    left: list[tuple[int, ...]] = []
    right: list[tuple[int, ...]] = []
    names: list[str] = []

    def split_children(children: list[Node], prefix: str) -> None:
        if len(children) <= 1:
            return
        mid = len(children) // 2
        a, b = children[:mid], children[mid:]
        L = tuple(sorted(x for child in a for x in leaves(child)))
        R = tuple(sorted(x for child in b for x in leaves(child)))
        if L and R:
            left.append(L)
            right.append(R)
            names.append(prefix + ":" + "+".join(x.name for x in a) + "|" + "+".join(x.name for x in b))
        split_children(a, prefix + "L")
        split_children(b, prefix + "R")

    def visit(node: Node, prefix: str) -> None:
        children = sorted(node.children.values(), key=lambda x: x.name)
        split_children(children, prefix)
        for child in children:
            visit(child, prefix + "/" + child.name)

    visit(root, "root")
    if not left:
        raise RuntimeError("No internal tree node was constructed")
    return BinaryTree(tuple(left), tuple(right), tuple(names))


@dataclass
class IntervalResult:
    value: float
    lo: float
    hi: float
    sign: float
    count: int


def worst_interval(scores: np.ndarray, residuals: np.ndarray, min_count: int = 1) -> IntervalResult:
    """Exact maximum absolute contiguous interval after preserving score ties."""
    s = np.asarray(scores, dtype=float)
    r = np.asarray(residuals, dtype=float)
    n = len(s)
    if n == 0:
        return IntervalResult(0.0, 0.0, 0.0, 1.0, 0)
    order = np.argsort(s, kind="mergesort")
    ss, rr = s[order], r[order]
    starts = np.r_[0, 1 + np.flatnonzero(ss[1:] != ss[:-1])]
    ends = np.r_[starts[1:], n]
    vals = np.add.reduceat(rr, starts)
    counts = ends - starts
    uniq = ss[starts]

    best_abs = -1.0
    best = (0, 0, 0.0)
    prefix_sum = np.r_[0.0, np.cumsum(vals)]
    prefix_count = np.r_[0, np.cumsum(counts)]
    for end in range(1, len(vals) + 1):
        eligible = np.flatnonzero(prefix_count[end] - prefix_count[:end] >= min_count)
        if len(eligible) == 0:
            continue
        seg = prefix_sum[end] - prefix_sum[eligible]
        for idx in (int(np.argmax(seg)), int(np.argmin(seg))):
            start = int(eligible[idx])
            total = float(seg[idx])
            if abs(total) > best_abs + 1.0e-15:
                best_abs = abs(total)
                best = (start, end - 1, total)
    if best_abs < 0:
        return IntervalResult(0.0, float(uniq[0]), float(uniq[-1]), 1.0, n)
    a, b, total = best
    return IntervalResult(abs(total) / n, float(uniq[a]), float(uniq[b]), 1.0 if total >= 0 else -1.0, int(counts[a:b + 1].sum()))


def discrepancy(kind: str, tree: BinaryTree, probs: np.ndarray, labels: np.ndarray, groups: np.ndarray, min_count: int = 1) -> tuple[float, dict[str, Any]]:
    score = utility_score(probs)
    use_c = kind.startswith("C-")
    base = kind[2:] if use_c else kind
    c_indices = range(groups.shape[1]) if use_c else [0]
    best = (-1.0, {})
    if base == "UC":
        residual = utility_residual(probs, labels)
        for c in c_indices:
            ans = worst_interval(score, groups[:, c] * residual, min_count)
            if ans.value > best[0]:
                best = (ans.value, {"c": int(c), "node": None, **ans.__dict__})
    elif base == "HUC":
        for node in range(tree.n_nodes):
            contribution = tree.contribution(probs, labels, node)
            for c in c_indices:
                ans = worst_interval(score, groups[:, c] * contribution, min_count)
                if ans.value > best[0]:
                    best = (ans.value, {"c": int(c), "node": int(node), **ans.__dict__})
    else:
        raise KeyError(kind)
    return max(0.0, float(best[0])), best[1]


def evaluate(tree: BinaryTree, probs: np.ndarray, labels: np.ndarray, groups: np.ndarray, min_count: int = 1) -> dict[str, float]:
    p = normalize_probs(probs)
    y = np.asarray(labels, dtype=int)
    out = {}
    for name in ("UC", "HUC", "C-UC", "C-HUC"):
        out[name] = discrepancy(name, tree, p, y, groups, min_count)[0]
    pred = np.argmax(p, axis=1)
    out["Accuracy"] = float(np.mean(pred == y))
    try:
        out["AUC"] = float(roc_auc_score(y, p, labels=np.arange(p.shape[1]), multi_class="ovr", average="macro"))
    except Exception:
        out["AUC"] = float("nan")
    out["Brier"] = brier(p, y)
    out["NLL"] = nll(p, y)
    return out


class Parametric:
    name = "Parametric"
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Parametric":
        raise NotImplementedError
    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def state(self) -> dict[str, Any]:
        return {}


class Raw(Parametric):
    name = "Raw"
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Raw":
        return self
    def transform(self, probs: np.ndarray) -> np.ndarray:
        return normalize_probs(probs)


class Temperature(Parametric):
    name = "Temperature"
    def __init__(self):
        self.temperature = 1.0
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Temperature":
        logp = np.log(np.clip(probs, EPS, 1.0))
        y = np.asarray(labels, dtype=int)
        def objective(log_t: float) -> float:
            t = float(np.exp(log_t))
            z = logp / t
            return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
        res = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded", options={"maxiter": 500})
        if np.isfinite(res.x):
            self.temperature = float(np.exp(res.x))
        return self
    def transform(self, probs: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(probs, EPS, 1.0)) / self.temperature
        z -= logsumexp(z, axis=1)[:, None]
        return np.exp(z)
    def state(self) -> dict[str, Any]:
        return {"temperature": self.temperature}


class Vector(Parametric):
    name = "Vector"
    def __init__(self, l2: float = 1.0e-4):
        self.l2 = l2
        self.scale: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.success = False
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Vector":
        x = np.log(np.clip(probs, EPS, 1.0))
        y = np.asarray(labels, dtype=int)
        k = x.shape[1]
        init = np.r_[np.ones(k), np.zeros(k - 1)]
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a = theta[:k]
            b = np.r_[theta[k:], 0.0]
            return a, b
        def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = unpack(theta)
            z = x * a + b
            q = np.exp(z - logsumexp(z, axis=1)[:, None])
            loss = -np.mean(np.log(np.clip(q[np.arange(len(y)), y], EPS, 1.0))) + self.l2 * float(np.sum((a - 1.0) ** 2) + np.sum(b ** 2))
            one = np.eye(k)[y]
            dz = (q - one) / len(y)
            ga = np.sum(dz * x, axis=0) + 2 * self.l2 * (a - 1.0)
            gb = np.sum(dz[:, :-1], axis=0) + 2 * self.l2 * b[:-1]
            return float(loss), np.r_[ga, gb]
        res = minimize(lambda t: fun(t)[0], init, jac=lambda t: fun(t)[1], method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 20000})
        theta = res.x if np.all(np.isfinite(res.x)) else init
        self.scale, self.bias = unpack(theta)
        self.success = bool(res.success)
        return self
    def transform(self, probs: np.ndarray) -> np.ndarray:
        assert self.scale is not None and self.bias is not None
        z = np.log(np.clip(probs, EPS, 1.0)) * self.scale + self.bias
        z -= logsumexp(z, axis=1)[:, None]
        return np.exp(z)
    def state(self) -> dict[str, Any]:
        return {"scale": self.scale, "bias": self.bias, "optimizer_success": self.success}


class Dirichlet(Parametric):
    name = "Dirichlet"
    def __init__(self, l2: float = 1.0e-3):
        self.l2 = l2
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.success = False
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Dirichlet":
        x = np.log(np.clip(probs, EPS, 1.0))
        y = np.asarray(labels, dtype=int)
        n, k = x.shape
        init_W = np.eye(k)
        init = np.r_[init_W.ravel(), np.zeros(k - 1)]
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            W = theta[:k * k].reshape(k, k)
            b = np.r_[theta[k * k:], 0.0]
            return W, b
        def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
            W, b = unpack(theta)
            z = x @ W.T + b
            q = np.exp(z - logsumexp(z, axis=1)[:, None])
            loss = -np.mean(np.log(np.clip(q[np.arange(n), y], EPS, 1.0))) + self.l2 * float(np.sum((W - np.eye(k)) ** 2) + np.sum(b ** 2))
            one = np.eye(k)[y]
            dz = (q - one) / n
            gW = dz.T @ x + 2 * self.l2 * (W - np.eye(k))
            gb = np.sum(dz[:, :-1], axis=0) + 2 * self.l2 * b[:-1]
            return float(loss), np.r_[gW.ravel(), gb]
        res = minimize(lambda t: fun(t)[0], init, jac=lambda t: fun(t)[1], method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 30000})
        theta = res.x if np.all(np.isfinite(res.x)) else init
        self.W, self.b = unpack(theta)
        self.success = bool(res.success)
        return self
    def transform(self, probs: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None
        z = np.log(np.clip(probs, EPS, 1.0)) @ self.W.T + self.b
        z -= logsumexp(z, axis=1)[:, None]
        return np.exp(z)
    def state(self) -> dict[str, Any]:
        return {"W": self.W, "b": self.b, "optimizer_success": self.success}


def make_parametric(name: str) -> Parametric:
    if name == "Raw":
        return Raw()
    if name == "Temperature":
        return Temperature()
    if name == "Vector":
        return Vector()
    if name == "Dirichlet":
        return Dirichlet()
    raise KeyError(name)


@dataclass
class Update:
    kind: str
    c: int
    node: int | None
    lo: float
    hi: float
    sign: float
    step: float
    value: float


class Iterative:
    def __init__(self, kind: str, tree: BinaryTree, threshold: float = 5.0e-4, max_iter: int = 25, initial_step: float = 0.5, shrink: float = 0.7, armijo: float = 1.0e-4, min_step: float = 1.0e-8, min_count: int = 1):
        self.name = kind
        self.kind = kind
        self.tree = tree
        self.threshold = threshold
        self.max_iter = max_iter
        self.initial_step = initial_step
        self.shrink = shrink
        self.armijo = armijo
        self.min_step = min_step
        self.min_count = min_count
        self.updates: list[Update] = []
        self.selected_round = 0
        self.attempted_rounds = 0
        self.stop_reason = "not_fitted"
        self.validation_history: list[dict[str, float]] = []

    def _direction(self, probs: np.ndarray, update: Update, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score = utility_score(probs)
        active = (score >= update.lo - 1.0e-15) & (score <= update.hi + 1.0e-15) & (groups[:, update.c] > 0)
        d = np.zeros_like(probs)
        if not np.any(active):
            return d, active
        p = probs[active]
        if update.node is None:
            pred = np.argmax(p, axis=1)
            k = p.shape[1]
            local = np.full_like(p, -1.0 / max(k - 1, 1))
            local[np.arange(len(p)), pred] = 1.0
            local *= update.sign
        else:
            node = update.node
            L = np.asarray(self.tree.left[node], dtype=int)
            R = np.asarray(self.tree.right[node], dtype=int)
            u = utility_values(p)
            ml = p[:, L].sum(axis=1)
            mr = p[:, R].sum(axis=1)
            gl = np.divide((p[:, L] * u[:, L]).sum(axis=1), ml, out=np.zeros(len(p)), where=ml > EPS)
            gr = np.divide((p[:, R] * u[:, R]).sum(axis=1), mr, out=np.zeros(len(p)), where=mr > EPS)
            gap_sign = np.sign(gr - gl)
            gap_sign[gap_sign == 0] = 1.0
            local = self.tree.branch_direction(p, node, update.sign * gap_sign)
        d[active] = local
        return d, active

    def _apply_one(self, probs: np.ndarray, groups: np.ndarray, update: Update) -> np.ndarray:
        d, active = self._direction(probs, update, groups)
        if not np.any(active):
            return probs.copy()
        q = probs.copy()
        q[active] = simplex_projection(q[active] + update.step * d[active])
        return q

    def apply(self, probs: np.ndarray, groups: np.ndarray, rounds: int | None = None) -> np.ndarray:
        q = normalize_probs(probs)
        upto = self.selected_round if rounds is None else rounds
        for update in self.updates[:upto]:
            q = self._apply_one(q, groups, update)
        return q

    def fit(self, probs: np.ndarray, labels: np.ndarray, groups: np.ndarray, validation_probs: np.ndarray | None = None, validation_labels: np.ndarray | None = None, validation_groups: np.ndarray | None = None) -> "Iterative":
        q = normalize_probs(probs)
        y = np.asarray(labels, dtype=int)
        self.updates = []
        self.stop_reason = "max_iter"
        for t in range(self.max_iter):
            value, info = discrepancy(self.kind, self.tree, q, y, groups, self.min_count)
            self.attempted_rounds = t + 1
            if value <= self.threshold:
                self.stop_reason = "threshold"
                break
            proto = Update(self.kind, int(info["c"]), info["node"], float(info["lo"]), float(info["hi"]), float(info["sign"]), 0.0, float(value))
            direction, active = self._direction(q, proto, groups)
            if not np.any(active):
                self.stop_reason = "empty_update"
                break
            old = brier(q, y)
            norm = float(np.mean(np.sum(direction[active] ** 2, axis=1)))
            step = self.initial_step
            accepted = None
            while step >= self.min_step:
                candidate = q.copy()
                candidate[active] = simplex_projection(candidate[active] + step * direction[active])
                new = brier(candidate, y)
                if new <= old - self.armijo * step * max(norm, EPS):
                    accepted = candidate
                    break
                step *= self.shrink
            if accepted is None:
                self.stop_reason = "armijo"
                break
            update = Update(proto.kind, proto.c, proto.node, proto.lo, proto.hi, proto.sign, float(step), proto.value)
            self.updates.append(update)
            q = accepted
        if validation_probs is None:
            self.selected_round = len(self.updates)
            return self
        assert validation_labels is not None and validation_groups is not None
        best_round = 0
        best_key = (float("inf"), float("inf"))
        self.validation_history = []
        for r in range(len(self.updates) + 1):
            vq = self.apply(validation_probs, validation_groups, rounds=r)
            val = discrepancy(self.kind, self.tree, vq, validation_labels, validation_groups, self.min_count)[0]
            br = brier(vq, validation_labels)
            self.validation_history.append({"round": r, "value": val, "brier": br})
            key = (val, br)
            if key < best_key:
                best_key = key
                best_round = r
        self.selected_round = best_round
        return self

    def state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selected_round": self.selected_round,
            "attempted_rounds": self.attempted_rounds,
            "stop_reason": self.stop_reason,
            "updates": [u.__dict__ for u in self.updates],
            "validation_history": self.validation_history,
        }


def run_postprocessing(name: str, tree: BinaryTree, probs: dict[str, np.ndarray], labels: dict[str, np.ndarray], groups: dict[str, np.ndarray], max_iter: int = 25, threshold: float = 5.0e-4) -> tuple[np.ndarray, dict[str, Any]]:
    if "→" not in name:
        if name in {"Raw", "Temperature", "Vector", "Dirichlet"}:
            method = make_parametric(name)
            method.fit(probs["stage1"], labels["stage1"])
            return method.transform(probs["test"]), {"name": name, **method.state()}
        method = Iterative(name, tree, threshold=threshold, max_iter=max_iter)
        method.fit(probs["stage2"], labels["stage2"], groups["stage2"], probs["validation"], labels["validation"], groups["validation"])
        return method.apply(probs["test"], groups["test"]), method.state()
    first_name, second_name = name.split("→", 1)
    first = make_parametric(first_name)
    first.fit(probs["stage1"], labels["stage1"])
    transformed = {key: first.transform(value) for key, value in probs.items()}
    second = Iterative(second_name, tree, threshold=threshold, max_iter=max_iter)
    second.fit(transformed["stage2"], labels["stage2"], groups["stage2"], transformed["validation"], labels["validation"], groups["validation"])
    out = second.apply(transformed["test"], groups["test"])
    return out, {"name": name, "first": first.state(), "second": second.state()}


METHODS = [
    "Raw", "Temperature", "Vector", "Dirichlet", "UC", "HUC", "C-UC", "C-HUC",
    "Temperature→C-UC", "Temperature→C-HUC", "Vector→C-UC", "Vector→C-HUC",
    "Dirichlet→C-UC", "Dirichlet→C-HUC",
]
