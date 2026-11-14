"""分组（Mondrian）共形的小样本平滑：向全局 / 相邻桶借强度。

v1.2 发现 Mondrian-辐照桶把覆盖极差砍半，但稀疏桶（P1 强光桶校准 n≈22）
的 q̂ 噪声大，测试覆盖直接过冲到 1.0。本模块提供三种"借强度"手段，
全部为纯函数：

  1. min_n    硬阈值：组内样本 < n_min 时整组回退到全局 q̂
  2. shrink   软收缩：q_g = w_g·q̂_g + (1−w_g)·q_global，w_g = n_g/(n_g+k)
  3. pool     邻桶池化：把有序相邻桶的校准分数按高斯距离核加权后合并取分位数，
              有效样本量用 Kish n_eff = (Σw)²/Σw² 做有限样本修正
  4. pool_shrink  两者叠加

另提供诊断工具：
  - bootstrap_q_variance  组内 bootstrap 量化 q̂ 的抽样标准差
  - oracle_group_quantiles 用评估集自身分数算 q̂（不可部署，仅作"分布已知"上限）
"""
from __future__ import annotations

import numpy as np

from .conformal import conformal_quantile, cqr_scores


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish 有效样本量：加权样本在分位数估计上的等价独立样本数。"""
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0.0:
        return 0.0
    return float(s * s / np.sum(w * w))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """加权分位数（type-7 / linear 推广）：权重视为样本占据的概率质量宽度。

    坐标 pos = level·(Σw−1)+1，在累积权重区间内线性插值。等权时与
    np.quantile(method="linear") 精确一致（保证 bandwidth→0 时邻桶池化
    能精确退化为组内独立校准）。
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.size == 0 or v.size != w.size:
        return 0.0
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    pos = float(np.clip(level, 0.0, 1.0)) * (total - 1.0) + 1.0
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, pos, side="left"))
    if idx >= v.size:
        return float(v[-1])
    if idx == 0:
        return float(v[0])
    step = w[idx]
    if step <= 0.0:
        return float(v[idx - 1])
    frac = float(np.clip((pos - cw[idx - 1]) / step, 0.0, 1.0))
    return float(v[idx - 1] + frac * (v[idx] - v[idx - 1]))


def weighted_conformal_quantile(values: np.ndarray, weights: np.ndarray,
                                nominal: float) -> float:
    """加权版有限样本分位数。

    level 用 Kish 有效样本量 n_eff 修正，并把权重整体缩放到 Σw = n_eff，
    使 type-7 坐标与 level 处于同一量纲；等权时 n_eff=Σw=n，精确退化为
    conformal_quantile。
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    n_eff = effective_sample_size(w)
    s = float(w.sum())
    if n_eff <= 0.0 or s <= 0.0:
        return 0.0
    level = min(1.0, np.ceil((n_eff + 1) * nominal) / n_eff)
    w_scaled = w * (n_eff / s)
    return weighted_quantile(values, w_scaled, level)


def neighbor_weights(order: list[str], center: str, bandwidth: float = 1.0) -> dict[str, float]:
    """有序桶的高斯距离核：本桶权重 1，相邻桶 exp(-0.5 d²/b²)。"""
    if center not in order:
        return {center: 1.0}
    i = order.index(center)
    b = max(float(bandwidth), 1e-8)
    return {g: float(np.exp(-0.5 * (abs(j - i) / b) ** 2)) for j, g in enumerate(order)}


def group_sizes(groups: np.ndarray) -> dict[str, int]:
    g = np.asarray(groups).astype(str)
    return {k: int((g == k).sum()) for k in np.unique(g)}


def shrink_quantiles(q_map: dict[str, float], n_map: dict[str, int],
                     q_global: float, k: float = 20.0) -> dict[str, float]:
    """James–Stein 式软收缩：小样本组更信任全局分位数。"""
    k = max(float(k), 0.0)
    return {
        g: (n / (n + k)) * q_map[g] + (k / (n + k)) * q_global
        for g, n in n_map.items() if g in q_map
    }


def pooled_quantiles(scores: np.ndarray, groups: np.ndarray, order: list[str],
                     nominal: float, bandwidth: float = 1.0) -> dict[str, float]:
    """邻桶池化：每桶用自身分数（权 1）+ 相邻桶分数（核权 <1）联合取 q̂。"""
    s = np.asarray(scores, dtype=np.float64).ravel()
    g = np.asarray(groups).astype(str).ravel()
    out: dict[str, float] = {}
    for center in order:
        wmap = neighbor_weights(order, center, bandwidth)
        w = np.zeros(s.size, dtype=np.float64)
        for name, wh in wmap.items():
            w[g == name] = wh
        m = w > 0
        out[center] = weighted_conformal_quantile(s[m], w[m], nominal) if m.any() else 0.0
    return out


def smoothed_quantiles(
    scores: np.ndarray,
    groups: np.ndarray,
    nominal: float,
    order: list[str] | None = None,
    mode: str = "raw",
    n_min: int = 30,
    k: float = 20.0,
    bandwidth: float = 1.0,
) -> tuple[dict[str, float], dict]:
    """统一入口：按 mode 产出各组 q̂，并返回诊断字典（n_map / q_global 等）。

    mode ∈ {global, raw, min_n, shrink, pool, pool_shrink}
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    g = np.asarray(groups).astype(str).ravel()
    if s.size != g.size:
        raise ValueError("分数与分组标签长度不一致")
    if order is None:
        order = sorted(np.unique(g).tolist())
    n_map = group_sizes(g)
    q_global = conformal_quantile(s, nominal)
    q_raw = {name: conformal_quantile(s[g == name], nominal) for name in order}

    if mode == "global":
        q_map = {name: q_global for name in order}
    elif mode == "raw":
        q_map = q_raw
    elif mode == "min_n":
        q_map = {name: (q_raw[name] if n_map.get(name, 0) >= n_min else q_global)
                 for name in order}
    elif mode == "shrink":
        q_map = shrink_quantiles(q_raw, n_map, q_global, k)
    elif mode == "pool":
        q_map = pooled_quantiles(s, g, order, nominal, bandwidth)
    elif mode == "pool_shrink":
        q_pool = pooled_quantiles(s, g, order, nominal, bandwidth)
        # 池化后有效样本量增大，收缩强度按 n_eff 折算
        n_eff_map: dict[str, int] = {}
        for center in order:
            wmap = neighbor_weights(order, center, bandwidth)
            w = np.zeros(s.size, dtype=np.float64)
            for name, wh in wmap.items():
                w[g == name] = wh
            n_eff_map[center] = int(round(effective_sample_size(w[w > 0])))
        q_map = shrink_quantiles(q_pool, n_eff_map, q_global, k)
    else:
        raise ValueError(f"未知 mode: {mode}")

    diag = {"q_global": q_global, "q_raw": q_raw, "n_map": n_map, "order": order}
    return q_map, diag


def bootstrap_q_variance(scores: np.ndarray, groups: np.ndarray, nominal: float,
                         mode: str = "raw", order: list[str] | None = None,
                         n_boot: int = 300, seed: int = 0,
                         **kwargs) -> dict[str, dict]:
    """组内 bootstrap 重采样，量化各桶 q̂ 的抽样标准差 / 均值。

    每组独立有放回抽样（保持组样本量不变），重跑 smoothed_quantiles。
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    g = np.asarray(groups).astype(str).ravel()
    if order is None:
        order = sorted(np.unique(g).tolist())
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in order}
    for _ in range(n_boot):
        sb, gb = [], []
        for name in order:
            idx = np.flatnonzero(g == name)
            if idx.size == 0:
                continue
            pick = rng.choice(idx, size=idx.size, replace=True)
            sb.append(s[pick])
            gb.append(np.full(idx.size, name, dtype=object))
        if not sb:
            continue
        q_map, _ = smoothed_quantiles(np.concatenate(sb), np.concatenate(gb),
                                      nominal, order=order, mode=mode, **kwargs)
        for name in order:
            draws[name].append(q_map[name])
    return {
        name: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_boot": len(v)}
        for name, v in draws.items() if v
    }


def oracle_group_quantiles(y_te: np.ndarray, lo_te: np.ndarray, hi_te: np.ndarray,
                           groups_te: np.ndarray, nominal: float,
                           order: list[str] | None = None) -> dict[str, float]:
    """诊断用"oracle"：拿评估集自身分数算组内 q̂（不可部署，仅作分布已知上限）。"""
    s = cqr_scores(y_te, lo_te, hi_te)
    g = np.asarray(groups_te).astype(str).ravel()
    if order is None:
        order = sorted(np.unique(g).tolist())
    return {name: conformal_quantile(s[g == name], nominal) for name in order}
