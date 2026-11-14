"""v1.3 E29 单测：分组共形小样本平滑。"""
import numpy as np
import pytest

from solar_diag.conformal import conformal_quantile
from solar_diag.smoothing import (
    bootstrap_q_variance,
    effective_sample_size,
    group_sizes,
    neighbor_weights,
    oracle_group_quantiles,
    pooled_quantiles,
    shrink_quantiles,
    smoothed_quantiles,
    weighted_conformal_quantile,
    weighted_quantile,
)

ORDER = ["g0", "g1", "g2", "g3"]


def test_effective_sample_size_uniform():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)


def test_effective_sample_size_concentrated():
    w = np.zeros(50)
    w[0] = 1.0
    assert effective_sample_size(w) == pytest.approx(1.0)


def test_weighted_quantile_matches_numpy_when_uniform():
    v = np.array([0.0, 1.0, 2.0, 3.0, 8.0])
    for level in (0.0, 0.1, 0.25, 0.5, 0.8, 1.0):
        assert weighted_quantile(v, np.ones(5), level) == pytest.approx(
            float(np.quantile(v, level)))


def test_weighted_quantile_respects_weights():
    v = np.array([0.0, 10.0])
    # 权重和视为等价样本数，故用整数计数语义（9 vs 1）
    assert weighted_quantile(v, np.array([9.0, 1.0]), 0.5) == pytest.approx(0.0)
    assert weighted_quantile(v, np.array([1.0, 9.0]), 0.5) > 0.0
    assert (weighted_quantile(v, np.array([9.0, 1.0]), 0.5)
            < weighted_quantile(v, np.array([1.0, 9.0]), 0.5))


def test_weighted_quantile_empty():
    assert weighted_quantile(np.array([]), np.array([]), 0.8) == 0.0


def test_weighted_conformal_quantile_uniform_matches_conformal():
    s = np.arange(30, dtype=float)
    assert weighted_conformal_quantile(s, np.ones(30), 0.8) == pytest.approx(
        conformal_quantile(s, 0.8))


def test_neighbor_weights_decay_with_distance():
    w = neighbor_weights(ORDER, "g1", bandwidth=1.0)
    assert w["g1"] == 1.0
    assert w["g0"] == pytest.approx(w["g2"])
    assert w["g0"] > w["g3"] > 0.0


def test_neighbor_weights_unknown_center():
    assert neighbor_weights(ORDER, "zz", 1.0) == {"zz": 1.0}


def test_group_sizes():
    g = np.array(["A", "A", "B"])
    assert group_sizes(g) == {"A": 2, "B": 1}


def test_shrink_small_group_pulled_to_global():
    q = {"A": 10.0, "B": -2.0}
    n = {"A": 2, "B": 500}
    out = shrink_quantiles(q, n, q_global=0.0, k=20.0)
    assert out["A"] < out["B"] or abs(out["A"]) < abs(q["A"])
    assert abs(out["A"]) < 2.0                     # 小样本被大幅拉回 0
    assert out["B"] == pytest.approx(q["B"], rel=0.05)  # 大样本基本不动


def test_shrink_zero_k_is_identity():
    q = {"A": 3.0}
    assert shrink_quantiles(q, {"A": 1}, 99.0, k=0.0)["A"] == pytest.approx(3.0)


def _toy_scores(seed=0):
    """4 个有序桶，g3 是稀疏强光桶（n=12）、分数整体偏大（需要更大 q̂）。"""
    rng = np.random.default_rng(seed)
    parts_s, parts_g = [], []
    for i, name in enumerate(ORDER):
        n = 300 if i < 3 else 12
        parts_s.append(rng.normal(loc=0.2 * i, scale=1.0, size=n))
        parts_g.append(np.full(n, name, dtype=object))
    return np.concatenate(parts_s), np.concatenate(parts_g)


def test_pooled_quantiles_degenerate_to_raw_when_bandwidth_tiny():
    s, g = _toy_scores()
    q_raw, _ = smoothed_quantiles(s, g, 0.8, order=ORDER, mode="raw")
    q_pool = pooled_quantiles(s, g, ORDER, 0.8, bandwidth=1e-6)
    for name in ORDER:
        assert q_pool[name] == pytest.approx(q_raw[name], abs=1e-6)


def test_smoothed_quantiles_min_n_fallback():
    s, g = _toy_scores()
    q_map, diag = smoothed_quantiles(s, g, 0.8, order=ORDER, mode="min_n", n_min=30)
    assert q_map["g3"] == pytest.approx(diag["q_global"])
    assert diag["n_map"]["g3"] == 12
    assert q_map["g0"] == pytest.approx(diag["q_raw"]["g0"])


def test_smoothed_quantiles_global_constant():
    s, g = _toy_scores()
    q_map, diag = smoothed_quantiles(s, g, 0.8, order=ORDER, mode="global")
    assert all(v == pytest.approx(diag["q_global"]) for v in q_map.values())


def test_smoothed_quantiles_unknown_mode():
    with pytest.raises(ValueError, match="未知 mode"):
        smoothed_quantiles(np.ones(3), np.array(["a"] * 3), 0.8, mode="bogus")


def test_smoothed_quantiles_length_mismatch():
    with pytest.raises(ValueError, match="长度不一致"):
        smoothed_quantiles(np.ones(3), np.array(["a"] * 2), 0.8)


def test_all_modes_return_full_order():
    s, g = _toy_scores()
    for mode in ("global", "raw", "min_n", "shrink", "pool", "pool_shrink"):
        q_map, _ = smoothed_quantiles(s, g, 0.8, order=ORDER, mode=mode)
        assert set(q_map) == set(ORDER)


def test_bootstrap_smoothing_reduces_sparse_group_variance():
    """核心：稀疏桶 q̂ 的抽样标准差，平滑后应显著低于独立校准。"""
    s, g = _toy_scores()
    common = dict(order=ORDER, n_boot=150, seed=0, n_min=30, k=20.0, bandwidth=1.0)
    raw = bootstrap_q_variance(s, g, 0.8, mode="raw", **common)
    shr = bootstrap_q_variance(s, g, 0.8, mode="shrink", **common)
    pool = bootstrap_q_variance(s, g, 0.8, mode="pool", **common)
    assert raw["g3"]["std"] > 0.05
    assert shr["g3"]["std"] < raw["g3"]["std"]
    assert pool["g3"]["std"] < raw["g3"]["std"]


def test_oracle_group_quantiles_uses_eval_scores():
    y = np.array([0.0, 10.0, 0.0, 10.0])
    lo = np.zeros(4)
    hi = np.ones(4)
    g = np.array(["A", "A", "B", "B"])
    q = oracle_group_quantiles(y, lo, hi, g, 0.5, order=["A", "B"])
    # 两组分数均为 [0, 9]（界内下沿贴边得 0，上溢出 9），n=2 → level=1 → q=9
    assert q["A"] == pytest.approx(9.0)
    assert q["B"] == pytest.approx(9.0)
