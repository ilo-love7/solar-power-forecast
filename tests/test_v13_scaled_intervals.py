"""v1.3 E30 单测：pinball + 共形二段缩放。"""
import numpy as np
import pytest

from solar_diag.conformal import conformal_quantile, cqr_scores
from solar_diag.evaluation import interval_coverage, pinball_loss
from solar_diag.scaled_intervals import (
    apply_scale,
    cqr_add_interval,
    group_scale_factors,
    normalized_residuals,
    pinball_sharpness,
    scale_factor,
    select_quantile_pair,
    two_stage_interval,
)


def test_normalized_residuals_sign_and_side():
    y = np.array([0.0, 3.0, -3.0])
    lo = np.full(3, -1.0)
    hi = np.full(3, 1.0)
    r = normalized_residuals(y, lo, hi)
    # r = (y − c)/h_side，|r|>1 即出界；c=0、左右半宽均为 1
    assert r[0] == 0.0
    assert r[1] == pytest.approx(3.0)
    assert r[2] == pytest.approx(-3.0)


def test_normalized_residuals_degenerate_interval_no_nan():
    r = normalized_residuals(np.array([5.0]), np.array([0.0]), np.array([0.0]))
    assert np.isfinite(r).all()


def test_apply_scale_identity_at_one():
    lo = np.array([-1.0, -10.0, 2.0])
    hi = np.array([1.0, 10.0, 8.0])
    lo2, hi2 = apply_scale(lo, hi, 1.0)
    np.testing.assert_allclose(lo2, lo)
    np.testing.assert_allclose(hi2, hi)


def test_apply_scale_preserves_center_and_width_ratio():
    """核心性质：乘性缩放保持逐点宽度比例（异方差结构），加法偏移不保持。"""
    lo = np.array([-1.0, -10.0])
    hi = np.array([1.0, 10.0])
    lo2, hi2 = apply_scale(lo, hi, 2.0)
    np.testing.assert_allclose((lo2 + hi2) / 2, (lo + hi) / 2)
    ratio_before = (hi - lo)[1] / (hi - lo)[0]
    ratio_after = (hi2 - lo2)[1] / (hi2 - lo2)[0]
    assert ratio_after == pytest.approx(ratio_before)
    q = 1.0
    w_add = (hi + q) - (lo - q)
    assert (w_add[1] / w_add[0]) < ratio_before   # 加成式把比例压小


def test_apply_scale_clamps_crossed_quantiles():
    """独立训练的 q10/q90 偶有交叉（lo>hi）；λ=1 时宽度被夹到 2·eps，不产生 NaN。"""
    lo = np.array([1.0, -1.0])
    hi = np.array([0.5, 1.0])
    lo2, hi2 = apply_scale(lo, hi, 1.0)
    assert np.isfinite(lo2).all() and np.isfinite(hi2).all()
    assert (hi2 - lo2)[0] >= 0.0
    np.testing.assert_allclose(lo2[1], -1.0)
    np.testing.assert_allclose(hi2[1], 1.0)


def test_apply_scale_group_lookup_with_fallback():
    lo = np.zeros(3)
    hi = np.ones(3)
    lo2, hi2 = apply_scale(lo, hi, 1.0,
                           np.array(["A", "B", "C"]), {"A": 2.0, "B": 0.5})
    assert (hi2 - lo2)[0] == pytest.approx(2.0)
    assert (hi2 - lo2)[1] == pytest.approx(0.5)
    assert (hi2 - lo2)[2] == pytest.approx(1.25)  # 未见组取已有 λ 均值


def test_apply_scale_group_without_groups_raises():
    with pytest.raises(ValueError, match="需要 groups"):
        apply_scale(np.zeros(2), np.ones(2), 1.0, None, {"A": 1.0})


def test_scale_factor_analytic_gaussian():
    """y~N(0,1)、固定区间 ±0.5 → λ 应为 2×z_{0.9} ≈ 2.563。"""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)
    lam = scale_factor(y, lo, hi, 0.8)
    assert lam == pytest.approx(2.0 * 1.28155, abs=0.05)
    lo2, hi2 = apply_scale(lo, hi, lam)
    assert interval_coverage(y, lo2, hi2) == pytest.approx(0.80, abs=0.02)


def test_scale_factor_mask_subset():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 400)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)
    mask = np.arange(400) < 100
    lam_all = scale_factor(y, lo, hi, 0.8)
    lam_sub = scale_factor(y, lo, hi, 0.8, mask)
    assert lam_sub != pytest.approx(lam_all, abs=1e-6)


def test_group_scale_factors_small_group_falls_back():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 503)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)
    g = np.array(["A"] * 500 + ["B"] * 3)
    lam_map = group_scale_factors(y, lo, hi, g, 0.8, n_min=30)
    lam_global = conformal_quantile(np.abs(normalized_residuals(y, lo, hi)), 0.8)
    assert lam_map["B"] == pytest.approx(lam_global)
    assert np.isfinite(lam_map["A"]) and lam_map["A"] > 0


def test_pinball_sharpness_positive():
    y = np.zeros(100)
    assert pinball_sharpness(y, np.full(100, -1.0), np.full(100, 1.0), 0.1, 0.9) > 0


def test_select_quantile_pair_picks_min_sharpness():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 800)
    wide = (np.full_like(y, -2.0), np.full_like(y, 2.0), 0.1, 0.9)
    good = (np.full_like(y, -1.28), np.full_like(y, 1.28), 0.1, 0.9)
    narrow = (np.full_like(y, -0.2), np.full_like(y, 0.2), 0.1, 0.9)
    best, table = select_quantile_pair({"wide": wide, "good": good, "narrow": narrow}, y, 0.8)
    assert best == "good"
    assert table["good"]["sharpness"] == min(v["sharpness"] for v in table.values())
    assert table["wide"]["coverage"] > table["good"]["coverage"] > table["narrow"]["coverage"]


def test_select_quantile_pair_require_coverage_filters_narrow():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 800)
    cands = {
        "narrow": (np.full_like(y, -0.2), np.full_like(y, 0.2), 0.1, 0.9),
        "wide": (np.full_like(y, -2.0), np.full_like(y, 2.0), 0.1, 0.9),
    }
    best, _ = select_quantile_pair(cands, y, 0.8, require_coverage=True)
    assert best == "wide"


def test_two_stage_interval_reaches_nominal_coverage():
    rng = np.random.default_rng(5)
    n = 4000
    y_cal = rng.normal(0, 1, n)
    cal = {"q10_q90": (np.full(n, -0.5), np.full(n, 0.5), 0.1, 0.9)}
    y_te = rng.normal(0, 1, 8000)
    lo_te, hi_te = np.full(8000, -0.5), np.full(8000, 0.5)
    lo, hi, params = two_stage_interval(
        y_cal, cal, {"q10_q90": lo_te}, {"q10_q90": hi_te}, 0.8)
    assert params["selected"] == "q10_q90"
    assert params["lambda"] == pytest.approx(2.0 * 1.28155, abs=0.1)
    assert interval_coverage(y_te, lo, hi) == pytest.approx(0.80, abs=0.02)


def test_two_stage_grouped_needs_groups():
    y = np.zeros(10)
    cal = {"p": (np.full(10, -1.0), np.full(10, 1.0), 0.1, 0.9)}
    with pytest.raises(ValueError, match="需要 groups_cal"):
        two_stage_interval(y, cal, {"p": np.full(5, -1.0)}, {"p": np.full(5, 1.0)},
                           0.8, grouped=True)


def test_cqr_add_matches_conformal():
    y = np.array([0.0, 5.0, -3.0])
    lo = np.full(3, -1.0)
    hi = np.full(3, 1.0)
    lo2, hi2, q = cqr_add_interval(y, lo, hi, np.zeros(2), np.ones(2), 0.8)
    assert q == pytest.approx(conformal_quantile(cqr_scores(y, lo, hi), 0.8))
    np.testing.assert_allclose(lo2, -q)
    np.testing.assert_allclose(hi2, 1.0 + q)


def test_pinball_loss_import_not_shadowed():
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.5) == pytest.approx(0.5)
