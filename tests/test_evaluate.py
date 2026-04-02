"""Tests for evaluate.py (calibration, subgroup analysis, ablation study)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import (
    calibration_analysis,
    subgroup_analysis,
    ablation_study,
    _feature_mask,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _perfect_scores(n=200, n_pos=20, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    # Perfect separation: positives all score 1, negatives all score 0
    scores = np.where(y == 1, 1.0, 0.0).astype(float)
    return y, scores


def _random_scores(n=200, n_pos=20, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    scores = rng.random(n)
    scores[:n_pos] += 0.3
    scores = np.clip(scores, 0, 1)
    return y, scores


def _metadata(n, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "sex": "M" if i % 2 == 0 else "F",
            "age_at_screening": float(50 + (i % 30)),
        }
        for i in range(n)
    ]


def _feature_names(n_cats=4):
    names = []
    for pfx in [
        "sld_", "neg_llk_", "pos_llk_", "llk_ratio_", "pscore_",
        "proportion_", "prevalence_", "first_incident_",
        "last_incident_", "mean_position_", "max_streak_",
    ]:
        for k in range(n_cats):
            names.append(f"{pfx}{k}")
    names += ["sld_mean", "pscore_mean", "age_at_screening", "sex_male"]
    return names


# ── calibration_analysis ─────────────────────────────────────────────────────

class TestCalibrationAnalysis:
    def test_returns_required_keys(self):
        y, scores = _random_scores()
        cal = calibration_analysis(y, scores, n_bins=5)
        for key in ("fraction_of_positives", "mean_predicted_value",
                    "bin_counts", "ece", "mce", "n_bins"):
            assert key in cal

    def test_n_bins_respected(self):
        y, scores = _random_scores()
        for n in (5, 10):
            cal = calibration_analysis(y, scores, n_bins=n)
            assert cal["n_bins"] == n
            assert len(cal["fraction_of_positives"]) == n
            assert len(cal["bin_counts"]) == n

    def test_ece_non_negative(self):
        y, scores = _random_scores()
        cal = calibration_analysis(y, scores)
        assert cal["ece"] >= 0

    def test_mce_non_negative(self):
        y, scores = _random_scores()
        cal = calibration_analysis(y, scores)
        assert cal["mce"] >= 0

    def test_mce_ge_ece(self):
        y, scores = _random_scores()
        cal = calibration_analysis(y, scores)
        assert cal["mce"] >= cal["ece"] - 1e-9

    def test_bin_counts_sum_to_n(self):
        y, scores = _random_scores(n=200)
        cal = calibration_analysis(y, scores, n_bins=10)
        assert sum(cal["bin_counts"]) == 200

    def test_perfect_model_low_ece(self):
        """A perfectly calibrated model should have low ECE."""
        # Synthetic: scores equal empirical prevalence within each bin
        rng = np.random.default_rng(42)
        n = 1000
        scores = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < scores).astype(int)
        cal = calibration_analysis(y, scores, n_bins=10)
        # ECE should be small for a well-calibrated model on large enough data
        assert cal["ece"] < 0.05


# ── subgroup_analysis ─────────────────────────────────────────────────────────

class TestSubgroupAnalysis:
    def test_returns_all_subgroups(self):
        y, scores = _random_scores(n=300, n_pos=30)
        meta = _metadata(300)
        result = subgroup_analysis(y, scores, meta)
        for key in ("sex_M", "sex_F", "age_45_54", "overall"):
            assert key in result

    def test_overall_auc_in_range(self):
        y, scores = _random_scores(n=300, n_pos=30)
        meta = _metadata(300)
        result = subgroup_analysis(y, scores, meta)
        auc = result["overall"]["auc"]
        assert 0.0 <= auc <= 1.0

    def test_skipped_subgroup_when_no_positives(self):
        """Subgroup with no positives should be skipped, not raise."""
        n = 200
        y = np.zeros(n, dtype=int)
        y[:20] = 1
        scores = np.random.default_rng(0).random(n)
        # All patients are male → female subgroup has no positives
        meta = [{"sex": "M", "age_at_screening": 60.0}] * n
        result = subgroup_analysis(y, scores, meta)
        assert result["sex_F"].get("skipped") is True

    def test_subgroup_sizes_consistent(self):
        n = 300
        y, scores = _random_scores(n=n, n_pos=30)
        meta = _metadata(n)
        result = subgroup_analysis(y, scores, meta)
        total = result["sex_M"]["n"] + result["sex_F"]["n"]
        assert total == n

    def test_sensitivity_in_range(self):
        y, scores = _random_scores(n=400, n_pos=40)
        meta = _metadata(400)
        result = subgroup_analysis(y, scores, meta)
        for name, r in result.items():
            if not r.get("skipped"):
                assert 0.0 <= r["sensitivity"] <= 1.0

    def test_ppv_in_range(self):
        y, scores = _random_scores(n=400, n_pos=40)
        meta = _metadata(400)
        result = subgroup_analysis(y, scores, meta)
        for name, r in result.items():
            if not r.get("skipped"):
                assert 0.0 <= r["ppv"] <= 1.0


# ── _feature_mask ─────────────────────────────────────────────────────────────

class TestFeatureMask:
    def test_mask_selects_correct_prefixes(self):
        names = _feature_names()
        mask = _feature_mask(names, ["sld_", "pscore_"])
        selected = [n for n, m in zip(names, mask) if m]
        assert all(n.startswith("sld_") or n.startswith("pscore_") for n in selected)

    def test_mask_excludes_other_prefixes(self):
        names = _feature_names()
        mask = _feature_mask(names, ["sld_"])
        selected = {n for n, m in zip(names, mask) if m}
        assert "neg_llk_0" not in selected

    def test_all_features_with_none_equivalent(self):
        """Full mask should equal all-True."""
        names = _feature_names()
        prefixes = [
            "sld_", "neg_llk_", "pos_llk_", "llk_ratio_", "pscore_",
            "proportion_", "prevalence_", "first_incident_",
            "last_incident_", "mean_position_", "max_streak_",
            "sld_mean", "pscore_mean", "age_at_screening", "sex_male",
        ]
        mask = _feature_mask(names, prefixes)
        assert mask.all()


# ── ablation_study ────────────────────────────────────────────────────────────

class TestAblationStudy:
    @pytest.fixture(scope="class")
    def ablation_result(self):
        rng = np.random.default_rng(7)
        names = _feature_names(n_cats=4)
        n = 240
        X = rng.random((n, len(names))).astype("float32")
        y = np.zeros(n, dtype=int)
        y[:24] = 1
        X[:24] += 0.4
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        result = ablation_study(X_tr, y_tr, X_te, y_te, names, n_folds=2)
        return result

    def test_returns_all_levels(self, ablation_result):
        assert len(ablation_result) == 5

    def test_each_level_has_required_keys(self, ablation_result):
        for r in ablation_result:
            assert "level" in r
            assert "n_features" in r
            assert "cv_auc_mean" in r
            assert "cv_auc_std" in r

    def test_auc_in_range(self, ablation_result):
        for r in ablation_result:
            assert 0.0 <= r["cv_auc_mean"] <= 1.0
            assert r["cv_auc_std"] >= 0.0

    def test_feature_count_non_decreasing(self, ablation_result):
        """Each ablation level should have at least as many features as the prior."""
        counts = [r["n_features"] for r in ablation_result]
        for a, b in zip(counts, counts[1:]):
            assert b >= a

    def test_full_model_has_most_features(self, ablation_result):
        counts = [r["n_features"] for r in ablation_result]
        assert counts[-1] == max(counts)
