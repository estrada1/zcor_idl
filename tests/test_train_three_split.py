"""Tests for ThreeSplitTrainer and related helpers in train.py."""

import numpy as np
import pytest


from zcor_idl.train import (
    ThreeSplitTrainer,
    _category_feature_indices,
    _aggregate_feature_indices,
)


def _make_feature_names(n_cats=4):
    """Build a minimal realistic feature name list for testing."""
    names = []
    for pfx in [
        "sld_", "neg_llk_", "pos_llk_", "llk_ratio_", "pscore_",
        "proportion_", "prevalence_", "first_incident_",
        "last_incident_", "mean_position_", "max_streak_",
    ]:
        for k in range(n_cats):
            names.append(f"{pfx}{k}")
    names += ["sld_mean", "sld_std", "sld_max", "pscore_mean", "age_at_screening", "sex_male"]
    return names


def _make_dataset(n=300, n_pos=30, n_feats=None, seed=0):
    feat_names = _make_feature_names()
    if n_feats is None:
        n_feats = len(feat_names)
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_feats)).astype("float32")
    y = np.zeros(n, dtype="int32")
    y[:n_pos] = 1
    return X, y, feat_names


class TestFeatureIndexHelpers:
    def test_category_feature_indices_returns_correct_names(self):
        names = _make_feature_names(n_cats=4)
        idx = _category_feature_indices(names, 2)
        selected = [names[i] for i in idx]
        assert "sld_2" in selected
        assert "pscore_2" in selected
        assert "sld_3" not in selected
        assert "age_at_screening" not in selected

    def test_aggregate_feature_indices_excludes_per_cat(self):
        names = _make_feature_names(n_cats=4)
        idx = _aggregate_feature_indices(names)
        selected = {names[i] for i in idx}
        assert "sld_mean" in selected
        assert "age_at_screening" in selected
        assert "sld_0" not in selected
        assert "proportion_3" not in selected

    def test_no_overlap_between_cat_and_agg(self):
        names = _make_feature_names(n_cats=4)
        cat_idx = set(_category_feature_indices(names, 0))
        agg_idx = set(_aggregate_feature_indices(names))
        assert cat_idx.isdisjoint(agg_idx)


class TestThreeSplitTrainer:
    @pytest.fixture(scope="class")
    def fitted_trainer(self):
        X, y, feat_names = _make_dataset(n=400, n_pos=40)
        trainer = ThreeSplitTrainer(n_leaves_cat=7, n_leaves_agg=15, random_state=0)
        trainer.fit(X, y, feat_names)
        return trainer, X, y

    def test_fit_produces_51_cat_models(self, fitted_trainer):
        trainer, _, _ = fitted_trainer
        assert len(trainer.cat_models) == ThreeSplitTrainer.N_CATEGORIES

    def test_agg_model_not_none(self, fitted_trainer):
        trainer, _, _ = fitted_trainer
        assert trainer.agg_model is not None

    def test_predict_returns_probabilities(self, fitted_trainer):
        trainer, X, _ = fitted_trainer
        scores = trainer.predict(X)
        assert scores.shape == (len(X),)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_predict_positive_higher_than_negative(self, fitted_trainer):
        trainer, X, y = fitted_trainer
        scores = trainer.predict(X)
        mean_pos = scores[y == 1].mean()
        mean_neg = scores[y == 0].mean()
        assert mean_pos > mean_neg

    def test_feature_importance_length(self, fitted_trainer):
        trainer, _, _ = fitted_trainer
        imp = trainer.feature_importance(top_n=10)
        assert len(imp) == 10

    def test_feature_importance_sorted_descending(self, fitted_trainer):
        trainer, _, _ = fitted_trainer
        imp = trainer.feature_importance(top_n=10)
        scores = [v for _, v in imp]
        assert scores == sorted(scores, reverse=True)

    def test_agg_feature_names_count(self, fitted_trainer):
        trainer, X, _ = fitted_trainer
        # should be N_CATEGORIES cat-prob features + aggregate feature count
        n_meta = len(trainer._agg_feature_names)
        assert n_meta > ThreeSplitTrainer.N_CATEGORIES

    def test_small_dataset_does_not_crash(self):
        """Verify trainer handles minimal viable dataset."""
        X, y, feat_names = _make_dataset(n=120, n_pos=12)
        trainer = ThreeSplitTrainer(n_leaves_cat=7, n_leaves_agg=15, random_state=1)
        trainer.fit(X, y, feat_names)
        scores = trainer.predict(X)
        assert scores.shape == (120,)
