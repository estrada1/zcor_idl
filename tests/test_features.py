"""Tests for feature engineering."""

import os
import sys
import numpy as np
import pytest


from zcor_idl.features import (
    compute_prevalence_scores,
    compute_sequence_features,
    compute_aggregate_features,
    build_feature_matrix,
)


def _make_toy_data(n_pos=10, n_ctrl=40, n_cats=5, n_weeks=20, seed=42):
    """Create toy data for feature tests."""
    rng = np.random.default_rng(seed)
    n_total = n_pos + n_ctrl
    series = rng.integers(0, 3, size=(n_total, n_cats, n_weeks)).astype(np.int8)
    sld = rng.standard_normal((n_total, n_cats)).astype(np.float64)
    neg_llk = np.abs(rng.standard_normal((n_total, n_cats))).astype(np.float64) + 0.1
    pos_llk = np.abs(rng.standard_normal((n_total, n_cats))).astype(np.float64) + 0.1

    metadata = []
    for i in range(n_total):
        metadata.append({
            "patient_id": f"P{i:04d}",
            "sex": "M" if i % 2 == 0 else "F",
            "label": 1 if i < n_pos else 0,
            "age_at_screening": 60.0 + rng.standard_normal() * 5,
        })

    return series, metadata, sld, neg_llk, pos_llk


# ── Prevalence scores ─────────────────────────────────────────

class TestPrevalenceScores:
    def test_shape(self):
        series, metadata, *_ = _make_toy_data()
        pscore_dict, patient_pscores = compute_prevalence_scores(series, metadata)
        assert pscore_dict.shape == (5,)
        assert patient_pscores.shape == (50, 5)

    def test_pscores_finite(self):
        series, metadata, *_ = _make_toy_data()
        pscore_dict, patient_pscores = compute_prevalence_scores(series, metadata)
        assert np.all(np.isfinite(pscore_dict))
        assert np.all(np.isfinite(patient_pscores))

    def test_pscore_ratio_positive(self):
        """P-scores should be positive (prevalence ratios)."""
        series, metadata, *_ = _make_toy_data()
        pscore_dict, _ = compute_prevalence_scores(series, metadata)
        assert np.all(pscore_dict > 0)


# ── Sequence features ─────────────────────────────────────────

class TestSequenceFeatures:
    def test_all_keys_present(self):
        series, *_ = _make_toy_data()
        feats = compute_sequence_features(series)
        expected_keys = {"proportion", "prevalence", "first_incident",
                         "last_incident", "mean_position", "max_streak"}
        assert set(feats.keys()) == expected_keys

    def test_shapes(self):
        series, *_ = _make_toy_data(n_pos=10, n_ctrl=40, n_cats=5)
        feats = compute_sequence_features(series)
        for name, arr in feats.items():
            assert arr.shape == (50, 5), f"{name} has wrong shape: {arr.shape}"

    def test_proportion_range(self):
        series, *_ = _make_toy_data()
        feats = compute_sequence_features(series)
        assert np.all(feats["proportion"] >= 0)
        assert np.all(feats["proportion"] <= 1)

    def test_first_incident_no_occurrence(self):
        """Categories never present should have first_incident = -1."""
        # All zeros → no category ever has value 1
        series = np.zeros((5, 3, 20), dtype=np.int8)
        feats = compute_sequence_features(series)
        assert np.all(feats["first_incident"] == -1)

    def test_first_before_last(self):
        """first_incident should be <= last_incident when both exist."""
        series, *_ = _make_toy_data()
        feats = compute_sequence_features(series)
        mask = feats["first_incident"] >= 0  # only where category appears
        assert np.all(feats["first_incident"][mask] <= feats["last_incident"][mask])


# ── Aggregate features ────────────────────────────────────────

class TestAggregateFeatures:
    def test_keys_present(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data()
        _, patient_pscores = compute_prevalence_scores(series, metadata)
        seq_feats = compute_sequence_features(series)
        agg = compute_aggregate_features(sld, neg_llk, pos_llk, patient_pscores, seq_feats)

        assert "sld_mean" in agg
        assert "sld_std" in agg
        assert "sld_max" in agg
        assert "record_density" in agg
        assert "pscore_dynamics" in agg

    def test_aggregate_shapes(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data()
        _, patient_pscores = compute_prevalence_scores(series, metadata)
        seq_feats = compute_sequence_features(series)
        agg = compute_aggregate_features(sld, neg_llk, pos_llk, patient_pscores, seq_feats)

        n = len(metadata)
        for name, arr in agg.items():
            assert arr.shape == (n,), f"{name} shape is {arr.shape}, expected ({n},)"


# ── Full feature matrix ──────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_output_shape(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data()
        X, names, y = build_feature_matrix(series, metadata, sld, neg_llk, pos_llk)
        assert X.shape[0] == len(metadata)
        assert X.shape[1] == len(names)
        assert y.shape == (len(metadata),)

    def test_labels_correct(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data(n_pos=10, n_ctrl=40)
        _, _, y = build_feature_matrix(series, metadata, sld, neg_llk, pos_llk)
        assert y.sum() == 10
        assert (1 - y).sum() == 40

    def test_no_nans(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data()
        X, _, _ = build_feature_matrix(series, metadata, sld, neg_llk, pos_llk)
        assert not np.any(np.isnan(X))

    def test_feature_names_unique(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data()
        _, names, _ = build_feature_matrix(series, metadata, sld, neg_llk, pos_llk)
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_demographics_included(self):
        series, metadata, sld, neg_llk, pos_llk = _make_toy_data()
        _, names, _ = build_feature_matrix(series, metadata, sld, neg_llk, pos_llk)
        assert "age_at_screening" in names
        assert "sex_male" in names
