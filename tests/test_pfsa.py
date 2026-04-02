"""Tests for PFSA training and SLD computation."""

import os
import sys
import numpy as np
import pytest


from zcor_idl.pfsa import PFSA, PFSAEnsemble


# ── Single PFSA model ─────────────────────────────────────────

class TestPFSA:
    def test_fit_and_score(self):
        model = PFSA(max_depth=2)
        seqs = np.array([[0, 1, 2, 0, 1, 2, 0, 1],
                         [0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.int8)
        model.fit(seqs)
        assert model.is_fitted

        llk = model.batch_log_likelihood(seqs)
        assert llk.shape == (2,)
        assert np.all(llk > 0)  # negative normalized log-likelihood → positive

    def test_unfitted_raises(self):
        model = PFSA(max_depth=2)
        seqs = np.array([[0, 1, 2]], dtype=np.int8)
        with pytest.raises(RuntimeError):
            model.batch_log_likelihood(seqs)

    def test_uniform_sequence_high_likelihood(self):
        """A model trained on all-zeros should score all-zeros lower (better)."""
        model = PFSA(max_depth=2)
        uniform = np.zeros((50, 20), dtype=np.int8)
        model.fit(uniform)

        # All-zeros should have lower NLL than random
        zero_llk = model.batch_log_likelihood(np.zeros((1, 20), dtype=np.int8))
        rng = np.random.default_rng(42)
        random_seqs = rng.integers(0, 3, size=(1, 20)).astype(np.int8)
        rand_llk = model.batch_log_likelihood(random_seqs)

        assert zero_llk[0] < rand_llk[0]

    def test_batch_shape(self):
        model = PFSA(max_depth=3)
        seqs = np.random.default_rng(0).integers(0, 3, size=(100, 50)).astype(np.int8)
        model.fit(seqs)
        llk = model.batch_log_likelihood(seqs)
        assert llk.shape == (100,)

    def test_different_max_depths(self):
        """Deeper models should not crash and should produce valid output."""
        seqs = np.random.default_rng(0).integers(0, 3, size=(30, 40)).astype(np.int8)
        for depth in [0, 1, 2, 3]:
            model = PFSA(max_depth=depth)
            model.fit(seqs)
            llk = model.batch_log_likelihood(seqs)
            assert llk.shape == (30,)
            assert np.all(np.isfinite(llk))


# ── PFSA Ensemble ─────────────────────────────────────────────

def _make_toy_data(n_pos=20, n_ctrl=80, n_cats=5, n_weeks=20, seed=42):
    """Create toy series + metadata for ensemble testing."""
    rng = np.random.default_rng(seed)
    n_total = n_pos + n_ctrl

    # Positive patients: slightly more 1s in category 0
    series = rng.integers(0, 3, size=(n_total, n_cats, n_weeks)).astype(np.int8)
    for i in range(n_pos):
        # Inject signal: category 0 has more 1s for positive patients
        series[i, 0, :] = rng.choice([0, 1, 1, 2], size=n_weeks).astype(np.int8)

    metadata = []
    for i in range(n_total):
        metadata.append({
            "patient_id": f"P{i:04d}",
            "sex": "M" if i % 2 == 0 else "F",
            "label": 1 if i < n_pos else 0,
            "age_at_screening": 65.0,
        })

    return series, metadata


class TestPFSAEnsemble:
    def test_fit_completes(self):
        series, metadata = _make_toy_data(n_cats=3)
        ensemble = PFSAEnsemble(n_categories=3, max_depth=2)
        ensemble.fit(series, metadata)

        # Check all models are fitted
        for sex in ["M", "F"]:
            for cohort in ["positive", "control"]:
                for cat in range(3):
                    assert ensemble.models[sex][cohort][cat].is_fitted

    def test_sld_shape(self):
        n_cats = 3
        series, metadata = _make_toy_data(n_cats=n_cats)
        ensemble = PFSAEnsemble(n_categories=n_cats, max_depth=2)
        ensemble.fit(series, metadata)

        sld, neg_llk, pos_llk = ensemble.compute_sld(series, metadata)
        assert sld.shape == (len(metadata), n_cats)
        assert neg_llk.shape == (len(metadata), n_cats)
        assert pos_llk.shape == (len(metadata), n_cats)

    def test_sld_is_finite(self):
        series, metadata = _make_toy_data(n_cats=3)
        ensemble = PFSAEnsemble(n_categories=3, max_depth=2)
        ensemble.fit(series, metadata)
        sld, _, _ = ensemble.compute_sld(series, metadata)
        assert np.all(np.isfinite(sld))

    def test_sld_separates_cohorts(self):
        """Positive patients should have different mean SLD than controls on injected signal."""
        series, metadata = _make_toy_data(n_pos=30, n_ctrl=70, n_cats=3, n_weeks=50)
        ensemble = PFSAEnsemble(n_categories=3, max_depth=2)
        ensemble.fit(series, metadata)
        sld, _, _ = ensemble.compute_sld(series, metadata)

        pos_mask = np.array([m["label"] == 1 for m in metadata])
        pos_sld_mean = sld[pos_mask, 0].mean()
        ctrl_sld_mean = sld[~pos_mask, 0].mean()
        # The injected signal in category 0 should create separation
        assert pos_sld_mean != ctrl_sld_mean
