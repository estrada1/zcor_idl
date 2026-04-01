"""Tests for training utilities (metrics, evaluation)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train import compute_likelihood_ratios, evaluate_at_specificity


class TestComputeLikelihoodRatios:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        m = compute_likelihood_ratios(y_true, y_pred)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert m["ppv"] == 1.0
        assert m["npv"] == 1.0

    def test_all_false_negatives(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        m = compute_likelihood_ratios(y_true, y_pred)
        assert m["sensitivity"] == 0.0
        assert m["specificity"] == 1.0

    def test_all_false_positives(self):
        y_true = np.array([1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        m = compute_likelihood_ratios(y_true, y_pred)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 0.0

    def test_lr_plus_formula(self):
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        m = compute_likelihood_ratios(y_true, y_pred)
        # sens = 2/4 = 0.5, spec = 5/6
        # LR+ = sens / (1 - spec) = 0.5 / (1/6) = 3.0
        assert abs(m["sensitivity"] - 0.5) < 1e-6
        assert abs(m["lr_plus"] - 3.0) < 1e-6


class TestEvaluateAtSpecificity:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 20 + [0] * 80)
        y_scores = rng.random(100)
        # Make positives score higher on average
        y_scores[:20] += 0.5

        m = evaluate_at_specificity(y_true, y_scores, target_spec=0.95)
        assert "sensitivity" in m
        assert "specificity" in m
        assert "lr_plus" in m
        assert "lr_minus" in m
        assert "ppv" in m
        assert "npv" in m
        assert "threshold" in m
        assert "target_specificity" in m

    def test_specificity_near_target(self):
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 50 + [0] * 200)
        y_scores = rng.random(250)
        y_scores[:50] += 0.5

        m = evaluate_at_specificity(y_true, y_scores, target_spec=0.95)
        # Specificity should be close to target (within tolerance)
        assert abs(m["specificity"] - 0.95) < 0.05

    def test_different_targets(self):
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 50 + [0] * 200)
        y_scores = rng.random(250)
        y_scores[:50] += 0.5

        m95 = evaluate_at_specificity(y_true, y_scores, 0.95)
        m99 = evaluate_at_specificity(y_true, y_scores, 0.99)

        # Higher specificity target → lower sensitivity (generally)
        assert m99["threshold"] >= m95["threshold"]
