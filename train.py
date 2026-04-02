"""
LightGBM training pipeline for ZCoR-IPF.

Two training modes are available:

1. **Single-LGBM** (default, --mode single):
   One LightGBM trained on all features with 75/25 split + cross-validation.
   Matches the prior implementation.

2. **Three-split** (--mode three-split):
   Implements the paper's three-split strategy (Methods):
     - Hyper-training split  (~40 %): PFSA models / p-score dicts were fitted
       on this partition *before* feature extraction; features in
       feature_matrix.npz are already computed with that split in mind.
     - Pre-aggregation split (~30 %): trains 51 per-phenotype LGBMs, one per
       disease category, each predicting the final label from category-specific
       features (SLD, log-likelihoods, p-score, and sequence features).
     - Aggregation split     (~30 %): trains a final LGBM whose inputs are the
       51 per-phenotype probability scores (and the original aggregate features).

   The held-out test set is kept completely separate from all three splits.
"""

import numpy as np
import os
import json
import time
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
import lightgbm as lgb


def compute_likelihood_ratios(y_true, y_pred_binary, prevalence=None):
    """Compute LR+, LR-, PPV, NPV."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    lr_plus = sensitivity / max(1 - specificity, 1e-10)
    lr_minus = (1 - sensitivity) / max(specificity, 1e-10)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "lr_plus": lr_plus,
        "lr_minus": lr_minus,
        "ppv": ppv,
        "npv": npv,
    }


def evaluate_at_specificity(y_true, y_scores, target_spec=0.95):
    """Find threshold achieving target specificity and compute metrics."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    spec = 1 - fpr

    # Find threshold closest to target specificity
    idx = np.argmin(np.abs(spec - target_spec))
    threshold = thresholds[idx]

    y_pred = (y_scores >= threshold).astype(int)
    metrics = compute_likelihood_ratios(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    metrics["target_specificity"] = target_spec

    return metrics


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names=None):
    """Train LightGBM and evaluate on test set."""

    # Handle class imbalance with scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Predictions
    y_scores = model.predict(X_test)
    auc = roc_auc_score(y_test, y_scores)

    # Metrics at 95% and 99% specificity
    metrics_95 = evaluate_at_specificity(y_test, y_scores, 0.95)
    metrics_99 = evaluate_at_specificity(y_test, y_scores, 0.99)

    # Feature importance
    importance = None
    if feature_names:
        imp = model.feature_importance(importance_type="gain")
        importance = sorted(
            zip(feature_names, imp),
            key=lambda x: x[1], reverse=True
        )

    return model, auc, y_scores, metrics_95, metrics_99, importance


def cross_validate(X, y, feature_names=None, n_folds=5):
    """Stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    all_metrics_95 = []
    all_metrics_99 = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{n_folds}:")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        _, auc, _, m95, m99, _ = train_and_evaluate(
            X_tr, y_tr, X_te, y_te, feature_names
        )
        aucs.append(auc)
        all_metrics_95.append(m95)
        all_metrics_99.append(m99)
        print(f"    AUC: {auc:.4f}")
        print(f"    @95% spec: sens={m95['sensitivity']:.3f}, "
              f"LR+={m95['lr_plus']:.1f}, LR-={m95['lr_minus']:.3f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    return aucs, all_metrics_95, all_metrics_99


def _lgbm_params(y_train, n_leaves=31):
    """Return LightGBM params tuned for class-imbalanced binary classification."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": n_leaves,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": n_neg / max(n_pos, 1),
        "min_child_samples": 10,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }


def _train_lgbm(X_train, y_train, X_val, y_val, n_leaves=31, rounds=300, early=30):
    """Train a single LightGBM and return (model, val_scores)."""
    params = _lgbm_params(y_train, n_leaves=n_leaves)
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=rounds,
        valid_sets=[ds_val],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(early, verbose=False)],
    )
    return model, model.predict(X_val)


# ── Category-feature slicing ─────────────────────────────────────────────────

# Feature prefixes that are per-category (one column per category, ordered 0..50)
_PER_CAT_PREFIXES = [
    "sld_", "neg_llk_", "pos_llk_", "llk_ratio_",
    "pscore_", "proportion_", "prevalence_", "first_incident_",
    "last_incident_", "mean_position_", "max_streak_",
]


def _category_feature_indices(feature_names, cat_id):
    """Return indices of all per-category features for category cat_id."""
    suffix = f"_{cat_id}"
    return [i for i, n in enumerate(feature_names)
            if any(n == pfx + str(cat_id) for pfx in _PER_CAT_PREFIXES)]


def _aggregate_feature_indices(feature_names):
    """Return indices of aggregate / demographic features (not per-category)."""
    per_cat_names = set()
    for pfx in _PER_CAT_PREFIXES:
        for cat in range(51):
            per_cat_names.add(f"{pfx}{cat}")
    return [i for i, n in enumerate(feature_names) if n not in per_cat_names]


# ── Three-split trainer ───────────────────────────────────────────────────────

class ThreeSplitTrainer:
    """
    Implements the paper's three-split training pipeline.

    Splits the (non-test) data as:
      hyper  : pre-aggregation : aggregation  ≈  40 : 30 : 30

    The hyper-training split is treated as already used (PFSA + p-scores were
    fit on it before feature extraction).  We use it here only as a validation
    set for the per-category LGBMs so we don't leak labels into the
    aggregation split.

    Per-category LGBMs
    ------------------
    For each of the 51 disease categories we train a small LightGBM that
    receives only the features for that category plus the demographic
    features, and predicts the binary label.  Its output probability on the
    aggregation split becomes a single meta-feature feeding the final model.

    Aggregation LGBM
    ----------------
    The final model receives the 51 per-category probability scores plus the
    aggregate features, and produces the ZCoR-IPF score.
    """

    N_CATEGORIES = 51

    def __init__(self, n_leaves_cat=15, n_leaves_agg=63, random_state=42):
        self.n_leaves_cat = n_leaves_cat
        self.n_leaves_agg = n_leaves_agg
        self.random_state = random_state
        self.cat_models = []        # list of 51 LightGBM models
        self.agg_model = None
        self.feature_names = None
        self._agg_feature_names = None

    def fit(self, X, y, feature_names):
        """
        Train all 51 per-category LGBMs and the final aggregation LGBM.

        X, y are the *training* data (test set must be kept separate by caller).
        Splits internally into hyper / pre-agg / agg partitions.
        """
        self.feature_names = feature_names
        rng = np.random.default_rng(self.random_state)

        # ── Partition ────────────────────────────────────────────────────────
        # hyper(40%) | pre_agg(30%) | agg(30%)
        idx = np.arange(len(y))
        pos_idx = idx[y == 1]
        neg_idx = idx[y == 0]

        def stratified_split(arr, frac, rng):
            n = max(1, int(len(arr) * frac))
            chosen = rng.choice(arr, size=n, replace=False)
            rest = arr[~np.isin(arr, chosen)]
            return chosen, rest

        hyper_pos, rest_pos = stratified_split(pos_idx, 0.40, rng)
        hyper_neg, rest_neg = stratified_split(neg_idx, 0.40, rng)

        pre_pos, agg_pos = stratified_split(rest_pos, 0.50, rng)
        pre_neg, agg_neg = stratified_split(rest_neg, 0.50, rng)

        hyper_idx = np.concatenate([hyper_pos, hyper_neg])
        pre_idx   = np.concatenate([pre_pos,   pre_neg])
        agg_idx   = np.concatenate([agg_pos,   agg_neg])

        X_hyper, y_hyper = X[hyper_idx], y[hyper_idx]
        X_pre,   y_pre   = X[pre_idx],   y[pre_idx]
        X_agg,   y_agg   = X[agg_idx],   y[agg_idx]

        print(f"  Split sizes — hyper: {len(y_hyper)} "
              f"({y_hyper.sum()} pos), pre-agg: {len(y_pre)} "
              f"({y_pre.sum()} pos), agg: {len(y_agg)} ({y_agg.sum()} pos)")

        # Pre-compute aggregate / demographic feature indices (shared)
        agg_feat_idx = np.array(_aggregate_feature_indices(feature_names))

        # ── Per-category LGBMs ───────────────────────────────────────────────
        print(f"\n  Training {self.N_CATEGORIES} per-category LGBMs...")
        self.cat_models = []
        cat_aucs = []

        for cat in range(self.N_CATEGORIES):
            cat_idx = np.array(_category_feature_indices(feature_names, cat))
            if len(cat_idx) == 0:
                # Fallback: use aggregate features only
                feat_idx = agg_feat_idx
            else:
                feat_idx = np.concatenate([cat_idx, agg_feat_idx])

            X_pre_cat   = X_pre[:, feat_idx]
            X_hyper_cat = X_hyper[:, feat_idx]

            model, val_scores = _train_lgbm(
                X_pre_cat, y_pre,
                X_hyper_cat, y_hyper,
                n_leaves=self.n_leaves_cat,
                rounds=200, early=20,
            )
            self.cat_models.append((model, feat_idx))

            if y_hyper.sum() > 0 and (y_hyper == 0).sum() > 0:
                cat_aucs.append(roc_auc_score(y_hyper, val_scores))

        mean_cat_auc = np.mean(cat_aucs) if cat_aucs else float("nan")
        print(f"  Per-category LGBMs done. Mean val AUC: {mean_cat_auc:.4f}")

        # ── Build meta-features for aggregation split ─────────────────────
        X_agg_meta   = self._make_meta_features(X_agg,   agg_feat_idx)
        X_hyper_meta = self._make_meta_features(X_hyper, agg_feat_idx)

        # ── Final aggregation LGBM ────────────────────────────────────────
        print(f"\n  Training aggregation LGBM "
              f"(meta-features: {X_agg_meta.shape[1]})...")

        meta_names = (
            [f"cat_prob_{k}" for k in range(self.N_CATEGORIES)]
            + [feature_names[i] for i in agg_feat_idx]
        )
        self._agg_feature_names = meta_names

        self.agg_model, _ = _train_lgbm(
            X_agg_meta, y_agg,
            X_hyper_meta, y_hyper,
            n_leaves=self.n_leaves_agg,
            rounds=300, early=30,
        )
        print(f"  Aggregation LGBM done.")
        return self

    def predict(self, X):
        """Return final ZCoR-IPF probability scores for rows in X."""
        agg_feat_idx = np.array(_aggregate_feature_indices(self.feature_names))
        X_meta = self._make_meta_features(X, agg_feat_idx)
        return self.agg_model.predict(X_meta)

    def _make_meta_features(self, X, agg_feat_idx):
        """Assemble [cat_prob_0 ... cat_prob_50, agg_feats] for each row."""
        cat_probs = np.column_stack([
            model.predict(X[:, feat_idx])
            for model, feat_idx in self.cat_models
        ])
        agg_feats = X[:, agg_feat_idx]
        return np.hstack([cat_probs, agg_feats]).astype(np.float32)

    def feature_importance(self, top_n=20):
        """Return top-N features by gain from the aggregation model."""
        imp = self.agg_model.feature_importance(importance_type="gain")
        pairs = sorted(zip(self._agg_feature_names, imp),
                       key=lambda x: x[1], reverse=True)
        return pairs[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Train ZCoR-IPF LightGBM model")
    parser.add_argument("--mode", choices=["single", "three-split"], default="single",
                        help="Training mode: 'single' (one LGBM, default) or "
                             "'three-split' (per-category + aggregation, per paper)")
    parser.add_argument("--cv-only", action="store_true",
                        help="Only run cross-validation, don't save a final model")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    features_dir = os.path.join("data", "processed", "features")
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    # Load features
    print("Loading features...")
    data = np.load(os.path.join(features_dir, "feature_matrix.npz"))
    X, y = data["X"], data["y"]

    with open(os.path.join(features_dir, "feature_names.txt")) as f:
        feature_names = f.read().strip().split("\n")

    n_pos = y.sum()
    n_ctrl = len(y) - n_pos
    print(f"  Features: {X.shape[1]}, Patients: {len(y)} "
          f"({n_pos} positive, {n_ctrl} control)")

    # Always hold out 25 % as the final test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"  Hold-out test set: {len(y_test)} patients ({y_test.sum()} pos)")

    t0 = time.time()

    # ── Cross-validation (single-LGBM mode only) ─────────────────────────────
    if not args.cv_only or args.mode == "single":
        print(f"\n{'='*60}")
        print(f"Cross-validation ({args.n_folds} folds) — single-LGBM")
        print(f"{'='*60}")
        aucs, metrics_95, metrics_99 = cross_validate(
            X_train, y_train, feature_names, n_folds=args.n_folds
        )
    else:
        aucs = []

    if args.cv_only:
        print(f"\nCV completed in {time.time()-t0:.1f}s")
        return

    # ── Train final model ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if args.mode == "three-split":
        print("Training three-split model (per-category LGBMs + aggregation)")
    else:
        print("Training final single-LGBM model (75/25 split)")
    print(f"{'='*60}")
    print(f"  Train: {len(y_train)} ({y_train.sum()} pos)")
    print(f"  Test:  {len(y_test)} ({y_test.sum()} pos)")

    if args.mode == "three-split":
        trainer = ThreeSplitTrainer()
        trainer.fit(X_train, y_train, feature_names)
        y_scores = trainer.predict(X_test)
        auc = roc_auc_score(y_test, y_scores)
        m95 = evaluate_at_specificity(y_test, y_scores, 0.95)
        m99 = evaluate_at_specificity(y_test, y_scores, 0.99)
        importance = trainer.feature_importance(top_n=20)
        save_name = "zcor_ipf_three_split"
    else:
        model, auc, y_scores, m95, m99, importance = train_and_evaluate(
            X_train, y_train, X_test, y_test, feature_names
        )
        save_name = "zcor_ipf_lgbm"

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"\nHeld-out test AUC: {auc:.4f}")
    if aucs:
        print(f"CV AUC (single):   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    print(f"\nAt 95% specificity:")
    print(f"  Sensitivity: {m95['sensitivity']:.3f}")
    print(f"  PPV:         {m95['ppv']:.3f}")
    print(f"  NPV:         {m95['npv']:.4f}")
    print(f"  LR+:         {m95['lr_plus']:.1f}")
    print(f"  LR-:         {m95['lr_minus']:.3f}")

    print(f"\nAt 99% specificity:")
    print(f"  Sensitivity: {m99['sensitivity']:.3f}")
    print(f"  PPV:         {m99['ppv']:.3f}")
    print(f"  NPV:         {m99['npv']:.4f}")
    print(f"  LR+:         {m99['lr_plus']:.1f}")
    print(f"  LR-:         {m99['lr_minus']:.3f}")

    if importance:
        print(f"\nTop 20 features by importance:")
        for name, imp in importance[:20]:
            print(f"  {name:40s} {imp:.1f}")

    # Save model and results
    if args.mode == "three-split":
        trainer.agg_model.save_model(
            os.path.join(models_dir, f"{save_name}_agg.txt")
        )
        for cat_id, (cat_model, _) in enumerate(trainer.cat_models):
            cat_model.save_model(
                os.path.join(models_dir, f"{save_name}_cat{cat_id:02d}.txt")
            )
    else:
        model.save_model(os.path.join(models_dir, f"{save_name}.txt"))

    results = {
        "mode": args.mode,
        "test_auc": float(auc),
        "cv_aucs": [float(a) for a in aucs],
        "cv_auc_mean": float(np.mean(aucs)) if aucs else None,
        "cv_auc_std": float(np.std(aucs)) if aucs else None,
        "metrics_95_spec": {k: float(v) for k, v in m95.items()},
        "metrics_99_spec": {k: float(v) for k, v in m99.items()},
        "n_features": X.shape[1],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_positive": int(n_pos),
        "n_control": int(n_ctrl),
    }
    with open(os.path.join(models_dir, f"{save_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if importance:
        import csv as _csv
        with open(os.path.join(models_dir, f"{save_name}_feature_importance.csv"),
                  "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["feature", "importance"])
            for name, imp in importance:
                writer.writerow([name, imp])

    print(f"\nModel saved to {models_dir}/")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
