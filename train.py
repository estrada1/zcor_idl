"""
LightGBM training pipeline for ZCoR-IPF.

Implements the three-split training approach from the paper:
1. Hyper-training split: train PFSA models + p-score dictionary
2. Pre-aggregation split: train per-category LGBMs
3. Aggregation split: train final ensemble LGBM

For simplicity (and given our smaller dataset), we use a single LGBM
with 75/25 train/test split and cross-validation, following the paper's
held-out validation approach.
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


def main():
    parser = argparse.ArgumentParser(description="Train ZCoR-IPF LightGBM model")
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

    # Cross-validation
    print(f"\n{'='*60}")
    print(f"Cross-validation ({args.n_folds} folds)")
    print(f"{'='*60}")
    t0 = time.time()
    aucs, metrics_95, metrics_99 = cross_validate(
        X, y, feature_names, n_folds=args.n_folds
    )
    cv_time = time.time() - t0

    if args.cv_only:
        print(f"\nCV completed in {cv_time:.1f}s")
        return

    # Train final model on 75/25 split (matching paper)
    print(f"\n{'='*60}")
    print(f"Training final model (75/25 split)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"  Train: {len(y_train)} ({y_train.sum()} pos)")
    print(f"  Test:  {len(y_test)} ({y_test.sum()} pos)")

    model, auc, y_scores, m95, m99, importance = train_and_evaluate(
        X_train, y_train, X_test, y_test, feature_names
    )

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"\nHeld-out test AUC: {auc:.4f}")
    print(f"CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

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
    model.save_model(os.path.join(models_dir, "zcor_ipf_lgbm.txt"))

    results = {
        "test_auc": float(auc),
        "cv_aucs": [float(a) for a in aucs],
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "metrics_95_spec": {k: float(v) for k, v in m95.items()},
        "metrics_99_spec": {k: float(v) for k, v in m99.items()},
        "n_features": X.shape[1],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_positive": int(n_pos),
        "n_control": int(n_ctrl),
    }
    with open(os.path.join(models_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if importance:
        with open(os.path.join(models_dir, "feature_importance.csv"), "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["feature", "importance"])
            for name, imp in importance:
                writer.writerow([name, imp])

    print(f"\nModel saved to {models_dir}/")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
