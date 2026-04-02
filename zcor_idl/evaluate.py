"""
Post-hoc evaluation for ZCoR-IPF.

Three analyses from NEXT_STEPS section 4:

1. calibration_analysis    — reliability diagram + Expected Calibration Error
2. subgroup_analysis       — AUC / sensitivity / PPV stratified by sex and age group
3. ablation_study          — AUC by feature group:
     SLD only → + p-scores → + sequence features → full feature set

All functions operate on numpy arrays (X, y, scores) and optionally on
metadata dicts so they can be run without re-training.

Typical usage
-------------
    python evaluate.py --features data/processed/features \
                       --model models/zcor_ipf_lgbm.txt \
                       --metadata data/processed/features/pfsa_metadata.csv
"""

import argparse
import csv
import json
import os

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from zcor_idl.train import (
    evaluate_at_specificity,
    train_and_evaluate,
    _category_feature_indices,
    _aggregate_feature_indices,
)


# ── 1. Calibration ────────────────────────────────────────────────────────────

def calibration_analysis(y_true, y_scores, n_bins=10):
    """
    Compute calibration statistics for a set of predicted probabilities.

    Returns a dict with:
      - fraction_of_positives: array of shape (n_bins,) — empirical positive rate
      - mean_predicted_value:  array of shape (n_bins,) — mean predicted probability
      - ece:                   Expected Calibration Error (uniform binning)
      - mce:                   Maximum Calibration Error
      - bin_counts:            number of samples per bin
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_scores, n_bins=n_bins, strategy="uniform"
    )

    # Bin edges
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_scores, bins[1:-1])
    bin_counts = np.bincount(bin_ids, minlength=n_bins)[:n_bins]

    # ECE: weighted average |empirical - predicted|
    n = len(y_true)
    # align arrays (calibration_curve may skip empty bins)
    full_frac  = np.zeros(n_bins)
    full_pred  = np.zeros(n_bins)
    full_frac[:len(frac_pos)]  = frac_pos
    full_pred[:len(mean_pred)] = mean_pred

    ece = np.sum(bin_counts / max(n, 1) * np.abs(full_frac - full_pred))
    mce = np.max(np.abs(full_frac - full_pred))

    return {
        "fraction_of_positives": full_frac.tolist(),
        "mean_predicted_value":  full_pred.tolist(),
        "bin_counts":            bin_counts.tolist(),
        "ece":                   float(ece),
        "mce":                   float(mce),
        "n_bins":                n_bins,
    }


def print_calibration(cal):
    """Pretty-print calibration results."""
    print(f"  ECE: {cal['ece']:.4f}   MCE: {cal['mce']:.4f}")
    print(f"  {'Bin':>5}  {'Pred':>6}  {'Emp':>6}  {'N':>6}")
    for i in range(cal["n_bins"]):
        print(f"  {i:5d}  {cal['mean_predicted_value'][i]:6.3f}"
              f"  {cal['fraction_of_positives'][i]:6.3f}"
              f"  {cal['bin_counts'][i]:6d}")


# ── 2. Subgroup analysis ──────────────────────────────────────────────────────

def subgroup_analysis(y_true, y_scores, metadata, target_spec=0.95):
    """
    Stratified performance by sex and age group.

    Args:
        y_true    : 1-D int array (0/1)
        y_scores  : 1-D float array of predicted probabilities
        metadata  : list of dicts with keys 'sex' and 'age_at_screening'
        target_spec: specificity target for threshold-dependent metrics

    Returns dict of subgroup → {auc, n, n_pos, sensitivity, ppv, lr_plus}
    """
    y_true   = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Define subgroup masks
    sexes = np.array([m["sex"] for m in metadata])
    ages  = np.array([float(m["age_at_screening"]) for m in metadata])

    subgroups = {
        "sex_M":        sexes == "M",
        "sex_F":        sexes == "F",
        "age_45_54":    (ages >= 45) & (ages < 55),
        "age_55_64":    (ages >= 55) & (ages < 65),
        "age_65_74":    (ages >= 65) & (ages < 75),
        "age_75_plus":  ages >= 75,
        "overall":      np.ones(len(y_true), dtype=bool),
    }

    results = {}
    for name, mask in subgroups.items():
        yt = y_true[mask]
        ys = y_scores[mask]
        n     = int(mask.sum())
        n_pos = int(yt.sum())
        if n_pos == 0 or (n - n_pos) == 0 or n < 10:
            results[name] = {"n": n, "n_pos": n_pos, "skipped": True}
            continue

        auc = roc_auc_score(yt, ys)
        m   = evaluate_at_specificity(yt, ys, target_spec)
        results[name] = {
            "n":           n,
            "n_pos":       n_pos,
            "auc":         float(auc),
            "sensitivity": float(m["sensitivity"]),
            "specificity": float(m["specificity"]),
            "ppv":         float(m["ppv"]),
            "npv":         float(m["npv"]),
            "lr_plus":     float(m["lr_plus"]),
            "lr_minus":    float(m["lr_minus"]),
        }

    return results


def print_subgroup(results, target_spec=0.95):
    """Pretty-print subgroup analysis."""
    hdr = f"{'Subgroup':<16}  {'N':>6}  {'Pos':>5}  {'AUC':>6}  "
    hdr += f"{'Sens':>6}  {'PPV':>6}  {'LR+':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        if r.get("skipped"):
            print(f"  {name:<14}  {r['n']:6d}  {r['n_pos']:5d}  (insufficient data)")
        else:
            print(f"  {name:<14}  {r['n']:6d}  {r['n_pos']:5d}"
                  f"  {r['auc']:6.3f}  {r['sensitivity']:6.3f}"
                  f"  {r['ppv']:6.3f}  {r['lr_plus']:6.1f}")


# ── 3. Ablation study ─────────────────────────────────────────────────────────

# Feature group definitions — each is a list of name prefixes/exact-names
# that belong to the group.  Groups are *cumulative*: each level adds on top
# of the previous so we can measure marginal contribution.
_ABLATION_LEVELS = [
    (
        "SLD only",
        ["sld_"],
    ),
    (
        "SLD + log-likelihoods",
        ["sld_", "neg_llk_", "pos_llk_", "llk_ratio_"],
    ),
    (
        "SLD + LLK + p-scores",
        ["sld_", "neg_llk_", "pos_llk_", "llk_ratio_", "pscore_"],
    ),
    (
        "SLD + LLK + p-scores + sequence",
        ["sld_", "neg_llk_", "pos_llk_", "llk_ratio_", "pscore_",
         "proportion_", "prevalence_", "first_incident_",
         "last_incident_", "mean_position_", "max_streak_"],
    ),
    (
        "Full (all features)",
        None,   # None means use all columns
    ),
]


def _feature_mask(feature_names, prefixes):
    """Return a boolean mask for columns whose name starts with any prefix."""
    names = np.array(feature_names)
    mask = np.zeros(len(names), dtype=bool)
    for pfx in prefixes:
        mask |= np.array([n.startswith(pfx) for n in names])
    return mask


def ablation_study(X_train, y_train, X_test, y_test, feature_names, n_folds=3):
    """
    Train a LightGBM on progressively richer feature sets and report AUC.

    For each ablation level a stratified k-fold CV AUC is computed on the
    training set; the held-out test AUC is computed only for the full model
    (to avoid over-interpreting results on a small test set).

    Returns list of dicts: [{level, n_features, cv_auc_mean, cv_auc_std}]
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for level_name, prefixes in _ABLATION_LEVELS:
        if prefixes is None:
            col_mask = np.ones(len(feature_names), dtype=bool)
        else:
            col_mask = _feature_mask(feature_names, prefixes)
            # Always include aggregate and demographic features
            agg_idx = _aggregate_feature_indices(feature_names)
            col_mask[agg_idx] = True

        X_sub = X_train[:, col_mask]
        n_feats = int(col_mask.sum())

        fold_aucs = []
        for train_idx, val_idx in skf.split(X_sub, y_train):
            X_tr, X_va = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            _, val_scores = train_and_evaluate(
                X_tr, y_tr, X_va, y_va
            )[:3:2]  # (model, auc, scores, ...) → take auc at [1] and scores at [2]
            # Actually train_and_evaluate returns (model, auc, y_scores, m95, m99, imp)
            fold_aucs.append(roc_auc_score(y_va, val_scores))

        results.append({
            "level":        level_name,
            "n_features":   n_feats,
            "cv_auc_mean":  float(np.mean(fold_aucs)),
            "cv_auc_std":   float(np.std(fold_aucs)),
        })
        print(f"  [{level_name}]  n_feat={n_feats}  "
              f"CV AUC={np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    return results


def _ablation_val_scores(X_tr, y_tr, X_va, y_va):
    """Helper: train and return val scores."""
    _, _, y_scores, _, _, _ = train_and_evaluate(X_tr, y_tr, X_va, y_va)
    return y_scores


def print_ablation(results):
    """Pretty-print ablation table."""
    print(f"  {'Feature set':<40}  {'N feat':>7}  {'CV AUC':>8}  {'Std':>6}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['level']:<40}  {r['n_features']:7d}"
              f"  {r['cv_auc_mean']:8.4f}  {r['cv_auc_std']:6.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZCoR-IPF evaluation suite")
    parser.add_argument("--features", default=os.path.join("data", "processed", "features"),
                        help="Directory containing feature_matrix.npz / feature_names.txt")
    parser.add_argument("--metadata", default=None,
                        help="Path to pfsa_metadata.csv (required for subgroup analysis)")
    parser.add_argument("--model", default=None,
                        help="Path to a saved LightGBM model (.txt). "
                             "If omitted, a model is trained on 75/25 split.")
    parser.add_argument("--analyses", nargs="+",
                        choices=["calibration", "subgroup", "ablation", "all"],
                        default=["all"])
    parser.add_argument("--output", default=None,
                        help="Directory to write JSON results (optional)")
    args = parser.parse_args()

    run_all    = "all" in args.analyses
    do_cal     = run_all or "calibration" in args.analyses
    do_subgrp  = run_all or "subgroup"    in args.analyses
    do_ablation = run_all or "ablation"   in args.analyses

    # ── Load features ─────────────────────────────────────────────────────────
    print("Loading features...")
    data = np.load(os.path.join(args.features, "feature_matrix.npz"))
    X, y = data["X"], data["y"]
    with open(os.path.join(args.features, "feature_names.txt")) as f:
        feature_names = f.read().strip().split("\n")
    print(f"  {X.shape[0]} patients, {X.shape[1]} features, {y.sum()} positive")

    # ── Load metadata ──────────────────────────────────────────────────────────
    metadata = None
    meta_path = args.metadata or os.path.join(args.features, "pfsa_metadata.csv")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = list(csv.DictReader(f))
        print(f"  Metadata loaded: {len(metadata)} patients")
    elif do_subgrp:
        print(f"  WARNING: metadata file not found at {meta_path}; "
              "skipping subgroup analysis")
        do_subgrp = False

    # ── Train/load model and get test scores ──────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    meta_test = None
    if metadata:
        # Reconstruct the same 75/25 split for metadata
        n = len(y)
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        # Replicate sklearn's stratified split deterministically
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        meta_test = [metadata[i] for i in test_idx]

    if args.model and os.path.exists(args.model):
        print(f"\nLoading model from {args.model}...")
        model = lgb.Booster(model_file=args.model)
        y_scores = model.predict(X_test)
    else:
        print("\nTraining model (75/25 split)...")
        model, test_auc, y_scores, _, _, _ = train_and_evaluate(
            X_train, y_train, X_test, y_test, feature_names
        )
        print(f"  Test AUC: {test_auc:.4f}")

    all_results = {}

    # ── 1. Calibration ────────────────────────────────────────────────────────
    if do_cal:
        print(f"\n{'='*60}")
        print("CALIBRATION ANALYSIS")
        print(f"{'='*60}")
        cal = calibration_analysis(y_test, y_scores)
        print_calibration(cal)
        all_results["calibration"] = cal

    # ── 2. Subgroup analysis ──────────────────────────────────────────────────
    if do_subgrp and meta_test:
        print(f"\n{'='*60}")
        print("SUBGROUP ANALYSIS  (@ 95% specificity)")
        print(f"{'='*60}")
        sg = subgroup_analysis(y_test, y_scores, meta_test)
        print_subgroup(sg)
        all_results["subgroup"] = sg

    # ── 3. Ablation study ─────────────────────────────────────────────────────
    if do_ablation:
        print(f"\n{'='*60}")
        print("ABLATION STUDY  (3-fold CV on training set)")
        print(f"{'='*60}")
        abl = ablation_study(X_train, y_train, X_test, y_test,
                             feature_names, n_folds=3)
        print_ablation(abl)
        all_results["ablation"] = abl

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
