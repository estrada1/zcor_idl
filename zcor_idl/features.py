"""
Feature engineering for ZCoR-IPF.

Computes all 667 features described in the paper (Extended Data Table 3):
- PFSA-based: SLD scores, log-likelihoods, likelihood ratios (51 × 3 = 153)
- Prevalence scores (p-scores): per-phenotype and aggregate (51 × 2 + ~20)
- Sequence scores: statistical summaries of trinary time series (51 × ~6 + ~20)
- Rare scores: subset of p-scores for extreme prevalence ratios
- Aggregate features: summary statistics across all phenotypes
- Demographics: age at screening

In practice we compute a comparable feature set (~400-500 features)
adapted to our dataset, following the same methodology.
"""

import numpy as np
import os
import csv
import time
import argparse


def compute_prevalence_scores(series_array, metadata):
    """
    Compute prevalence scores (p-scores) per disease category.

    P-score for category k = (prevalence of k in positive) / (prevalence of k overall)
    Higher p-score → category is over-represented in IPF patients.

    Returns:
        pscore_dict: array of shape (n_categories,) with p-scores
        patient_pscores: array of shape (n_patients, n_categories) with per-patient p-scores
    """
    n_patients, n_cats, n_weeks = series_array.shape
    pos_mask = np.array([m["label"] == 1 for m in metadata])
    ctrl_mask = ~pos_mask

    # Category prevalence: fraction of weeks with code present (value=1)
    cat_present = (series_array == 1).astype(np.float32)

    # Per-cohort prevalence
    pos_prev = cat_present[pos_mask].mean(axis=(0, 2)) if pos_mask.any() else np.zeros(n_cats)
    ctrl_prev = cat_present[ctrl_mask].mean(axis=(0, 2)) if ctrl_mask.any() else np.zeros(n_cats)

    # P-score: ratio of positive to control prevalence (with smoothing)
    eps = 1e-8
    pscore_dict = (pos_prev + eps) / (ctrl_prev + eps)

    # Per-patient p-scores: for each patient, the mean p-score of categories
    # present in their record
    patient_pscores = np.zeros((n_patients, n_cats), dtype=np.float32)
    for k in range(n_cats):
        # Fraction of weeks with this category present, weighted by p-score
        patient_pscores[:, k] = cat_present[:, k, :].mean(axis=1) * np.log2(pscore_dict[k] + eps)

    return pscore_dict, patient_pscores


def compute_sequence_features(series_array):
    """
    Compute per-category sequence features from trinary time series.

    For each of the 51 categories, computes:
    - proportion: fraction of weeks with category present (value=1)
    - prevalence: fraction of non-empty weeks with category present
    - first_incident: normalized time to first occurrence
    - last_incident: normalized time to last occurrence
    - mean_position: mean normalized position of occurrences
    - streak: longest consecutive run of category present
    """
    n_patients, n_cats, n_weeks = series_array.shape

    proportion = np.zeros((n_patients, n_cats), dtype=np.float32)
    prevalence = np.zeros((n_patients, n_cats), dtype=np.float32)
    first_incident = np.zeros((n_patients, n_cats), dtype=np.float32)
    last_incident = np.zeros((n_patients, n_cats), dtype=np.float32)
    mean_position = np.zeros((n_patients, n_cats), dtype=np.float32)
    max_streak = np.zeros((n_patients, n_cats), dtype=np.float32)

    cat_present = (series_array == 1)  # shape (n, 51, 104)
    any_present = (series_array != 0)  # shape (n, 51, 104)

    # Proportion: fraction of weeks with code
    proportion = cat_present.mean(axis=2).astype(np.float32)

    # Prevalence: fraction of non-empty weeks that have this category
    any_code_weeks = (series_array != 0).any(axis=1)  # (n, 104) — any category present
    n_active_weeks = any_code_weeks.sum(axis=1, keepdims=True).clip(min=1)  # (n, 1)
    prevalence = cat_present.sum(axis=2).astype(np.float32) / n_active_weeks.astype(np.float32)

    # First/last incident and mean position
    for k in range(n_cats):
        for i in range(n_patients):
            positions = np.where(cat_present[i, k, :])[0]
            if len(positions) > 0:
                first_incident[i, k] = positions[0] / n_weeks
                last_incident[i, k] = positions[-1] / n_weeks
                mean_position[i, k] = positions.mean() / n_weeks
                # Longest streak
                diffs = np.diff(positions)
                streak = 1
                cur = 1
                for d in diffs:
                    if d == 1:
                        cur += 1
                        streak = max(streak, cur)
                    else:
                        cur = 1
                max_streak[i, k] = streak / n_weeks
            else:
                first_incident[i, k] = -1
                last_incident[i, k] = -1
                mean_position[i, k] = -1

    return {
        "proportion": proportion,
        "prevalence": prevalence,
        "first_incident": first_incident,
        "last_incident": last_incident,
        "mean_position": mean_position,
        "max_streak": max_streak,
    }


def compute_aggregate_features(sld, neg_llk, pos_llk, patient_pscores, seq_features):
    """
    Compute aggregate features across all 51 categories.

    These are summary statistics (mean, std, max, range) of the per-category
    features, yielding a smaller set of patient-level features.
    """
    features = {}

    # SLD aggregates
    features["sld_mean"] = sld.mean(axis=1)
    features["sld_std"] = sld.std(axis=1)
    features["sld_max"] = sld.max(axis=1)
    features["sld_range"] = sld.max(axis=1) - sld.min(axis=1)

    # Log-likelihood aggregates
    features["neg_llk_mean"] = neg_llk.mean(axis=1)
    features["pos_llk_mean"] = pos_llk.mean(axis=1)
    features["neg_llk_range"] = neg_llk.max(axis=1) - neg_llk.min(axis=1)
    features["pos_llk_range"] = pos_llk.max(axis=1) - pos_llk.min(axis=1)

    # Likelihood ratio aggregates
    llk_ratio = pos_llk / np.clip(neg_llk, 1e-10, None)
    features["llk_ratio_mean"] = llk_ratio.mean(axis=1)
    features["llk_ratio_std"] = llk_ratio.std(axis=1)
    features["llk_ratio_range"] = llk_ratio.max(axis=1) - llk_ratio.min(axis=1)

    # P-score aggregates
    features["pscore_mean"] = patient_pscores.mean(axis=1)
    features["pscore_std"] = patient_pscores.std(axis=1)
    features["pscore_max"] = patient_pscores.max(axis=1)

    # Proportion of high/low p-score codes
    features["high_pscore_prop"] = (patient_pscores > 0.5).mean(axis=1).astype(np.float32)
    features["low_pscore_prop"] = (patient_pscores < -0.5).mean(axis=1).astype(np.float32)

    # Sequence feature aggregates
    for name, arr in seq_features.items():
        features[f"{name}_mean"] = arr.mean(axis=1)
        features[f"{name}_std"] = arr.std(axis=1)

    # Dynamics: compare first half vs second half of p-scores
    n_cats = patient_pscores.shape[1]
    mid = n_cats // 2
    first_half = patient_pscores[:, :mid].mean(axis=1)
    second_half = patient_pscores[:, mid:].mean(axis=1)
    features["pscore_dynamics"] = second_half - first_half

    # Density of record: fraction of weeks with any code
    features["record_density"] = seq_features["proportion"].sum(axis=1)

    return features


def build_feature_matrix(series_array, metadata, sld, neg_llk, pos_llk):
    """
    Build the full feature matrix for LightGBM training.

    Returns:
        X: feature matrix of shape (n_patients, n_features)
        feature_names: list of feature name strings
        y: labels array
    """
    n_patients = len(metadata)
    n_cats = series_array.shape[1]

    print("  Computing prevalence scores...")
    pscore_dict, patient_pscores = compute_prevalence_scores(series_array, metadata)

    print("  Computing sequence features...")
    seq_features = compute_sequence_features(series_array)

    print("  Computing aggregate features...")
    agg_features = compute_aggregate_features(
        sld, neg_llk, pos_llk, patient_pscores, seq_features
    )

    # ── Assemble feature matrix ──
    feature_blocks = []
    feature_names = []

    # 1. SLD per category (51 features)
    feature_blocks.append(sld)
    feature_names.extend([f"sld_{k}" for k in range(n_cats)])

    # 2. Negative log-likelihood per category (51 features)
    feature_blocks.append(neg_llk)
    feature_names.extend([f"neg_llk_{k}" for k in range(n_cats)])

    # 3. Positive log-likelihood per category (51 features)
    feature_blocks.append(pos_llk)
    feature_names.extend([f"pos_llk_{k}" for k in range(n_cats)])

    # 4. Log-likelihood ratio per category (51 features)
    llk_ratio = pos_llk / np.clip(neg_llk, 1e-10, None)
    feature_blocks.append(llk_ratio)
    feature_names.extend([f"llk_ratio_{k}" for k in range(n_cats)])

    # 5. Patient p-scores per category (51 features)
    feature_blocks.append(patient_pscores)
    feature_names.extend([f"pscore_{k}" for k in range(n_cats)])

    # 6. Sequence features per category (51 × 6 = 306 features)
    for name, arr in seq_features.items():
        feature_blocks.append(arr)
        feature_names.extend([f"{name}_{k}" for k in range(n_cats)])

    # 7. Aggregate features
    for name, arr in agg_features.items():
        feature_blocks.append(arr.reshape(-1, 1))
        feature_names.append(name)

    # 8. Age at screening
    ages = np.array([float(m["age_at_screening"]) for m in metadata]).reshape(-1, 1)
    feature_blocks.append(ages)
    feature_names.append("age_at_screening")

    # 9. Sex (binary: 1=M, 0=F)
    sex = np.array([1.0 if m["sex"] == "M" else 0.0 for m in metadata]).reshape(-1, 1)
    feature_blocks.append(sex)
    feature_names.append("sex_male")

    X = np.hstack(feature_blocks).astype(np.float32)
    y = np.array([m["label"] for m in metadata], dtype=np.int32)

    print(f"  Feature matrix: {X.shape} ({len(feature_names)} features)")
    return X, feature_names, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ZCoR-IPF feature matrix")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use only N patients (0=all)")
    args = parser.parse_args()

    encoded_dir = os.path.join("data", "processed", "encoded")
    features_dir = os.path.join("data", "processed", "features")

    print("Loading data...")
    data = np.load(os.path.join(encoded_dir, "trinary_series.npz"))
    series = data["series"]

    pfsa_data = np.load(os.path.join(features_dir, "pfsa_features.npz"))
    sld = pfsa_data["sld"]
    neg_llk = pfsa_data["neg_llk"]
    pos_llk = pfsa_data["pos_llk"]

    with open(os.path.join(features_dir, "pfsa_metadata.csv")) as f:
        metadata = list(csv.DictReader(f))
        for m in metadata:
            m["label"] = int(m["label"])

    # Align series with pfsa metadata (in case subset was used for PFSA)
    if series.shape[0] != len(metadata):
        # Re-load and match
        with open(os.path.join(encoded_dir, "patient_metadata.csv")) as f:
            full_meta = list(csv.DictReader(f))
        pfsa_ids = {m["patient_id"] for m in metadata}
        keep_idx = [i for i, m in enumerate(full_meta) if m["patient_id"] in pfsa_ids]
        series = series[keep_idx]

    if args.subset > 0 and args.subset < len(metadata):
        series = series[:args.subset]
        metadata = metadata[:args.subset]
        sld = sld[:args.subset]
        neg_llk = neg_llk[:args.subset]
        pos_llk = pos_llk[:args.subset]

    print(f"  Series: {series.shape}, SLD: {sld.shape}, Patients: {len(metadata)}")

    t0 = time.time()
    X, feature_names, y = build_feature_matrix(series, metadata, sld, neg_llk, pos_llk)
    elapsed = time.time() - t0
    print(f"\nFeature engineering completed in {elapsed:.1f}s")
    print(f"  Positive: {y.sum()}, Control: {(1-y).sum()}")

    # Save
    np.savez_compressed(
        os.path.join(features_dir, "feature_matrix.npz"),
        X=X, y=y,
    )
    with open(os.path.join(features_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    print(f"Saved to {features_dir}/feature_matrix.npz ({X.shape[1]} features)")
