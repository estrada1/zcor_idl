"""
Generate result plots for ZCoR-IPF reproduction.

Produces:
  1. ROC curve (held-out test set, by sex)
  2. Precision-recall curve
  3. Feature importance (top 30)
  4. SLD distribution (positive vs control)
  5. Cross-validation AUC summary
  6. LR+ vs LR- operating characteristics
  7. Calibration (reliability diagram)
  8. Subgroup analysis (AUC by sex and age group)
  9. Ablation study (AUC by feature group)
"""

import numpy as np
import os
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import lightgbm as lgb


FEATURES_DIR = os.path.join("data", "processed", "features")
MODELS_DIR = os.path.join("models")
PLOTS_DIR = os.path.join("results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color palette
C_POS = "#E74C3C"
C_CTRL = "#3498DB"
C_MALE = "#2C3E50"
C_FEMALE = "#E91E63"
C_ACCENT = "#F39C12"


def load_data():
    """Load features, metadata, and trained model."""
    data = np.load(os.path.join(FEATURES_DIR, "feature_matrix.npz"))
    X, y = data["X"], data["y"]

    with open(os.path.join(FEATURES_DIR, "feature_names.txt")) as f:
        feature_names = f.read().strip().split("\n")

    with open(os.path.join(FEATURES_DIR, "pfsa_metadata.csv")) as f:
        metadata = list(csv.DictReader(f))
        for m in metadata:
            m["label"] = int(m["label"])

    with open(os.path.join(MODELS_DIR, "results.json")) as f:
        results = json.load(f)

    model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "zcor_ipf_lgbm.txt"))

    return X, y, feature_names, metadata, results, model


def plot_roc_curves(X, y, metadata, model):
    """Plot ROC curves: overall and by sex."""
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata, test_size=0.25, stratify=y, random_state=42
    )
    y_scores = model.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Overall ROC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_val = roc_auc_score(y_test, y_scores)
    ax1.plot(fpr, tpr, color=C_POS, lw=2.5, label=f"ZCoR-IPF (AUC = {auc_val:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)

    # Mark 95% and 99% specificity points
    for spec_target, marker, ms in [(0.95, "o", 10), (0.99, "s", 10)]:
        idx = np.argmin(np.abs((1 - fpr) - spec_target))
        ax1.plot(fpr[idx], tpr[idx], marker, color=C_ACCENT, ms=ms, zorder=5,
                 markeredgecolor="white", markeredgewidth=1.5,
                 label=f"@ {int(spec_target*100)}% spec (sens={tpr[idx]:.2f})")

    ax1.set_xlabel("1 - Specificity (FPR)", fontsize=12)
    ax1.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax1.set_title("ROC Curve — Held-out Test Set", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # ROC by sex
    sex_arr = np.array([m["sex"] for m in meta_test])
    for sex, color, label in [("M", C_MALE, "Male"), ("F", C_FEMALE, "Female")]:
        mask = sex_arr == sex
        if mask.sum() == 0 or y_test[mask].sum() == 0:
            continue
        fpr_s, tpr_s, _ = roc_curve(y_test[mask], y_scores[mask])
        auc_s = roc_auc_score(y_test[mask], y_scores[mask])
        ax2.plot(fpr_s, tpr_s, color=color, lw=2.5, label=f"{label} (AUC = {auc_s:.3f})")

    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax2.set_xlabel("1 - Specificity (FPR)", fontsize=12)
    ax2.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax2.set_title("ROC Curve — By Sex", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_precision_recall(X, y, model):
    """Plot precision-recall curve."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    y_scores = model.predict(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    prevalence = y_test.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=C_POS, lw=2.5, label="ZCoR-IPF")
    ax.axhline(y=prevalence, color="gray", ls="--", lw=1, alpha=0.5,
               label=f"Prevalence ({prevalence:.3f})")

    # Annotate specific operating points
    for target_recall in [0.9, 0.8, 0.6]:
        idx = np.argmin(np.abs(recall - target_recall))
        if idx < len(precision):
            ax.plot(recall[idx], precision[idx], "o", color=C_ACCENT, ms=8,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"PPV={precision[idx]:.2f}\nSens={recall[idx]:.2f}",
                        (recall[idx], precision[idx]),
                        textcoords="offset points", xytext=(15, 5), fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_precision_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_feature_importance():
    """Plot top 30 feature importances."""
    features = []
    with open(os.path.join(MODELS_DIR, "feature_importance.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append((row["feature"], float(row["importance"])))

    features.sort(key=lambda x: x[1], reverse=True)
    top = features[:30]
    names = [f[0] for f in top][::-1]
    values = [f[1] for f in top][::-1]

    # Color by feature type
    def feature_color(name):
        if name.startswith("sld_"):
            return "#E74C3C"
        elif name.startswith("neg_llk_") or name.startswith("pos_llk_"):
            return "#3498DB"
        elif name.startswith("llk_ratio_"):
            return "#9B59B6"
        elif name.startswith("pscore_"):
            return "#2ECC71"
        elif name.startswith("first_incident") or name.startswith("last_incident"):
            return "#F39C12"
        elif name.startswith("mean_position") or name.startswith("proportion") or name.startswith("prevalence"):
            return "#1ABC9C"
        elif name.startswith("max_streak"):
            return "#E67E22"
        else:
            return "#95A5A6"

    colors = [feature_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_title("Top 30 Feature Importances", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E74C3C", label="SLD"),
        Patch(facecolor="#3498DB", label="Log-likelihood"),
        Patch(facecolor="#9B59B6", label="LLK ratio"),
        Patch(facecolor="#2ECC71", label="P-score"),
        Patch(facecolor="#F39C12", label="First/last incident"),
        Patch(facecolor="#1ABC9C", label="Sequence stats"),
        Patch(facecolor="#95A5A6", label="Aggregate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sld_distribution():
    """Plot SLD score distribution for positive vs control."""
    pfsa_data = np.load(os.path.join(FEATURES_DIR, "pfsa_features.npz"))
    sld = pfsa_data["sld"]

    with open(os.path.join(FEATURES_DIR, "pfsa_metadata.csv")) as f:
        metadata = list(csv.DictReader(f))
        for m in metadata:
            m["label"] = int(m["label"])

    pos_mask = np.array([m["label"] == 1 for m in metadata])
    ctrl_mask = ~pos_mask

    # Mean SLD across all 51 categories per patient
    mean_sld = sld.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of mean SLD
    bins = np.linspace(mean_sld.min(), mean_sld.max(), 80)
    ax1.hist(mean_sld[ctrl_mask], bins=bins, alpha=0.7, color=C_CTRL,
             label=f"Control (n={ctrl_mask.sum():,})", density=True)
    ax1.hist(mean_sld[pos_mask], bins=bins, alpha=0.7, color=C_POS,
             label=f"IPF Positive (n={pos_mask.sum():,})", density=True)
    ax1.axvline(x=0, color="gray", ls="--", lw=1, alpha=0.5)
    ax1.set_xlabel("Mean SLD (across 51 categories)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("SLD Distribution: Positive vs Control", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Per-category SLD comparison (top discriminating categories)
    pos_mean_by_cat = sld[pos_mask].mean(axis=0)
    ctrl_mean_by_cat = sld[ctrl_mask].mean(axis=0)
    separation = pos_mean_by_cat - ctrl_mean_by_cat

    # Import category names
    import sys
    sys.path.insert(0, ".")
    from icd9_categories import get_category_name

    top_cats = np.argsort(np.abs(separation))[::-1][:20]
    cat_names = [get_category_name(c) for c in top_cats][::-1]
    sep_vals = separation[top_cats][::-1]

    colors = [C_POS if v > 0 else C_CTRL for v in sep_vals]
    ax2.barh(range(len(cat_names)), sep_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(cat_names)))
    ax2.set_yticklabels(cat_names, fontsize=9)
    ax2.axvline(x=0, color="gray", lw=1)
    ax2.set_xlabel("SLD Separation (positive − control)", fontsize=12)
    ax2.set_title("Top 20 Discriminating Categories", fontsize=14, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_sld_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cv_summary(results):
    """Plot cross-validation AUC summary."""
    aucs = results["cv_aucs"]
    mean_auc = results["cv_auc_mean"]
    std_auc = results["cv_auc_std"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # CV AUC bars
    folds = range(1, len(aucs) + 1)
    bars = ax1.bar(folds, aucs, color=C_CTRL, edgecolor="white", linewidth=1, alpha=0.8)
    ax1.axhline(y=mean_auc, color=C_POS, ls="--", lw=2,
                label=f"Mean = {mean_auc:.4f} ± {std_auc:.4f}")
    ax1.fill_between([0.5, len(aucs) + 0.5], mean_auc - std_auc, mean_auc + std_auc,
                     color=C_POS, alpha=0.1)
    for i, v in enumerate(aucs):
        ax1.text(i + 1, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax1.set_xlabel("Fold", fontsize=12)
    ax1.set_ylabel("AUC", fontsize=12)
    ax1.set_title("5-Fold Cross-Validation AUC", fontsize=14, fontweight="bold")
    ax1.set_ylim(0.94, 1.0)
    ax1.legend(fontsize=11)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, axis="y", alpha=0.3)

    # Comparison with paper
    comparisons = {
        "This work\n(SynPUF)": results["test_auc"],
        "Paper: Truven\n(M/F)": 0.88,
        "Paper: UCM\n(M/F)": 0.91,
        "Paper: MAYO": 0.86,
    }
    colors_comp = [C_POS, C_CTRL, C_CTRL, C_CTRL]
    hatches = ["", "//", "//", "//"]

    bars2 = ax2.bar(range(len(comparisons)), list(comparisons.values()),
                    color=colors_comp, edgecolor="white", linewidth=1, alpha=0.8)
    for bar, hatch in zip(bars2, hatches):
        bar.set_hatch(hatch)
    ax2.set_xticks(range(len(comparisons)))
    ax2.set_xticklabels(list(comparisons.keys()), fontsize=10)
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_title("AUC Comparison with Original Paper", fontsize=14, fontweight="bold")
    ax2.set_ylim(0.7, 1.0)
    for i, v in enumerate(comparisons.values()):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_cv_and_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_operating_characteristics(X, y, model):
    """Plot sensitivity, PPV, LR+, LR- as functions of specificity threshold."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    y_scores = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    spec = 1 - fpr

    # Compute metrics at various specificity levels
    spec_levels = np.linspace(0.80, 0.995, 100)
    sensitivities = []
    lr_plus_vals = []
    lr_minus_vals = []
    ppv_vals = []

    prevalence = y_test.mean()
    for s in spec_levels:
        idx = np.argmin(np.abs(spec - s))
        sens = tpr[idx]
        sp = spec[idx]
        sensitivities.append(sens)
        lr_p = sens / max(1 - sp, 1e-10)
        lr_m = (1 - sens) / max(sp, 1e-10)
        lr_plus_vals.append(min(lr_p, 200))
        lr_minus_vals.append(lr_m)
        ppv = (sens * prevalence) / max(sens * prevalence + (1 - sp) * (1 - prevalence), 1e-10)
        ppv_vals.append(ppv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Sensitivity vs Specificity
    ax = axes[0, 0]
    ax.plot(spec_levels * 100, sensitivities, color=C_POS, lw=2.5)
    ax.axvline(x=95, color="gray", ls="--", lw=1, alpha=0.5)
    ax.axvline(x=99, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Specificity (%)", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("Sensitivity vs Specificity", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # LR+ vs Specificity
    ax = axes[0, 1]
    ax.plot(spec_levels * 100, lr_plus_vals, color=C_ACCENT, lw=2.5)
    ax.axhline(y=30, color="gray", ls="--", lw=1, alpha=0.5, label="LR+ = 30 (paper target)")
    ax.axvline(x=95, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axvline(x=99, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Specificity (%)", fontsize=12)
    ax.set_ylabel("LR+", fontsize=12)
    ax.set_title("Positive Likelihood Ratio", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # LR- vs Specificity
    ax = axes[1, 0]
    ax.plot(spec_levels * 100, lr_minus_vals, color=C_CTRL, lw=2.5)
    ax.axhline(y=0.7, color="gray", ls="--", lw=1, alpha=0.5, label="LR- = 0.7 (paper target)")
    ax.axvline(x=95, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axvline(x=99, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Specificity (%)", fontsize=12)
    ax.set_ylabel("LR−", fontsize=12)
    ax.set_title("Negative Likelihood Ratio", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # PPV vs Specificity
    ax = axes[1, 1]
    ax.plot(spec_levels * 100, ppv_vals, color="#2ECC71", lw=2.5)
    ax.axvline(x=95, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axvline(x=99, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Specificity (%)", fontsize=12)
    ax.set_ylabel("PPV", fontsize=12)
    ax.set_title("Positive Predictive Value", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Operating Characteristics Across Specificity Thresholds",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_operating_characteristics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_calibration(X, y, model):
    """Plot reliability diagram (calibration curve)."""
    from sklearn.model_selection import train_test_split as _tts
    from evaluate import calibration_analysis

    _, X_test, _, y_test = _tts(X, y, test_size=0.25, stratify=y, random_state=42)
    y_scores = model.predict(X_test)

    cal = calibration_analysis(y_test, y_scores, n_bins=10)
    frac = cal["fraction_of_positives"]
    pred = cal["mean_predicted_value"]
    counts = cal["bin_counts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reliability diagram
    # Only plot bins with data
    has_data = [c > 0 for c in counts]
    x_pts = [p for p, h in zip(pred, has_data) if h]
    y_pts = [f for f, h in zip(frac, has_data) if h]

    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")
    ax1.plot(x_pts, y_pts, "o-", color=C_POS, lw=2, ms=8,
             markeredgecolor="white", markeredgewidth=1.5, label="ZCoR-IPF")
    ax1.fill_between(x_pts, x_pts, y_pts, alpha=0.15, color=C_POS)
    ax1.text(0.05, 0.88,
             f"ECE = {cal['ece']:.4f}\nMCE = {cal['mce']:.4f}",
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title("Reliability Diagram (Calibration)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    ax2.hist(y_scores[y_test == 0], bins=50, color=C_CTRL, alpha=0.7,
             density=True, label=f"Control (n={(y_test==0).sum():,})")
    ax2.hist(y_scores[y_test == 1], bins=20, color=C_POS, alpha=0.8,
             density=True, label=f"Positive (n={(y_test==1).sum():,})")
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Score Distribution by Label", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "07_calibration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_subgroup_analysis(X, y, metadata, model):
    """Bar chart of AUC by sex and age group."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from evaluate import subgroup_analysis

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _, test_idx = next(sss.split(X, y))
    X_test = X[test_idx]
    y_test = y[test_idx]
    meta_test = [metadata[i] for i in test_idx]

    y_scores = model.predict(X_test)
    sg = subgroup_analysis(y_test, y_scores, meta_test)

    # Arrange subgroups in display order
    sex_groups  = ["sex_M", "sex_F"]
    age_groups  = ["age_45_54", "age_55_64", "age_65_74", "age_75_plus"]
    all_groups  = sex_groups + age_groups + ["overall"]
    labels      = ["Male", "Female", "45–54", "55–64", "65–74", "75+", "Overall"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def _bar_group(ax, groups, group_labels, metric, ylabel, title, ylim=(0, 1.05)):
        vals, errs, colors_g = [], [], []
        for g, lbl in zip(groups, group_labels):
            r = sg.get(g, {})
            if r.get("skipped") or metric not in r:
                vals.append(0); errs.append(0); colors_g.append("#cccccc")
            else:
                vals.append(r[metric]); errs.append(0)
                colors_g.append(C_POS if g == "overall" else C_CTRL)
        x = range(len(vals))
        bars = ax.bar(x, vals, color=colors_g, edgecolor="white", linewidth=1, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(*ylim)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate N
        for i, g in enumerate(groups):
            r = sg.get(g, {})
            n = r.get("n", 0)
            np_ = r.get("n_pos", 0)
            ax.text(i, -0.06, f"n={n}\n({np_}+)", ha="center", va="top",
                    fontsize=8, color="gray", transform=ax.get_xaxis_transform())

    _bar_group(ax1, all_groups, labels, "auc", "AUC",
               "AUC by Subgroup", ylim=(0.85, 1.02))
    _bar_group(ax2, all_groups, labels, "sensitivity",
               "Sensitivity @ 95% specificity",
               "Sensitivity by Subgroup (@ 95% spec)", ylim=(0, 1.1))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_subgroup_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_ablation(X, y, feature_names, model):
    """Grouped bar chart of CV AUC by feature set."""
    from sklearn.model_selection import train_test_split as _tts
    from evaluate import ablation_study

    X_train, _, y_train, _ = _tts(X, y, test_size=0.25, stratify=y, random_state=42)

    print("    (Running 3-fold CV ablation — may take a minute...)")
    results = ablation_study(X_train, y_train, X_train, y_train,
                             feature_names, n_folds=3)

    levels = [r["level"] for r in results]
    means  = [r["cv_auc_mean"] for r in results]
    stds   = [r["cv_auc_std"] for r in results]
    n_feats = [r["n_features"] for r in results]

    # Short display labels
    short = ["SLD", "SLD\n+LLK", "SLD+LLK\n+P-score",
             "SLD+LLK\nP-score\n+Seq", "Full"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(means))
    colors_ab = [C_CTRL] * (len(means) - 1) + [C_POS]
    bars = ax1.bar(x, means, yerr=stds, capsize=5,
                   color=colors_ab, edgecolor="white", linewidth=1, alpha=0.85,
                   error_kw=dict(ecolor="gray", lw=1.5))
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(short, fontsize=9)
    ax1.set_ylabel("CV AUC (3-fold)", fontsize=12)
    ax1.set_title("Ablation Study: Feature Group Contribution",
                  fontsize=13, fontweight="bold")
    ymin = max(0.5, min(means) - 0.05)
    ax1.set_ylim(ymin, min(1.01, max(means) + 0.05))
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + s + 0.003,
                 f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Marginal gain
    gains = [means[0]] + [means[i] - means[i - 1] for i in range(1, len(means))]
    gain_colors = [C_ACCENT if g > 0 else "#cccccc" for g in gains]
    ax2.bar(x, gains, color=gain_colors, edgecolor="white", linewidth=1, alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(short, fontsize=9)
    ax2.set_ylabel("Marginal AUC gain", fontsize=12)
    ax2.set_title("Marginal Contribution per Feature Group",
                  fontsize=13, fontweight="bold")
    ax2.axhline(y=0, color="black", lw=0.8)
    for i, (bar, g) in enumerate(zip(ax2.patches, gains)):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 g + (0.001 if g >= 0 else -0.003),
                 f"{g:+.4f}", ha="center",
                 va="bottom" if g >= 0 else "top", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "09_ablation_study.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    print("Generating result plots...")
    print("=" * 50)

    X, y, feature_names, metadata, results, model = load_data()
    print(f"  Loaded {len(y)} patients, {X.shape[1]} features")

    print("\n[1/9] ROC curves...")
    plot_roc_curves(X, y, metadata, model)

    print("[2/9] Precision-recall curve...")
    plot_precision_recall(X, y, model)

    print("[3/9] Feature importance...")
    plot_feature_importance()

    print("[4/9] SLD distribution...")
    plot_sld_distribution()

    print("[5/9] CV summary & paper comparison...")
    plot_cv_summary(results)

    print("[6/9] Operating characteristics...")
    plot_operating_characteristics(X, y, model)

    print("[7/9] Calibration...")
    plot_calibration(X, y, model)

    print("[8/9] Subgroup analysis...")
    plot_subgroup_analysis(X, y, metadata, model)

    print("[9/9] Ablation study...")
    plot_ablation(X, y, feature_names, model)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
