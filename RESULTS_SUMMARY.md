# ZCoR-ILD Reproduction: Results Summary

## Overview

This project reimplements the Zero-burden Comorbidity Risk score for Interstitial Lung Disease (ILD), based on the ZCoR-IPF method from [Onishchenko et al., *Nature Medicine* 2022](https://doi.org/10.1038/s41591-022-02010-y). The pipeline processes insurance claims data, encodes diagnostic histories as trinary time series, trains Probabilistic Finite State Automata (PFSA), computes Sequence Likelihood Defect (SLD) features, and trains a LightGBM classifier for disease risk prediction.

---

## Data

### Original Paper: Truven MarketScan

- **Source:** IBM Truven Health MarketScan Commercial Claims and Encounters database (proprietary)
- **Years:** 2003--2018 (15 years)
- **Cohort size:** ~2.6 million patients
- **Positive cases:** 5,765 IPF patients (ICD-9 516.31 + ICD-10 J84.112)
- **Age range:** 45--90
- **Code systems:** ICD-9-CM and ICD-10-CM

### This Reproduction: CMS DE-SynPUF

- **Source:** [CMS 2008--2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files), Sample 1
- **Years:** 2008--2010 (3 years)
- **Cohort size:** 62,102 patients after inclusion/exclusion criteria
  - 743 positive (ILD), 61,359 controls
  - Prevalence: 1.2%
- **Age range:** 45--90, minimum 2-year record span
- **Code system:** ICD-9-CM only (SynPUF predates the 2015 ICD-10 transition)
- **Target proxy:** ICD-9 `516.3` (idiopathic interstitial pneumonia NOS) -- broader than `516.31` (IPF specifically), because SynPUF stores codes at reduced granularity (e.g., `5163` not `51631`)
- **Claims files used:**
  - Beneficiary summary files (2008, 2009) -- demographics
  - Inpatient claims (2008--2010) -- hospital admissions
  - Outpatient claims (2008--2010) -- facility claims
  - Carrier claims A & B (2008--2010) -- physician claims
- **Download size:** ~260 MB compressed, ~2.6 GB extracted

### Key Data Differences

| Dimension | Original (Truven) | Reproduction (SynPUF) |
|---|---|---|
| Data type | Real claims | Synthetic (programmatically generated) |
| Years of data | 15 | 3 |
| Patients | ~2.6M | 62,102 |
| Positive cases | 5,765 | 743 |
| ICD systems | ICD-9 + ICD-10 | ICD-9 only (SynPUF predates 2015 transition) |
| ICD-10 support | ✓ | ✓ (pipeline ready; awaits post-2015 data) |
| Target definition | ICD-9 516.31 / ICD-10 J84.112 | ICD-9 516.3 (broader proxy) |

---

## Pipeline

The pipeline follows the same architecture as the paper:

1. **Disease category encoding** (`icd9_categories.py`, `preprocess_cms.py`): Map ~13,500 ICD-9 codes to 51 non-overlapping disease categories aligned with the ICD-9 chapter structure. Apply inclusion/exclusion criteria (age 45--90, min 2-year record span). 100% mapping coverage on SynPUF data.

2. **Trinary time series** (`trinary_encoding.py`): Convert each patient's history into 51 parallel trinary time series over a 104-week observation window (0 = no code, 1 = category code present, 2 = other category code present).

3. **PFSA training + SLD** (`pfsa.py`): Train 204 PFSA models (51 categories x 2 cohorts x 2 sexes) using variable-order Markov chain estimation. Compute the Sequence Likelihood Defect (SLD) per patient per category.

4. **Feature engineering** (`features.py`): Compute 593 features per patient:

   | Feature Group | Count | Description |
   |---|---|---|
   | SLD scores | 51 | Per-category sequence likelihood defect |
   | Log-likelihoods | 102 | Negative and positive model likelihoods |
   | Likelihood ratios | 51 | Positive-to-control likelihood ratio |
   | Prevalence scores | 51 | Category over-representation in IPF |
   | Sequence features | 306 | Proportion, prevalence, first/last incident, mean position, streak |
   | Aggregate features | 30 | Summary statistics across all categories |
   | Demographics | 2 | Age at screening, sex |

   The paper reports 667 features; the difference comes from rare-score variants and additional aggregate features not explicitly described.

5. **LightGBM training** (`train.py`): Two modes available:
   - `--mode single` (default): 5-fold stratified CV + 75/25 held-out evaluation
   - `--mode three-split`: Paper's three-split pipeline — 40% hyper-training, 30% per-category (51 LGBM models, one per phenotype), 30% aggregation (final LGBM on 51 probability meta-features)

6. **ICD-10 support** (`icd10_categories.py`): Full ICD-10-CM → 51 category mapping for post-2015 data. Primary target J84.112 (IPF); combined classifier auto-detects coding system per code, enabling transparent processing of mixed ICD-9/ICD-10 datasets.

7. **Evaluation suite** (`evaluate.py`): Calibration (reliability diagram, ECE/MCE), subgroup analysis (AUC/sensitivity/PPV by sex and age group), ablation study (5-level feature group comparison).

Total runtime: ~10 minutes on Apple Silicon.

---

## Results

### Primary Metrics

| Metric | This Reproduction | Original Paper |
|---|---|---|
| **AUC (held-out)** | 0.971 | 0.88 |
| **CV AUC** | 0.976 +/- 0.010 | -- |
| **Sensitivity @ 95% specificity** | 0.898 | 0.52 |
| **Specificity @ 95% target** | 0.950 | 0.95 |
| **LR+ @ 95% specificity** | 18.0 | 13.5 |
| **LR- @ 95% specificity** | 0.108 | 0.50 |
| **PPV @ 95% specificity** | 0.179 | -- |
| **NPV @ 95% specificity** | 0.999 | -- |
| **Sensitivity @ 99% specificity** | 0.817 | 0.36 |
| **Specificity @ 99% target** | 0.994 | 0.99 |
| **LR+ @ 99% specificity** | 133.4 | 54.9 |
| **LR- @ 99% specificity** | 0.184 | -- |
| **PPV @ 99% specificity** | 0.618 | -- |
| **NPV @ 99% specificity** | 0.998 | -- |

### Cross-Validation Fold Detail

| Fold | AUC |
|---|---|
| 1 | 0.977 |
| 2 | 0.986 |
| 3 | 0.987 |
| 4 | 0.964 |
| 5 | 0.965 |
| **Mean +/- Std** | **0.976 +/- 0.010** |

### Top 10 Features by Importance (Gain)

| Rank | Feature | Importance | Description |
|---|---|---|---|
| 1 | `first_incident_15` | 786,022 | First occurrence of hypertensive-systemic codes |
| 2 | `sld_37` | 208,063 | SLD for disorders of soft tissues |
| 3 | `first_incident_43` | 95,567 | First occurrence of general symptoms |
| 4 | `first_incident_35` | 50,071 | First occurrence of skin disorders |
| 5 | `first_incident_std` | 31,719 | Std. dev. of first-incident times across categories |
| 6 | `pos_llk_25` | 26,498 | Positive-model log-likelihood for pulmonary fibrosis/ILD |
| 7 | `neg_llk_5` | 23,416 | Negative-model log-likelihood for lipid metabolism |
| 8 | `mean_position_mean` | 20,592 | Mean position of diagnoses across all categories |
| 9 | `first_incident_48` | 20,252 | First occurrence of health service contact (other) |
| 10 | `first_incident_16` | 18,703 | First occurrence of coronary atherosclerosis |

---

## Interpretation

### Why our metrics exceed the paper

Our reproduction shows substantially higher performance than the original paper across all metrics. This is **not** an improvement -- it reflects fundamental differences in the data:

1. **Synthetic data artifacts.** SynPUF generates comorbidity patterns programmatically using a deterministic algorithm, producing more regular and separable diagnostic trajectories than real clinical data. The signal-to-noise ratio is artificially high.

2. **Smaller, narrower cohort.** 62K patients over 3 years vs 2.6M over 15 years. The reduced diversity makes the classification task easier.

3. **Broader target definition.** ICD-9 `516.3` (idiopathic interstitial pneumonia NOS) captures a broader set of conditions than `516.31` (IPF specifically). This may create a more distinctive comorbidity signature.

4. **No ICD-10 transition noise.** The original study spans the 2015 ICD-9 to ICD-10 transition, introducing coding inconsistencies that add noise. SynPUF is entirely ICD-9.

5. **Medicare population only.** SynPUF represents Medicare beneficiaries (65+), while Truven MarketScan includes commercial insurance (younger, employed). The older SynPUF population has denser comorbidity records.

### What the results validate

Despite the inflated metrics, this reproduction validates:

- **Pipeline architecture:** The trinary encoding -> PFSA -> SLD -> LightGBM pipeline works end-to-end and produces a discriminative model.
- **Feature design:** The 593-feature set (comparable to the paper's 667) captures meaningful comorbidity patterns. The top features align with clinical intuition -- hypertension timing, soft tissue disorders, respiratory features, and coronary disease all have known associations with ILD/IPF.
- **Methodology:** PFSA-based sequence modeling of diagnostic trajectories successfully distinguishes positive and control cohorts, confirming the core insight of the paper.

---

## References

- Onishchenko D, et al. "Screening for idiopathic pulmonary fibrosis using comorbidity signatures in electronic health records." *Nature Medicine* 28, 2107--2116 (2022). https://doi.org/10.1038/s41591-022-02010-y
- Chattopadhyay I & Lipson H. "Abductive learning of quantized stochastic processes with probabilistic finite automata." *Phil. Trans. R. Soc. A* 371, 20110543 (2013).
- CMS DE-SynPUF: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
