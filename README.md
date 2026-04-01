# ZCoR-ILD

A Python reimplementation of the Zero-burden Comorbidity Risk score for Interstitial Lung Disease (ILD), based on the ZCoR-IPF method described in [Onishchenko et al., Nature Medicine 2022](https://doi.org/10.1038/s41591-022-02010-y).

The pipeline processes claims data, encodes diagnostic histories as trinary time series, trains Probabilistic Finite State Automata (PFSA) models, computes Sequence Likelihood Defect (SLD) features, and trains a LightGBM classifier for disease risk prediction.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- macOS: `brew install libomp` (required by LightGBM)

## Installation

```bash
uv sync
```

This installs all dependencies (numpy, pandas, scikit-learn, lightgbm, scipy) into a local virtual environment.

## Data Download

The pipeline uses the **CMS DE-SynPUF** (2008-2010 Synthetic Medicare Claims) as a freely available substitute for the proprietary Truven MarketScan database used in the original paper.

Run the download script to fetch Sample 1 (~260 MB download, ~2.6 GB extracted):

```bash
bash download_data.sh
```

This downloads and extracts the following files into `data/raw/extracted/`:

| File | Description | Size |
|---|---|---|
| `DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv` | Patient demographics (2008) | 14 MB |
| `DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv` | Patient demographics (2009) | 14 MB |
| `DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv` | Hospital inpatient claims | 16 MB |
| `DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv` | Outpatient facility claims | 154 MB |
| `DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv` | Physician claims (part A) | 1.2 GB |
| `DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv` | Physician claims (part B) | 1.2 GB |

Source: [CMS SynPUF website](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf). Also downloads the codebook and data user's guide PDFs.

### Manual Download (if script fails)

Visit the [CMS DE-SynPUF Sample 1 page](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1), download all zip files, and extract CSVs into `data/raw/extracted/`.

## Usage

Run each step sequentially:

```bash
# 1. Preprocess claims data → ZCoR format (~3 min)
uv run python preprocess_cms.py

# 2. Encode trinary time series (~2 min)
uv run python trinary_encoding.py

# 3. Train PFSA models + compute SLD features (~4 min)
uv run python pfsa.py

# 4. Build feature matrix (~10 sec)
uv run python features.py

# 5. Train LightGBM + evaluate (~40 sec)
uv run python train.py
```

Total runtime: approximately 10 minutes on Apple Silicon.

### Subset Mode (Fast Iteration)

For development or testing, use `--subset N` to limit patient count. All positive (IPF) patients are always retained; only controls are subsampled.

```bash
uv run python pfsa.py --subset 2000       # ~9 sec instead of ~4 min
uv run python features.py --subset 2000
uv run python train.py
```

## Pipeline Overview

### Step 1: Disease Category Encoding

**`icd9_categories.py`** — Maps ~13,500 unique ICD-9 codes to 51 non-overlapping disease categories, aligning with the ICD-9 chapter structure. Achieves 100% coverage on SynPUF claims data.

**`preprocess_cms.py`** — Reads raw CMS files, applies inclusion/exclusion criteria (age 45-90, minimum 2-year record span), identifies positive (ICD-9 516.3) and control cohorts, and writes `.dat` files.

**`trinary_encoding.py`** — Converts each patient's history into 51 parallel trinary time series over a 104-week observation window:
- `0` = no diagnosis code that week
- `1` = code from this category present
- `2` = code from a different category present

### Step 2: PFSA Inference + SLD Computation

**`pfsa.py`** — Trains **204 PFSA models** (51 categories x 2 cohorts x 2 sexes) using variable-order Markov chain estimation. Computes the **Sequence Likelihood Defect (SLD)** per patient per category, measuring how much more likely a patient's diagnostic trajectory matches the IPF cohort vs. controls. Uses vectorized batch scoring for performance.

### Step 3: Feature Engineering + LightGBM

**`features.py`** — Computes **593 features** per patient:

| Feature Group | Count | Description |
|---|---|---|
| SLD scores | 51 | Per-category sequence likelihood defect |
| Log-likelihoods | 102 | Negative and positive model likelihoods |
| Likelihood ratios | 51 | Positive-to-control likelihood ratio |
| Prevalence scores | 51 | Category over-representation in IPF |
| Sequence features | 306 | Proportion, prevalence, first/last incident, mean position, streak |
| Aggregate features | 30 | Summary statistics across all categories |
| Demographics | 2 | Age at screening, sex |

**`train.py`** — Trains a LightGBM classifier with 5-fold stratified cross-validation and 75/25 held-out evaluation. Handles class imbalance via `scale_pos_weight`. Reports AUC, sensitivity, specificity, LR+, LR-, PPV, NPV at 95% and 99% specificity thresholds.

## Results on CMS DE-SynPUF

| Metric | This Reproduction | Original Paper (Truven) |
|---|---|---|
| AUC (held-out) | 0.971 | 0.88 |
| CV AUC | 0.976 +/- 0.010 | — |
| Sensitivity @ 95% specificity | 0.898 | 0.52 |
| LR+ @ 95% specificity | 18.0 | 13.5 |
| LR- @ 95% specificity | 0.108 | 0.50 |
| Sensitivity @ 99% specificity | 0.817 | 0.36 |
| LR+ @ 99% specificity | 133.4 | 54.9 |

Our higher metrics on synthetic data likely reflect that SynPUF's programmatic generation creates more regular comorbidity patterns than real clinical data. These numbers should not be taken as an improvement over the original paper.

## Data Caveats

- **Synthetic data**: comorbidity co-occurrence patterns are programmatically generated, not from real patients
- **ICD-9 only**: SynPUF predates the US ICD-10 transition (2015); codes are stored at reduced granularity (e.g., `5163` not `51631`)
- **3 years of data**: 2008-2010 vs 15 years in the original study
- **Target proxy**: ICD-9 `516.3` (idiopathic interstitial pneumonia NOS) is broader than `516.31` (IPF specifically)
- **No ICD-10 codes**: the original study used both ICD-9 and ICD-10

## Directory Structure

```
├── download_data.sh           # Downloads CMS DE-SynPUF data
├── preprocess_cms.py          # Raw CSV → ZCoR .dat format
├── icd9_categories.py         # ICD-9 → 51 disease categories
├── trinary_encoding.py        # .dat → trinary time series
├── pfsa.py                    # PFSA training + SLD computation
├── features.py                # Feature engineering (593 features)
├── train.py                   # LightGBM training + evaluation
├── pyproject.toml             # Python dependencies
├── data/
│   ├── raw/extracted/         # CMS CSV files (not committed)
│   └── processed/
│       ├── synpuf_all.dat     # All patients in ZCoR format
│       ├── synpuf_labels.csv  # Patient labels and metadata
│       ├── encoded/           # Trinary series arrays
│       └── features/          # Feature matrices, PFSA outputs
└── models/
    ├── zcor_ipf_lgbm.txt      # Trained LightGBM model
    ├── results.json            # Evaluation metrics
    └── feature_importance.csv  # Feature ranking
```

## References

- Onishchenko D, et al. "Screening for idiopathic pulmonary fibrosis using comorbidity signatures in electronic health records." *Nature Medicine* 28, 2107-2116 (2022). https://doi.org/10.1038/s41591-022-02010-y
- Chattopadhyay I & Lipson H. "Abductive learning of quantized stochastic processes with probabilistic finite automata." *Phil. Trans. R. Soc. A* 371, 20110543 (2013).
- CMS DE-SynPUF: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
