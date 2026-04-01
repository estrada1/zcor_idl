# ZCoR-ILD

A Python reimplementation of the Zero-burden Comorbidity Risk score for Interstitial Lung Disease (ILD), based on the ZCoR-IPF method. The pipeline processes claims data, encodes diagnostic histories as trinary time series, trains Probabilistic Finite State Automata (PFSA) models, and computes Sequence Likelihood Defect (SLD) features for disease risk prediction.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
uv sync
```

This installs all dependencies (numpy, pandas, scikit-learn, lightgbm) into a local virtual environment.

## Data Setup

The pipeline expects CMS DE-SynPUF sample files placed under `data/raw/extracted/`:

- `DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv`
- `DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv`
- `DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv`
- `DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv`
- `DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv`
- `DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv`

These files are available from the [CMS SynPUF website](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf).

## Usage

Run each step sequentially:

### 1. Preprocess claims data

Reads raw CMS files, applies inclusion/exclusion criteria (age 45-90, minimum 2-year record span), and writes `.dat` files in ZCoR-IPF format.

```bash
uv run python preprocess_cms.py
```

Output: `data/processed/synpuf_all.dat`, `data/processed/synpuf_labels.csv`

### 2. Encode trinary time series

Transforms each patient's diagnostic history into 51 parallel trinary time series (one per disease category) over a 104-week observation window.

```bash
uv run python trinary_encoding.py
```

Output: `data/processed/encoded/trinary_series.npz`, `data/processed/encoded/patient_metadata.csv`

### 3. Train PFSA models and compute SLD features

Trains 204 PFSA models (51 categories x 2 cohorts x 2 sexes) and computes Sequence Likelihood Defect scores.

```bash
uv run python pfsa.py
```

Output: `data/processed/features/pfsa_features.npz`

## Pipeline Overview

1. **ICD-9 categorization** (`icd9_categories.py`): Maps ICD-9 codes to 51 non-overlapping disease categories
2. **Preprocessing** (`preprocess_cms.py`): Extracts and filters patient claims from CMS SynPUF data
3. **Trinary encoding** (`trinary_encoding.py`): Converts per-patient diagnostic histories into week-by-week trinary sequences (0 = no code, 1 = category present, 2 = other category present)
4. **PFSA / SLD** (`pfsa.py`): Learns cohort-specific stochastic models and computes the likelihood defect that serves as the risk feature vector
