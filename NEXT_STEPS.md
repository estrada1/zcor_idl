# Next Steps

## 1. Better Public Data Sources

The current reproduction uses CMS DE-SynPUF (synthetic Medicare data, 2008--2010), which inflates metrics due to programmatic comorbidity patterns. Candidates for real or higher-quality data:

### Tier 1: Best Fit

- **CMS Medicare Limited Data Set (LDS)** -- Real Medicare claims with the same file format as SynPUF. Inpatient + outpatient + carrier claims, multi-year. Requires DUA with CMS + per-year fees. `preprocess_cms.py` would need minimal changes.

- **NIH All of Us Research Program** -- Free, ~440K patients with longitudinal EHR data in OMOP format. ICD-9/ICD-10 codes mapped to SNOMED. All analysis runs in the cloud Researcher Workbench (no data download). Pipeline would need adaptation from CMS flat files to OMOP `condition_occurrence` table.

### Tier 2: Viable with Access

- **State All-Payer Claims Databases (APCDs)** -- Massachusetts (CHIA), Colorado, New Hampshire offer individual-level longitudinal claims with ICD codes across all payers. Application + privacy review required. Single-state only.

- **OHDSI network** -- If institutional access exists to a node with real claims data (Optum, IQVIA), the OMOP CDM format provides standardized ICD codes. Check with your institution.

### Tier 3: Supplementary Only

- **MIMIC-IV** -- Free via PhysioNet. ~223K patients but hospital/inpatient only. Lacks the outpatient/carrier claims that ZCoR relies on for weekly time series. Useful for validation, not primary reproduction.

- **HCUP SID/NIS** -- Inpatient discharge data only. Misses outpatient encounters critical for ZCoR.

## 2. ICD-10 Support

The current pipeline is ICD-9 only. To work with post-2015 data:
- Add ICD-10 category mapping (parallel to `icd9_categories.py`)
- Map ICD-10 J84.112 (IPF) as the primary target code
- Handle the ICD-9/ICD-10 transition period (2015 crosswalk)

## 3. Pipeline Improvements

- **Three-split training**: Implement the paper's hyper-training / pre-aggregation / aggregation split instead of the current single LightGBM
- **Per-category LGBMs**: Train 51 per-phenotype models before the final aggregation model
- **PFSA depth tuning**: Systematic comparison of PFSA `max_depth` (currently fixed at 3)
- **Feature selection**: The paper reports 667 features; we have 593. Identify missing rare-score variants and additional aggregate features

## 4. Validation

- **Calibration analysis**: The current model outputs raw probabilities; assess calibration curves
- **Temporal validation**: Train on earlier years, test on later (if multi-year data available)
- **Subgroup analysis**: Performance by sex, age group, comorbidity burden
- **Ablation study**: Feature group contribution (SLD alone vs. SLD + sequence features vs. full)
