#!/usr/bin/env python3
"""
Preprocess CMS DE-SynPUF data into ZCoR-IPF input format.

Output format (per the ZCoR IPF Manual):
  Each line: PATIENT_ID,SEX,BIRTH_DATE,DATE,CODE,CODE,...,DATE,CODE,...
  - SEX: "M" or "F"
  - Dates: YYYY-MM-DD
  - ICD codes grouped by day

Target ICD-9 codes for IPF (narrow): 51631
Broader target codes: 515, 5163, 51630, 51631, 51632, 51633, 51634, 51635
(ICD-9 codes in SynPUF lack dots)
"""

import csv
import sys
import os
from collections import defaultdict
from datetime import datetime, date

# ── Configuration ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw", "extracted")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

# IPF target codes (ICD-9, no dots, as they appear in SynPUF)
# SynPUF stores codes at reduced granularity (e.g. 5163 not 51631),
# so we use the broadest available codes that map to IPF/ILD.
#
# Narrow target: 5163 (=516.3, idiopathic interstitial pneumonia) is the
# closest available proxy for 516.31 (IPF) in this dataset.
NARROW_TARGET_CODES = {"5163"}

# Broad target — includes ILD-related codes
BROAD_TARGET_CODES = {
    "515",   # Postinflammatory pulmonary fibrosis
    "5160",  # Pulmonary alveolar proteinosis
    "5161",  # Idiopathic pulmonary hemosiderosis
    "5162",  # Pulmonary alveolar microlithiasis
    "5163",  # Idiopathic interstitial pneumonia (includes IPF)
    "5168",  # Other alveolar/parietoalveolar pneumonopathy
    "5169",  # Unspecified alveolar/parietoalveolar pneumonopathy
}

# The binary also looks for these ICD-10 equivalents (not in SynPUF which is ICD-9 only):
# J84.112, J84.113, J84.1, J84.10, J84.11

# Age filter: 45-90 years (paper requirement)
MIN_AGE = 45
MAX_AGE = 90

# Minimum record span: paper requires 3 years (156 weeks), but SynPUF only
# covers 2008-2010 (~3 years max). We relax to 2 years (104 weeks) which is
# the inference window size used by ZCoR-IPF.
MIN_RECORD_SPAN_DAYS = 2 * 365

BENEFICIARY_FILES = [
    os.path.join(DATA_DIR, "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"),
    os.path.join(DATA_DIR, "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"),
]

CLAIMS_FILES = {
    "inpatient": {
        "path": os.path.join(DATA_DIR, "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"),
        "date_col": "CLM_FROM_DT",
        "diag_cols": [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)] + ["ADMTNG_ICD9_DGNS_CD"],
    },
    "outpatient": {
        "path": os.path.join(DATA_DIR, "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"),
        "date_col": "CLM_FROM_DT",
        "diag_cols": [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)] + ["ADMTNG_ICD9_DGNS_CD"],
    },
    "carrier_a": {
        "path": os.path.join(DATA_DIR, "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv"),
        "date_col": "CLM_FROM_DT",
        "diag_cols": [f"ICD9_DGNS_CD_{i}" for i in range(1, 9)]
                     + [f"LINE_ICD9_DGNS_CD_{i}" for i in range(1, 14)],
    },
    "carrier_b": {
        "path": os.path.join(DATA_DIR, "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv"),
        "date_col": "CLM_FROM_DT",
        "diag_cols": [f"ICD9_DGNS_CD_{i}" for i in range(1, 9)]
                     + [f"LINE_ICD9_DGNS_CD_{i}" for i in range(1, 14)],
    },
}


def parse_cms_date(s):
    """Parse YYYYMMDD date string from CMS data."""
    s = s.strip().strip('"')
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def format_date(d):
    """Format date as YYYY-MM-DD for ZCoR output."""
    return d.strftime("%Y-%m-%d")


def load_beneficiaries():
    """
    Load patient demographics from beneficiary summary files.
    Returns dict: patient_id -> {birth_date, sex, death_date}
    Uses the most recent beneficiary file for each patient.
    """
    patients = {}
    for filepath in BENEFICIARY_FILES:
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping")
            continue
        print(f"  Reading {os.path.basename(filepath)}...")
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["DESYNPUF_ID"].strip().strip('"')
                birth_str = row["BENE_BIRTH_DT"].strip().strip('"')
                sex_code = row["BENE_SEX_IDENT_CD"].strip().strip('"')
                death_str = row.get("BENE_DEATH_DT", "").strip().strip('"')

                birth_date = parse_cms_date(birth_str)
                if birth_date is None:
                    continue

                sex = "M" if sex_code == "1" else "F"
                death_date = parse_cms_date(death_str) if death_str else None

                # Keep/update record
                patients[pid] = {
                    "birth_date": birth_date,
                    "sex": sex,
                    "death_date": death_date,
                }

    print(f"  Loaded {len(patients)} unique patients")
    return patients


def load_claims():
    """
    Load all claims and extract (patient_id, date, [icd_codes]).
    Returns dict: patient_id -> {date -> set of ICD codes}
    """
    patient_codes = defaultdict(lambda: defaultdict(set))
    total_claims = 0

    for name, config in CLAIMS_FILES.items():
        filepath = config["path"]
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping")
            continue

        print(f"  Reading {name} ({os.path.basename(filepath)})...")
        claim_count = 0
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            available_diag_cols = [c for c in config["diag_cols"] if c in reader.fieldnames]

            for row in reader:
                pid = row["DESYNPUF_ID"].strip().strip('"')
                claim_date = parse_cms_date(row[config["date_col"]])
                if claim_date is None:
                    continue

                codes = set()
                for col in available_diag_cols:
                    val = row.get(col, "").strip().strip('"')
                    if val:
                        codes.add(val)

                if codes:
                    patient_codes[pid][claim_date].update(codes)
                    claim_count += 1

        total_claims += claim_count
        print(f"    {claim_count} claims with codes")

    print(f"  Total: {total_claims} claims across {len(patient_codes)} patients")
    return patient_codes


def identify_cohorts(patients, patient_codes, target_codes):
    """
    Identify positive and control cohorts.
    Positive: has at least one target code in their history.
    Control: no target codes anywhere.
    Returns (positive_pids, control_pids)
    """
    positive = set()
    for pid, date_codes in patient_codes.items():
        for dt, codes in date_codes.items():
            if codes & target_codes:
                positive.add(pid)
                break

    all_with_claims = set(patient_codes.keys()) & set(patients.keys())
    control = all_with_claims - positive

    return positive, control


def filter_patients(patients, patient_codes, pids, reference_date=None):
    """
    Apply inclusion/exclusion criteria:
    - Age 45-90 at reference_date (default: 2010-12-31, end of SynPUF)
    - At least 3 years of medical history
    """
    if reference_date is None:
        reference_date = date(2010, 12, 31)

    filtered = set()
    excluded_age = 0
    excluded_span = 0

    for pid in pids:
        if pid not in patients:
            continue

        info = patients[pid]
        birth = info["birth_date"]

        # Age check
        age = (reference_date - birth).days / 365.25
        if age < MIN_AGE or age > MAX_AGE:
            excluded_age += 1
            continue

        # Record span check
        if pid in patient_codes and patient_codes[pid]:
            dates = sorted(patient_codes[pid].keys())
            span = (dates[-1] - dates[0]).days
            if span < MIN_RECORD_SPAN_DAYS:
                excluded_span += 1
                continue
        else:
            excluded_span += 1
            continue

        filtered.add(pid)

    print(f"    Excluded {excluded_age} for age, {excluded_span} for insufficient span")
    print(f"    Remaining: {len(filtered)}")
    return filtered


def write_zcor_dat(patients, patient_codes, pids, output_path):
    """
    Write patient records in ZCoR-IPF .dat format:
    PATIENT_ID,SEX,BIRTH_DATE,DATE,CODE,CODE,...,DATE,CODE,...
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    with open(output_path, "w") as f:
        for pid in sorted(pids):
            info = patients[pid]
            birth_str = format_date(info["birth_date"])
            sex = info["sex"]

            # Build date-grouped codes, sorted by date
            parts = [pid, sex, birth_str]
            for dt in sorted(patient_codes[pid].keys()):
                codes = sorted(patient_codes[pid][dt])
                parts.append(format_date(dt))
                parts.extend(codes)

            f.write(",".join(parts) + "\n")
            count += 1

    print(f"  Wrote {count} patients to {output_path}")
    return count


def write_labels(patients, patient_codes, positive_pids, control_pids, output_path):
    """Write a CSV with patient_id, label (1=positive, 0=control), sex, age, first_target_date."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ref = date(2010, 12, 31)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "sex", "age", "first_target_date",
                         "record_start", "record_end", "record_span_weeks"])

        for pid in sorted(positive_pids | control_pids):
            info = patients[pid]
            label = 1 if pid in positive_pids else 0
            age = round((ref - info["birth_date"]).days / 365.25, 1)

            dates = sorted(patient_codes[pid].keys())
            record_start = format_date(dates[0])
            record_end = format_date(dates[-1])
            span_weeks = round((dates[-1] - dates[0]).days / 7, 1)

            # Find first target code date for positive patients
            first_target = ""
            if label == 1:
                for dt in dates:
                    if patient_codes[pid][dt] & NARROW_TARGET_CODES:
                        first_target = format_date(dt)
                        break

            writer.writerow([pid, label, info["sex"], age, first_target,
                             record_start, record_end, span_weeks])

    print(f"  Wrote labels to {output_path}")


def main():
    print("=" * 60)
    print("CMS DE-SynPUF → ZCoR-IPF Preprocessor")
    print("=" * 60)

    # Step 1: Load beneficiary demographics
    print("\n[1/5] Loading beneficiary demographics...")
    patients = load_beneficiaries()

    # Step 2: Load all claims with ICD codes
    print("\n[2/5] Loading claims data...")
    patient_codes = load_claims()

    # Step 3: Identify cohorts
    print("\n[3/5] Identifying IPF cohorts...")
    print("  Narrow target (516.31):")
    pos_narrow, ctrl_narrow = identify_cohorts(patients, patient_codes, NARROW_TARGET_CODES)
    print(f"    Positive: {len(pos_narrow)}, Control: {len(ctrl_narrow)}")

    print("  Broad target (515, 516.x):")
    pos_broad, ctrl_broad = identify_cohorts(patients, patient_codes, BROAD_TARGET_CODES)
    print(f"    Positive: {len(pos_broad)}, Control: {len(ctrl_broad)}")

    # Step 4: Apply inclusion/exclusion criteria
    print("\n[4/5] Applying inclusion/exclusion filters...")
    print("  Filtering narrow positive cohort:")
    pos_narrow_filtered = filter_patients(patients, patient_codes, pos_narrow)
    print("  Filtering narrow control cohort:")
    ctrl_narrow_filtered = filter_patients(patients, patient_codes, ctrl_narrow)
    print("  Filtering broad positive cohort:")
    pos_broad_filtered = filter_patients(patients, patient_codes, pos_broad)

    # Step 5: Write output
    print("\n[5/5] Writing output files...")

    # Combine all patients for the .dat file (the binary handles cohort assignment internally)
    all_pids = pos_narrow_filtered | ctrl_narrow_filtered
    write_zcor_dat(
        patients, patient_codes, all_pids,
        os.path.join(OUTPUT_DIR, "synpuf_all.dat"),
    )

    # Also write separate files for analysis
    write_zcor_dat(
        patients, patient_codes, pos_narrow_filtered,
        os.path.join(OUTPUT_DIR, "synpuf_positive_narrow.dat"),
    )
    write_zcor_dat(
        patients, patient_codes, ctrl_narrow_filtered,
        os.path.join(OUTPUT_DIR, "synpuf_control.dat"),
    )

    # Write labels file for our own training pipeline
    write_labels(
        patients, patient_codes,
        pos_narrow_filtered, ctrl_narrow_filtered,
        os.path.join(OUTPUT_DIR, "synpuf_labels.csv"),
    )

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total patients with claims: {len(patient_codes)}")
    print(f"Narrow IPF positive (pre-filter):  {len(pos_narrow)}")
    print(f"Narrow IPF positive (post-filter): {len(pos_narrow_filtered)}")
    print(f"Control (post-filter):             {len(ctrl_narrow_filtered)}")
    print(f"Broad IPF positive (post-filter):  {len(pos_broad_filtered)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
