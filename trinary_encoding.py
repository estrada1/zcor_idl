"""
Trinary time series encoding for ZCoR-IPF.

For each patient, transforms their diagnostic history into 51 parallel
trinary time series (one per disease category), week by week.

Per the paper (Methods, equation 1):
  z_i^k = 0  if no diagnosis codes in week i
  z_i^k = 1  if there exists a diagnosis of category k in week i
  z_i^k = 2  otherwise (diagnosis from a different category)

The observation window is 104 weeks (≈2 years) before the prediction point.
"""

import os
import csv
import numpy as np
from datetime import date, timedelta
from collections import defaultdict

from icd9_categories import build_category_lookup, NUM_CATEGORIES, get_category_name
from icd10_categories import (
    build_icd10_category_lookup,
    is_icd10_code,
    ICD10_NARROW_TARGET_CODES,
    ICD10_BROAD_TARGET_CODES,
)

# ── Configuration ──
INFERENCE_WINDOW_WEEKS = 104  # 2 years back from prediction point
TARGET_CODES = {"5163"}       # Narrow ICD-9 target (matching preprocess_cms.py)

# ICD-10 narrow target codes (post-2015 claims)
TARGET_CODES_ICD10 = ICD10_NARROW_TARGET_CODES

# Combined target codes (ICD-9 + ICD-10) for mixed-year data
TARGET_CODES_COMBINED = TARGET_CODES | TARGET_CODES_ICD10


def build_combined_category_lookup():
    """
    Build a classifier that handles both ICD-9 and ICD-10 codes.

    Detects which coding system a code belongs to via is_icd10_code()
    and routes to the appropriate lookup.  Returns None for unrecognised codes.
    """
    icd9_classify = build_category_lookup()
    icd10_classify = build_icd10_category_lookup()

    def classify(code):
        if is_icd10_code(code):
            return icd10_classify(code)
        return icd9_classify(code)

    return classify


def parse_dat_file(filepath):
    """
    Parse a ZCoR-IPF .dat file.
    Format: PATIENT_ID,SEX,BIRTH_DATE,DATE,CODE,CODE,...,DATE,CODE,...

    Returns list of dicts with keys: id, sex, birth_date, encounters
    where encounters is a list of (date, [codes]).
    """
    classify = build_category_lookup()
    patients = []

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 4:
                continue

            pid = parts[0]
            sex = parts[1]
            birth_str = parts[2]
            birth_date = date.fromisoformat(birth_str)

            # Parse remaining: alternating dates and codes
            encounters = []
            current_date = None
            current_codes = []

            for token in parts[3:]:
                # Try to parse as date
                try:
                    dt = date.fromisoformat(token)
                    # Save previous encounter
                    if current_date is not None and current_codes:
                        encounters.append((current_date, current_codes))
                    current_date = dt
                    current_codes = []
                except ValueError:
                    # It's an ICD code
                    if current_date is not None:
                        current_codes.append(token)

            # Don't forget last encounter
            if current_date is not None and current_codes:
                encounters.append((current_date, current_codes))

            patients.append({
                "id": pid,
                "sex": sex,
                "birth_date": birth_date,
                "encounters": encounters,
            })

    return patients


def determine_prediction_point(patient, target_codes=TARGET_CODES):
    """
    Determine the prediction point for a patient.

    Per the paper:
    - Positive: 52 weeks (1 year) before first target code appearance
    - Control: last record date (end of observation)

    Returns (prediction_date, is_positive, first_target_date)
    """
    first_target_date = None
    last_date = None

    for dt, codes in patient["encounters"]:
        if last_date is None or dt > last_date:
            last_date = dt
        for code in codes:
            if code in target_codes:
                if first_target_date is None or dt < first_target_date:
                    first_target_date = dt

    if first_target_date is not None:
        # Positive: prediction point = 52 weeks before first target code
        prediction_date = first_target_date - timedelta(weeks=52)
        return prediction_date, True, first_target_date
    else:
        # Control: prediction point = last record
        return last_date, False, None


def encode_patient(patient, classify_fn, prediction_date, window_weeks=INFERENCE_WINDOW_WEEKS):
    """
    Encode a single patient's history as 51 trinary time series.

    Returns numpy array of shape (NUM_CATEGORIES, window_weeks) with values 0, 1, 2.
    """
    # Define the observation window
    window_start = prediction_date - timedelta(weeks=window_weeks)
    window_end = prediction_date

    # Initialize: all zeros (no codes)
    series = np.zeros((NUM_CATEGORIES, window_weeks), dtype=np.int8)

    # Track which weeks have any codes at all
    week_has_codes = np.zeros(window_weeks, dtype=bool)
    # Track which (week, category) pairs have codes
    week_cat_present = np.zeros((NUM_CATEGORIES, window_weeks), dtype=bool)

    for dt, codes in patient["encounters"]:
        if dt < window_start or dt >= window_end:
            continue

        # Determine which week this falls in
        week_idx = (dt - window_start).days // 7
        if week_idx < 0 or week_idx >= window_weeks:
            continue

        # Classify each code
        cats_this_encounter = set()
        for code in codes:
            cat = classify_fn(code)
            if cat is not None:
                cats_this_encounter.add(cat)

        if cats_this_encounter:
            week_has_codes[week_idx] = True
            for cat in cats_this_encounter:
                week_cat_present[cat][week_idx] = True

    # Apply trinary encoding rules
    for k in range(NUM_CATEGORIES):
        for i in range(window_weeks):
            if not week_has_codes[i]:
                series[k][i] = 0  # no codes this week
            elif week_cat_present[k][i]:
                series[k][i] = 1  # category k present this week
            else:
                series[k][i] = 2  # other category present, not k

    return series


def encode_cohort(dat_filepath, output_dir, target_codes=None,
                  window_weeks=INFERENCE_WINDOW_WEEKS, icd_version="auto"):
    """
    Encode all patients from a .dat file into trinary time series.

    Args:
        dat_filepath: path to the .dat file produced by preprocess_cms.py
        output_dir:   directory to write trinary_series.npz + patient_metadata.csv
        target_codes: set of target ICD codes; defaults to TARGET_CODES_COMBINED
                      (ICD-9 "5163" + ICD-10 "J84112") so both coding eras work
        window_weeks: length of the observation window (default 104 = 2 years)
        icd_version:  "auto" (detect per-code), "icd9", or "icd10"

    Saves:
    - {output_dir}/trinary_series.npz: compressed arrays
    - {output_dir}/patient_metadata.csv: patient info + labels
    """
    if target_codes is None:
        target_codes = TARGET_CODES_COMBINED

    os.makedirs(output_dir, exist_ok=True)

    if icd_version == "icd9":
        classify = build_category_lookup()
    elif icd_version == "icd10":
        classify = build_icd10_category_lookup()
    else:
        classify = build_combined_category_lookup()

    print(f"Parsing {dat_filepath}...")
    patients = parse_dat_file(dat_filepath)
    print(f"  {len(patients)} patients loaded")

    all_series = []
    metadata = []
    skipped = 0

    for i, patient in enumerate(patients):
        if (i + 1) % 5000 == 0:
            print(f"  Encoding patient {i+1}/{len(patients)}...")

        pred_date, is_positive, first_target = determine_prediction_point(
            patient, target_codes
        )

        if pred_date is None:
            skipped += 1
            continue

        # Check we have enough history before prediction point
        earliest = min(dt for dt, _ in patient["encounters"])
        available_weeks = (pred_date - earliest).days / 7
        if available_weeks < window_weeks * 0.5:
            # Require at least half the window to be populated
            skipped += 1
            continue

        series = encode_patient(patient, classify, pred_date, window_weeks)
        all_series.append(series)

        age_at_pred = (pred_date - patient["birth_date"]).days / 365.25
        metadata.append({
            "patient_id": patient["id"],
            "sex": patient["sex"],
            "label": 1 if is_positive else 0,
            "age_at_screening": round(age_at_pred, 1),
            "prediction_date": pred_date.isoformat(),
            "first_target_date": first_target.isoformat() if first_target else "",
        })

    print(f"  Encoded {len(all_series)} patients (skipped {skipped})")

    # Save as compressed numpy
    series_array = np.array(all_series, dtype=np.int8)
    npz_path = os.path.join(output_dir, "trinary_series.npz")
    np.savez_compressed(npz_path, series=series_array)
    print(f"  Saved series: {series_array.shape} to {npz_path}")

    # Save metadata
    meta_path = os.path.join(output_dir, "patient_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    print(f"  Saved metadata to {meta_path}")

    return series_array, metadata


def print_encoding_summary(series_array, metadata):
    """Print summary statistics of the encoded data."""
    n_patients, n_cats, n_weeks = series_array.shape
    n_pos = sum(1 for m in metadata if m["label"] == 1)
    n_ctrl = n_patients - n_pos

    print(f"\n{'='*60}")
    print(f"ENCODING SUMMARY")
    print(f"{'='*60}")
    print(f"Patients:     {n_patients} ({n_pos} positive, {n_ctrl} control)")
    print(f"Categories:   {n_cats}")
    print(f"Weeks:        {n_weeks}")
    print(f"Array shape:  {series_array.shape}")
    print(f"Memory:       {series_array.nbytes / 1024 / 1024:.1f} MB")

    # Sparsity
    zeros = np.sum(series_array == 0)
    ones = np.sum(series_array == 1)
    twos = np.sum(series_array == 2)
    total = series_array.size
    print(f"\nValue distribution:")
    print(f"  0 (no code):    {zeros/total*100:.1f}%")
    print(f"  1 (this cat):   {ones/total*100:.1f}%")
    print(f"  2 (other cat):  {twos/total*100:.1f}%")

    # Activity by category
    print(f"\nTop 10 most active categories (by '1' frequency):")
    cat_activity = np.mean(series_array == 1, axis=(0, 2))
    top_cats = np.argsort(cat_activity)[::-1][:10]
    for cat_id in top_cats:
        name = get_category_name(cat_id)
        print(f"  {cat_id:2d} {name:40s} {cat_activity[cat_id]*100:.2f}%")


if __name__ == "__main__":
    import sys

    dat_file = os.path.join("data", "processed", "synpuf_all.dat")
    out_dir = os.path.join("data", "processed", "encoded")

    print("Trinary Encoding Pipeline")
    print("=" * 60)

    series, meta = encode_cohort(dat_file, out_dir)
    print_encoding_summary(series, meta)
