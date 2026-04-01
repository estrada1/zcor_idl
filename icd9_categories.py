"""
ICD-9 → 51 disease category mapping for ZCoR-IPF.

Based on the paper's description (Methods, Supplementary Table 3):
"We partitioned the human disease spectrum into 51 non-overlapping broad
diagnostic categories... approximately aligning with the categories defined
within the ICD framework."

The 51 categories are derived by subdividing the standard ICD-9 chapters
into clinically meaningful groups, with finer granularity for categories
most relevant to IPF (respiratory, cardiovascular, etc.).

Category names visible in Fig. 2e of the paper are used where possible.
"""

# Each entry: (category_id, name, list of (start_code, end_code) ranges)
# Ranges are inclusive, using numeric ICD-9 codes (no dots).
# V-codes and E-codes handled separately.

DISEASE_CATEGORIES = [
    # ── Infectious diseases (001-139) ──
    (0, "All infections", [("001", "139")]),

    # ── Neoplasms (140-239) ──
    (1, "Neoplasms - malignant", [("140", "208")]),
    (2, "Neoplasms - benign/uncertain", [("209", "239")]),

    # ── Endocrine, nutritional, metabolic, immunity (240-279) ──
    (3, "Thyroid disorders", [("240", "246")]),
    (4, "Diabetes mellitus", [("249", "250")]),
    (5, "Lipid metabolism disorders", [("272", "272")]),
    (6, "Metabolic disorders", [("247", "248"), ("251", "271"), ("273", "277")]),
    (7, "Immune disorders", [("278", "279")]),

    # ── Blood and blood-forming organs (280-289) ──
    (8, "Blood disorders", [("280", "289")]),

    # ── Mental disorders (290-319) ──
    (9, "Psychiatric - organic/dementias", [("290", "294")]),
    (10, "Psychiatric - other", [("295", "319")]),

    # ── Nervous system (320-359) ──
    (11, "Central nervous system", [("320", "349")]),
    (12, "Peripheral nervous system", [("350", "359")]),

    # ── Eye and adnexa (360-379) ──
    (13, "Ophthalmological", [("360", "379")]),

    # ── Ear and mastoid (380-389) ──
    (14, "Ear and mastoid", [("380", "389")]),

    # ── Circulatory system (390-459) ──
    (15, "Hypertensive - systemic", [("401", "405")]),
    (16, "Coronary atherosclerosis", [("410", "414")]),
    (17, "Chronic ischemia", [("440", "440")]),
    (18, "Pulmonary heart disease", [("415", "417")]),
    (19, "Heart failure", [("428", "428")]),
    (20, "Cardiac arrhythmias", [("426", "427")]),
    (21, "Cardiovascular - other", [("390", "400"), ("406", "409"),
                                     ("418", "425"), ("429", "439"),
                                     ("441", "459")]),

    # ── Respiratory system (460-519) — most granular ──
    (22, "Upper respiratory infections", [("460", "466")]),
    (23, "Respiratory - chronic/COPD", [("490", "496")]),
    (24, "Pneumonia", [("480", "487")]),
    (25, "Respiratory - pulmonary fibrosis/ILD", [("515", "516")]),
    (26, "Respiratory - pleural", [("510", "514"), ("517", "519")]),
    (27, "Respiratory - other", [("467", "479"), ("488", "489"),
                                  ("500", "509")]),
    (28, "Dyspnea & respiratory abnormalities", [("786", "786")]),
    (29, "Allergies", [("477", "477")]),  # overlap handled: specific allergy codes

    # ── Digestive system (520-579) ──
    (30, "Esophagus disorders", [("530", "530")]),
    (31, "Digestive - gastric/duodenal", [("531", "537")]),
    (32, "Digestive - other", [("520", "529"), ("538", "579")]),

    # ── Genitourinary system (580-629) ──
    (33, "Kidney diseases", [("580", "593")]),
    (34, "Genitourinary - other", [("594", "629")]),

    # ── Skin and subcutaneous tissue (680-709) ──
    (35, "Skin disorders", [("680", "709")]),

    # ── Musculoskeletal and connective tissue (710-739) ──
    (36, "Connective tissue disorders", [("710", "710")]),
    (37, "Disorders of soft tissues", [("725", "729")]),
    (38, "Musculoskeletal - other", [("711", "724"), ("730", "739")]),

    # ── Congenital anomalies (740-759) ──
    (39, "Congenital anomaly", [("740", "759")]),

    # ── Perinatal conditions (760-779) ──
    (40, "Perinatal conditions", [("760", "779")]),

    # ── Symptoms, signs, ill-defined (780-799) ──
    (41, "Chest pain", [("7865", "7865")]),  # special: 786.5x
    (42, "Blood examination abnormal findings", [("790", "796")]),
    (43, "Symptoms - general", [("780", "785"), ("787", "789")]),
    (44, "Ill-defined conditions", [("797", "799")]),

    # ── Injury and poisoning (800-999) ──
    (45, "Injuries - fractures/trauma", [("800", "869")]),
    (46, "Injuries - other", [("870", "999")]),

    # ── Supplementary classifications ──
    (47, "Health service contact - aftercare", []),  # V-codes: V50-V59
    (48, "Health service contact - other", []),      # V-codes: all other V
    (49, "External causes", []),                     # E-codes

    # ── Special: hypertensive pulmonary ──
    (50, "Hypertensive - pulmonary", [("4160", "4169")]),
    # Note: 416.x overlaps with cat 18. We handle 416 specifically as pulmonary hypertension.
]

# Total: 51 categories (0-50)

# Special range for dyspnea within symptoms chapter
# 786 = Symptoms involving respiratory system and other chest symptoms
# 786.0x = dyspnea/respiratory abnormalities
# 786.5x = chest pain
# We handle 786 as category 28 (dyspnea & respiratory abnormalities)
# except 786.5x which goes to category 41 (chest pain)


def _code_to_numeric(code_str):
    """Convert ICD-9 code string to a comparable numeric string."""
    return code_str.replace(".", "").strip()


def _in_range(code, start, end):
    """Check if a code falls within a numeric range."""
    # Handle codes of different lengths by comparing prefixes
    code_num = code.lstrip("0") or "0"
    start_num = start.lstrip("0") or "0"
    end_num = end.lstrip("0") or "0"

    # Pad to same length for comparison
    max_len = max(len(code_num), len(start_num), len(end_num))
    code_padded = code_num.ljust(max_len, "0")
    start_padded = start_num.ljust(max_len, "0")
    end_padded = end_num.ljust(max_len, "9")

    return start_padded <= code_padded <= end_padded


def build_category_lookup():
    """
    Build a lookup function that maps an ICD-9 code to a category index (0-50).
    Returns None if the code doesn't match any category.
    """
    # Pre-build a prefix tree for fast lookup
    # For most codes, the first 3 digits determine the chapter
    def classify(code):
        code = _code_to_numeric(code)
        if not code:
            return None

        # Handle V-codes
        if code.startswith("V"):
            v_num = code[1:]
            if v_num and v_num[:2].isdigit():
                vn = int(v_num[:2])
                if 50 <= vn <= 59:
                    return 47  # Health service contact - aftercare
            return 48  # Health service contact - other

        # Handle E-codes
        if code.startswith("E"):
            return 49  # External causes

        # Numeric codes
        if not code[0].isdigit():
            return None

        # Special cases first (more specific overrides general)
        # Pulmonary hypertension: 416.x
        if code.startswith("416"):
            return 50

        # Chest pain: 786.5x
        if code.startswith("7865"):
            return 41

        # Dyspnea & respiratory: 786.x (except 786.5)
        if code.startswith("786"):
            return 28

        # Allergic rhinitis: 477
        if code.startswith("477"):
            return 29

        # General range matching
        try:
            code3 = code[:3]
            code3_int = int(code3)
        except (ValueError, IndexError):
            return None

        # Map by ICD-9 chapter ranges
        range_map = [
            (1, 139, 0),     # Infections
            (140, 208, 1),   # Malignant neoplasms
            (209, 239, 2),   # Benign/uncertain neoplasms
            (240, 246, 3),   # Thyroid
            (249, 250, 4),   # Diabetes
            (247, 248, 6),   # Metabolic (other endocrine)
            (251, 271, 6),   # Metabolic (other)
            (272, 272, 5),   # Lipid metabolism
            (273, 277, 6),   # Metabolic
            (278, 279, 7),   # Immune
            (280, 289, 8),   # Blood
            (290, 294, 9),   # Psychiatric - organic
            (295, 319, 10),  # Psychiatric - other
            (320, 349, 11),  # CNS
            (350, 359, 12),  # PNS
            (360, 379, 13),  # Eye
            (380, 389, 14),  # Ear
            (390, 400, 21),  # CV other
            (401, 405, 15),  # Hypertensive
            (406, 409, 21),  # CV other
            (410, 414, 16),  # Coronary
            (415, 415, 18),  # Pulmonary heart (excl 416 handled above)
            (417, 417, 18),  # Pulmonary heart
            (418, 425, 21),  # CV other
            (426, 427, 20),  # Arrhythmias
            (428, 428, 19),  # Heart failure
            (429, 439, 21),  # CV other
            (440, 440, 17),  # Chronic ischemia
            (441, 459, 21),  # CV other
            (460, 466, 22),  # Upper resp infections
            (467, 476, 27),  # Resp other
            (478, 479, 27),  # Resp other
            (480, 487, 24),  # Pneumonia
            (488, 489, 27),  # Resp other
            (490, 496, 23),  # COPD/chronic
            (500, 509, 27),  # Resp other (pneumoconioses)
            (510, 514, 26),  # Resp pleural
            (515, 516, 25),  # Pulmonary fibrosis/ILD
            (517, 519, 26),  # Resp pleural/other
            (520, 529, 32),  # Digestive other
            (530, 530, 30),  # Esophagus
            (531, 537, 31),  # Gastric/duodenal
            (538, 579, 32),  # Digestive other
            (580, 593, 33),  # Kidney
            (594, 629, 34),  # GU other
            (630, 679, 34),  # Pregnancy → GU other (rare in 45-90 cohort)
            (680, 709, 35),  # Skin
            (710, 710, 36),  # Connective tissue
            (711, 724, 38),  # MSK other
            (725, 729, 37),  # Soft tissue
            (730, 739, 38),  # MSK other
            (740, 759, 39),  # Congenital
            (760, 779, 40),  # Perinatal
            (780, 785, 43),  # Symptoms general
            (787, 789, 43),  # Symptoms general
            (790, 796, 42),  # Abnormal findings
            (797, 799, 44),  # Ill-defined
            (800, 869, 45),  # Fractures/trauma
            (870, 999, 46),  # Injuries other
        ]

        for start, end, cat in range_map:
            if start <= code3_int <= end:
                return cat

        return None

    return classify


# Category names for reference
CATEGORY_NAMES = {cat_id: name for cat_id, name, _ in DISEASE_CATEGORIES}
NUM_CATEGORIES = 51


def get_category_name(cat_id):
    return CATEGORY_NAMES.get(cat_id, f"Unknown ({cat_id})")


# ── Self-test ──
if __name__ == "__main__":
    classify = build_category_lookup()

    # Test cases
    tests = [
        ("4019", 15, "Hypertension NOS"),
        ("25000", 4, "Diabetes"),
        ("2724", 5, "Hyperlipidemia"),
        ("42731", 20, "Atrial fibrillation"),
        ("5163", 25, "IPF/ILD"),
        ("515", 25, "Postinflammatory pulm fibrosis"),
        ("496", 23, "COPD"),
        ("78650", 41, "Chest pain"),
        ("78605", 28, "Dyspnea"),
        ("4168", 50, "Pulmonary hypertension"),
        ("V5869", 47, "Aftercare"),
        ("E8859", 49, "External cause"),
        ("4280", 19, "Heart failure"),
        ("41401", 16, "Coronary atherosclerosis"),
        ("51881", 26, "Respiratory failure"),
        ("7106", 36, "Connective tissue"),
    ]

    print("Category mapping self-test:")
    print("-" * 60)
    ok = 0
    for code, expected_cat, desc in tests:
        result = classify(code)
        status = "OK" if result == expected_cat else "FAIL"
        if status == "FAIL":
            print(f"  {status}: {code} ({desc}) → {result} (expected {expected_cat})")
        else:
            ok += 1
    print(f"  {ok}/{len(tests)} passed")

    # Coverage test: check what fraction of real codes get mapped
    import csv
    from collections import Counter
    unmapped = Counter()
    mapped = 0
    total = 0
    for fname in ['data/raw/extracted/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv']:
        with open(fname) as f:
            reader = csv.DictReader(f)
            diag_cols = [c for c in reader.fieldnames if 'ICD9_DGNS' in c]
            for row in reader:
                for c in diag_cols:
                    v = row[c].strip().strip('"')
                    if v:
                        total += 1
                        cat = classify(v)
                        if cat is not None:
                            mapped += 1
                        else:
                            unmapped[v] += 1

    print(f"\nCoverage on inpatient claims: {mapped}/{total} ({100*mapped/total:.1f}%)")
    if unmapped:
        print(f"Top unmapped codes:")
        for code, count in unmapped.most_common(10):
            print(f"  {code}: {count}")

    # Distribution across categories
    cat_counts = Counter()
    for fname in ['data/raw/extracted/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv']:
        with open(fname) as f:
            reader = csv.DictReader(f)
            diag_cols = [c for c in reader.fieldnames if 'ICD9_DGNS' in c]
            for row in reader:
                for c in diag_cols:
                    v = row[c].strip().strip('"')
                    if v:
                        cat = classify(v)
                        if cat is not None:
                            cat_counts[cat] += 1

    print(f"\nCategory distribution (inpatient):")
    for cat_id in range(NUM_CATEGORIES):
        name = get_category_name(cat_id)
        count = cat_counts.get(cat_id, 0)
        bar = "█" * min(50, count // 500)
        print(f"  {cat_id:2d} {name:40s} {count:7d} {bar}")
