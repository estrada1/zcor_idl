"""
ICD-10 → 51 disease category mapping for ZCoR-IPF.

Parallel to icd9_categories.py; maps ICD-10-CM codes to the same 51
non-overlapping diagnostic categories so the PFSA/feature pipeline works
transparently with post-2015 claims data.

ICD-10 primary target for IPF: J84.112 (Idiopathic pulmonary fibrosis)
Broader ILD targets: J84.1, J84.10, J84.11, J84.112, J84.113, J84.17

Category numbering is identical to icd9_categories.py (0–50).
"""

from icd9_categories import CATEGORY_NAMES, NUM_CATEGORIES, get_category_name  # noqa: F401

# ── ICD-10 target codes ──────────────────────────────────────────────────────

# Narrow: the paper's primary outcome (UIP-pattern IPF)
ICD10_NARROW_TARGET_CODES = {
    "J84112",   # Idiopathic pulmonary fibrosis
}

# Broad: all ILD/pulmonary-fibrosis codes that may signal IPF
ICD10_BROAD_TARGET_CODES = {
    "J841",     # Other interstitial pulmonary diseases with fibrosis
    "J8410",    # Pulmonary fibrosis, unspecified
    "J8411",    # Idiopathic interstitial pneumonia, NOS
    "J84110",   # Idiopathic interstitial pneumonia, NOS (alt)
    "J84111",   # Idiopathic interstitial pneumonia, NOS
    "J84112",   # Idiopathic pulmonary fibrosis ← primary
    "J84113",   # Idiopathic non-specific interstitial pneumonitis
    "J84114",   # Acute interstitial pneumonitis
    "J84115",   # Respiratory bronchiolitis interstitial lung disease
    "J84116",   # Cryptogenic organizing pneumonia
    "J84117",   # Desquamative interstitial pneumonitis
    "J8417",    # Other interstitial pulmonary diseases with fibrosis in systemic CTD
    "J842",     # Lymphoid interstitial pneumonia
    "J848",     # Other specified interstitial pulmonary diseases
    "J849",     # Interstitial pulmonary disease, unspecified
}


# ── Classifier ───────────────────────────────────────────────────────────────

def _strip(code):
    """Remove dots, whitespace, and upper-case."""
    return code.replace(".", "").strip().upper()


def build_icd10_category_lookup():
    """
    Build a function that maps an ICD-10-CM code to a category index (0–50).
    Returns None if the code doesn't match any category.

    Input codes may contain or omit decimal points; both "J84.112" and
    "J84112" are accepted.
    """

    def classify(raw_code):
        if not raw_code:
            return None
        code = _strip(raw_code)
        if not code:
            return None

        c0 = code[0]

        # ── Chapter I: Infectious & parasitic — A00-B99 ──
        if c0 in ("A", "B"):
            return 0

        # ── Chapter II: Neoplasms — C00-D49 ──
        if c0 == "C":
            return 1  # all C codes = malignant neoplasms
        if c0 == "D":
            # D00-D49 benign/uncertain; D50-D89 blood & immune
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 2
            if num <= 49:
                return 2   # Neoplasms - benign/uncertain
            if num <= 79:
                return 8   # Blood disorders (D50-D79)
            return 7       # Immune disorders (D80-D89)

        # ── Chapter IV: Endocrine, nutritional, metabolic — E00-E89 ──
        if c0 == "E":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 6
            if num <= 7:
                return 3   # Thyroid disorders (E00-E07)
            if num <= 13:
                return 4   # Diabetes mellitus (E08-E13)
            if num == 78:
                return 5   # Lipid metabolism disorders (E78)
            return 6       # Metabolic disorders (everything else in E14-E89)

        # ── Chapter V: Mental & behavioural — F00-F99 ──
        if c0 == "F":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 10
            if num <= 9:
                return 9   # Psychiatric - organic/dementias (F00-F09)
            return 10      # Psychiatric - other (F10-F99)

        # ── Chapter VI: Nervous system — G00-G99 ──
        if c0 == "G":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 11
            if num <= 47:
                return 11  # Central nervous system (G00-G47)
            return 12      # Peripheral nervous system (G48-G99)

        # ── Chapter VII: Eye — H00-H59 ──
        # ── Chapter VIII: Ear — H60-H99 ──
        if c0 == "H":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 13
            if num <= 59:
                return 13  # Ophthalmological
            return 14      # Ear and mastoid

        # ── Chapter IX: Circulatory — I00-I99 ──
        if c0 == "I":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 21

            # Pulmonary hypertension (I27) — most specific, check first
            if num == 27:
                return 50

            if num <= 9:
                return 21  # Cardiovascular - other (I00-I09: rheumatic)
            if num <= 16:
                return 15  # Hypertensive - systemic (I10-I16)
            if num <= 19:
                return 21  # CV other (I17-I19)
            if num <= 25:
                return 16  # Coronary atherosclerosis (I20-I25)
            if num <= 28:
                return 18  # Pulmonary heart disease (I26-I28, excl I27→50)
            if num <= 43:
                return 21  # CV other (I29-I43)
            if num <= 49:
                return 20  # Cardiac arrhythmias (I44-I49)
            if num == 50:
                return 19  # Heart failure
            if num <= 69:
                return 21  # CV other (I51-I69: cerebrovascular etc.)
            if num == 70:
                return 17  # Chronic ischemia / atherosclerosis
            return 21      # CV other (I71-I99)

        # ── Chapter X: Respiratory — J00-J99 ──
        if c0 == "J":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 27

            if num <= 6:
                return 22  # Upper respiratory infections (J00-J06)
            if num <= 11:
                return 27  # Respiratory - other (J07-J11: flu, other URI)
            if num <= 18:
                return 24  # Pneumonia (J12-J18)
            if num <= 22:
                return 27  # Respiratory - other (J19-J22)
            if num == 30:
                return 29  # Allergies (J30: allergic rhinitis)
            if num <= 39:
                return 27  # Respiratory - other (J31-J39)
            if num <= 47:
                return 23  # Respiratory - chronic/COPD (J40-J47)
            if num <= 70:
                return 27  # Respiratory - other (J60-J70: pneumoconioses etc.)
            if num <= 83:
                return 27  # Respiratory - other (J80-J83: respiratory failure etc.)
            if num == 84:
                return 25  # Pulmonary fibrosis/ILD (J84) ← includes J84.112
            if num <= 86:
                return 26  # Respiratory - pleural (J85-J86)
            if num <= 89:
                return 27  # Respiratory - other (J87-J89)
            return 26      # Respiratory - pleural/other (J90-J99)

        # ── Chapter XI: Digestive — K00-K95 ──
        if c0 == "K":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 32
            if num <= 19:
                return 32  # Digestive - other (K00-K19: oral, teeth, salivary)
            if num <= 22:
                return 30  # Esophagus disorders (K20-K22)
            if num <= 24:
                return 32  # Digestive - other (K23-K24: esophagus in diseases elsewhere)
            if num <= 31:
                return 31  # Gastric/duodenal (K25-K31: peptic ulcer, gastritis)
            return 32      # Digestive - other (K32-K95)

        # ── Chapter XII: Skin — L00-L99 ──
        if c0 == "L":
            return 35

        # ── Chapter XIII: Musculoskeletal — M00-M99 ──
        if c0 == "M":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 38
            if num <= 29:
                return 38  # MSK - other (M00-M29: arthropathies)
            if num <= 36:
                return 36  # Connective tissue disorders (M30-M36: systemic CTD)
            if num <= 59:
                return 38  # MSK - other (M37-M59: spinal, dorsopathies)
            if num <= 79:
                return 37  # Soft tissue disorders (M60-M79)
            return 38      # MSK - other (M80-M99: bone density etc.)

        # ── Chapter XIV: Genitourinary — N00-N99 ──
        if c0 == "N":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 34
            if num <= 29:
                return 33  # Kidney diseases (N00-N29)
            return 34      # Genitourinary - other (N30-N99)

        # ── Chapter XV: Pregnancy — O00-O99 ──
        if c0 == "O":
            return 34  # → Genitourinary - other (rare in 45-90 cohort)

        # ── Chapter XVI: Perinatal — P00-P99 ──
        if c0 == "P":
            return 40

        # ── Chapter XVII: Congenital — Q00-Q99 ──
        if c0 == "Q":
            return 39

        # ── Chapter XVIII: Symptoms / ill-defined — R00-R99 ──
        if c0 == "R":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 43

            if num == 6:
                return 28  # Dyspnea & respiratory abnormalities (R06)
            if num == 7:
                return 41  # Chest pain (R07)
            if 70 <= num <= 79:
                return 42  # Blood examination abnormal findings (R70-R79)
            if num >= 95:
                return 44  # Ill-defined conditions (R95-R99)
            return 43      # Symptoms - general

        # ── Chapter XIX: Injury & poisoning — S00-T98 ──
        if c0 == "S":
            return 45  # Injuries - fractures/trauma
        if c0 == "T":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 46
            if num <= 14:
                return 45  # Fractures/trauma (T00-T14)
            return 46      # Injuries - other (T15-T98)

        # ── Chapter XX: External causes — V00-Y99 ──
        if c0 in ("V", "W", "X", "Y"):
            return 49

        # ── Chapter XXI: Factors influencing health — Z00-Z99 ──
        if c0 == "Z":
            try:
                num = int(code[1:3])
            except (ValueError, IndexError):
                return 48
            if 40 <= num <= 53:
                return 47  # Health service contact - aftercare (Z40-Z53)
            return 48      # Health service contact - other

        # Unknown
        return None

    return classify


# ── Utility: detect whether a raw code string looks like ICD-10 ─────────────

def is_icd10_code(code):
    """
    Heuristic to distinguish ICD-10-CM codes from ICD-9-CM codes.

    ICD-10: starts with A-Z letter, then 2 digits, optional alphanumeric suffix
            e.g. J84.112, E11.9, I50.9
    ICD-9:  purely numeric 3-5 digits, or V##... or E###...
            e.g. 4019, 25000, V5869, E8159

    Edge cases handled:
    - E codes: ICD-9 E codes have 3 numeric digits after E (E8xx, E9xx, E0xx).
               ICD-10 E codes have 2 numeric digits (E00-E89) — metabolic disease.
               Disambiguate by total length of numeric portion.
    - V codes: ICD-9 V codes are V + digits only. ICD-10 V codes (external
               causes) include letter suffixes after the 3rd character (e.g.
               V43.52XA). Detect by presence of any alpha char after position 1.
    """
    if not code:
        return False
    c0 = code[0].upper()
    body = code[1:].replace(".", "")

    # Letters that exclusively indicate ICD-10
    if c0 in "ABCDFGHIJKLMNOPQRSTUWXYZ":
        return True

    if c0 == "E":
        # ICD-9 E-codes: E + 3 digits before decimal (E800-E999 external causes).
        # ICD-10 E-codes: E + 2 digits before decimal (E00-E89 metabolic).
        # Check the raw digits before the decimal point.
        raw_body = code[1:]
        digits_before_dot = raw_body.split(".")[0]
        if len(digits_before_dot) >= 3 and digits_before_dot[:3].isdigit():
            return False  # ICD-9 pattern (3+ pre-decimal digits)
        return True       # ICD-10 pattern (2 pre-decimal digits)

    if c0 == "V":
        # ICD-10 V-codes contain letters after the numeric stem
        if any(c.isalpha() for c in body):
            return True
        return False  # ICD-9 V-code

    return False  # purely numeric → ICD-9


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    classify = build_icd10_category_lookup()

    tests = [
        # (code, expected_cat, description)
        ("J84.112", 25, "IPF — primary target"),
        ("J84112",  25, "IPF no-dot"),
        ("J84.9",   25, "ILD unspecified"),
        ("J44.1",   23, "COPD exacerbation"),
        ("J18.9",   24, "Pneumonia NOS"),
        ("I10",     15, "Essential hypertension"),
        ("I50.9",   19, "Heart failure"),
        ("I27.2",   50, "Pulmonary arterial hypertension"),
        ("I25.10",  16, "Coronary atherosclerosis"),
        ("E11.9",    4, "Type 2 diabetes"),
        ("E78.5",    5, "Hyperlipidemia"),
        ("R06.0",   28, "Dyspnea"),
        ("R07.9",   41, "Chest pain NOS"),
        ("Z87.39",  48, "Personal history"),
        ("Z48.89",  47, "Aftercare"),
        ("A41.9",    0, "Sepsis"),
        ("C34.90",   1, "Lung cancer"),
        ("D64.9",    8, "Anemia NOS"),
        ("M34.9",   36, "Systemic sclerosis (scleroderma)"),
        ("M06.9",   38, "Rheumatoid arthritis"),
        ("K21.0",   30, "GERD with esophagitis"),
        ("L40.0",   35, "Psoriasis"),
        ("N18.6",   33, "CKD stage 5"),
        ("G47.33",  11, "Sleep apnea"),
        ("V43.52XA", 49, "Car occupant injured"),
    ]

    print("ICD-10 category mapping self-test:")
    print("-" * 60)
    ok = 0
    for code, expected, desc in tests:
        result = classify(code)
        status = "OK" if result == expected else "FAIL"
        if status != "OK":
            print(f"  {status}: {code} ({desc}) → cat {result} (expected {expected})")
        else:
            ok += 1
    print(f"  {ok}/{len(tests)} passed")

    # Test is_icd10_code heuristic
    print("\nis_icd10_code heuristic:")
    icd10_cases = [
        ("J84.112", True), ("E11.9", True), ("I50.9", True),
        ("V43.52XA", True), ("Z48.89", True), ("A41.9", True),
        ("4019", False), ("25000", False), ("V5869", False),
        ("E8159", False), ("78605", False), ("5163", False),
    ]
    ok2 = 0
    for code, expected in icd10_cases:
        result = is_icd10_code(code)
        status = "OK" if result == expected else "FAIL"
        if status != "OK":
            print(f"  {status}: {code!r} → {result} (expected {expected})")
        else:
            ok2 += 1
    print(f"  {ok2}/{len(icd10_cases)} passed")
