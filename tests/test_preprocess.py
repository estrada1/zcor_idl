"""Tests for CMS data preprocessing (download config, parsing, cohort logic)."""

import os
import csv
import tempfile
from datetime import date
from collections import defaultdict

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess_cms import (
    parse_cms_date,
    format_date,
    identify_cohorts,
    filter_patients,
    write_zcor_dat,
    write_labels,
    NARROW_TARGET_CODES,
    BROAD_TARGET_CODES,
    CLAIMS_FILES,
    BENEFICIARY_FILES,
    MIN_AGE,
    MAX_AGE,
    MIN_RECORD_SPAN_DAYS,
)


# ── Date parsing ──────────────────────────────────────────────

class TestParseCmsDate:
    def test_standard_format(self):
        assert parse_cms_date("20090115") == date(2009, 1, 15)

    def test_with_whitespace(self):
        assert parse_cms_date("  20100301  ") == date(2010, 3, 1)

    def test_with_quotes(self):
        assert parse_cms_date('"20081231"') == date(2008, 12, 31)

    def test_empty_string(self):
        assert parse_cms_date("") is None

    def test_whitespace_only(self):
        assert parse_cms_date("   ") is None

    def test_invalid_date(self):
        assert parse_cms_date("99999999") is None

    def test_wrong_format(self):
        assert parse_cms_date("2009-01-15") is None

    def test_partial_date(self):
        assert parse_cms_date("200901") is None


class TestFormatDate:
    def test_iso_format(self):
        assert format_date(date(2009, 1, 15)) == "2009-01-15"

    def test_roundtrip(self):
        d = parse_cms_date("20100630")
        assert format_date(d) == "2010-06-30"


# ── Download/config validation ────────────────────────────────

class TestDataConfig:
    def test_beneficiary_files_listed(self):
        assert len(BENEFICIARY_FILES) == 2
        for f in BENEFICIARY_FILES:
            assert "Beneficiary" in f or "beneficiary" in f.lower()

    def test_claims_file_types(self):
        expected = {"inpatient", "outpatient", "carrier_a", "carrier_b"}
        assert set(CLAIMS_FILES.keys()) == expected

    def test_claims_have_required_keys(self):
        for name, config in CLAIMS_FILES.items():
            assert "path" in config, f"{name} missing 'path'"
            assert "date_col" in config, f"{name} missing 'date_col'"
            assert "diag_cols" in config, f"{name} missing 'diag_cols'"
            assert len(config["diag_cols"]) > 0, f"{name} has empty diag_cols"

    def test_inpatient_has_10_dgns_plus_admitting(self):
        cols = CLAIMS_FILES["inpatient"]["diag_cols"]
        assert "ICD9_DGNS_CD_1" in cols
        assert "ICD9_DGNS_CD_10" in cols
        assert "ADMTNG_ICD9_DGNS_CD" in cols

    def test_carrier_has_line_diag_cols(self):
        cols = CLAIMS_FILES["carrier_a"]["diag_cols"]
        line_cols = [c for c in cols if c.startswith("LINE_ICD9")]
        assert len(line_cols) == 13  # LINE_ICD9_DGNS_CD_1 through 13

    def test_date_col_consistent(self):
        for name, config in CLAIMS_FILES.items():
            assert config["date_col"] == "CLM_FROM_DT"

    def test_target_codes_are_strings(self):
        for code in NARROW_TARGET_CODES:
            assert isinstance(code, str)
        for code in BROAD_TARGET_CODES:
            assert isinstance(code, str)

    def test_narrow_is_subset_of_broad(self):
        assert NARROW_TARGET_CODES <= BROAD_TARGET_CODES

    def test_age_bounds(self):
        assert MIN_AGE == 45
        assert MAX_AGE == 90
        assert MIN_AGE < MAX_AGE


# ── Cohort identification ─────────────────────────────────────

def _make_patient_codes(pid_code_map):
    """Helper: build patient_codes dict from {pid: [(date, {codes}), ...]}."""
    pc = defaultdict(lambda: defaultdict(set))
    for pid, encounters in pid_code_map.items():
        for dt, codes in encounters:
            pc[pid][dt].update(codes)
    return pc


def _make_patients(pid_list):
    """Helper: minimal patients dict."""
    return {pid: {"birth_date": date(1950, 1, 1), "sex": "M", "death_date": None}
            for pid in pid_list}


class TestIdentifyCohorts:
    def test_positive_patient_detected(self):
        patients = _make_patients(["P1", "P2"])
        pc = _make_patient_codes({
            "P1": [(date(2009, 6, 1), {"5163", "4019"})],
            "P2": [(date(2009, 6, 1), {"4019"})],
        })
        pos, ctrl = identify_cohorts(patients, pc, NARROW_TARGET_CODES)
        assert "P1" in pos
        assert "P2" in ctrl

    def test_no_false_positives(self):
        patients = _make_patients(["P1"])
        pc = _make_patient_codes({
            "P1": [(date(2009, 1, 1), {"4019", "25000"})],
        })
        pos, ctrl = identify_cohorts(patients, pc, NARROW_TARGET_CODES)
        assert len(pos) == 0
        assert "P1" in ctrl

    def test_broad_targets_expand_positives(self):
        patients = _make_patients(["P1", "P2", "P3"])
        pc = _make_patient_codes({
            "P1": [(date(2009, 1, 1), {"515"})],   # postinflammatory pulm fibrosis
            "P2": [(date(2009, 1, 1), {"5169"})],   # unspecified alveolar pneumonopathy
            "P3": [(date(2009, 1, 1), {"4019"})],   # hypertension
        })
        pos, ctrl = identify_cohorts(patients, pc, BROAD_TARGET_CODES)
        assert pos == {"P1", "P2"}
        assert ctrl == {"P3"}

    def test_multiple_encounters_any_positive(self):
        patients = _make_patients(["P1"])
        pc = _make_patient_codes({
            "P1": [
                (date(2008, 1, 1), {"4019"}),
                (date(2009, 6, 1), {"25000"}),
                (date(2010, 3, 1), {"5163"}),  # target on last encounter
            ],
        })
        pos, ctrl = identify_cohorts(patients, pc, NARROW_TARGET_CODES)
        assert "P1" in pos

    def test_patients_without_claims_excluded(self):
        patients = _make_patients(["P1", "P2"])
        pc = _make_patient_codes({
            "P1": [(date(2009, 1, 1), {"4019"})],
            # P2 has no claims
        })
        pos, ctrl = identify_cohorts(patients, pc, NARROW_TARGET_CODES)
        assert "P2" not in ctrl  # not in patient_codes

    def test_empty_input(self):
        pos, ctrl = identify_cohorts({}, defaultdict(lambda: defaultdict(set)), NARROW_TARGET_CODES)
        assert len(pos) == 0
        assert len(ctrl) == 0


# ── Patient filtering ─────────────────────────────────────────

class TestFilterPatients:
    def _make_long_history(self, pid, birth_year=1950):
        """Create a patient with 3 years of claims."""
        patients = {pid: {
            "birth_date": date(birth_year, 1, 1), "sex": "M", "death_date": None
        }}
        pc = _make_patient_codes({
            pid: [
                (date(2008, 1, 15), {"4019"}),
                (date(2010, 6, 15), {"25000"}),
            ],
        })
        return patients, pc

    def test_valid_patient_passes(self):
        patients, pc = self._make_long_history("P1", 1950)
        result = filter_patients(patients, pc, {"P1"}, reference_date=date(2010, 12, 31))
        assert "P1" in result

    def test_too_young_excluded(self):
        # Born 1970 → age 40 at 2010-12-31 → under MIN_AGE=45
        patients, pc = self._make_long_history("P1", 1970)
        result = filter_patients(patients, pc, {"P1"}, reference_date=date(2010, 12, 31))
        assert "P1" not in result

    def test_too_old_excluded(self):
        # Born 1915 → age 95 at 2010-12-31 → over MAX_AGE=90
        patients, pc = self._make_long_history("P1", 1915)
        result = filter_patients(patients, pc, {"P1"}, reference_date=date(2010, 12, 31))
        assert "P1" not in result

    def test_short_record_excluded(self):
        patients = {"P1": {
            "birth_date": date(1950, 1, 1), "sex": "M", "death_date": None
        }}
        # Only 3 months of history → under MIN_RECORD_SPAN_DAYS
        pc = _make_patient_codes({
            "P1": [
                (date(2010, 7, 1), {"4019"}),
                (date(2010, 10, 1), {"25000"}),
            ],
        })
        result = filter_patients(patients, pc, {"P1"}, reference_date=date(2010, 12, 31))
        assert "P1" not in result

    def test_boundary_age_included(self):
        # Born 1965-01-01 → age ~45.99 at 2010-12-31 → inside MIN_AGE..MAX_AGE
        patients = {"P1": {
            "birth_date": date(1965, 1, 1), "sex": "F", "death_date": None
        }}
        pc = _make_patient_codes({
            "P1": [
                (date(2008, 1, 1), {"4019"}),
                (date(2010, 12, 1), {"25000"}),
            ],
        })
        result = filter_patients(patients, pc, {"P1"}, reference_date=date(2010, 12, 31))
        assert "P1" in result

    def test_no_claims_excluded(self):
        patients = {"P1": {
            "birth_date": date(1950, 1, 1), "sex": "M", "death_date": None
        }}
        pc = defaultdict(lambda: defaultdict(set))  # empty claims
        result = filter_patients(patients, pc, {"P1"}, reference_date=date(2010, 12, 31))
        assert "P1" not in result


# ── Output writing ────────────────────────────────────────────

class TestWriteZcorDat:
    def test_dat_file_format(self):
        patients = {"P1": {
            "birth_date": date(1950, 6, 15), "sex": "M", "death_date": None
        }}
        pc = _make_patient_codes({
            "P1": [
                (date(2009, 3, 1), {"4019", "25000"}),
                (date(2009, 6, 1), {"2724"}),
            ],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            path = f.name

        try:
            count = write_zcor_dat(patients, pc, {"P1"}, path)
            assert count == 1

            with open(path) as f:
                line = f.readline().strip()

            parts = line.split(",")
            assert parts[0] == "P1"
            assert parts[1] == "M"
            assert parts[2] == "1950-06-15"
            # First encounter date
            assert parts[3] == "2009-03-01"
            # Codes are sorted
            codes_1 = [p for p in parts[4:] if not p.startswith("200")]
            assert "25000" in parts or "4019" in parts
        finally:
            os.unlink(path)

    def test_dat_file_sorted_by_date(self):
        patients = {"P1": {
            "birth_date": date(1950, 1, 1), "sex": "F", "death_date": None
        }}
        pc = _make_patient_codes({
            "P1": [
                (date(2010, 1, 1), {"496"}),
                (date(2008, 6, 1), {"4019"}),
            ],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            path = f.name

        try:
            write_zcor_dat(patients, pc, {"P1"}, path)
            with open(path) as f:
                line = f.readline().strip()
            parts = line.split(",")
            # 2008 date should come before 2010 date
            dates_in_line = [p for p in parts[3:] if len(p) == 10 and p[4] == "-"]
            assert dates_in_line[0] == "2008-06-01"
            assert dates_in_line[1] == "2010-01-01"
        finally:
            os.unlink(path)


class TestWriteLabels:
    def test_labels_csv_format(self):
        patients = {
            "P1": {"birth_date": date(1950, 1, 1), "sex": "M", "death_date": None},
            "P2": {"birth_date": date(1955, 6, 1), "sex": "F", "death_date": None},
        }
        pc = _make_patient_codes({
            "P1": [
                (date(2008, 1, 1), {"5163", "4019"}),
                (date(2010, 6, 1), {"25000"}),
            ],
            "P2": [
                (date(2008, 3, 1), {"4019"}),
                (date(2010, 9, 1), {"2724"}),
            ],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "labels.csv")
            write_labels(patients, pc, {"P1"}, {"P2"}, path)

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            p1 = next(r for r in rows if r["patient_id"] == "P1")
            p2 = next(r for r in rows if r["patient_id"] == "P2")

            assert p1["label"] == "1"
            assert p2["label"] == "0"
            assert p1["sex"] == "M"
            assert p2["sex"] == "F"
            assert "record_start" in p1
            assert "record_span_weeks" in p1
            assert float(p1["record_span_weeks"]) > 0
