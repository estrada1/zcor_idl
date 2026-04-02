"""Tests for trinary time series encoding."""

import os
import sys
import tempfile
import csv

import numpy as np
import pytest


from datetime import date, timedelta
from zcor_idl.encoding import (
    parse_dat_file,
    determine_prediction_point,
    encode_patient,
    INFERENCE_WINDOW_WEEKS,
    TARGET_CODES,
)
from zcor_idl.icd9 import build_category_lookup, NUM_CATEGORIES


@pytest.fixture
def classify():
    return build_category_lookup()


# ── .dat file parsing ─────────────────────────────────────────

def _write_dat(lines):
    """Write lines to a temp .dat file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    for line in lines:
        f.write(line + "\n")
    f.close()
    return f.name


class TestParseDatFile:
    def test_single_patient(self):
        path = _write_dat([
            "P001,M,1950-01-01,2009-03-01,4019,25000,2009-06-15,2724"
        ])
        try:
            patients = parse_dat_file(path)
            assert len(patients) == 1
            p = patients[0]
            assert p["id"] == "P001"
            assert p["sex"] == "M"
            assert p["birth_date"] == date(1950, 1, 1)
            assert len(p["encounters"]) == 2
            assert p["encounters"][0] == (date(2009, 3, 1), ["4019", "25000"])
            assert p["encounters"][1] == (date(2009, 6, 15), ["2724"])
        finally:
            os.unlink(path)

    def test_multiple_patients(self):
        path = _write_dat([
            "P001,M,1950-01-01,2009-03-01,4019",
            "P002,F,1955-06-15,2008-01-01,25000,2009-12-01,2724",
        ])
        try:
            patients = parse_dat_file(path)
            assert len(patients) == 2
            assert patients[0]["id"] == "P001"
            assert patients[1]["id"] == "P002"
            assert patients[1]["sex"] == "F"
        finally:
            os.unlink(path)

    def test_empty_lines_skipped(self):
        path = _write_dat(["", "P001,M,1950-01-01,2009-03-01,4019", ""])
        try:
            patients = parse_dat_file(path)
            assert len(patients) == 1
        finally:
            os.unlink(path)

    def test_short_line_skipped(self):
        """Lines with < 4 parts (no encounter data) are skipped by the parser."""
        path = _write_dat(["P001,M,1950-01-01"])  # no encounters → < 4 parts
        try:
            patients = parse_dat_file(path)
            assert len(patients) == 0  # skipped because len(parts) < 4
        finally:
            os.unlink(path)


# ── Prediction point determination ────────────────────────────

class TestDeterminePredictionPoint:
    def test_positive_patient(self):
        """Positive: prediction = 52 weeks before first target code."""
        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [
                (date(2008, 3, 1), ["4019"]),
                (date(2009, 6, 1), ["5163"]),  # target code
                (date(2010, 1, 1), ["25000"]),
            ],
        }
        pred_date, is_pos, first_target = determine_prediction_point(patient)
        assert is_pos is True
        assert first_target == date(2009, 6, 1)
        expected_pred = date(2009, 6, 1) - timedelta(weeks=52)
        assert pred_date == expected_pred

    def test_control_patient(self):
        """Control: prediction = last record date."""
        patient = {
            "id": "P2", "sex": "F", "birth_date": date(1955, 1, 1),
            "encounters": [
                (date(2008, 3, 1), ["4019"]),
                (date(2010, 9, 15), ["2724"]),
            ],
        }
        pred_date, is_pos, first_target = determine_prediction_point(patient)
        assert is_pos is False
        assert first_target is None
        assert pred_date == date(2010, 9, 15)

    def test_earliest_target_used(self):
        """If multiple target codes, use the earliest date."""
        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [
                (date(2008, 6, 1), ["5163"]),  # first target
                (date(2009, 6, 1), ["5163"]),   # second target
            ],
        }
        _, _, first_target = determine_prediction_point(patient)
        assert first_target == date(2008, 6, 1)


# ── Trinary encoding logic ───────────────────────────────────

class TestEncodePatient:
    def test_output_shape(self, classify):
        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [(date(2009, 6, 1), ["4019"])],
        }
        pred_date = date(2010, 6, 1)
        series = encode_patient(patient, classify, pred_date)
        assert series.shape == (NUM_CATEGORIES, INFERENCE_WINDOW_WEEKS)

    def test_values_are_trinary(self, classify):
        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [
                (date(2009, 1, 1), ["4019", "25000"]),
                (date(2009, 6, 1), ["2724"]),
                (date(2009, 9, 1), ["486"]),
            ],
        }
        pred_date = date(2010, 1, 1)
        series = encode_patient(patient, classify, pred_date)
        unique_vals = set(np.unique(series))
        assert unique_vals <= {0, 1, 2}

    def test_no_encounters_all_zero(self, classify):
        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [],
        }
        pred_date = date(2010, 1, 1)
        series = encode_patient(patient, classify, pred_date)
        assert np.all(series == 0)

    def test_encounter_outside_window_ignored(self, classify):
        """Encounters before the observation window should be ignored."""
        pred_date = date(2010, 1, 1)
        window_start = pred_date - timedelta(weeks=INFERENCE_WINDOW_WEEKS)

        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [
                (window_start - timedelta(days=30), ["4019"]),  # before window
            ],
        }
        series = encode_patient(patient, classify, pred_date)
        assert np.all(series == 0)

    def test_category_1_and_2_encoding(self, classify):
        """When hypertension code appears, cat 15 should be 1, others should be 2."""
        pred_date = date(2010, 1, 1)
        window_start = pred_date - timedelta(weeks=INFERENCE_WINDOW_WEEKS)
        encounter_date = window_start + timedelta(days=7)  # week 1

        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [(encounter_date, ["4019"])],  # hypertension → cat 15
        }
        series = encode_patient(patient, classify, pred_date)

        week_idx = (encounter_date - window_start).days // 7
        # Cat 15 (hypertension) should be 1
        assert series[15, week_idx] == 1
        # Other categories with codes in this week should be 2
        for k in range(NUM_CATEGORIES):
            if k != 15:
                assert series[k, week_idx] == 2

    def test_multiple_categories_same_week(self, classify):
        """Multiple codes in same week: each category gets 1, others get 2."""
        pred_date = date(2010, 1, 1)
        window_start = pred_date - timedelta(weeks=INFERENCE_WINDOW_WEEKS)
        encounter_date = window_start + timedelta(days=14)

        patient = {
            "id": "P1", "sex": "M", "birth_date": date(1950, 1, 1),
            "encounters": [(encounter_date, ["4019", "25000"])],  # HTN + diabetes
        }
        series = encode_patient(patient, classify, pred_date)

        week_idx = (encounter_date - window_start).days // 7
        assert series[15, week_idx] == 1  # hypertension
        assert series[4, week_idx] == 1   # diabetes
        # All other categories should be 2 (other code present)
        for k in range(NUM_CATEGORIES):
            if k not in (15, 4):
                assert series[k, week_idx] == 2
