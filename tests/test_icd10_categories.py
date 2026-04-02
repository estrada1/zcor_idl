"""Tests for icd10_categories module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from icd10_categories import (
    build_icd10_category_lookup,
    is_icd10_code,
    ICD10_NARROW_TARGET_CODES,
    ICD10_BROAD_TARGET_CODES,
)


@pytest.fixture(scope="module")
def classify():
    return build_icd10_category_lookup()


class TestIcd10CategoryLookup:
    # ── Primary IPF target ────────────────────────────────────────────────────
    def test_ipf_primary_with_dot(self, classify):
        assert classify("J84.112") == 25

    def test_ipf_primary_no_dot(self, classify):
        assert classify("J84112") == 25

    def test_ild_unspecified(self, classify):
        assert classify("J84.9") == 25

    # ── Respiratory ───────────────────────────────────────────────────────────
    def test_copd(self, classify):
        assert classify("J44.1") == 23

    def test_asthma(self, classify):
        assert classify("J45.9") == 23

    def test_pneumonia(self, classify):
        assert classify("J18.9") == 24

    def test_upper_resp_infection(self, classify):
        assert classify("J06.9") == 22

    def test_allergic_rhinitis(self, classify):
        assert classify("J30.1") == 29

    def test_pleural_effusion(self, classify):
        assert classify("J90") == 26

    def test_respiratory_failure(self, classify):
        assert classify("J96.0") == 26  # J90-J99

    # ── Circulatory ───────────────────────────────────────────────────────────
    def test_essential_hypertension(self, classify):
        assert classify("I10") == 15

    def test_heart_failure(self, classify):
        assert classify("I50.9") == 19

    def test_pulmonary_hypertension(self, classify):
        assert classify("I27.2") == 50

    def test_coronary_artery_disease(self, classify):
        assert classify("I25.10") == 16

    def test_afib(self, classify):
        assert classify("I48.0") == 20

    def test_atherosclerosis(self, classify):
        assert classify("I70.0") == 17

    def test_cerebrovascular(self, classify):
        assert classify("I63.9") == 21  # I51-I69 CV other

    # ── Endocrine ─────────────────────────────────────────────────────────────
    def test_diabetes_t2(self, classify):
        assert classify("E11.9") == 4

    def test_diabetes_t1(self, classify):
        assert classify("E10.9") == 4

    def test_hypothyroidism(self, classify):
        assert classify("E03.9") == 3

    def test_hyperlipidemia(self, classify):
        assert classify("E78.5") == 5

    def test_obesity(self, classify):
        assert classify("E66.9") == 6   # metabolic other

    # ── Symptoms ─────────────────────────────────────────────────────────────
    def test_dyspnea(self, classify):
        assert classify("R06.0") == 28

    def test_chest_pain(self, classify):
        assert classify("R07.9") == 41

    def test_abnormal_blood_finding(self, classify):
        assert classify("R73.0") == 42

    def test_ill_defined(self, classify):
        assert classify("R99") == 44

    def test_general_symptom(self, classify):
        assert classify("R00.0") == 43  # palpitations

    # ── Other chapters ───────────────────────────────────────────────────────
    def test_infection(self, classify):
        assert classify("A41.9") == 0

    def test_malignant_neoplasm(self, classify):
        assert classify("C34.90") == 1

    def test_benign_neoplasm(self, classify):
        assert classify("D12.0") == 2

    def test_blood_disorder(self, classify):
        assert classify("D64.9") == 8

    def test_immune_disorder(self, classify):
        assert classify("D84.9") == 7

    def test_dementias(self, classify):
        assert classify("F03.9") == 9

    def test_psych_other(self, classify):
        assert classify("F32.9") == 10

    def test_cns(self, classify):
        assert classify("G47.33") == 11   # sleep apnea

    def test_peripheral_ns(self, classify):
        assert classify("G60.0") == 12

    def test_eye(self, classify):
        assert classify("H26.9") == 13

    def test_ear(self, classify):
        assert classify("H66.9") == 14

    def test_connective_tissue(self, classify):
        assert classify("M34.9") == 36   # systemic sclerosis

    def test_soft_tissue(self, classify):
        assert classify("M79.3") == 37

    def test_msk_other(self, classify):
        assert classify("M06.9") == 38

    def test_kidney(self, classify):
        assert classify("N18.6") == 33

    def test_gu_other(self, classify):
        assert classify("N40.1") == 34

    def test_skin(self, classify):
        assert classify("L40.0") == 35

    def test_congenital(self, classify):
        assert classify("Q25.0") == 39

    def test_perinatal(self, classify):
        assert classify("P05.0") == 40

    def test_fracture(self, classify):
        assert classify("S52.5") == 45

    def test_injury_other(self, classify):
        assert classify("T79.3") == 46

    def test_external_cause_v(self, classify):
        assert classify("V43.52XA") == 49

    def test_external_cause_w(self, classify):
        assert classify("W19.XXXA") == 49

    def test_aftercare(self, classify):
        assert classify("Z48.89") == 47

    def test_health_contact_other(self, classify):
        assert classify("Z87.39") == 48

    def test_empty_string(self, classify):
        assert classify("") is None

    def test_unknown_code(self, classify):
        assert classify("99999") is None


class TestIsIcd10Code:
    def test_j84_with_dot(self):
        assert is_icd10_code("J84.112") is True

    def test_j84_no_dot(self):
        assert is_icd10_code("J84112") is True

    def test_e11_with_dot(self):
        assert is_icd10_code("E11.9") is True

    def test_i10(self):
        assert is_icd10_code("I10") is True

    def test_z_code(self):
        assert is_icd10_code("Z48.89") is True

    def test_v_code_icd10(self):
        assert is_icd10_code("V43.52XA") is True

    def test_icd9_numeric(self):
        assert is_icd10_code("4019") is False

    def test_icd9_long_numeric(self):
        assert is_icd10_code("25000") is False

    def test_icd9_v_code(self):
        assert is_icd10_code("V5869") is False

    def test_icd9_e_code(self):
        assert is_icd10_code("E8159") is False

    def test_icd9_diag_numeric(self):
        assert is_icd10_code("5163") is False

    def test_empty(self):
        assert is_icd10_code("") is False


class TestTargetCodes:
    def test_narrow_contains_j84112(self):
        assert "J84112" in ICD10_NARROW_TARGET_CODES

    def test_broad_is_superset(self):
        assert ICD10_NARROW_TARGET_CODES.issubset(ICD10_BROAD_TARGET_CODES)

    def test_broad_contains_unspecified_ild(self):
        assert "J849" in ICD10_BROAD_TARGET_CODES
