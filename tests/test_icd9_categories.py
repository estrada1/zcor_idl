"""Tests for ICD-9 → 51 disease category mapping."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from icd9_categories import (
    build_category_lookup,
    DISEASE_CATEGORIES,
    CATEGORY_NAMES,
    NUM_CATEGORIES,
    get_category_name,
)


@pytest.fixture
def classify():
    return build_category_lookup()


# ── Category structure ────────────────────────────────────────

class TestCategoryStructure:
    def test_51_categories(self):
        assert NUM_CATEGORIES == 51

    def test_category_ids_contiguous(self):
        ids = sorted(cat_id for cat_id, _, _ in DISEASE_CATEGORIES)
        assert ids == list(range(51))

    def test_all_categories_named(self):
        for i in range(NUM_CATEGORIES):
            name = get_category_name(i)
            assert name and "Unknown" not in name

    def test_category_names_dict_complete(self):
        assert len(CATEGORY_NAMES) == NUM_CATEGORIES


# ── Known code mappings (from paper Fig 2e) ───────────────────

class TestKnownMappings:
    """Verify codes map to their expected categories per the paper."""

    @pytest.mark.parametrize("code,expected_cat,description", [
        ("4019",  15, "Hypertension NOS"),
        ("25000", 4,  "Diabetes type 2"),
        ("2724",  5,  "Hyperlipidemia"),
        ("42731", 20, "Atrial fibrillation"),
        ("5163",  25, "IPF/ILD"),
        ("515",   25, "Postinflammatory pulmonary fibrosis"),
        ("496",   23, "COPD NOS"),
        ("78650", 41, "Chest pain NOS"),
        ("78605", 28, "Dyspnea"),
        ("4168",  50, "Pulmonary hypertension"),
        ("4280",  19, "Heart failure NOS"),
        ("41401", 16, "Coronary atherosclerosis"),
        ("51881", 26, "Respiratory failure"),
        ("7106",  36, "Connective tissue disorder"),
        ("486",   24, "Pneumonia NOS"),
        ("27800", 7,  "Obesity → immune"),
        ("2859",  8,  "Anemia NOS"),
        ("311",   10, "Depressive disorder"),
        ("5990",  34, "UTI"),
        ("7242",  38, "Lumbago"),
    ])
    def test_code_mapping(self, classify, code, expected_cat, description):
        result = classify(code)
        assert result == expected_cat, (
            f"{code} ({description}): expected cat {expected_cat} "
            f"({get_category_name(expected_cat)}), got {result} "
            f"({get_category_name(result) if result is not None else 'None'})"
        )


# ── V-codes and E-codes ──────────────────────────────────────

class TestSpecialCodes:
    def test_v_code_aftercare(self, classify):
        assert classify("V5869") == 47  # aftercare V50-V59

    def test_v_code_other(self, classify):
        assert classify("V1000") == 48  # other V-codes

    def test_e_code(self, classify):
        assert classify("E8859") == 49

    def test_v_code_short(self, classify):
        # Short V-code should still map
        result = classify("V10")
        assert result == 48


# ── Edge cases ────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self, classify):
        assert classify("") is None

    def test_nonsense_code(self, classify):
        assert classify("ZZZZZ") is None

    def test_pulmonary_hypertension_override(self, classify):
        """416.x should go to cat 50 (pulmonary HTN), not cat 18 (pulmonary heart)."""
        assert classify("4160") == 50
        assert classify("4169") == 50

    def test_chest_pain_override(self, classify):
        """786.5x goes to cat 41, not cat 28 (dyspnea)."""
        assert classify("7865") == 41
        assert classify("78650") == 41

    def test_dyspnea_not_chest_pain(self, classify):
        """786.0x goes to cat 28 (dyspnea), not cat 41."""
        assert classify("7860") == 28
        assert classify("78609") == 28

    def test_ild_range(self, classify):
        """515-516 should all map to cat 25 (pulmonary fibrosis/ILD)."""
        assert classify("515") == 25
        assert classify("5160") == 25
        assert classify("5163") == 25
        assert classify("5169") == 25


# ── Coverage: all ICD-9 chapters map somewhere ────────────────

class TestChapterCoverage:
    """Ensure representative codes from each ICD-9 chapter get classified."""

    @pytest.mark.parametrize("code,chapter", [
        ("001", "Infectious"),
        ("140", "Neoplasms"),
        ("250", "Endocrine"),
        ("280", "Blood"),
        ("290", "Mental"),
        ("320", "Nervous"),
        ("360", "Eye"),
        ("380", "Ear"),
        ("401", "Circulatory"),
        ("460", "Respiratory"),
        ("520", "Digestive"),
        ("580", "Genitourinary"),
        ("680", "Skin"),
        ("710", "Musculoskeletal"),
        ("740", "Congenital"),
        ("780", "Symptoms"),
        ("800", "Injury"),
    ])
    def test_chapter_maps(self, classify, code, chapter):
        result = classify(code)
        assert result is not None, f"Code {code} ({chapter}) returned None"
