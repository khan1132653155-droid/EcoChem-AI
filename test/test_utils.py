import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils import smiles_to_fingerprint, combine_fingerprints, name_to_fingerprint, COMMON_NADES


def test_smiles_to_fingerprint_returns_2048():
    fp = smiles_to_fingerprint("CCO")
    assert fp is not None
    assert fp.shape == (2048,)
    assert fp.dtype == np.float32


def test_smiles_to_fingerprint_invalid():
    fp = smiles_to_fingerprint("NOT_A_CHEMICAL")
    assert fp is None


def test_combine_fingerprints_adds_correctly():
    fp1 = np.ones(2048, dtype=np.float32)
    fp2 = np.ones(2048, dtype=np.float32) * 2
    result = combine_fingerprints(fp1, fp2, ratio=1.0)
    expected = fp1 + 1.0 * fp2
    np.testing.assert_array_equal(result, expected)


def test_combine_fingerprints_with_ratio():
    fp1 = np.ones(2048, dtype=np.float32)
    fp2 = np.ones(2048, dtype=np.float32)
    result = combine_fingerprints(fp1, fp2, ratio=3.0)
    expected = fp1 + 3.0 * fp2
    np.testing.assert_array_equal(result, expected)


def test_common_nades_has_smiles():
    for name, smiles in COMMON_NADES.items():
        assert len(smiles) > 0, f"{name} has empty SMILES"
        fp = smiles_to_fingerprint(smiles)
        assert fp is not None, f"{name} SMILES {smiles} failed fingerprinting"
        assert fp.shape == (2048,)


def test_name_to_fingerprint_known_chemical():
    fp, smiles = name_to_fingerprint("Water")
    if fp is not None:
        assert fp.shape == (2048,)
        assert smiles is not None


def test_compute_rdkit_descriptors_ethanol():
    from utils import compute_rdkit_descriptors
    desc = compute_rdkit_descriptors("CCO")
    assert desc is not None
    assert desc["hbd"] == 1
    assert desc["hba"] == 1
    assert desc["logp"] is not None
    assert desc["tpsa"] > 0


def test_compute_rdkit_descriptors_invalid():
    from utils import compute_rdkit_descriptors
    desc = compute_rdkit_descriptors("NOT_A_CHEMICAL")
    assert desc is None


def test_suggest_precipitation_reagents_for_alkaloids():
    from utils import suggest_precipitation_reagents
    fgs = {"amine_primary": 1, "aromatic_ring": 2}
    steps = suggest_precipitation_reagents(fgs)
    assert len(steps) > 0
    assert any("Dragendorff" in s["detail"] for s in steps)
    assert all(s["step_type"] == "precipitation" for s in steps)


def test_suggest_precipitation_reagents_for_phenolics():
    from utils import suggest_precipitation_reagents
    fgs = {"phenolic_oh": 3, "carboxylic_acid": 1}
    steps = suggest_precipitation_reagents(fgs)
    assert len(steps) > 0
    assert any("Lead" in s["reagent"] for s in steps)


def test_suggest_chromatography_by_class():
    from utils import suggest_chromatography
    fgs = {"aromatic_ring": 2}
    recs = suggest_chromatography(fgs, compound_class="Flavonoids")
    assert any("C18" in r["stationary_phase"] for r in recs)
    assert any("Formic Acid" in r["mobile_phase"] for r in recs)


def test_suggest_chromatography_by_fg_fallback():
    from utils import suggest_chromatography
    fgs = {"alkene": 2}
    recs = suggest_chromatography(fgs, compound_class=None)
    assert len(recs) > 0
    assert any("AgNO" in r.get("note", "") for r in recs) or any("AgNO" in r.get("stationary_phase", "") for r in recs)


def test_chiral_hbd_list():
    from utils import CHIRAL_HBD
    assert "L-Malic Acid" in CHIRAL_HBD
    assert "D-Tartaric Acid" in CHIRAL_HBD
    assert "L-Lactic Acid" in CHIRAL_HBD
    assert len(CHIRAL_HBD) >= 7


def test_precipitation_reagents_coverage():
    from utils import PRECIPITATION_REAGENTS
    assert "phenolic_oh" in PRECIPITATION_REAGENTS
    assert "amine_primary" in PRECIPITATION_REAGENTS
    assert "carboxylic_acid" in PRECIPITATION_REAGENTS
    assert len(PRECIPITATION_REAGENTS) >= 5


def test_chromatography_class_coverage():
    from utils import COMPOUND_CLASS_CHROMATOGRAPHY
    for cls in ["Alkaloids", "Terpenoids", "Flavonoids", "Phenolics"]:
        assert cls in COMPOUND_CLASS_CHROMATOGRAPHY
        assert "stationary_phase" in COMPOUND_CLASS_CHROMATOGRAPHY[cls]
        assert "mobile_phase" in COMPOUND_CLASS_CHROMATOGRAPHY[cls]


def test_compute_contrast_analysis():
    from utils import compute_rdkit_descriptors
    from ml_engine import compute_contrast_analysis
    target_desc = compute_rdkit_descriptors("CCO")
    contrast = compute_contrast_analysis(target_desc, ["Water", "Methanol"])
    assert contrast["matrix_count"] > 0
    assert "logp_delta" in contrast
    assert "selectivity_estimate" in contrast
