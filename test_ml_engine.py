import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ml_engine import (
    generate_synthetic_dataset,
    build_model,
    train,
    save_model,
    load_model,
    model_exists,
    predict,
    initialize,
    N_FEATURES,
    N_TARGETS,
    TARGET_NAMES,
)
import tempfile


def test_synthetic_data_shape():
    X, y = generate_synthetic_dataset(n_samples=100)
    assert X.shape == (100, N_FEATURES)
    assert y.shape == (100, N_TARGETS)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_synthetic_targets_ranges():
    X, y = generate_synthetic_dataset(n_samples=200)
    assert np.all(y[:, 0] >= -50.0)
    assert np.all(y[:, 0] <= 120.0)
    assert np.all((y[:, 1] == 0.0) | (y[:, 1] == 1.0))
    assert np.all(y[:, 2] >= 0.0)
    assert np.all(y[:, 2] <= 1.0)


def test_model_build_and_train():
    X, y = generate_synthetic_dataset(n_samples=50)
    model, scaler = train(X, y)
    assert model is not None
    assert scaler is not None


def test_save_load_roundtrip():
    X, y = generate_synthetic_dataset(n_samples=50)
    model, scaler = train(X, y)
    with tempfile.TemporaryDirectory() as tmp:
        orig_path = os.path.join(tmp, "test_model.pkl")
        scaler_path = os.path.join(tmp, "test_scaler.pkl")
        import joblib
        joblib.dump(model, orig_path)
        joblib.dump(scaler, scaler_path)
        loaded_model = joblib.load(orig_path)
        loaded_scaler = joblib.load(scaler_path)
        assert loaded_model is not None
        assert loaded_scaler is not None


def test_predict_output_format():
    initialize()
    fp = np.random.randint(0, 2, N_FEATURES).astype(np.float32)
    result = predict(fp)
    expected_keys = {"Eutectic_Temp_C", "Phase_Stability", "EcoToxicity_Index", "Liquid_At_Room_Temp"}
    assert set(result.keys()) == expected_keys
    assert isinstance(result["Eutectic_Temp_C"], float)
    assert isinstance(result["Phase_Stability"], bool)
    assert isinstance(result["EcoToxicity_Index"], float)
    assert isinstance(result["Liquid_At_Room_Temp"], bool)
    assert 0.0 <= result["EcoToxicity_Index"] <= 1.0


def test_initialize_creates_model():
    if model_exists():
        os.remove(os.path.join(os.path.dirname(__file__), "..", "models", "ecochem_model.pkl"))
        os.remove(os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl"))
    initialize()
    assert model_exists()


def test_recommend_nades_with_path_fast_path():
    from ml_engine import recommend_nades_with_path
    result = recommend_nades_with_path(
        target_name="Quercetin",
        target_smiles="C1=CC(=C(C2=C1C(=O)C(=C(O2)O)O)O)O",
        compound_class="Flavonoids",
        preferences={"liquid_rt": 1.0, "low_eco_tox": 0.5, "stability": 0.5},
        mode="targeted",
    )
    assert result["path"] == "fast"
    assert "Literature" in result["path_label"]
    assert len(result["recommendations"]) >= 1
    r = result["recommendations"][0]
    assert r["HBA"] == "Choline Chloride"
    assert r["HBD"] == "Citric Acid"
    assert result["eco_toxicity"] is not None


def test_recommend_nades_with_path_slow_path():
    from ml_engine import recommend_nades_with_path
    result = recommend_nades_with_path(
        target_name="UnknownCompoundX",
        target_smiles="CCCCCCCO",
        compound_class="Terpenoids",
        bulk_matrix_names=["Water", "Cellulose"],
        preferences={"liquid_rt": 1.0, "low_eco_tox": 0.5, "stability": 0.5},
        mode="targeted",
    )
    assert result["path"] == "slow"
    assert "Dynamic" in result["path_label"]
    assert len(result["recommendations"]) >= 1


def test_recommend_nades_with_path_total_extraction():
    from ml_engine import recommend_nades_with_path
    result = recommend_nades_with_path(
        target_name=None,
        target_smiles=None,
        compound_class=None,
        mode="total",
    )
    assert result["path"] == "slow"
    assert result["mode"] == "total"


def test_recommend_nades_with_path_eco_toxicity_present():
    from ml_engine import recommend_nades_with_path
    result = recommend_nades_with_path(
        target_name="Caffeine",
        target_smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        compound_class="Alkaloids",
        mode="targeted",
    )
    r = result["recommendations"][0]
    assert 0.0 <= r["EcoToxicity_Index"] <= 1.0
    assert result["eco_toxicity"] is not None


def test_compute_contrast_analysis_function():
    from ml_engine import compute_contrast_analysis
    target_desc = {"hbd": 1, "hba": 1, "tpsa": 20.0, "logp": 0.5,
                   "mol_wt": 46, "rotatable_bonds": 0, "hbd_hba_ratio": 1.0}
    contrast = compute_contrast_analysis(target_desc, ["Water", "Methanol"])
    assert contrast["matrix_count"] == 2
    assert "logp_delta" in contrast
    assert "selectivity_estimate" in contrast
    assert contrast["target_logp"] == 0.5


def test_compute_contrast_analysis_empty():
    from ml_engine import compute_contrast_analysis
    contrast = compute_contrast_analysis({"logp": 1.0, "tpsa": 30}, [])
    assert contrast["matrix_count"] == 0


def test_nades_literature_db():
    from ml_engine import NADES_LITERATURE_DB
    assert "Quercetin" in NADES_LITERATURE_DB
    assert "Berberine" in NADES_LITERATURE_DB
    entry = NADES_LITERATURE_DB["Quercetin"]
    assert "hba" in entry
    assert "hbd" in entry
    assert "ratio" in entry
    assert "citation" in entry