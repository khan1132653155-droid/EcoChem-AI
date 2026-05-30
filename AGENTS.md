# EcoChem-AI — Agent Instructions

## Quick Start
```sh
pip install -r requirements.txt
streamlit run app.py              # launch GUI
python -c "from ml_engine import demo; demo()"  # smoke test ML pipeline
python -m pytest tests/ -v        # run unit tests
```

## Entrypoints
- **`app.py`** — Streamlit GUI. Launch with `streamlit run app.py`. Dark theme configured in `.streamlit/config.toml`.
- **`ml_engine.py`** — Multi-output Random Forest (scikit-learn `MultiOutputRegressor`). No GPU needed. Targets: `[Eutectic_Temp_C, Phase_Stability, EcoToxicity_Index]`.
- **`utils.py`** — PubChem name→SMILES resolver (cached to `pubchem_cache.json`), RDKit 2048-bit Morgan fingerprinting (radius=2), molecular diagram rendering.

## Architecture
- Input: 2048-bit Morgan fingerprint from combined HBA+HBD molecules (ratio-weighted)
- Model: `RandomForestRegressor(n_estimators=100, max_depth=20)` wrapped in `MultiOutputRegressor`
- Targets predicted: Eutectic Temperature (°C), Phase Stability (binary), EcoToxicity Index [0–1]
- Liquid-at-room-temperature flag: true when Eutectic Temp < 25°C

## Data Pipeline
- PubChem rate limit: `time.sleep(0.5)` between requests. Cache file: `pubchem_cache.json` (name→SMILES dict).
- Fingerprint: `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)` → `np.float32` array
- Combine: `fp_hba + ratio * fp_hbd`
- Built-in common NADES lookup (`COMMON_NADES` dict in `utils.py`) with verified SMILES for 20 chemicals

## ML Engine
- Synthetic data: `generate_synthetic_dataset(n=500)` creates random fingerprints + linear-heuristic targets with noise
- Auto-initialize: `initialize()` trains model on 800 synthetic samples if no saved model exists
- Persistence: `models/ecochem_model.pkl` + `models/scaler.pkl` (joblib)
- Prediction returns dict with `Eutectic_Temp_C`, `Phase_Stability`, `EcoToxicity_Index`, `Liquid_At_Room_Temp`

## Streamlit GUI
- Sidebar: HBA/HBD dropdowns (from `COMMON_NADES`), molar ratio slider, Predict button, Clear History
- 3 tabs: **Solvent Screening** (results table + bar chart + radar), **Structural Diagrams** (RDKit 2D renders), **Methodology** (expandable docs)
- Session state: `st.session_state.history` tracks prediction history

## Testing
- Tests in `tests/` directory. Run: `python -m pytest tests/ -v`
- Test framework: `pytest` (install via requirements.txt)
- `test_utils.py`: tests fingerprint generation, SMILES lookup fallback, fingerprint combination
- `test_ml_engine.py`: tests synthetic data shapes, model train/save/load, prediction format

## Dependencies (no PyTorch)
streamlit, rdkit, pubchempy, scikit-learn, pandas, matplotlib, numpy, joblib

## Conventions
- Fingerprints: `np.ndarray` of `float32`, shape `(2048,)`
- All chemical IDs stored as `int` (PubChem CIDs)
- Predictions stored as dict with string keys
- Cache is JSON — human-readable for debugging