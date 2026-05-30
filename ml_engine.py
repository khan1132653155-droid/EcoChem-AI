import os
import pickle
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from nades_db import get_all_valid_records
from utils import (
    CHIRAL_HBD,
    COMMON_NADES,
    COMPOUND_CLASS_PREFERENCES,
    ALL_HBA,
    ALL_HBD,
    combine_fingerprints,
    compute_rdkit_descriptors,
    name_to_fingerprint,
    name_to_smiles,
    smiles_to_fingerprint,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ecochem_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

N_FEATURES = 2048
N_TARGETS = 3
TARGET_NAMES = ["Eutectic_Temp_C", "Phase_Stability", "EcoToxicity_Index"]

NADES_LITERATURE_DB = {
    "Quercetin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 2.0, "citation": "Dai et al., Anal. Chim. Acta 766, 61\u201368 (2013)"},
    "Berberine": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 1.0, "citation": "Roch\u00edn-Wong et al., Food Chem. 239, 578\u2013586 (2018)"},
    "Curcumin": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 1.0, "citation": "Jelinski et al., New J. Chem. 43, 11740\u201311748 (2019)"},
    "Rutin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 2.0, "citation": "Huang et al., J. Mol. Liq. 241, 405\u2013411 (2017)"},
    "Gallic Acid": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 1.0, "citation": "Bosiljkov et al., Food Chem. 234, 144\u2013153 (2017)"},
    "Caffeine": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 1.0, "citation": "Santos et al., ACS Sustainable Chem. Eng. 6, 7500\u20137508 (2018)"},
    "Resveratrol": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 2.0, "citation": "Ruesgas-Ram\u00f3n et al., Food Chem. 312, 126086 (2020)"},
    "Rutin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 2.0, "citation": "Huang et al., J. Mol. Liq. 241, 405\u2013411 (2017)"},
    "Luteolin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 2.0, "citation": "Peng et al., Sep. Purif. Technol. 250, 117162 (2020)"},
    "Naringenin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 2.0, "citation": "Peng et al., Sep. Purif. Technol. 250, 117162 (2020)"},
    "Kaempferol": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 1.0, "citation": "Khezeli et al., Talanta 191, 454\u2013463 (2019)"},
    "Apigenin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 2.0, "citation": "Peng et al., Sep. Purif. Technol. 250, 117162 (2020)"},
    "Betanin": {"hba": "Choline Chloride", "hbd": "Citric Acid", "ratio": 1.0, "citation": "Dai et al., Food Chem. 187, 14\u201319 (2015)"},
    "Glycyrrhizic Acid": {"hba": "Choline Chloride", "hbd": "Urea", "ratio": 2.0, "citation": "Huang et al., J. Mol. Liq. 241, 405\u2013411 (2017)"},
    "Chlorogenic Acid": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 1.0, "citation": "Ruesgas-Ram\u00f3n et al., Food Chem. 312, 126086 (2020)"},
    "Caffeic Acid": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 1.0, "citation": "Ruesgas-Ram\u00f3n et al., Food Chem. 312, 126086 (2020)"},
    "Ferulic Acid": {"hba": "Choline Chloride", "hbd": "Glycerol", "ratio": 1.0, "citation": "Ruesgas-Ram\u00f3n et al., Food Chem. 312, 126086 (2020)"},
}


def _generate_synthetic_fingerprint() -> np.ndarray:
    return np.random.randint(0, 2, N_FEATURES).astype(np.float32)


def _synthetic_targets(fp: np.ndarray) -> np.ndarray:
    ones_count = np.sum(fp)
    eutectic_temp = 120.0 - (ones_count / N_FEATURES) * 180.0 + np.random.normal(0, 5.0)
    eutectic_temp = np.clip(eutectic_temp, -50.0, 120.0)
    phase_stab = 1.0 if ones_count > N_FEATURES * 0.4 else 0.0
    if np.random.random() < 0.05:
        phase_stab = 1.0 - phase_stab
    eco_tox = (ones_count / N_FEATURES) * 0.6 + np.random.uniform(-0.05, 0.05)
    eco_tox = np.clip(eco_tox, 0.0, 1.0)
    return np.array([eutectic_temp, phase_stab, eco_tox], dtype=np.float32)


def generate_synthetic_dataset(n_samples: int = 500) -> tuple:
    X = np.zeros((n_samples, N_FEATURES), dtype=np.float32)
    y = np.zeros((n_samples, N_TARGETS), dtype=np.float32)
    for i in range(n_samples):
        fp = _generate_synthetic_fingerprint()
        X[i] = fp
        y[i] = _synthetic_targets(fp)
    return X, y


def build_model(n_estimators: int = 100, max_depth: int = 20, n_jobs: int = -1):
    base = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=42,
        min_samples_leaf=2,
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    return model


def train(X: np.ndarray, y: np.ndarray) -> tuple:
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)
    model = build_model()
    model.fit(X, y_scaled)
    return model, scaler


def save_model(model, scaler) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


def load_model() -> tuple:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def predict(fp: np.ndarray) -> dict:
    model, scaler = load_model()
    fp_2d = fp.reshape(1, -1)
    y_scaled = model.predict(fp_2d)
    y = scaler.inverse_transform(y_scaled)[0]
    eutectic_temp = round(float(y[0]), 1)
    phase_stab = bool(round(y[1]))
    eco_tox = round(float(np.clip(y[2], 0.0, 1.0)), 3)
    liquid_at_rt = eutectic_temp < 25.0
    return {
        "Eutectic_Temp_C": eutectic_temp,
        "Phase_Stability": phase_stab,
        "EcoToxicity_Index": eco_tox,
        "Liquid_At_Room_Temp": liquid_at_rt,
    }


def initialize() -> None:
    if not model_exists():
        print("Generating synthetic training data...")
        X, y = generate_synthetic_dataset(n_samples=800)
        print(f"Training on {X.shape[0]} samples...")
        model, scaler = train(X, y)
        save_model(model, scaler)
        print("Model trained and saved.")


def recommend_nades(
    compound_class: str,
    functional_groups: dict = None,
    is_chiral: bool = False,
    chiral_hbd_priority: bool = True,
    preferences: dict = None,
    top_k: int = 5,
) -> list[dict]:
    if preferences is None:
        preferences = {"liquid_rt": 0.5, "low_eco_tox": 0.5, "stability": 0.5}
    class_prefs = COMPOUND_CLASS_PREFERENCES.get(compound_class, {})
    preferred_hba = class_prefs.get("preferred_hba", {})
    preferred_hbd = class_prefs.get("preferred_hbd", {})
    ratio_range = class_prefs.get("preferred_ratio_range", (1.0, 2.0))
    hba_candidates = ALL_HBA
    hbd_candidates = ALL_HBD
    candidates = []
    for hba in hba_candidates:
        hba_score = preferred_hba.get(hba, 0.1)
        if hba_score <= 0:
            continue
        for hbd in hbd_candidates:
            if hbd == hba:
                continue
            hbd_score = preferred_hbd.get(hbd, 0.1)
            if hbd_score <= 0:
                continue
            chiral_bonus = 0.0
            if is_chiral and chiral_hbd_priority:
                if hbd in CHIRAL_HBD:
                    chiral_bonus = 0.3
            base_score = (hba_score + hbd_score) / 2.0 + chiral_bonus
            candidates.append({
                "hba": hba, "hbd": hbd, "base_score": base_score,
                "hba_score": hba_score, "hbd_score": hbd_score,
                "chiral_bonus": chiral_bonus, "is_chiral_hbd": hbd in CHIRAL_HBD,
            })
    candidates.sort(key=lambda c: c["base_score"], reverse=True)
    top_candidates = candidates[:20]
    ratio_options = np.arange(ratio_range[0], ratio_range[1] + 0.5, 0.5).tolist()
    ratio_options = [r for r in ratio_options if r >= 0.5]
    recommendations = []
    for cand in top_candidates:
        for ratio in ratio_options:
            fp_hba, _ = name_to_fingerprint(cand["hba"])
            fp_hbd, _ = name_to_fingerprint(cand["hbd"])
            if fp_hba is None or fp_hbd is None:
                continue
            combined_fp = combine_fingerprints(fp_hba, fp_hbd, ratio=ratio)
            props = predict(combined_fp)
            liquid_score = 1.0 if props["Liquid_At_Room_Temp"] else 0.0
            eco_score = 1.0 - props["EcoToxicity_Index"]
            stability_score = 1.0 if props["Phase_Stability"] else 0.0
            pref_score = (
                preferences.get("liquid_rt", 0.5) * liquid_score
                + preferences.get("low_eco_tox", 0.5) * eco_score
                + preferences.get("stability", 0.5) * stability_score
            )
            composite_score = 0.4 * cand["base_score"] + 0.6 * pref_score
            recommendations.append({
                "HBA": cand["hba"], "HBD": cand["hbd"],
                "Ratio": f"{ratio:.1f} : 1.0" if ratio >= 1.0 else f"1.0 : {1.0/ratio:.1f}",
                "Molar_Ratio": ratio, "Composite_Score": round(composite_score, 3),
                "Eutectic_Temp_C": props["Eutectic_Temp_C"],
                "Phase_Stability": props["Phase_Stability"],
                "EcoToxicity_Index": props["EcoToxicity_Index"],
                "Liquid_At_Room_Temp": props["Liquid_At_Room_Temp"],
                "Chiral_HBD": cand["is_chiral_hbd"],
            })
    recommendations.sort(key=lambda r: r["Composite_Score"], reverse=True)
    seen = set()
    unique = []
    for r in recommendations:
        key = (r["HBA"], r["HBD"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:top_k]


def _predict_eco_toxicity(hba_name: str, hbd_name: str, ratio: float) -> dict:
    fp_hba, _ = name_to_fingerprint(hba_name)
    fp_hbd, _ = name_to_fingerprint(hbd_name)
    if fp_hba is None or fp_hbd is None:
        return {"EcoToxicity_Index": 0.5, "Eutectic_Temp_C": 25.0,
                "Phase_Stability": True, "Liquid_At_Room_Temp": True}
    combined = combine_fingerprints(fp_hba, fp_hbd, ratio=ratio)
    return predict(combined)


def _score_nades_descriptor_match(target_desc: dict, nades_desc: dict) -> float:
    hbd_diff = abs(target_desc["hbd"] - nades_desc["hbd"]) / max(target_desc["hbd"], 1)
    hba_diff = abs(target_desc["hba"] - nades_desc["hba"]) / max(target_desc["hba"], 1)
    tpsa_diff = abs(target_desc["tpsa"] - nades_desc["tpsa"]) / max(target_desc["tpsa"], 1)
    similarity = 1.0 - min(1.0, (hbd_diff * 0.4 + hba_diff * 0.3 + tpsa_diff * 0.3))
    return similarity


def compute_contrast_analysis(target_desc: dict, bulk_matrix_names: list[str]) -> dict:
    if not bulk_matrix_names or not target_desc:
        return {"target_logp": target_desc.get("logp", 0) if target_desc else 0,
                "target_tpsa": target_desc.get("tpsa", 0) if target_desc else 0,
                "matrix_count": 0, "avg_matrix_logp": 0, "avg_matrix_tpsa": 0,
                "logp_delta": 0, "tpsa_delta": 0, "selectivity_estimate": "unknown"}
    matrix_descs = []
    for name in bulk_matrix_names:
        smi = name_to_smiles(name.strip())
        if smi:
            d = compute_rdkit_descriptors(smi)
            if d:
                matrix_descs.append(d)
    if not matrix_descs:
        return {"target_logp": target_desc["logp"], "target_tpsa": target_desc["tpsa"],
                "matrix_count": len(bulk_matrix_names), "error": "could not resolve matrix SMILES"}
    avg_logp = float(np.mean([d["logp"] for d in matrix_descs]))
    avg_tpsa = float(np.mean([d["tpsa"] for d in matrix_descs]))
    logp_delta = abs(target_desc["logp"] - avg_logp)
    tpsa_delta = abs(target_desc["tpsa"] - avg_tpsa)
    if logp_delta > 2.0:
        selectivity = "high (large LogP difference)"
    elif logp_delta > 1.0:
        selectivity = "moderate"
    elif logp_delta > 0.5:
        selectivity = "low \u2014 NADES should exploit polarity difference"
    else:
        selectivity = "very low \u2014 rely on functional-group-specific NADES interactions"
    return {
        "target_logp": target_desc["logp"], "target_tpsa": target_desc["tpsa"],
        "matrix_count": len(bulk_matrix_names),
        "matrix_compounds": [{"name": n, "logp": d["logp"], "tpsa": d["tpsa"]}
                            for n, d in zip(bulk_matrix_names, matrix_descs)],
        "avg_matrix_logp": round(avg_logp, 2), "avg_matrix_tpsa": round(avg_tpsa, 1),
        "logp_delta": round(logp_delta, 2), "tpsa_delta": round(tpsa_delta, 1),
        "selectivity_estimate": selectivity,
    }


def _match_nades_from_descriptors(target_desc: dict, matrix_descs: list[dict],
                                   is_chiral: bool = False) -> tuple[str, str, float]:
    hba_candidates = ALL_HBA
    hbd_candidates = ALL_HBD
    hba_scores = []
    for hba_name in hba_candidates:
        smi = COMMON_NADES.get(hba_name) or CHIRAL_HBD.get(hba_name)
        if not smi:
            continue
        nd = compute_rdkit_descriptors(smi)
        if nd is None:
            continue
        score = _score_nades_descriptor_match(target_desc, nd)
        hba_scores.append((hba_name, score, nd))
    hbd_scores = []
    for hbd_name in hbd_candidates:
        smi = COMMON_NADES.get(hbd_name) or CHIRAL_HBD.get(hbd_name)
        if not smi:
            continue
        nd = compute_rdkit_descriptors(smi)
        if nd is None:
            continue
        chiral_bonus = 0.0
        if is_chiral and hbd_name in CHIRAL_HBD:
            chiral_bonus = 0.15
        score = _score_nades_descriptor_match(target_desc, nd) + chiral_bonus
        hbd_scores.append((hbd_name, score, nd))

    hba_scores.sort(key=lambda x: x[1], reverse=True)
    hbd_scores.sort(key=lambda x: x[1], reverse=True)

    if matrix_descs:
        avg_matrix_logp = float(np.mean([d["logp"] for d in matrix_descs]))
        hba_scores = [(n, s - 0.3 * (1 - abs(nd["logp"] - avg_matrix_logp) / 10), nd)
                      for n, s, nd in hba_scores]
        hba_scores.sort(key=lambda x: x[1], reverse=True)

    best_hba = hba_scores[0][0] if hba_scores else "Choline Chloride"
    best_hbd = hbd_scores[0][0] if hbd_scores else "Glycerol"
    target_hba_count = max(target_desc["hba"], 1)
    target_hbd_count = max(target_desc["hbd"], 1)
    ratio = np.clip(target_hba_count / target_hbd_count, 0.5, 5.0)
    ratio = round(ratio, 1)
    return best_hba, best_hbd, ratio


def recommend_nades_with_path(
    target_name: str = None,
    target_smiles: str = None,
    compound_class: str = None,
    bulk_matrix_names: list[str] = None,
    functional_groups: dict = None,
    is_chiral: bool = False,
    preferences: dict = None,
    mode: str = "targeted",
) -> dict:
    if preferences is None:
        preferences = {"liquid_rt": 0.5, "low_eco_tox": 0.5, "stability": 0.5}
    result = {"path": None, "path_label": None, "recommendations": [],
              "eco_toxicity": None, "contrast_analysis": None,
              "target_descriptors": None, "mode": mode}

    target_desc = None
    if target_smiles:
        target_desc = compute_rdkit_descriptors(target_smiles)
        result["target_descriptors"] = target_desc

    if mode == "targeted" and target_name:
        if target_name in NADES_LITERATURE_DB:
            entry = NADES_LITERATURE_DB[target_name]
            props = _predict_eco_toxicity(entry["hba"], entry["hbd"], entry["ratio"])
            result["path"] = "fast"
            result["path_label"] = f"\U0001f4da Literature-backed ({entry['citation']})"
            result["recommendations"] = [{
                "HBA": entry["hba"], "HBD": entry["hbd"],
                "Ratio": f"{entry['ratio']:.1f} : 1.0",
                "Molar_Ratio": entry["ratio"],
                "Eutectic_Temp_C": props["Eutectic_Temp_C"],
                "Phase_Stability": props["Phase_Stability"],
                "EcoToxicity_Index": props["EcoToxicity_Index"],
                "Liquid_At_Room_Temp": props["Liquid_At_Room_Temp"],
                "Chiral_HBD": entry["hbd"] in CHIRAL_HBD,
                "Source": entry["citation"],
            }]
            result["eco_toxicity"] = props["EcoToxicity_Index"]
            if bulk_matrix_names:
                result["contrast_analysis"] = compute_contrast_analysis(target_desc, bulk_matrix_names)
            return result

        if compound_class and target_name:
            cls_match = [v for k, v in NADES_LITERATURE_DB.items()
                        if k == compound_class or k.lower() == compound_class.lower()]
            if cls_match:
                entry = cls_match[0]
                props = _predict_eco_toxicity(entry["hba"], entry["hbd"], entry["ratio"])
                result["path"] = "fast"
                result["path_label"] = f"\U0001f4da Class-based literature reference"
                result["recommendations"] = [{
                    "HBA": entry["hba"], "HBD": entry["hbd"],
                    "Ratio": f"{entry['ratio']:.1f} : 1.0",
                    "Molar_Ratio": entry["ratio"],
                    "Eutectic_Temp_C": props["Eutectic_Temp_C"],
                    "Phase_Stability": props["Phase_Stability"],
                    "EcoToxicity_Index": props["EcoToxicity_Index"],
                    "Liquid_At_Room_Temp": props["Liquid_At_Room_Temp"],
                    "Chiral_HBD": entry["hbd"] in CHIRAL_HBD,
                    "Source": "Compound class match in literature DB",
                }]
                result["eco_toxicity"] = props["EcoToxicity_Index"]
                return result

    matrix_descs = []
    if bulk_matrix_names:
        for name in bulk_matrix_names:
            smi = name_to_smiles(name.strip())
            if smi:
                d = compute_rdkit_descriptors(smi)
                if d:
                    matrix_descs.append(d)

    if target_desc is None:
        target_desc = {"hbd": 3, "hba": 3, "tpsa": 60, "logp": 1.5,
                       "mol_wt": 300, "rotatable_bonds": 3, "hbd_hba_ratio": 1.0}

    hba_name, hbd_name, ratio = _match_nades_from_descriptors(target_desc, matrix_descs, is_chiral)
    props = _predict_eco_toxicity(hba_name, hbd_name, ratio)
    result["path"] = "slow"
    result["path_label"] = "\U0001f9ec Dynamic prediction (RDKit descriptor-matched)"
    result["recommendations"] = [{
        "HBA": hba_name, "HBD": hbd_name,
        "Ratio": f"{ratio:.1f} : 1.0" if ratio >= 1.0 else f"1.0 : {1.0/ratio:.1f}",
        "Molar_Ratio": ratio,
        "Eutectic_Temp_C": props["Eutectic_Temp_C"],
        "Phase_Stability": props["Phase_Stability"],
        "EcoToxicity_Index": props["EcoToxicity_Index"],
        "Liquid_At_Room_Temp": props["Liquid_At_Room_Temp"],
        "Chiral_HBD": hbd_name in CHIRAL_HBD,
        "Source": "RDKit descriptor-matched NADES",
    }]
    result["eco_toxicity"] = props["EcoToxicity_Index"]

    if mode == "total" and target_desc:
        result["contrast_analysis"] = compute_contrast_analysis(target_desc, bulk_matrix_names)
    elif mode == "targeted" and bulk_matrix_names and target_desc:
        result["contrast_analysis"] = compute_contrast_analysis(target_desc, bulk_matrix_names)

    return result


def retrain_from_lab_data() -> dict:
    records = get_all_valid_records()
    if len(records) < 3:
        return {"success": False, "message": f"Need at least 3 valid records, found {len(records)}"}
    X_list = []
    y_list = []
    skipped = 0
    for rec in records:
        fp_hba, _ = name_to_fingerprint(rec["hba_name"])
        fp_hbd, _ = name_to_fingerprint(rec["hbd_name"])
        if fp_hba is None or fp_hbd is None:
            skipped += 1
            continue
        ratio = rec["molar_ratio"]
        combined = combine_fingerprints(fp_hba, fp_hbd, ratio=ratio)
        X_list.append(combined)
        temp = rec.get("temperature_c", 25.0)
        phase_stab = 1.0
        eco_tox = max(0.0, min(1.0, 0.5 - (rec.get("yield_percent", 50) - 50) / 200))
        y_list.append([temp, phase_stab, eco_tox])
    if len(X_list) < 3:
        return {"success": False, "message": f"Only {len(X_list)} usable records after fingerprint resolution"}
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    model, scaler = train(X, y)
    save_model(model, scaler)
    return {
        "success": True,
        "message": f"Retrained on {len(X)} records ({skipped} skipped due to unresolved SMILES)",
        "samples": len(X),
    }


def demo() -> None:
    initialize()
    fp = _generate_synthetic_fingerprint()
    result = predict(fp)
    print("Demo prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
