import json
import os
import time
from typing import Optional, Tuple

import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

CACHE_PATH = os.path.join(os.path.dirname(__file__), "pubchem_cache.json")


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def name_to_smiles(chemical_name: str) -> Optional[str]:
    cache = _load_cache()
    key = chemical_name.strip().lower()
    if key in cache:
        return cache[key]
    try:
        compounds = pcp.get_compounds(key, "name")
        if compounds:
            smiles = compounds[0].canonical_smiles
            cache[key] = smiles
            _save_cache(cache)
            time.sleep(0.5)
            return smiles
    except Exception:
        pass
    return None


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs = __import__("rdkit.DataStructs", fromlist=["DataStructs"])
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def name_to_fingerprint(chemical_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    smiles = name_to_smiles(chemical_name)
    if smiles is None:
        return None, None
    fp = smiles_to_fingerprint(smiles)
    return fp, smiles


def generate_molecular_diagram(smiles: str, width: int = 300, height: int = 250):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(width, height))
    return img


def combine_fingerprints(fp1: np.ndarray, fp2: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    return fp1 + ratio * fp2


CHIRAL_HBD = {
    "L-Malic Acid": "C([C@@H](C(=O)O)O)C(=O)O",
    "D-Malic Acid": "C([C@H](C(=O)O)O)C(=O)O",
    "L-Tartaric Acid": "[C@@H]([C@H](C(=O)O)O)(C(=O)O)O",
    "D-Tartaric Acid": "[C@H]([C@@H](C(=O)O)O)(C(=O)O)O",
    "L-Lactic Acid": "C[C@@H](C(=O)O)O",
    "D-Lactic Acid": "C[C@H](C(=O)O)O",
    "L-Citronellic Acid": "C[C@@H](CCC=C(C)C)C(=O)O",
}

FG_PATTERNS = {
    "alcohol_oh": "[OX2H]",
    "phenolic_oh": "c[OH]",
    "carboxylic_acid": "C(=O)[OH]",
    "amine_primary": "[NH2]",
    "amine_secondary": "[NH]",
    "amine_tertiary": "[N](C)(C)C",
    "aromatic_ring": "c1ccccc1",
    "ester": "C(=O)O",
    "ether": "COC",
    "ketone": "C(=O)C",
    "aldehyde": "[CH]=O",
    "alkene": "C=C",
    "alkyne": "C#C",
    "nitro": "[NX3](=O)=O",
    "sulfide": "CS",
    "lactone": "O=C1COC1",
    "furan_ring": "o1cccc1",
    "pyran_ring": "O1CCCCC1",
}


def detect_functional_groups(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    result = {}
    for name, smarts in FG_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            result[name] = len(matches)
    return result


def detect_chirality(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"is_chiral": False, "centers": [], "natural_enantiomer": "racemic"}
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    chiral_centers = []
    for atom_idx, chirality_label in centers:
        chiral_centers.append({"atom_index": int(atom_idx), "chirality": chirality_label})
    is_chiral = len(chiral_centers) > 0
    return {
        "is_chiral": is_chiral,
        "centers": chiral_centers,
        "natural_enantiomer": "racemic",
    }


def is_chiral_hbd(name: str) -> bool:
    return name in CHIRAL_HBD


def compute_rdkit_descriptors(smiles: str) -> Optional[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem import rdMolDescriptors
    return {
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "mol_wt": Descriptors.MolWt(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "hbd_hba_ratio": max(Descriptors.NumHDonors(mol), 1) / max(Descriptors.NumHAcceptors(mol), 1),
    }


PRECIPITATION_REAGENTS = {
    "alcohol_oh": [
        {"reagent": "Acetic Anhydride", "method": "Acetylation",
         "detail": "Acetylate hydroxyl groups with Ac\u2082O in anhydrous pyridine (1:2 molar ratio, 60\u00b0C, 2 h); phase separation upon water addition (3 volumes) — acetylated derivative partitions into organic layer"},
    ],
    "phenolic_oh": [
        {"reagent": "Lead Acetate", "method": "Lead precipitation",
         "detail": "Add 10% (w/v) lead acetate solution dropwise with stirring; phenolic compounds form insoluble lead complexes; centrifuge at 5000\u00d7g for 10 min"},
        {"reagent": "Gelatin", "method": "Gelatin precipitation",
         "detail": "Add 1% (w/v) gelatin solution in 10% NaCl; tannins and polyphenols form flocculent precipitate; stand 30 min at 4\u00b0C"},
    ],
    "amine_primary": [
        {"reagent": "Dragendorff's Reagent", "method": "Alkaloid precipitation",
         "detail": "Add Dragendorff's reagent (KBil\u2084) dropwise; alkaloids form orange-red precipitate; collect by centrifugation"},
        {"reagent": "pH Shift (NH\u2084OH)", "method": "Alkaline precipitation",
         "detail": "Adjust pH to 8\u20139 with 25% NH\u2084OH; free alkaloid bases precipitate at reduced solubility; extract with CHCl\u2083 (3\u00d720 mL)"},
    ],
    "amine_secondary": [
        {"reagent": "Dragendorff's Reagent", "method": "Alkaloid precipitation",
         "detail": "Same as primary amines; Dragendorff's is non-selective for amine substitution"},
        {"reagent": "Mayer's Reagent", "method": "Mercuric precipitation",
         "detail": "Add Mayer's reagent (K\u2082HgI\u2084); alkaloids form cream-colored precipitate; sensitive to 1:20000 dilution"},
    ],
    "amine_tertiary": [
        {"reagent": "Silicotungstic Acid", "method": "Heteropoly acid precipitation",
         "detail": "Add 5% (w/v) silicotungstic acid in 0.1N HCl; tertiary alkaloids form white flocculent precipitate"},
        {"reagent": "Dragendorff's Reagent", "method": "Alkaloid precipitation",
         "detail": "Standard Dragendorff test; orange-red precipitate"},
    ],
    "carboxylic_acid": [
        {"reagent": "Calcium Chloride", "method": "Calcium salt precipitation",
         "detail": "Add 10% CaCl\u2082 at pH 7\u20138 (adjusted with dilute NaOH); carboxylic acids form insoluble Ca salts; digest at 60\u00b0C for 30 min"},
        {"reagent": "Lead Acetate", "method": "Lead salt precipitation",
         "detail": "Add 10% lead acetate; acidic compounds form insoluble lead salts; adjust pH to 5\u20136 for optimal precipitation"},
    ],
    "ester": [
        {"reagent": "NaOH/Ethanol", "method": "Saponification",
         "detail": "Hydrolyse with 0.5N NaOH in 70% ethanol at 60\u00b0C for 30 min; acidify to pH 2\u20133 with HCl to precipitate the carboxylic acid product"},
    ],
    "amine_any": [
        {"reagent": "Picric Acid", "method": "Picrate formation",
         "detail": "Add saturated picric acid solution in ethanol; amine picrates form yellow crystals; recrystallise from ethanol for purification"},
    ],
}

CHROMATOGRAPHY_FG_MAP = {
    "alcohol_oh": {
        "stationary_phase": "Silica Gel 60 (230\u2013400 mesh)",
        "mobile_phase": "CHCl\u2083:MeOH gradient (95:5 \u2192 80:20)",
        "detection": "UV 254 nm or Liebermann\u2013Burchard reagent",
        "rf_range": "0.2\u20130.5",
    },
    "phenolic_oh": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:H\u2082O (70:30) + 0.1% Formic Acid",
        "detection": "UV 280 nm (DAD)",
        "rf_range": "0.3\u20130.6",
    },
    "carboxylic_acid": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:H\u2082O (50:50) + 0.1% H\u2083PO\u2084 (pH 2.5)",
        "detection": "UV 210 nm or RI",
        "rf_range": "0.2\u20130.5",
    },
    "amine_primary": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:H\u2082O (60:40) + 0.1% Triethylamine",
        "detection": "UV 254 nm",
        "note": "Triethylamine (0.1%) suppresses silanol\u2013amine tailing",
    },
    "amine_secondary": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "ACN:H\u2082O (40:60) + 0.1% TEA, pH 8.0",
        "detection": "UV 254 nm",
        "note": "Basic pH ensures amines are unionised for better retention",
    },
    "amine_tertiary": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "ACN:H\u2082O (30:70) + 0.1% TFA, pH 3.0",
        "detection": "UV 254 nm (DAD)",
        "note": "Low pH ion-suppresses tertiary amines; add TEA if tailing persists",
    },
    "aromatic_ring": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "ACN:H\u2082O gradient (30:70 \u2192 70:30 over 30 min)",
        "detection": "UV 254 nm (DAD)",
        "rf_range": "0.3\u20130.7",
    },
    "alkene": {
        "stationary_phase": "Silica Gel 60 (AgNO\u2083-impregnated, 10% w/w)",
        "mobile_phase": "Hexane:EtOAc gradient (95:5 \u2192 80:20)",
        "detection": "UV 220 nm",
        "note": "AgNO\u2083 selectively complexes \u03c0-bonds for alkene isomer separation",
    },
    "ketone": {
        "stationary_phase": "Silica Gel 60 (230\u2013400 mesh)",
        "mobile_phase": "Hexane:EtOAc gradient (90:10 \u2192 70:30)",
        "detection": "UV 280 nm or 2,4-DNP spray",
    },
    "ester": {
        "stationary_phase": "Silica Gel 60 (230\u2013400 mesh)",
        "mobile_phase": "Hexane:EtOAc gradient (95:5 \u2192 80:20)",
        "detection": "UV 220 nm or hydroxylamine\u2013FeCl\u2083 spray",
    },
}

COMPOUND_CLASS_CHROMATOGRAPHY = {
    "Alkaloids": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "ACN:H\u2082O (40:60) + 0.1% Triethylamine, pH 8.0",
        "detection": "UV 254 nm (DAD)",
        "note": "Basic mobile phase suppresses silanol\u2013alkaloid cation exchange",
    },
    "Terpenoids": {
        "stationary_phase": "Silica Gel 60 (230\u2013400 mesh, 250\u00d710 mm semi-prep)",
        "mobile_phase": "Hexane:EtOAc gradient (95:5 \u2192 70:30 in 40 min)",
        "detection": "UV 210 nm or Liebermann\u2013Burchard spray",
        "note": "Gradient elution separates mono-, sesqui-, and di-terpenoids by polarity",
    },
    "Flavonoids": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:H\u2082O (65:35) + 0.1% Formic Acid",
        "detection": "UV 254/280 nm (DAD)",
        "note": "Formic acid sharpens peak shape for flavonoid glycosides",
    },
    "Phenolics": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:H\u2082O (70:30) + 0.1% Formic Acid",
        "detection": "UV 280 nm (DAD)",
        "note": "Phenolic acids and their esters resolved in 25 min isocratic run",
    },
    "Anthocyanins": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "H\u2082O:HCOOH:ACN (87:10:3, pH 1.8) isocratic",
        "detection": "VIS 520 nm (DAD)",
        "note": "Low pH (pH < 2) ensures anthocyanins are in flavylium cation form for sharp peaks",
    },
    "Carotenoids": {
        "stationary_phase": "C30 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:MTBE:H\u2082O (81:15:4) isocratic",
        "detection": "VIS 450 nm (DAD)",
        "note": "C30 phase provides shape selectivity for carotenoid isomers",
    },
    "Saponins": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "ACN:H\u2082O (35:65) + 0.05% TFA",
        "detection": "ELSD or UV 203 nm",
        "note": "ELSD preferred as saponins lack strong chromophores",
    },
    "Coumarins": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "MeOH:H\u2082O (50:50) isocratic",
        "detection": "UV 320 nm (DAD) \u2014 intense fluorescence under UV 366 nm",
    },
    "Lignans": {
        "stationary_phase": "C18 Reverse Phase (250\u00d74.6 mm, 5 \u00b5m)",
        "mobile_phase": "ACN:H\u2082O (45:55) + 0.1% Formic Acid",
        "detection": "UV 280 nm (DAD)",
    },
    "Quinones": {
        "stationary_phase": "Silica Gel 60 (230\u2013400 mesh)",
        "mobile_phase": "Hexane:EtOAc (80:20) isocratic",
        "detection": "VIS 430 nm \u2014 quinones are naturally coloured",
    },
}


def suggest_precipitation_reagents(functional_groups: dict) -> list[dict]:
    steps = []
    used_reagents = set()
    if functional_groups.get("amine_primary") or functional_groups.get("amine_secondary") or functional_groups.get("amine_tertiary"):
        fg_name = next((g for g in ["amine_primary", "amine_secondary", "amine_tertiary"] if functional_groups.get(g, 0) > 0), None)
        if fg_name and fg_name in PRECIPITATION_REAGENTS:
            for rec in PRECIPITATION_REAGENTS[fg_name]:
                if rec["reagent"] not in used_reagents:
                    steps.append({
                        "step_type": "precipitation",
                        "reagent": rec["reagent"],
                        "method": rec["method"],
                        "detail": rec["detail"],
                    })
                    used_reagents.add(rec["reagent"])
    for fg_name in ["phenolic_oh", "carboxylic_acid", "alcohol_oh", "ester"]:
        if functional_groups.get(fg_name, 0) > 0 and fg_name in PRECIPITATION_REAGENTS:
            for rec in PRECIPITATION_REAGENTS[fg_name]:
                if rec["reagent"] not in used_reagents:
                    steps.append({
                        "step_type": "precipitation",
                        "reagent": rec["reagent"],
                        "method": rec["method"],
                        "detail": rec["detail"],
                    })
                    used_reagents.add(rec["reagent"])
    return steps


def suggest_chromatography(functional_groups: dict, compound_class: str = None) -> list[dict]:
    results = []
    if compound_class and compound_class in COMPOUND_CLASS_CHROMATOGRAPHY:
        rec = COMPOUND_CLASS_CHROMATOGRAPHY[compound_class]
        results.append({
            "step_type": "chromatography",
            "source": f"{compound_class} class recommendation",
            "stationary_phase": rec["stationary_phase"],
            "mobile_phase": rec["mobile_phase"],
            "detection": rec["detection"],
            "note": rec.get("note", ""),
            "column_type": rec.get("column_type", ""),
        })
    for fg_name in ["phenolic_oh", "carboxylic_acid", "amine_primary", "amine_secondary",
                     "amine_tertiary", "aromatic_ring", "alkene", "ketone", "ester", "alcohol_oh"]:
        if functional_groups.get(fg_name, 0) > 0 and fg_name in CHROMATOGRAPHY_FG_MAP:
            rec = CHROMATOGRAPHY_FG_MAP[fg_name]
            results.append({
                "step_type": "chromatography",
                "source": f"Functional group: {fg_name}",
                "stationary_phase": rec["stationary_phase"],
                "mobile_phase": rec["mobile_phase"],
                "detection": rec["detection"],
                "note": rec.get("note", ""),
                "rf_range": rec.get("rf_range", ""),
            })
    return results


def suggest_isolation_strategy(functional_groups: dict, chirality_info: dict = None, compound_class: str = None) -> list[dict]:
    steps = []
    has_basic_amine = any(
        functional_groups.get(g, 0) > 0
        for g in ["amine_primary", "amine_secondary", "amine_tertiary"]
    )
    has_acidic = functional_groups.get("carboxylic_acid", 0) > 0
    has_phenolic = functional_groups.get("phenolic_oh", 0) > 0
    has_polyol = functional_groups.get("alcohol_oh", 0) > 2
    has_aromatic = functional_groups.get("aromatic_ring", 0) > 0
    is_chiral = chirality_info and chirality_info.get("is_chiral", False)

    # Phase 1: NADES extraction
    if has_basic_amine:
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Extract with acidic NADES (citric or oxalic acid) to form ion pairs with basic amine groups",
            "rationale": "Ion-pair extraction improves recovery of alkaloids and amines",
        })
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Adjust pH to 8\u20139 with NH\u2084OH to precipitate the target compound",
            "rationale": "Alkaline pH deprotonates the amine, reducing solubility",
        })
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Back-extract into ethyl acetate or chloroform (3 \u00d7 20 mL)",
            "rationale": "Liquid-liquid partitioning enriches the free base in organic phase",
        })
    if has_acidic:
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Use choline chloride-based NADES for enhanced solubility of carboxylic acids",
            "rationale": "Choline acts as a counter-ion, forming deep eutectic complexes",
        })
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Acidify to pH 2\u20133 with HCl to precipitate the compound",
            "rationale": "Low pH protonates carboxylate, reducing aqueous solubility",
        })
    if has_phenolic:
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Extract with NADES at pH 5\u20136 to target phenolic hydroxyl groups via H-bonding",
            "rationale": "Neutral pH preserves phenolic-OH for optimal H-bond interactions",
        })
    if has_polyol:
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Use glycerol or sorbitol-based NADES to match polyol H-bond donor capacity",
            "rationale": "Similar H-bonding profiles improve solubility and extraction yield",
        })
    if has_aromatic and not has_basic_amine and not has_acidic:
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Extract with choline chloride:glycerol (1:2) NADES for aromatic compound solubility",
            "rationale": "Aromatic interactions with choline and glycerol enhance extraction",
        })
    if is_chiral:
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Include a chiral HBD (L-Malic Acid, L-Tartaric Acid, or L-Lactic Acid) to enable enantioselective extraction",
            "rationale": "Chiral selectors in NADES form diastereomeric complexes favoring one enantiomer",
        })
        steps.append({
            "step": len(steps) + 1, "phase": "extraction",
            "action": "Monitor enantiomeric excess via chiral HPLC or polarimetry after extraction",
            "rationale": "Verify enantioselectivity of the NADES system",
        })

    # Phase 2: Precipitation / chemical derivatisation
    precip_steps = suggest_precipitation_reagents(functional_groups)
    for ps in precip_steps:
        steps.append({
            "step": len(steps) + 1,
            "phase": "precipitation",
            "action": f"{ps['method']}: add {ps['reagent']} — {ps['detail']}",
            "rationale": f"Selective {ps['method'].lower()} isolates target compound via {ps['reagent']}",
        })

    # Phase 3: Chromatography purification
    chroma_recs = suggest_chromatography(functional_groups, compound_class)
    for cr in chroma_recs:
        steps.append({
            "step": len(steps) + 1,
            "phase": "chromatography",
            "action": f"Column: {cr['stationary_phase']} | Mobile: {cr['mobile_phase']} | Detection: {cr['detection']}",
            "rationale": cr.get("note", f"Recommended based on {cr['source']}"),
        })

    # Phase 4: Final workup
    steps.append({
        "step": len(steps) + 1, "phase": "workup",
        "action": "Concentrate pooled fractions under reduced pressure at 40\u201350 \u00b0C",
        "rationale": "Gentle evaporation avoids thermal degradation of target compound",
    })
    steps.append({
        "step": len(steps) + 1, "phase": "workup",
        "action": "Dry under high vacuum overnight and characterise by MS and NMR",
        "rationale": "Confirm identity and purity of isolated compound",
    })
    return steps


NATURAL_ENANTIOMERS = {
    "Limonene": "R",
    "Carvone": "S",
    "Camphor": "R",
    "Menthol": "R",
    "Quinine": "S",
    "Cocaine": "R",
    "Morphine": "R",
    "Codeine": "S",
    "Cytisine": "R",
    "Hyoscyamine": "S",
    "Ephedrine": "R",
    "Thalidomide": "racemic",
    "Ibuprofen": "S",
    "Naproxen": "S",
}


COMPOUND_CLASS_PREFERENCES = {
    "Alkaloids": {
        "examples": ["Berberine", "Caffeine", "Nicotine", "Quinine", "Morphine"],
        "preferred_hba": {"Choline Chloride": 0.9, "Betaine": 0.6, "Citric Acid": 0.7},
        "preferred_hbd": {"Citric Acid": 0.8, "Malic Acid": 0.7, "Glycerol": 0.5, "Oxalic Acid": 0.6},
        "preferred_ratio_range": (1.0, 3.0),
    },
    "Terpenoids": {
        "examples": ["Limonene", "Menthol", "Camphor", "Carvone", "Thymol"],
        "preferred_hba": {"Menthol": 0.8, "Choline Chloride": 0.5, "Betaine": 0.4},
        "preferred_hbd": {"Glycerol": 0.6, "Ethylene Glycol": 0.5, "Propylene Glycol": 0.5, "Lactic Acid": 0.7},
        "preferred_ratio_range": (1.0, 2.0),
    },
    "Flavonoids": {
        "examples": ["Quercetin", "Kaempferol", "Rutin", "Apigenin", "Naringenin"],
        "preferred_hba": {"Choline Chloride": 0.9, "Betaine": 0.7},
        "preferred_hbd": {"Citric Acid": 0.8, "Malic Acid": 0.6, "Glycerol": 0.7},
        "preferred_ratio_range": (1.0, 2.0),
    },
    "Phenolics": {
        "examples": ["Gallic Acid", "Caffeic Acid", "Ferulic Acid", "Curcumin", "Resveratrol"],
        "preferred_hba": {"Choline Chloride": 0.8, "Betaine": 0.6},
        "preferred_hbd": {"Glycerol": 0.8, "Citric Acid": 0.7, "Malic Acid": 0.6, "Lactic Acid": 0.5},
        "preferred_ratio_range": (1.0, 3.0),
    },
    "Anthocyanins": {
        "examples": ["Cyanidin", "Delphinidin", "Pelargonidin", "Malvidin"],
        "preferred_hba": {"Choline Chloride": 0.8, "Citric Acid": 0.6},
        "preferred_hbd": {"Citric Acid": 0.9, "Malic Acid": 0.7, "Lactic Acid": 0.6},
        "preferred_ratio_range": (1.0, 2.0),
    },
    "Carotenoids": {
        "examples": ["Beta-Carotene", "Lycopene", "Lutein", "Astaxanthin"],
        "preferred_hba": {"Menthol": 0.9, "Choline Chloride": 0.3},
        "preferred_hbd": {"Glycerol": 0.3, "Ethylene Glycol": 0.2},
        "preferred_ratio_range": (0.5, 2.0),
    },
    "Saponins": {
        "examples": ["Ginsenoside Rb1", "Ginsenoside Rg1", "Diosgenin", "Glycyrrhizic Acid"],
        "preferred_hba": {"Choline Chloride": 0.9, "Betaine": 0.5},
        "preferred_hbd": {"Citric Acid": 0.6, "Malic Acid": 0.5, "Glycerol": 0.8, "Urea": 0.6},
        "preferred_ratio_range": (1.0, 2.0),
    },
    "Coumarins": {
        "examples": ["Umbelliferone", "Scopoletin", "Psoralen", "Bergapten"],
        "preferred_hba": {"Choline Chloride": 0.8, "Betaine": 0.5},
        "preferred_hbd": {"Glycerol": 0.7, "Citric Acid": 0.6, "Malic Acid": 0.5},
        "preferred_ratio_range": (1.0, 2.0),
    },
    "Lignans": {
        "examples": ["Podophyllotoxin", "Secoisolariciresinol", "Matairesinol", "Arctigenin"],
        "preferred_hba": {"Choline Chloride": 0.7, "Betaine": 0.5},
        "preferred_hbd": {"Glycerol": 0.6, "Citric Acid": 0.5, "Lactic Acid": 0.6},
        "preferred_ratio_range": (1.0, 2.0),
    },
    "Quinones": {
        "examples": ["Thymoquinone", "Plumbagin", "Shikonin", "Anthraquinone"],
        "preferred_hba": {"Choline Chloride": 0.6, "Menthol": 0.4},
        "preferred_hbd": {"Citric Acid": 0.5, "Glycerol": 0.4, "Lactic Acid": 0.5},
        "preferred_ratio_range": (0.5, 2.0),
    },
}


COMMON_NADES = {
    "Choline Chloride": "C[N+](C)(C)CCO.[Cl-]",
    "Urea": "C(=O)(N)N",
    "Glycerol": "C(C(CO)O)O",
    "Ethylene Glycol": "C(CO)O",
    "Citric Acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    "Malic Acid": "C(C(C(=O)O)O)C(=O)O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Fructose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Lactic Acid": "CC(C(=O)O)O",
    "Oxalic Acid": "C(=O)(C(=O)O)O",
    "Betaine": "C[N+](C)(C)CC(=O)[O-]",
    "Sorbitol": "C(C(C(C(C(CO)O)O)O)O)O",
    "Xylitol": "C(C(C(C(CO)O)O)O)O",
    "Water": "O",
    "Acetic Acid": "CC(=O)O",
    "Formic Acid": "C(=O)O",
    "Propylene Glycol": "CC(CO)O",
    "Triethylene Glycol": "C(COCCOCCO)O",
    "Methanol": "CO",
    "Ethanol": "CCO",
}

ALL_HBA = sorted(set(COMMON_NADES.keys()) - set(CHIRAL_HBD.keys()))
ALL_HBD = sorted(set(COMMON_NADES.keys()) | set(CHIRAL_HBD.keys()))
