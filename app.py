import io
import os
import re
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml_engine import (
    initialize,
    predict,
    model_exists,
    recommend_nades,
    recommend_nades_with_path,
    compute_contrast_analysis,
    retrain_from_lab_data as retrain_model,
)
from nades_db import (
    get_compound_names,
    get_compound_classes_with_data,
    get_extraction_records,
    get_target_compound_by_name,
    init_db,
    insert_extraction_record,
    insert_target_compound,
    extraction_record_count,
)
from utils import (
    CHIRAL_HBD,
    COMMON_NADES,
    ALL_HBA,
    ALL_HBD,
    COMPOUND_CLASS_CHROMATOGRAPHY,
    combine_fingerprints,
    compute_rdkit_descriptors,
    detect_chirality,
    detect_functional_groups,
    generate_molecular_diagram,
    name_to_fingerprint,
    name_to_smiles,
    smiles_to_fingerprint,
    suggest_isolation_strategy,
    suggest_precipitation_reagents,
    suggest_chromatography,
)

st.set_page_config(page_title="EcoChem-AI", page_icon="🧪", layout="wide")

init_db()

if "history" not in st.session_state:
    st.session_state.history = []
if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "target_info" not in st.session_state:
    st.session_state.target_info = None
if "chirality_info" not in st.session_state:
    st.session_state.chirality_info = None
if "protocol_steps" not in st.session_state:
    st.session_state.protocol_steps = []
if "paper_extracted" not in st.session_state:
    st.session_state.paper_extracted = None
if "bulk_matrix" not in st.session_state:
    st.session_state.bulk_matrix = []
if "extraction_mode" not in st.session_state:
    st.session_state.extraction_mode = "targeted"
if "path_indicator" not in st.session_state:
    st.session_state.path_indicator = None
if "contrast_analysis" not in st.session_state:
    st.session_state.contrast_analysis = None
if "target_descriptors" not in st.session_state:
    st.session_state.target_descriptors = None
if "precipitation_steps" not in st.session_state:
    st.session_state.precipitation_steps = []
if "chromatography_steps" not in st.session_state:
    st.session_state.chromatography_steps = []

if not st.session_state.model_initialized:
    with st.spinner("Initializing ML engine..."):
        initialize()
        st.session_state.model_initialized = True

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>\U0001f9ea EcoChem-AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; color: #888;'>NADES Recommendation & Extraction Planning System</h4>",
    unsafe_allow_html=True,
)

compound_classes = [
    "Alkaloids", "Terpenoids", "Flavonoids", "Phenolics",
    "Anthocyanins", "Carotenoids", "Saponins", "Coumarins",
    "Lignans", "Quinones",
]

with st.sidebar:
    st.markdown("### \U0001f4cb Bulk Sample Matrix")
    st.markdown("---")
    bulk_input = st.text_area(
        "Matrix Compounds (comma-separated)",
        placeholder="e.g. cellulose, lignin, chlorophyll, proteins, starch",
        key="bulk_matrix_input",
        height=80,
    )
    st.markdown("---")
    st.markdown("### \U0001f3af Extraction Mode")
    extraction_mode = st.radio(
        "Mode",
        ["Targeted Compound", "Total Extraction"],
        index=0,
        key="extraction_mode_radio",
    )
    st.markdown("---")
    target_name = None
    compound_class = "Flavonoids"
    target_smiles_input = ""
    if extraction_mode == "Targeted Compound":
        target_name = st.text_input(
            "Target Compound Name",
            placeholder="e.g. Quercetin, Berberine, Menthol",
            key="target_name_input",
        )
        target_smiles_input = st.text_input(
            "SMILES (optional — paste directly if PubChem fails)",
            placeholder="e.g. CC(=O)N[C@H]1CCC2=CC...",
            key="target_smiles_input",
        )
        compound_class = st.selectbox(
            "Compound Class", compound_classes, index=2, key="class_dropdown"
        )
    st.markdown("### \U0001f3af Preferences")
    pref_liquid_rt = st.checkbox("Prefer Liquid at Room Temp", value=True)
    pref_low_eco_tox = st.checkbox("Minimize EcoToxicity", value=True)
    pref_stability = st.checkbox("Maximize Phase Stability", value=True)
    st.markdown("---")
    recommend_btn = st.button(
        "\U0001f52c Run Analysis", type="primary", use_container_width=True
    )
    st.markdown("---")
    st.caption(f"DB records: {extraction_record_count()}")

if recommend_btn:
    bulk_list = [c.strip() for c in bulk_input.split(",") if c.strip()]

    if extraction_mode == "Targeted Compound" and not target_name:
        st.warning("Enter a target compound name in 'Targeted Compound' mode.")
    elif extraction_mode == "Total Extraction" and not bulk_list:
        st.warning("Enter at least one bulk matrix compound in 'Total Extraction' mode.")
    else:
        st.session_state.bulk_matrix = bulk_list
        st.session_state.extraction_mode = "targeted" if extraction_mode == "Targeted Compound" else "total"

        with st.spinner("Analyzing..."):
            fgs = {}
            chirality_info = None
            target_smiles = None

            if extraction_mode == "Targeted Compound" and target_name:
                if target_smiles_input and len(target_smiles_input) > 5:
                    target_smiles = target_smiles_input.strip()
                else:
                    target_smiles = name_to_smiles(target_name)
                if target_smiles is None:
                    st.error(f"Could not resolve '{target_name}'. Paste the SMILES string in the 'SMILES (optional)' field above, or check the spelling.")
                    st.stop()
                fgs = detect_functional_groups(target_smiles)
                chirality_info = detect_chirality(target_smiles)
                from utils import NATURAL_ENANTIOMERS
                if target_name in NATURAL_ENANTIOMERS:
                    chirality_info["natural_enantiomer"] = NATURAL_ENANTIOMERS[target_name]
                compound_id = insert_target_compound(
                    name=target_name, smiles=target_smiles,
                    compound_class=compound_class, functional_groups=fgs,
                    is_chiral=1 if chirality_info["is_chiral"] else 0,
                    natural_enantiomer=chirality_info.get("natural_enantiomer", "racemic"),
                )
                st.session_state.target_info = {
                    "name": target_name, "smiles": target_smiles,
                    "functional_groups": fgs, "compound_class": compound_class,
                    "compound_id": compound_id,
                }
                st.session_state.chirality_info = chirality_info

            elif extraction_mode == "Total Extraction" and bulk_list:
                target_smiles = None
                resolved = []
                for c in bulk_list:
                    smi = name_to_smiles(c)
                    if smi:
                        resolved.append((c, smi))
                    else:
                        resolved.append((c, None))
                st.session_state.target_info = {
                    "name": " + ".join(c for c, _ in resolved),
                    "smiles": "; ".join(s for _, s in resolved if s),
                    "functional_groups": {},
                    "compound_class": "Total Extraction",
                    "compound_id": None,
                }

            preferences = {
                "liquid_rt": 1.0 if pref_liquid_rt else 0.0,
                "low_eco_tox": 1.0 if pref_low_eco_tox else 0.0,
                "stability": 1.0 if pref_stability else 0.0,
            }

            try:
                path_result = recommend_nades_with_path(
                    target_name=target_name if extraction_mode == "Targeted Compound" else None,
                    target_smiles=target_smiles,
                    compound_class=compound_class if extraction_mode == "Targeted Compound" else None,
                    bulk_matrix_names=bulk_list if bulk_list else None,
                    functional_groups=fgs,
                    is_chiral=chirality_info["is_chiral"] if chirality_info else False,
                    preferences=preferences,
                    mode="targeted" if extraction_mode == "Targeted Compound" else "total",
                )
                st.session_state.path_indicator = path_result["path_label"]
                st.session_state.recommendations = path_result["recommendations"]
                st.session_state.contrast_analysis = path_result.get("contrast_analysis")
                st.session_state.target_descriptors = path_result.get("target_descriptors")
            except Exception as e:
                st.error(f"NADES recommendation error: {e}")
                st.code(traceback.format_exc())
                st.session_state.recommendations = []

            if extraction_mode == "Targeted Compound" and target_smiles:
                iso_steps = suggest_isolation_strategy(fgs, chirality_info, compound_class)
                st.session_state.protocol_steps = iso_steps
                st.session_state.precipitation_steps = suggest_precipitation_reagents(fgs)
                st.session_state.chromatography_steps = suggest_chromatography(fgs, compound_class)
            else:
                st.session_state.protocol_steps = []
                st.session_state.precipitation_steps = []
                st.session_state.chromatography_steps = []

tab_recs, tab_protocol, tab_diagrams, tab_lab, tab_methodology = st.tabs(
    ["\U0001f50d NADES Recommendations", "\U0001f4cb Extraction Protocol",
     "\U0001f9ec Structural Diagrams", "\U0001f4ca Lab Data", "\U0001f4d6 Methodology"]
)

with tab_recs:
    if st.session_state.target_info is None:
        st.info("Enter a target or matrix in the sidebar and click 'Run Analysis' to begin.")
    else:
        info = st.session_state.target_info
        left_col, right_col = st.columns(2)

        with left_col:
            if st.session_state.extraction_mode == "targeted":
                st.markdown(f"**\U0001f3af Target:** {info['name']}")
                if info.get("smiles"):
                    st.markdown(f"**SMILES:** `{info['smiles']}`")
                st.markdown(f"**Class:** {info.get('compound_class', 'N/A')}")
                chir = st.session_state.chirality_info
                if chir and chir.get("is_chiral"):
                    st.markdown(f"**Chiral:** \u2705 Yes \u2014 {chir.get('natural_enantiomer', '?')}-dominant")
                else:
                    st.markdown(f"**Chiral:** \u274c No")
                fgs = info.get("functional_groups", {})
                if fgs:
                    active = {k: v for k, v in fgs.items() if v > 0}
                    st.markdown("**Functional Groups:**")
                    for k, v in list(active.items())[:8]:
                        st.markdown(f"- `{k}`: {v}")

            td = st.session_state.target_descriptors
            if td:
                st.markdown("**RDKit Descriptors:**")
                st.markdown(f"- HBD: {td['hbd']} | HBA: {td['hba']}")
                st.markdown(f"- TPSA: {td['tpsa']} \u00c5\u00b2 | LogP: {td['logp']:.2f}")
                st.markdown(f"- MW: {td['mol_wt']:.1f} | Rot. bonds: {td['rotatable_bonds']}")

        with right_col:
            if st.session_state.bulk_matrix:
                st.markdown(f"**\U0001f4cb Bulk Matrix ({len(st.session_state.bulk_matrix)} components):**")
                for c in st.session_state.bulk_matrix:
                    st.markdown(f"- {c}")

            ca = st.session_state.contrast_analysis
            if ca:
                st.markdown("---")
                st.markdown("**\U0001f9ee Matrix Contrast Analysis:**")
                st.markdown(f"- LogP \u0394: `{ca.get('logp_delta', '?')}`")
                st.markdown(f"- TPSA \u0394: `{ca.get('tpsa_delta', '?')}` \u00c5\u00b2")
                sel = ca.get("selectivity_estimate", "")
                if "high" in sel:
                    st.markdown(f"- Selectivity: \U0001f7e2 **{sel}**")
                elif "low" in sel or "very low" in sel:
                    st.markdown(f"- Selectivity: \U0001f534 **{sel}**")
                else:
                    st.markdown(f"- Selectivity: \U0001f7e1 **{sel}**")
                if ca.get("matrix_compounds"):
                    st.markdown("**Component descriptors:**")
                    mc_df = pd.DataFrame(ca["matrix_compounds"])
                    st.dataframe(mc_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        pi = st.session_state.path_indicator
        if pi:
            path_color = "#4CAF50" if "Literature" in pi or "Class" in pi else "#2196F3"
            st.markdown(
                f"<div style='background:{path_color};padding:6px 12px;border-radius:6px;"
                f"color:white;font-weight:bold;display:inline-block;'>{pi}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(" ")

        if st.session_state.recommendations:
            recs = st.session_state.recommendations
            df_recs = pd.DataFrame(recs)
            display_cols = [
                "HBA", "HBD", "Ratio", "Eutectic_Temp_C", "Phase_Stability",
                "EcoToxicity_Index", "Liquid_At_Room_Temp", "Chiral_HBD", "Source",
            ]
            df_display = df_recs[[c for c in display_cols if c in df_recs.columns]].copy()
            if "Phase_Stability" in df_display.columns:
                df_display["Phase_Stability"] = df_display["Phase_Stability"].map(
                    {True: "\u2705 Yes", False: "\u274c No"}
                )
            if "Liquid_At_Room_Temp" in df_display.columns:
                df_display["Liquid_At_Room_Temp"] = df_display["Liquid_At_Room_Temp"].map(
                    {True: "\u2705 Yes", False: "\u274c No"}
                )
            if "Chiral_HBD" in df_display.columns:
                df_display["Chiral_HBD"] = df_display["Chiral_HBD"].map(
                    {True: "\U0001f9ec Chiral", False: "\u2014"}
                )
            df_display = df_display.rename(columns={
                "Eutectic_Temp_C": "Eutectic (\u00b0C)",
                "Phase_Stability": "Stable",
                "EcoToxicity_Index": "EcoTox",
                "Liquid_At_Room_Temp": "Liquid RT",
                "Chiral_HBD": "HBD Type",
            })

            st.markdown("### Recommended NADES System")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            if len(recs) > 0:
                eco_val = recs[0].get("EcoToxicity_Index", 0.5)
                if eco_val < 0.3:
                    st.success(f"\U0001f7e2 **EcoToxicity Index: {eco_val}** \u2014 Low environmental impact")
                elif eco_val < 0.6:
                    st.warning(f"\U0001f7e1 **EcoToxicity Index: {eco_val}** \u2014 Moderate environmental impact")
                else:
                    st.error(f"\U0001f534 **EcoToxicity Index: {eco_val}** \u2014 High environmental impact")

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Property Bar")
                bar_labels = [f"{r['HBA']}/{r['HBD']}" for r in recs]
                bar_scores = [r["Eutectic_Temp_C"] for r in recs]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(bar_labels, bar_scores, color="#4CAF50")
                ax.axvline(25, color="orange", linestyle="--", label="Room Temp (25\u00b0C)")
                ax.set_xlabel("Eutectic Temperature (\u00b0C)")
                ax.legend(fontsize=8)
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close(fig)
            with col_chart2:
                st.markdown("#### Property Radar")
                best = recs[0]
                categories = ["Eutectic Temp", "Phase Stability", "EcoToxicity"]
                eutectic_norm = min(best["Eutectic_Temp_C"] / 120.0, 1.0)
                phase_norm = 1.0 if best["Phase_Stability"] else 0.0
                eco_norm = best["EcoToxicity_Index"]
                values = [eutectic_norm, phase_norm, 1.0 - eco_norm]
                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                ax.fill(angles, values, color="#4CAF50", alpha=0.25)
                ax.plot(angles, values, color="#4CAF50", linewidth=2)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_title(f"Best: {recs[0]['HBA']} / {recs[0]['HBD']}", fontsize=9)
                st.pyplot(fig)
                plt.close(fig)

            if st.session_state.history:
                st.markdown("---")
                st.markdown("### Prediction History")
                st.dataframe(
                    pd.DataFrame(st.session_state.history),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.warning("No recommendations generated. Try a different compound or matrix.")

with tab_protocol:
    st.markdown("## \U0001f4cb Extraction & Isolation Protocol")
    has_any = bool(
        st.session_state.protocol_steps
        or st.session_state.precipitation_steps
        or st.session_state.chromatography_steps
    )
    if not has_any:
        if st.session_state.target_info is None:
            st.info("Run an analysis first to see the extraction protocol.")
        else:
            st.warning("No protocol generated for this input. Try a targeted compound.")
    else:
        if st.session_state.protocol_steps:
            st.markdown("### \U0001f3ed Phase 1: NADES Extraction")
            for step in st.session_state.protocol_steps:
                if step.get("phase") == "extraction":
                    with st.expander(f"**Step {step['step']}:** {step['action'][:90]}..."):
                        st.markdown(f"**Action:** {step['action']}")
                        st.markdown(f"**Rationale:** {step['rationale']}")

        if st.session_state.precipitation_steps:
            st.markdown("---")
            st.markdown("### \U0001f9ea Phase 2: Precipitation / Chemical Derivatisation")
            for i, ps in enumerate(st.session_state.precipitation_steps, 1):
                with st.expander(f"**Precip. {i}:** {ps['method']} with {ps['reagent']}"):
                    st.markdown(f"**Reagent:** {ps['reagent']}")
                    st.markdown(f"**Method:** {ps['method']}")
                    st.markdown(f"**Detail:** {ps['detail']}")

        if st.session_state.chromatography_steps:
            st.markdown("---")
            st.markdown("### \U0001f9f0 Phase 3: Chromatography Purification")
            for i, cr in enumerate(st.session_state.chromatography_steps, 1):
                label = f"**Chroma. {i}:** {cr['stationary_phase'][:60]}..."
                with st.expander(label):
                    st.markdown(f"**Source:** {cr['source']}")
                    st.markdown(f"**Column:** {cr['stationary_phase']}")
                    st.markdown(f"**Mobile Phase:** {cr['mobile_phase']}")
                    st.markdown(f"**Detection:** {cr['detection']}")
                    if cr.get("note"):
                        st.markdown(f"**Note:** {cr['note']}")
                    if cr.get("rf_range"):
                        st.markdown(f"**Expected Rf:** {cr['rf_range']}")

        st.markdown("---")
        st.markdown("### \U0001f527 Phase 4: Final Workup")
        if st.session_state.protocol_steps:
            for step in st.session_state.protocol_steps:
                if step.get("phase") == "workup":
                    with st.expander(f"**Final {step['step']}:** {step['action'][:80]}..."):
                        st.markdown(f"**Action:** {step['action']}")
                        st.markdown(f"**Rationale:** {step['rationale']}")

        st.markdown("---")
        st.markdown("**References:**")
        st.markdown("""
        - Dai, Y., et al. *Anal. Chim. Acta* 766, 61\u201368 (2013)
        - Choi, Y. H., et al. *Trends Anal. Chem.* 118, 158\u2013169 (2019)
        - Fern\u00e1ndez, M. \u00c1., et al. *J. Clean. Prod.* 201, 572\u2013582 (2018)
        - Stahl, E. *Thin-Layer Chromatography*, 2nd ed., Springer (1969)
        - Harborne, J. B. *Phytochemical Methods*, 3rd ed., Chapman & Hall (1998)
        """)

with tab_diagrams:
    st.markdown("### \U0001f9ec Molecular Structure Viewer")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.target_info and st.session_state.target_info.get("smiles"):
            st.markdown(f"**Target:** {st.session_state.target_info['name']}")
            smi = st.session_state.target_info["smiles"]
            if ";" in smi:
                smi = smi.split(";")[0]
            elif " " in smi:
                smi = smi.split()[0]
            img = generate_molecular_diagram(smi)
            if img:
                st.image(img, use_container_width=True)
        else:
            st.info("No target compound selected.")
    with col2:
        st.markdown("**NADES Components**")
        if st.session_state.recommendations:
            viewed_hba = st.selectbox(
                "HBA",
                list(dict.fromkeys([r["HBA"] for r in st.session_state.recommendations])),
                key="dia_hba",
            )
            viewed_hbd = st.selectbox(
                "HBD",
                list(dict.fromkeys([r["HBD"] for r in st.session_state.recommendations])),
                key="dia_hbd",
            )
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                _, smi_hba = name_to_fingerprint(viewed_hba)
                if smi_hba:
                    img = generate_molecular_diagram(smi_hba)
                    if img:
                        st.image(img, caption=f"HBA: {viewed_hba}", use_container_width=True)
            with sub_c2:
                _, smi_hbd = name_to_fingerprint(viewed_hbd)
                if smi_hbd:
                    img = generate_molecular_diagram(smi_hbd)
                    if img:
                        st.image(img, caption=f"HBD: {viewed_hbd}", use_container_width=True)
        else:
            st.info("Get recommendations first to view NADES structures.")

with tab_lab:
    st.markdown("## \U0001f4ca Lab Data Management")
    tab_entry, tab_browse, tab_paper, tab_retrain = st.tabs(
        ["\U0001f4dd Enter Data", "\U0001f4c2 Browse Records",
         "\U0001f4c4 Upload Paper", "\U0001f504 Retrain Model"]
    )

    with tab_entry:
        st.markdown("### Enter Experimental Results")
        existing_names = get_compound_names()
        lab_target = st.text_input("Target Compound Name", placeholder="e.g. Quercetin", key="lab_target")
        if existing_names:
            lab_target = st.selectbox("Or select existing", [""] + existing_names, key="lab_target_sel")
        lab_smiles = st.text_input("SMILES (optional)", key="lab_smiles")
        lab_class = st.selectbox("Compound Class", compound_classes, key="lab_class")
        col_a, col_b = st.columns(2)
        with col_a:
            lab_hba = st.selectbox("HBA", [""] + sorted(ALL_HBA), key="lab_hba")
        with col_b:
            lab_hbd = st.selectbox("HBD", [""] + sorted(ALL_HBD), key="lab_hbd")
        lab_ratio = st.number_input("Molar Ratio", 0.1, 10.0, 1.0, 0.1, key="lab_ratio")
        lab_method = st.selectbox("Method", ["UAE", "MAE", "Maceration", "HRE", "Soxhlet", "Other"], key="lab_method")
        col_c, col_d, col_e, col_f = st.columns(4)
        with col_c:
            lab_temp = st.number_input("Temp (\u00b0C)", -20.0, 200.0, 25.0, key="lab_temp")
        with col_d:
            lab_time = st.number_input("Time (min)", 1, 1440, 30, key="lab_time")
        with col_e:
            lab_yield = st.number_input("Yield (%)", 0.0, 100.0, 50.0, key="lab_yield")
        with col_f:
            lab_purity = st.number_input("Purity (%)", 0.0, 100.0, 90.0, key="lab_purity")
        lab_steps = st.text_area("Isolation Steps (one per line)", height=80, key="lab_steps")
        lab_citation = st.text_input("Citation (optional)", key="lab_citation")
        if st.button("\U0001f4be Save to Database", type="primary", key="save_lab"):
            if not lab_target or not lab_hba or not lab_hbd:
                st.warning("Target, HBA, and HBD required.")
            else:
                if not lab_smiles:
                    lab_smiles = name_to_smiles(lab_target) or ""
                fgs = detect_functional_groups(lab_smiles) if lab_smiles else {}
                chir = detect_chirality(lab_smiles) if lab_smiles else {}
                cid = insert_target_compound(
                    name=lab_target, smiles=lab_smiles or "Unknown",
                    compound_class=lab_class, functional_groups=fgs,
                    is_chiral=1 if chir.get("is_chiral") else 0,
                    natural_enantiomer=chir.get("natural_enantiomer", "racemic"),
                )
                steps_list = [s.strip() for s in lab_steps.strip().split("\n") if s.strip()]
                rid = insert_extraction_record(
                    compound_id=cid, hba_name=lab_hba, hbd_name=lab_hbd,
                    molar_ratio=lab_ratio, method=lab_method,
                    temperature_c=lab_temp, time_minutes=int(lab_time),
                    yield_percent=lab_yield, purity_percent=lab_purity,
                    isolation_steps=steps_list if steps_list else None,
                    source_type="lab", citation=lab_citation or None,
                )
                st.success(f"Record saved (ID: {rid})!")
                st.rerun()

    with tab_browse:
        st.markdown("### Stored Extraction Records")
        records = get_extraction_records()
        if records:
            df = pd.DataFrame(records)
            df["created_at"] = pd.to_datetime(df["created_at"])
            display_cols = [
                "record_id", "hba_name", "hbd_name", "molar_ratio", "method",
                "temperature_c", "time_minutes", "yield_percent", "purity_percent",
                "source_type", "citation", "created_at",
            ]
            df_display = df[[c for c in display_cols if c in df.columns]]
            df_display = df_display.rename(columns={
                "record_id": "ID", "hba_name": "HBA", "hbd_name": "HBD",
                "molar_ratio": "Ratio", "method": "Method",
                "temperature_c": "Temp (\u00b0C)", "time_minutes": "Time (min)",
                "yield_percent": "Yield (%)", "purity_percent": "Purity (%)",
                "source_type": "Source", "citation": "Citation", "created_at": "Date",
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            csv = df_display.to_csv(index=False).encode()
            st.download_button("\U0001f4e5 Download CSV", csv, "ecochem_data.csv", "text/csv")
        else:
            st.info("No records yet.")

    with tab_paper:
        st.markdown("### Upload Research Paper (Semi-Automatic)")
        uploaded_pdf = st.file_uploader("Choose PDF", type=["pdf"], key="pdf_uploader")
        if uploaded_pdf is not None:
            paper_class = st.selectbox("Compound Class", compound_classes, key="paper_class")
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(uploaded_pdf.read())) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                st.success(f"Extracted {len(text)} chars")
                if st.session_state.paper_extracted is None:
                    ex = {"hba": "", "hbd": "", "ratio": 1.0, "method": "UAE",
                          "temp": 25.0, "time": 30, "yield": 50.0, "purity": 90.0,
                          "steps": "", "citation": ""}
                    temp_m = re.findall(r'(\d+[.,]?\d*)\s*[°]?\s*C', text, re.IGNORECASE)
                    if temp_m:
                        ex["temp"] = float(temp_m[0].replace(",", "."))
                    ratio_m = re.findall(r'(\d+[.:]\d+)', text)
                    if ratio_m:
                        parts = ratio_m[0].replace(":", ".").split(".")
                        if len(parts) >= 2:
                            ex["ratio"] = float(parts[0]) / float(parts[1])
                            if ex["ratio"] < 0.1 or ex["ratio"] > 10:
                                ex["ratio"] = 1.0
                    yield_m = re.findall(r'(\d+[.,]?\d*)\s*%', text)
                    for ym in yield_m:
                        val = float(ym.replace(",", "."))
                        if 0 < val <= 100:
                            ex["yield"] = val
                            break
                    time_m = re.findall(r'(\d+)\s*(min|minutes|hour|h)', text, re.IGNORECASE)
                    if time_m:
                        val = int(time_m[0][0])
                        if time_m[0][1].startswith("h"):
                            val *= 60
                        ex["time"] = val
                    for name in sorted(ALL_HBA) + sorted(ALL_HBD):
                        if name.lower() in text.lower():
                            if name in ALL_HBA and not ex["hba"]:
                                ex["hba"] = name
                            elif name in ALL_HBD and not ex["hbd"]:
                                ex["hbd"] = name
                    st.session_state.paper_extracted = ex
                ex = st.session_state.paper_extracted
                st.markdown("### Review Extracted Data")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    paper_hba = st.text_input("HBA", value=ex.get("hba", ""), key="paper_hba")
                    paper_hbd = st.text_input("HBD", value=ex.get("hbd", ""), key="paper_hbd")
                    paper_ratio = st.number_input("Ratio", 0.1, 10.0, float(ex.get("ratio", 1.0)), 0.1, key="paper_ratio")
                    paper_method = st.text_input("Method", value=ex.get("method", "UAE"), key="paper_method")
                with col_p2:
                    paper_temp = st.number_input("Temp", -20.0, 200.0, float(ex.get("temp", 25.0)), key="paper_temp")
                    paper_time = st.number_input("Time (min)", 1, 1440, int(ex.get("time", 30)), key="paper_time")
                    paper_yield = st.number_input("Yield", 0.0, 100.0, float(ex.get("yield", 50.0)), key="paper_yield")
                    paper_purity = st.number_input("Purity", 0.0, 100.0, float(ex.get("purity", 90.0)), key="paper_purity")
                paper_steps = st.text_area("Steps", value=ex.get("steps", ""), height=80, key="paper_steps")
                paper_citation = st.text_input("Citation", value=ex.get("citation", ""), key="paper_citation")
                paper_target = st.text_input("Target Compound", key="paper_target")
                if st.button("\U0001f4be Save Paper Data", type="primary", key="save_paper"):
                    if not paper_hba or not paper_hbd or not paper_target:
                        st.warning("HBA, HBD, and target required.")
                    else:
                        tgt_smi = name_to_smiles(paper_target) or ""
                        fgs = detect_functional_groups(tgt_smi) if tgt_smi else {}
                        cid = insert_target_compound(
                            name=paper_target, smiles=tgt_smi or "Unknown",
                            compound_class=paper_class, functional_groups=fgs,
                        )
                        steps_l = [s.strip() for s in paper_steps.strip().split("\n") if s.strip()]
                        rid = insert_extraction_record(
                            compound_id=cid, hba_name=paper_hba, hbd_name=paper_hbd,
                            molar_ratio=paper_ratio, method=paper_method,
                            temperature_c=paper_temp, time_minutes=int(paper_time),
                            yield_percent=paper_yield, purity_percent=paper_purity,
                            isolation_steps=steps_l if steps_l else None,
                            source_type="literature", citation=paper_citation or None,
                        )
                        st.success(f"Literature record saved (ID: {rid})!")
                        st.session_state.paper_extracted = None
                        st.rerun()
            except ImportError:
                st.error("pdfplumber required. Install: pip install pdfplumber")
            except Exception as e:
                st.error(f"PDF error: {e}")
                st.code(traceback.format_exc())

    with tab_retrain:
        st.markdown("### Retrain Model from Lab Data")
        records = get_extraction_records()
        valid_count = sum(
            1 for r in records
            if r.get("yield_percent") is not None and r.get("temperature_c") is not None
        )
        st.markdown(f"- Total records: {len(records)}")
        st.markdown(f"- Valid records: {valid_count}")
        if valid_count < 3:
            st.warning("Need at least 3 valid records.")
        if st.button("\U0001f504 Retrain Model", type="primary", disabled=valid_count < 3):
            with st.spinner("Retraining..."):
                result = retrain_model()
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])

with tab_methodology:
    st.markdown("## Methodology")
    with st.expander("\U0001f4ca Model Architecture", expanded=True):
        st.markdown("""
        **EcoChem-AI** uses a **Multi-Output Random Forest Regressor** (scikit-learn)
        to predict NADES properties from 2048-bit Morgan molecular fingerprints (RDKit, radius=2).

        **Fast/Slow Path Architecture:**
        1. **Fast Path** \u2014 Checks literature-backed NADES database for known compound matches
        2. **Slow Path** \u2014 Dynamically predicts ideal HBA/HBD/ratio using RDKit descriptors (HBD, HBA, TPSA, LogP)
        3. **Matrix Contrast** \u2014 Compares target LogP/TPSA against bulk matrix for selectivity estimation
        """)
    with st.expander("\U0001f9ec Isolation Strategy Engine"):
        st.markdown("""
        **Phase 1 \u2014 NADES Extraction:** Functional-group-specific NADES selection
        **Phase 2 \u2014 Precipitation:** Classical reagents (Dragendorff's, lead acetate, CaCl\u2082, etc.)
        **Phase 3 \u2014 Chromatography:** Compound-class-specific column configurations
        **Phase 4 \u2014 Final Workup:** Concentration, drying, characterisation
        """)
    with st.expander("\U0001f4da References"):
        st.markdown("""
        - PubChem \u2014 Kim et al., *Nucleic Acids Res.* (2023)
        - RDKit \u2014 https://www.rdkit.org/
        - scikit-learn \u2014 Pedregosa et al., *JMLR* (2011)
        - Dai, Y., et al. *Anal. Chim. Acta* 766, 61\u201368 (2013)
        - Choi, Y. H., et al. *Trends Anal. Chem.* 118, 158\u2013169 (2019)
        - Fern\u00e1ndez, M. \u00c1., et al. *J. Clean. Prod.* 201, 572\u2013582 (2018)
        """)
