"""
Streamlit demo. Run with: streamlit run app/app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import run_pipeline  # noqa: E402
from paths import CHECKPOINT_PATH, OUTPUTS_DIR  # noqa: E402
from report_pdf import generate_pdf_report  # noqa: E402

st.set_page_config(page_title="Chest X-ray Diagnostic Assistant", layout="wide")
st.title("Explainable Chest X-ray Diagnostic Assistant")
st.caption(
    "Prototype for offline, low-resource clinics. This is a decision-support "
    "second opinion, not a replacement for a qualified clinician."
)

uploaded = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

with st.expander("Patient details (optional, included in downloadable report)"):
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        patient_name = st.text_input("Name")
        patient_age = st.text_input("Age")
    with p_col2:
        patient_gender = st.text_input("Gender")
        patient_id = st.text_input("Patient / Case ID")
    patient_notes = st.text_area("Notes (symptoms, history, etc.)")

checkpoint_path = CHECKPOINT_PATH

if uploaded is not None:
    if not os.path.exists(checkpoint_path):
        st.error(f"No trained model found at {checkpoint_path}.")
    else:
        temp_path = os.path.join(OUTPUTS_DIR, "_uploaded_temp.png")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        Image.open(uploaded).convert("L").save(temp_path)

        with st.spinner("Analyzing..."):
            result = run_pipeline(temp_path, checkpoint=checkpoint_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(Image.open(temp_path), use_container_width=True)
        with col2:
            st.subheader("Heatmap (attention)")
            st.image(result["overlay_image"], use_container_width=True)
        with col3:
            st.subheader("Marked region")
            st.image(result["contour_image"], use_container_width=True)

        tier = result["triage"]["tier"]
        tier_color = {"Routine": "green", "Soon": "blue", "Urgent": "orange", "Immediate": "red"}
        st.markdown(f"### Triage: :{tier_color.get(tier, 'gray')}[{tier}]")
        st.write(result["triage"]["message"])

        st.subheader("Generated report")
        st.markdown(result["report"])

        if result.get("blip_caption"):
            st.info(f"**General image caption (BLIP v1, not medical-specific):** {result['blip_caption']}")

        if result.get("clip_scores"):
            clip_top = max(result["clip_scores"].items(), key=lambda x: x[1])
            st.caption(f"CLIP zero-shot cross-check: top match = {clip_top[0]} ({clip_top[1]*100:.1f}%)")

        flagged_summary = ", ".join(
            f"{c} ({conf*100:.0f}%)" for c, conf in result["all_flagged"]
        ) or "None above threshold"
        st.caption(f"Flagged conditions: {flagged_summary} | Primary region: {result['region']}")

        st.divider()
        if st.button("Generate Downloadable PDF Report"):
            patient_info = {
                "name": patient_name, "age": patient_age, "gender": patient_gender,
                "patient_id": patient_id, "notes": patient_notes,
            }
            with st.spinner("Building PDF..."):
                pdf_path = generate_pdf_report(result, patient_info, temp_path)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Report (PDF)", data=f.read(),
                                    file_name="xray_report.pdf", mime="application/pdf")
else:
    st.info("Upload an X-ray image to get started.")
