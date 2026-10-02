"""
Builds a downloadable PDF report combining everything a patient and a
reviewing doctor need.
"""

import os
from datetime import datetime

from fpdf import FPDF

from paths import OUTPUTS_DIR


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "AI-Assisted Chest X-Ray Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.multi_cell(
            0, 4,
            "This is an AI-assisted screening tool intended to support, not replace, "
            "professional medical evaluation. All findings should be reviewed by a "
            "qualified clinician.",
            align="C",
        )

    def section_title(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(235, 240, 245)
        self.cell(0, 8, text, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5.5, text)
        self.ln(1)


TRIAGE_COLORS = {
    "Routine": (76, 175, 80),
    "Soon": (33, 150, 243),
    "Urgent": (255, 152, 0),
    "Immediate": (244, 67, 54),
}


def generate_pdf_report(result, patient_info, original_image_path, out_path=None):
    if out_path is None:
        out_path = os.path.join(OUTPUTS_DIR, "xray_report.pdf")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.section_title("Patient Information")
    pdf.set_font("Helvetica", "", 10)
    info_rows = [
        ("Name", patient_info.get("name") or "Not provided"),
        ("Age", patient_info.get("age") or "Not provided"),
        ("Gender", patient_info.get("gender") or "Not provided"),
        ("Patient / Case ID", patient_info.get("patient_id") or "Not provided"),
    ]
    for label, value in info_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, f"{label}:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, str(value), ln=True)
    if patient_info.get("notes"):
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Notes: {patient_info['notes']}")
    pdf.ln(3)

    pdf.section_title("X-Ray Images")
    img_w = 58
    y_before = pdf.get_y()
    heatmap_path = os.path.join(OUTPUTS_DIR, "_pdf_heatmap_temp.png")
    contour_path = os.path.join(OUTPUTS_DIR, "_pdf_contour_temp.png")
    result["overlay_image"].save(heatmap_path)
    result["contour_image"].save(contour_path)

    pdf.image(original_image_path, x=10, y=y_before, w=img_w)
    pdf.image(heatmap_path, x=10 + img_w + 4, y=y_before, w=img_w)
    pdf.image(contour_path, x=10 + 2 * (img_w + 4), y=y_before, w=img_w)
    pdf.set_y(y_before + img_w / 1.0 + 2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(img_w, 5, "Original", align="C")
    pdf.cell(img_w, 5, "Heatmap (attention)", align="C")
    pdf.cell(img_w, 5, "Marked region", align="C")
    pdf.ln(10)

    pdf.section_title("AI-Assisted Findings")
    if result["all_flagged"]:
        for cond, conf in result["all_flagged"]:
            pdf.body_text(f"- {cond}: confidence {round(conf * 100, 1)}%")
    else:
        pdf.body_text("No conditions were flagged above the detection threshold (No Finding).")
    pdf.body_text(f"Primary finding region (attention-based): {result['region']}")

    tier = result["triage"]["tier"]
    color = TRIAGE_COLORS.get(tier, (0, 0, 0))
    pdf.section_title("Triage Recommendation")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, tier, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.body_text(result["triage"]["message"])

    pdf.section_title("Precautions Before Seeing a Doctor")
    seen = set()
    from report_generator import PRECAUTIONS
    for cond, _ in result["all_flagged"]:
        p = PRECAUTIONS.get(cond)
        if p and p not in seen:
            pdf.body_text(f"- {p}")
            seen.add(p)
    if not result["all_flagged"]:
        pdf.body_text(PRECAUTIONS.get("No Finding", ""))

    pdf.add_page()
    pdf.section_title("Technical Notes (For Clinical Staff)")
    pdf.body_text(
        "Model: ViT-B/16 (Vision Transformer), ImageNet-pretrained, fully "
        "fine-tuned on the complete NIH ChestX-ray14 dataset (112,120 images, "
        "15 classes) via cloud GPU training. Explainability: attention-based "
        "visualization (CLS token attention over image patches from the final "
        "transformer block). Report text for each flagged condition is "
        "retrieved from similar real radiology reports (Indiana University / "
        "OpenI corpus) via TF-IDF similarity, not generated by an "
        "unconstrained language model."
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Per-class model confidence:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for cond, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        pdf.cell(0, 5, f"  {cond}: {round(prob * 100, 1)}%", ln=True)

    if result.get("clip_scores"):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "CLIP zero-shot cross-check (independent model):", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for cond, prob in sorted(result["clip_scores"].items(), key=lambda x: -x[1])[:3]:
            pdf.cell(0, 5, f"  {cond}: {round(prob * 100, 1)}%", ln=True)

    if result.get("blip_caption"):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "BLIP v1 general image caption (not medical-specific):", ln=True)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, result["blip_caption"])

    pdf.output(out_path)

    for p in (heatmap_path, contour_path):
        if os.path.exists(p):
            os.remove(p)

    return out_path
