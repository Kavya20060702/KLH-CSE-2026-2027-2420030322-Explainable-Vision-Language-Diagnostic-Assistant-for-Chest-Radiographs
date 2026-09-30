"""
Turns (predicted condition, confidence, Grad-CAM region) into a readable
report. Tries retrieval from real OpenI report text first (if the index
exists); falls back to a clean template if the index isn't built yet or
nothing relevant is found. Always keep the template fallback -- it's what
guarantees the demo works even if the retrieval index has issues.
"""

import os
import pickle
import re

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from paths import REPORT_INDEX_PATH


def clean_report_text(text):
    """
    IU/OpenI reports use 'XXXX' as a de-identification placeholder for
    redacted terms. Strip it out for readability in the final report.
    """
    text = re.sub(r"\bXXXX\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,])", r"\1", text)
    return text


TEMPLATE = (
    "Findings suggest {condition_phrase}, most notably in the {region}. "
    "Model confidence: {confidence_pct}%. {urgency_note}"
)

CONDITION_PHRASES = {
    "No Finding": "no significant abnormality",
    "Infiltration": "an area of pulmonary infiltration",
    "Effusion": "findings consistent with pleural effusion",
    "Atelectasis": "an area of atelectasis (partial lung collapse)",
    "Nodule": "a pulmonary nodule",
    "Mass": "a pulmonary mass requiring further evaluation",
    "Pneumothorax": "findings suggestive of pneumothorax",
    "Consolidation": "an area of pulmonary consolidation",
    "Pleural_Thickening": "pleural thickening",
    "Cardiomegaly": "cardiac silhouette enlargement (cardiomegaly)",
    "Pneumonia": "signs consistent with pneumonia",
}

# IU/OpenI's "Problems" field uses MeSH-style terms that don't always match
# our classifier's label names exactly (e.g. "Pleural Effusion" vs "Effusion").
# Map each of our labels to the search terms likely to appear in that field.
CONDITION_SYNONYMS = {
    "No Finding": ["normal"],
    "Infiltration": ["infiltrate", "infiltration", "opacity"],
    "Effusion": ["effusion", "pleural effusion"],
    "Atelectasis": ["atelectasis", "atelectatic"],
    "Nodule": ["nodule", "nodular"],
    "Mass": ["mass", "neoplasm"],
    "Pneumothorax": ["pneumothorax"],
    "Consolidation": ["consolidation", "consolidative"],
    "Pleural_Thickening": ["pleural thickening", "thickening", "pleural"],
    "Cardiomegaly": ["cardiomegaly", "cardiac"],
    "Pneumonia": ["pneumonia"],
}

# General, non-prescriptive safety-net guidance for the wait before a
# clinician is seen. Deliberately avoids medication names/dosages or
# specific treatment instructions -- this is a triage aid, not a
# treatment plan, and always pairs with the "see a doctor" triage message.
PRECAUTIONS = {
    "No Finding": "No specific precautions needed based on this image. Continue routine health monitoring.",
    "Infiltration": "Rest and stay hydrated. Watch for fever, worsening cough, or shortness of breath. Avoid smoke or dust exposure.",
    "Effusion": "Avoid strenuous activity. Sitting upright may ease breathing. Seek urgent care if breathlessness or chest pain increases.",
    "Atelectasis": "Avoid smoking and strenuous exertion. Seek care promptly if breathing becomes more difficult.",
    "Nodule": "This often requires follow-up imaging rather than immediate action, but should still be reviewed by a clinician. Try not to be alarmed -- many nodules are benign.",
    "Mass": "Seek prompt clinical evaluation without delay. Note any weight loss, persistent cough, or chest pain to report to the doctor.",
    "Pneumothorax": "Avoid strenuous activity and air travel until evaluated. Seek IMMEDIATE care if sudden chest pain or severe breathlessness occurs.",
    "Consolidation": "Rest, stay hydrated, and monitor temperature. Seek urgent care if breathing difficulty increases.",
    "Pleural_Thickening": "Follow up with a clinician for further evaluation. Avoid smoke exposure in the meantime.",
    "Cardiomegaly": "Avoid strenuous exertion. Watch for leg swelling, breathlessness, or chest discomfort. Limit salty foods until evaluated.",
    "Pneumonia": "Rest, stay hydrated, and monitor temperature regularly. Seek IMMEDIATE care if breathing becomes difficult or lips/fingertips look bluish.",
}

PRECAUTION_DISCLAIMER = (
    "These are general precautions only, not a treatment plan, and do not "
    "replace professional medical evaluation. Please consult a healthcare "
    "provider as indicated by the triage recommendation above."
)


def load_index(index_path=REPORT_INDEX_PATH):
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve_similar_report(index, condition, top_k=1):
    """Find the most similar real report mentioning this condition, if any."""
    if index is None:
        return None
    reports = index["reports"]

    synonyms = CONDITION_SYNONYMS.get(condition, [condition])
    mask = pd.Series(False, index=reports.index)
    for term in synonyms:
        mask = mask | reports["labels"].str.contains(term, case=False, na=False)

    if not mask.any():
        return None

    query_vec = index["vectorizer"].transform([" ".join(synonyms)])
    subset_matrix = index["tfidf_matrix"][mask.values]
    sims = cosine_similarity(query_vec, subset_matrix)[0]
    best_idx = sims.argmax()
    matched_reports = reports[mask].reset_index(drop=True)
    return matched_reports.iloc[best_idx]["text"]


def truncate_at_sentence(text, max_chars=350):
    """Cut retrieved report text at a sentence boundary instead of a hard
    character cutoff, so snippets don't end mid-word/mid-clause."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > 80:  # avoid cutting too short if punctuation is sparse
        return truncated[: last_period + 1]
    return truncated.rstrip() + "..."


def generate_report(condition, confidence, region, urgency_note="", index=None):
    condition_phrase = CONDITION_PHRASES.get(condition, condition.lower())
    confidence_pct = round(confidence * 100, 1)
    precaution = PRECAUTIONS.get(condition, "Please consult a healthcare provider for guidance.")

    retrieved = retrieve_similar_report(index, condition) if index else None

    if retrieved:
        # Ground the retrieved real report text with our own structured facts
        # up front, so the output stays tied to this specific prediction.
        cleaned = truncate_at_sentence(clean_report_text(retrieved))
        report = (
            f"AI-assisted finding: {condition_phrase}, focused in the {region} "
            f"(confidence {confidence_pct}%).\n\n"
            f"Similar documented finding pattern: {cleaned}\n\n"
            f"{urgency_note}\n\n"
            f"Precautions before seeing a doctor: {precaution}\n"
            f"{PRECAUTION_DISCLAIMER}"
        )
    else:
        report = TEMPLATE.format(
            condition_phrase=condition_phrase,
            region=region,
            confidence_pct=confidence_pct,
            urgency_note=urgency_note,
        )
        report += f"\n\nPrecautions before seeing a doctor: {precaution}\n{PRECAUTION_DISCLAIMER}"

    return report


def generate_multi_condition_report(flagged_conditions, region, triage_result, index=None):
    """
    Builds a structured, multi-section report covering every condition that
    cleared its threshold (not just the single highest-confidence one).

    flagged_conditions: list of (condition_name, confidence) tuples, sorted
                         by confidence descending. The first entry is treated
                         as the primary finding (the one Grad-CAM visualized).
    triage_result: dict from triage.assess_multiple() -- the combined,
                   worst-case urgency across all flagged conditions.
    """
    if not flagged_conditions:
        return generate_report("No Finding", 1.0, region, "", index=index)

    lines = ["## AI-Assisted Findings\n"]
    for cond, conf in flagged_conditions:
        phrase = CONDITION_PHRASES.get(cond, cond.lower())
        lines.append(f"- **{cond}**: {phrase} (confidence {round(conf*100, 1)}%)")

    primary_cond, primary_conf = flagged_conditions[0]
    lines.append(f"\nPrimary finding region (Grad-CAM): {region}\n")

    lines.append("## Similar Documented Patterns\n")
    seen_precautions = []
    for cond, conf in flagged_conditions:
        retrieved = retrieve_similar_report(index, cond) if index else None
        if retrieved:
            snippet = truncate_at_sentence(clean_report_text(retrieved), max_chars=250)
            lines.append(f"- **{cond}**: {snippet}")
        precaution = PRECAUTIONS.get(cond)
        if precaution and precaution not in seen_precautions:
            seen_precautions.append(precaution)

    lines.append(f"\n## Triage\n{triage_result['tier']} -- {triage_result['message']}\n")

    lines.append("## Precautions Before Seeing a Doctor\n")
    for p in seen_precautions:
        lines.append(f"- {p}")
    lines.append(f"\n{PRECAUTION_DISCLAIMER}")

    return "\n".join(lines)


if __name__ == "__main__":
    idx = load_index()
    print(generate_report("Pneumonia", 0.81, "lower right lung field", index=idx))