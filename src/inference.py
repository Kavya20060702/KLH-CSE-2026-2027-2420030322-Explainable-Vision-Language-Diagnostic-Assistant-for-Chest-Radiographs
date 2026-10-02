"""
End-to-end pipeline: X-ray image in -> marked-up image + readable report +
urgency tier out.
"""

import argparse
import json
import os

import torch
from PIL import Image

from dataset import eval_transform, CONDITIONS
from model import build_model
from gradcam import GradCAM, overlay_heatmap, contour_overlay, region_description
from report_generator import generate_multi_condition_report, load_index
from triage import assess_multiple
from paths import CHECKPOINT_PATH, THRESHOLDS_PATH, REPORT_INDEX_PATH, OUTPUTS_DIR


def load_thresholds(conditions, path=THRESHOLDS_PATH):
    """Per-class thresholds, falling back to 0.5 for any class not covered."""
    if not os.path.exists(path):
        return {c: 0.5 for c in conditions}
    with open(path) as f:
        tuned = json.load(f)
    return {c: tuned.get(c, 0.5) for c in conditions}


def run_pipeline(
    image_path,
    checkpoint=CHECKPOINT_PATH,
    report_index_path=REPORT_INDEX_PATH,
):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    conditions = ckpt.get("conditions", CONDITIONS)
    model = build_model(num_classes=len(conditions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image = Image.open(image_path).convert("L")
    input_tensor = eval_transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0]

    class_thresholds = load_thresholds(conditions)

    flagged = []
    for i, c in enumerate(conditions):
        if c != "No Finding" and probs[i].item() >= class_thresholds[c]:
            flagged.append((c, probs[i].item()))
    flagged.sort(key=lambda x: x[1], reverse=True)

    if flagged:
        primary_idx = conditions.index(flagged[0][0])
        primary_condition, primary_confidence = flagged[0]
    else:
        primary_idx = conditions.index("No Finding") if "No Finding" in conditions else int(torch.argmax(probs).item())
        primary_condition = conditions[primary_idx]
        primary_confidence = probs[primary_idx].item()

    cam_engine = GradCAM(model)
    cam, _ = cam_engine.generate(input_tensor, primary_idx)
    overlay = overlay_heatmap(image, cam)
    contour_image = contour_overlay(image, cam)
    region = region_description(cam)

    triage_result = assess_multiple(flagged)

    report_index = load_index(report_index_path)
    report_text = generate_multi_condition_report(
        flagged, region, triage_result, index=report_index,
    )

    all_probs = {conditions[i]: probs[i].item() for i in range(len(conditions))}

    clip_scores = None
    try:
        from clip_verify import clip_agreement
        clip_scores = clip_agreement(image_path, conditions)
    except Exception as e:
        print(f"[warning] CLIP verification skipped: {e}")

    blip_caption = None
    try:
        from blip_caption import generate_caption
        blip_caption = generate_caption(image_path)
    except Exception as e:
        print(f"[warning] BLIP captioning skipped: {e}")

    return {
        "condition": primary_condition,
        "confidence": primary_confidence,
        "all_flagged": flagged,
        "all_probs": all_probs,
        "region": region,
        "overlay_image": overlay,
        "contour_image": contour_image,
        "report": report_text,
        "triage": triage_result,
        "clip_scores": clip_scores,
        "blip_caption": blip_caption,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    result = run_pipeline(args.image, checkpoint=args.checkpoint)

    out_path = os.path.join(OUTPUTS_DIR, "marked_xray.png")
    result["overlay_image"].save(out_path)

    print(f"Condition: {result['condition']} ({result['confidence']:.2f})")
    print(f"Triage: {result['triage']['tier']}")
    print("\nReport:\n" + result["report"])
    print(f"\nMarked-up image saved to {out_path}")