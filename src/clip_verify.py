"""
Zero-shot CLIP verification. Independently checks the ViT classifier's
prediction by comparing the X-ray image against short text descriptions
of each candidate condition using CLIP's joint image-text embedding
space. This is a genuinely separate model with different pretraining
than the ViT classifier -- no fine-tuning is done here (CLIP is used
purely zero-shot), so it acts as an independent cross-check rather than
a second trained classifier.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_model = None
_processor = None

CLIP_PROMPTS = {
    "No Finding": "a normal chest x-ray with no abnormality",
    "Infiltration": "a chest x-ray showing pulmonary infiltration",
    "Effusion": "a chest x-ray showing pleural effusion",
    "Atelectasis": "a chest x-ray showing atelectasis, a partially collapsed lung",
    "Nodule": "a chest x-ray showing a pulmonary nodule",
    "Mass": "a chest x-ray showing a pulmonary mass",
    "Pneumothorax": "a chest x-ray showing pneumothorax",
    "Consolidation": "a chest x-ray showing pulmonary consolidation",
    "Pleural_Thickening": "a chest x-ray showing pleural thickening",
    "Cardiomegaly": "a chest x-ray showing cardiomegaly, an enlarged heart",
    "Pneumonia": "a chest x-ray showing pneumonia",
    "Edema": "a chest x-ray showing pulmonary edema",
    "Emphysema": "a chest x-ray showing emphysema",
    "Fibrosis": "a chest x-ray showing pulmonary fibrosis",
    "Hernia": "a chest x-ray showing a diaphragmatic hernia",
}


def _load():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
    return _model, _processor


def clip_agreement(image_path, conditions):
    model, processor = _load()
    image = Image.open(image_path).convert("RGB")
    texts = [CLIP_PROMPTS.get(c, f"a chest x-ray showing {c.lower()}") for c in conditions]

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    return {c: probs[i].item() for i, c in enumerate(conditions)}