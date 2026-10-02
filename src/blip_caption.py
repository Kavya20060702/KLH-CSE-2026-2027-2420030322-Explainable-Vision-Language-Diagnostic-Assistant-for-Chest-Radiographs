"""
Image captioning using BLIP (v1) -- explicitly NOT BLIP-2. BLIP-2 needs
a >=3.7B-parameter language model backbone (~15GB+ RAM), infeasible for
CPU-only deployment. BLIP v1 (~220M params, ~1GB RAM) is the genuine
alternative used here. It is pretrained on natural photos, NOT medical
images, so its captions are supplementary, not diagnostic.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _model.eval()
    return _model, _processor


def generate_caption(image_path, max_new_tokens=30):
    model, processor = _load()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True)
