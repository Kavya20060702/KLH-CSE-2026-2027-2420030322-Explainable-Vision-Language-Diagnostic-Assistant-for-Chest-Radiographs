"""
Explainability for the ViT-B/32 classifier via attention visualization.
Rather than adapting CNN-style Grad-CAM to a transformer, this extracts
the CLS token's attention over patch tokens -- the standard way ViT
models are visualized, and more robust than gradient-based attribution
given this model's mostly-frozen backbone (see GradCAM class docstring).
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import eval_transform, CONDITIONS
from model import build_model
from paths import CHECKPOINT_PATH, OUTPUTS_DIR


class GradCAM:
    """
    Attention-based explainability for ViT. Rather than adapting
    gradient-based Grad-CAM (designed for CNN feature maps) to a
    transformer, this extracts the CLS token's attention distribution
    over patch tokens from the last transformer block -- the standard
    way Vision Transformers are visualized in the literature, since the
    CLS token is exactly what the classifier head reads to make its
    prediction. This also sidesteps a real failure mode observed with
    gradient-based Grad-CAM on this model: with only 2 of 12 transformer
    blocks fine-tuned, patch-token representations in the final block can
    collapse to near-identical values ("token homogenization"), leaving
    gradient-weighted activations with almost no spatial variation to
    visualize. Attention weights, being a normalized probability
    distribution over patches, don't suffer the same collapse.
    """

    def __init__(self, model, block_idx=-1):
        self.model = model
        self.model.eval()
        self.attn_weights = None
        target_block = model.encoder.layers[block_idx]
        self.attention_module = target_block.self_attention
        self.attention_module.register_forward_pre_hook(self._force_need_weights, with_kwargs=True)
        self.attention_module.register_forward_hook(self._save_attention, with_kwargs=True)

    def _force_need_weights(self, module, args, kwargs):
        # torchvision's EncoderBlock calls self_attention with
        # need_weights=False by default, which discards the attention
        # matrix entirely. Force it on so we can capture it below.
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return args, kwargs

    def _save_attention(self, module, args, kwargs, output):
        # output is (attn_output, attn_output_weights); weights shape
        # with average_attn_weights=True: (batch, seq_len, seq_len)
        self.attn_weights = output[1].detach()

    def generate(self, input_tensor, class_idx):
        with torch.no_grad():
            logits = self.model(input_tensor)

        if self.attn_weights is None:
            raise RuntimeError("Attention weights were not captured -- hook did not fire.")

        # Row 0 = CLS token's attention distribution over all tokens;
        # drop column 0 (CLS attending to itself) to keep only patches.
        cls_to_patches = self.attn_weights[0, 0, 1:]
        n_patches = cls_to_patches.numel()
        g = int(round(n_patches ** 0.5))
        cam = cls_to_patches.reshape(g, g).cpu().numpy()

        lo, hi = np.percentile(cam, 5), np.percentile(cam, 95)
        if hi > lo:
            cam = np.clip((cam - lo) / (hi - lo), 0, 1)
        else:
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
        return cam, torch.sigmoid(logits)[0, class_idx].item()


def overlay_heatmap(original_pil_image, cam, alpha=0.4):
    """Resize cam to the original image size and blend it in as a heatmap."""
    img = np.array(original_pil_image.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def contour_overlay(original_pil_image, cam, threshold=0.5, color=(255, 0, 0), thickness=3):
    """
    Draws a clean red outline around the region where the CAM is 'hot'
    (above `threshold`), on top of the plain original image -- distinct
    from the full heatmap wash, closer to how a radiologist annotation
    circles a finding.
    """
    img = np.array(original_pil_image.convert("RGB")).copy()
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    mask = (cam_resized >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return Image.fromarray(img)


def region_description(cam):
    """
    Rough quadrant description of where the CAM is concentrated, used later
    by the report generator (e.g. 'lower right lung field').
    """
    h, w = cam.shape
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    vertical = "upper" if y < h / 2 else "lower"
    # Note: X-rays are typically displayed with patient's left on the viewer's
    # right -- adjust this mapping if your dataset convention differs.
    horizontal = "right" if x < w / 2 else "left"
    return f"{vertical} {horizontal} lung field"


def run(image_path, checkpoint=CHECKPOINT_PATH, class_idx=None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    conditions = ckpt.get("conditions", CONDITIONS)
    model = build_model(num_classes=len(conditions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cam_engine = GradCAM(model)

    image = Image.open(image_path).convert("L")
    input_tensor = eval_transform(image).unsqueeze(0)
    input_tensor.requires_grad_(False)

    if class_idx is None:
        with torch.no_grad():
            probs = torch.sigmoid(model(input_tensor))[0]
        class_idx = int(torch.argmax(probs).item())

    cam, confidence = cam_engine.generate(input_tensor, class_idx)
    print(f"[diagnostic] cam stats: min={cam.min():.4f} max={cam.max():.4f} std={cam.std():.4f}")
    overlay = overlay_heatmap(image, cam)
    region = region_description(cam)

    print(f"Predicted: {conditions[class_idx]} (confidence {confidence:.2f})")
    print(f"Region of concern: {region}")
    out_path = os.path.join(OUTPUTS_DIR, "gradcam_overlay.png")
    overlay.save(out_path)
    print(f"Saved overlay to {out_path}")

    return overlay, conditions[class_idx], confidence, region


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=False)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if args.image:
        run(args.image, checkpoint=args.checkpoint)
    elif args.test:
        print("Pass --image path/to/xray.png to test Grad-CAM on a real file.")