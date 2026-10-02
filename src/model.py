"""
Vision Transformer (ViT-B/16) classifier.

The deployed checkpoint is trained via full fine-tuning (all parameters
trainable) on a cloud GPU (Kaggle T4/P100) against the complete NIH
ChestX-ray14 dataset (112,120 images, 15 classes) -- see
kaggle/train_vit_kaggle.py. Patch size /16 (196 patches) is used here
since full-dataset, full-fine-tune GPU training doesn't have the same
per-token attention-cost constraint that motivated the earlier /32
CPU-only prototype.

`freeze_backbone=True` is kept as an option for anyone later doing
CPU-only local fine-tuning (e.g. a smaller experiment or ablation) --
it reproduces the earlier lightweight-training approach. The main
deployed checkpoint does NOT use this; it was fully fine-tuned.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_model(num_classes, freeze_backbone=False, num_trainable_blocks=2):
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for block in model.encoder.layers[-num_trainable_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.encoder.ln.parameters():
            param.requires_grad = True
    # else: full fine-tuning, all parameters trainable (matches the
    # deployed Kaggle-trained checkpoint).

    # Replace classifier head (always trainable). hidden_dim is 768 for
    # both ViT-B/16 and ViT-B/32 ("Base" variants), so this is unaffected
    # by the patch-size choice.
    hidden_dim = model.hidden_dim
    model.heads = nn.Linear(hidden_dim, num_classes)

    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model(num_classes=15)
    print(f"Trainable params: {count_trainable_params(m):,}")