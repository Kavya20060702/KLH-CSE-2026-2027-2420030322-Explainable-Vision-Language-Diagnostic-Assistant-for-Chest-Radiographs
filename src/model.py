"""
Vision Transformer (ViT-B/32) classifier, ImageNet-pretrained, with most
transformer blocks frozen so fine-tuning is realistic on CPU. Only the
last few encoder blocks + classifier head are trainable.

Using the /32 patch-size variant (not /16) deliberately: self-attention
cost scales with the square of the number of patch tokens, and /32 gives
49 tokens vs /16's 196 -- roughly a 4x cheaper attention computation,
which matters a lot on CPU. Still a genuine ViT architecture.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights


def build_model(num_classes, num_trainable_blocks=2):
    weights = ViT_B_32_Weights.IMAGENET1K_V1
    model = vit_b_32(weights=weights)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the last `num_trainable_blocks` transformer encoder
    # blocks, plus the final layer norm and classifier head.
    for block in model.encoder.layers[-num_trainable_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.encoder.ln.parameters():
        param.requires_grad = True

    # Replace classifier head (always trainable)
    hidden_dim = model.hidden_dim
    model.heads = nn.Linear(hidden_dim, num_classes)

    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model(num_classes=11)
    print(f"Trainable params: {count_trainable_params(m):,}")