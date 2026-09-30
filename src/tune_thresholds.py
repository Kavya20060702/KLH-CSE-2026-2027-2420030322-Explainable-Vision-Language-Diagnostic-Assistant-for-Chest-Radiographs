"""
Per-class threshold tuning. A flat 0.5 cutoff is rarely optimal across
imbalanced classes -- some rare-but-learned classes may need a lower
threshold to ever fire. This finds the threshold that maximizes F1 for
each class individually, using the same held-out validation split as
evaluate.py. No retraining required.

Usage:
    python tune_thresholds.py --checkpoint models/chest_classifier.pt
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, eval_transform
from model import build_model
from paths import CHECKPOINT_PATH, THRESHOLDS_PATH


def tune(
    checkpoint=CHECKPOINT_PATH,
    csv_path=r"D:\Datasets\nih\sample_labels.csv",
    image_dir=r"D:\Datasets\nih\sample\images",
    val_split=0.15,
    out_json=THRESHOLDS_PATH,
):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    conditions = ckpt["conditions"]
    model = build_model(num_classes=len(conditions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    full_ds = ChestXrayDataset(csv_path, image_dir, transform=eval_transform, conditions=conditions)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(42)
    _, val_idx = torch.utils.data.random_split(range(len(full_ds)), [n_train, n_val], generator=generator)
    val_ds = torch.utils.data.Subset(full_ds, val_idx.indices)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            probs = torch.sigmoid(model(images))
            all_probs.append(probs)
            all_targets.append(targets)
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    thresholds = {}
    print(f"{'Condition':22s} {'best_thr':>9s} {'F1@0.5':>8s} {'F1@best':>8s}")
    for i, cond in enumerate(conditions):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            # No positive examples in val set for this class -- can't tune meaningfully.
            thresholds[cond] = 0.5
            print(f"{cond:22s} {'n/a':>9s} {'n/a':>8s} {'n/a':>8s} (no positives in val set)")
            continue

        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        # thr has len = len(precision)-1; align by dropping last precision/recall point
        f1_scores = np.where(
            (precision[:-1] + recall[:-1]) > 0,
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12),
            0,
        )
        if len(f1_scores) == 0:
            thresholds[cond] = 0.5
            continue

        best_idx = f1_scores.argmax()
        best_thr = float(thr[best_idx])
        f1_at_best = f1_scores[best_idx]
        f1_at_half = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)

        thresholds[cond] = best_thr
        print(f"{cond:22s} {best_thr:9.3f} {f1_at_half:8.3f} {f1_at_best:8.3f}")

    with open(out_json, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved tuned thresholds to {out_json}")

    # Show overall report using tuned thresholds instead of flat 0.5
    thr_array = np.array([thresholds[c] for c in conditions])
    preds_tuned = (probs >= thr_array).astype(int)
    print("\n--- Evaluation report using TUNED thresholds ---")
    print(classification_report(targets, preds_tuned, target_names=conditions, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--csv", default=r"D:\Datasets\nih\sample_labels.csv")
    parser.add_argument("--images", default=r"D:\Datasets\nih\sample\images")
    args = parser.parse_args()
    tune(checkpoint=args.checkpoint, csv_path=args.csv, image_dir=args.images)