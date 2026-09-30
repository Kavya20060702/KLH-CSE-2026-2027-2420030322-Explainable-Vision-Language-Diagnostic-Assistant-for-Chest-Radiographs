# Explainable Vision-Language Diagnostic Assistant for Chest Radiographs

CPU-only and offline-first.

## Team Members
- P. Kavya Sai - 2420030322
- P. Meghna - 2420030401
- S. Anjana - 2420030499
- D. Poojitha - 2420030697

## Supervisor
Dr. K Swanthana

## Abstract
An offline, explainable vision-language assistant for chest X-rays that supports rural clinics by providing interpretable diagnoses and urgency guidance without needing a radiologist or internet.

## Phases
<img width="1168" height="559" alt="image" src="https://github.com/user-attachments/assets/3134dc26-f7b5-4725-83f7-78468f79f92c" />

## Current Status
Phase 1 – Data Prep & Alignment- Completed

## Why this design
- Vision model: ViT-B/32 (Vision Transformer, 32x32 patches), ImageNet-pretrained,
  with most transformer blocks frozen so fine-tuning is realistic on CPU. Patch
  size 32 (vs the more common 16) cuts self-attention cost roughly 4x by using
  49 patch tokens instead of 196 -- a deliberate CPU-feasibility tradeoff.
- Explainability: Grad-CAM on the last convolutional block -> heatmap overlay
  showing the region the model focused on.
- Report generation (the NLP core): predicted labels + Grad-CAM region are used
  to retrieve and adapt real radiology report sentences from the OpenI dataset
  (TF-IDF similarity). This gives clinically-worded, fluent text without needing
  to train a multi-billion-parameter VLM.
- Triage: a deterministic rule table (condition x confidence -> urgency tier).
  No model needed here on purpose -- it must be explainable and auditable.

## Folder layout
```
data/            put your downloaded datasets here (see below)
models/          saved model checkpoints go here
src/
  dataset.py     PyTorch Dataset for the NIH sample CSV
  model.py       ViT-B/32 classifier definition
  train.py       CPU-friendly training loop
  gradcam.py     Grad-CAM implementation + overlay utility
  report_index.py   builds a retrieval index from OpenI reports
  report_generator.py  turns predictions into a report using the index
  triage.py      rule-based urgency scoring
  inference.py   end-to-end: image in -> marked image + report + triage out
app/
  app.py         Streamlit demo UI
```

## Datasets you need to download yourself
(My sandbox cannot reach these hosts -- download locally, then copy into `data/`.)

1. **NIH ChestX-ray14 sample** (Kaggle "nih-chest-xrays-sample", ~5,606 images,
   much more CPU-training-friendly than the full 45GB set).
   Put images in `data/nih_sample/images/` and the labels CSV at
   `data/nih_sample/labels.csv`.

2. **OpenI (Indiana University) Chest X-Ray Collection** -- has real paired
   findings/impression report text. Put the reports (whatever export format
   you download, XML or CSV) in `data/openi/`. `src/report_index.py` has a
   loader stub you'll need to adjust to match the exact file format you get,
   since OpenI's download format varies.

## Order of operations (matches the 7-day plan)
1. `python src/dataset.py --check` -- verify your CSV/image paths line up
2. `python src/train.py` -- fine-tunes ViT-B/32 head on your subset
3. `python src/gradcam.py --test` -- sanity-check heatmaps on a few images
4. `python src/report_index.py` -- builds the OpenI retrieval index (one-time)
5. `python src/inference.py --image path/to/xray.png` -- full pipeline
6. `streamlit run app/app.py` -- interactive demo
