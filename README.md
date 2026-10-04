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

# Explainable Chest X-Ray Diagnostic Assistant

An AI-assisted screening tool for chest radiographs, built for low-resource
clinics without a radiologist on site. A health worker uploads a chest X-ray
and receives: a predicted condition (or conditions), a visual explanation of
which region of the image drove the prediction, a readable report grounded in
real radiology text, an independent cross-check from a second model, and a
deterministic triage/urgency recommendation.

**This is a decision-support prototype, not a diagnostic device.** All
outputs are intended to help a non-specialist health worker decide whether
and how urgently to refer a patient to a doctor -- not to replace one.

## Current results

The classifier (ViT-B/16) was fully fine-tuned on the complete NIH
ChestX-ray14 dataset (112,120 images, 15 classes) on a Kaggle GPU, then
evaluated on the official, patient-disjoint test split (25,596 images).

| Metric | Value |
|---|---|
| Mean AUROC (15 classes) | **0.7884** |
| Mean AUROC (14 classes, excluding "No Finding" -- directly comparable to CheXNet) | **0.7926** |
| CheXNet published mean AUROC (Rajpurkar et al., 2017), for reference | 0.841 |

Strongest classes: Cardiomegaly (0.889), Emphysema (0.870), Edema (0.847),
Pneumothorax (0.844). Weakest: Infiltration (0.699), Pneumonia (0.713),
Nodule (0.720) -- consistent with known difficulty patterns for these
conditions in the wider chest X-ray classification literature.

Precision/recall/F1 per class: see `models/thresholds.json`-based
evaluation output from `kaggle/addendum_same_session.py` or
`kaggle/addendum_standalone.py`.

## How it works

1. **Classification** -- ViT-B/16 (Vision Transformer), ImageNet-pretrained,
   fully fine-tuned on NIH ChestX-ray14. Outputs a probability for each of
   15 conditions (14 pathologies + "No Finding").
2. **Explainability** -- attention-based visualization: the CLS token's
   attention over image patches (final transformer block) is extracted and
   rendered as a heatmap and a red-outline "marked region." This is used
   instead of CNN-style Grad-CAM, which can produce degenerate, near-flat
   heatmaps on lightly fine-tuned transformer backbones.
3. **Report generation** -- for every condition that clears its tuned
   threshold, the system retrieves the most similar real radiology report
   from the Indiana University / OpenI corpus (7,426 reports) via TF-IDF
   similarity, grounding the generated text in authentic clinical language
   rather than free-form generation.
4. **Independent cross-checks**:
   - **CLIP** (zero-shot, no fine-tuning) compares the image against short
     text descriptions of each condition as a second, architecturally
     different opinion.
   - **BLIP v1** (*not* BLIP-2 -- see note below) produces a general
     image caption as supplementary context.
5. **Triage** -- a deterministic rule table maps the most severe flagged
   condition to an urgency tier (Routine / Soon / Urgent / Immediate),
   deliberately rule-based rather than a second learned model, so the
   recommendation stays auditable.
6. **PDF export** -- a downloadable report combining patient details, all
   three images, findings, triage, precautions, and a technical appendix
   for reviewing clinical staff.

### Why BLIP v1, not BLIP-2

BLIP-2 requires a multi-billion-parameter language-model backbone
(>=3.7B params, ~15GB+ RAM just to load), which doesn't fit the CPU-only,
limited-RAM deployment target this project is built for. BLIP v1
(~220M params, ~1-2GB RAM) is used instead as a disclosed, deliberate
engineering tradeoff. It is pretrained on natural photographs, not
radiographs, so its captions are supplementary context, not a diagnostic
claim -- the report's actual findings come from the ViT classifier and
the retrieval-based report generator.

## Repository structure

```
app/
  app.py                 Streamlit demo UI
src/
  paths.py                Shared path resolution (models/, outputs/)
  dataset.py               Dataset loader, 15-class label list
  model.py                  ViT-B/16 model definition
  gradcam.py                 Attention-based explainability
  report_index.py             Builds the TF-IDF retrieval index (IU/OpenI)
  report_generator.py          Report text generation + precautions
  triage.py                     Rule-based urgency scoring
  clip_verify.py                 Zero-shot CLIP cross-check
  blip_caption.py                 BLIP v1 captioning
  report_pdf.py                    Downloadable PDF report builder
  inference.py                      End-to-end pipeline
kaggle/
  train_vit_kaggle.py        Full-dataset GPU training script (run on Kaggle)
  addendum_same_session.py    Precision/recall/F1 addendum (same session)
  addendum_standalone.py       Precision/recall/F1 addendum (new session)
models/                   Trained checkpoint, tuned thresholds, retrieval index
outputs/                  Generated images/reports land here
data/                     Local datasets (not included in repo)
```

## Setup (local, CPU-only inference)

```bash
cd xray-assistant
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download CLIP and BLIP v1 once (requires internet; cached afterward):

```bash
cd src
python download_models.py
```

## Training the classifier (Kaggle GPU, not local)

CPU-only hardware cannot realistically fine-tune a Vision Transformer on
the full 112,120-image NIH dataset. Training is done on a free Kaggle GPU
notebook instead:

1. New Kaggle Notebook -> Settings -> Accelerator: GPU, Internet: On
2. Add Data -> "NIH Chest X-rays" (`nih-chest-xrays/data`)
3. Paste `kaggle/train_vit_kaggle.py` into one cell and run
4. Download the three output files from `/kaggle/working/`:
   `chest_classifier.pt`, `thresholds.json`, `test_metrics.json`
5. Place the first two into this project's `models/` folder (overwriting
   any existing ones). `test_metrics.json` is for reference/reporting only
   -- the app doesn't read it.

For precision/recall/F1 (not computed by the main training script), run
`kaggle/addendum_same_session.py` (if that Kaggle session is still open)
or `kaggle/addendum_standalone.py` (fresh session, needs the checkpoint
and thresholds uploaded as a Kaggle Dataset).

## Building the report-retrieval index (one-time)

Requires the Indiana University / OpenI dataset
(`indiana_reports.csv` + `indiana_projections.csv`):

```bash
cd src
python report_index.py --reports path\to\indiana_reports.csv --projections path\to\indiana_projections.csv
```

## Running the app

```bash
streamlit run app\app.py
```

Upload a chest X-ray, review the prediction/heatmap/report/triage, and
optionally generate a downloadable PDF.

## Known limitations

- **Class imbalance**: rarer conditions (e.g. Hernia, Pneumonia) have far
  fewer training examples than common ones, limiting achievable recall
  despite class-weighted loss.
- **Attention-based explainability** is more robust than gradient-based
  Grad-CAM on this architecture but is a coarser signal than pixel-level
  attribution methods; it can occasionally attend to non-anatomical image
  artifacts (e.g. positioning markers).
- **BLIP v1 captions** are general-purpose, not radiology-specific -- treat
  them as supplementary color, not a finding.
- **Single-source training data**: NIH ChestX-ray14 originates from one
  health system; performance on images from other scanners/populations is
  not yet independently validated.
- **CPU inference speed**: BLIP v1 captioning in particular can take several
  seconds to ~20s per image on CPU, since generation is token-by-token.

## Datasets used

- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) -- classifier
  training/evaluation
- [Indiana University / OpenI](https://openi.nlm.nih.gov/faq#collection) --
  retrieval-based report generation only (not used for classifier training)

## Order of operations (matches the 7-day plan)
1. `python src/dataset.py --check` -- verify your CSV/image paths line up
2. `python src/train.py` -- fine-tunes ViT-B/32 head on your subset
3. `python src/gradcam.py --test` -- sanity-check heatmaps on a few images
4. `python src/report_index.py` -- builds the OpenI retrieval index (one-time)
5. `python src/inference.py --image path/to/xray.png` -- full pipeline
6. `streamlit run app/app.py` -- interactive demo
