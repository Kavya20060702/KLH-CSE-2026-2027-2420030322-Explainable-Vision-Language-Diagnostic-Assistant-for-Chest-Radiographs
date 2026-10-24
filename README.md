# Explainable Vision-Language Diagnostic Assistant for Chest Radiographs

### From Image to Insight: An Explainable Multimodal Framework for AI-Assisted Chest X-ray Analysis and Triage

An AI-assisted clinical decision-support prototype that combines deep learning, vision-language models, explainable AI, clinical-report retrieval, and deterministic triage to support preliminary chest radiograph assessment in rural and resource-constrained healthcare environments.

The framework integrates **ViT-B/16, CLIP, BLIP v1, attention-based visual explanations, TF-IDF report retrieval, and rule-based triage** into a unified workflow with an interactive Streamlit application and downloadable PDF reports.

> **Important clinical safety notice:** This project is a research prototype for AI-assisted screening, preliminary reporting, and triage support. It is not an autonomous diagnostic system and must not replace radiologists, physicians, or established clinical workflows. Its outputs require review by qualified healthcare professionals.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Research Problem](#-research-problem)
- [Research Objectives](#-research-objectives)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Technology Stack](#-technology-stack)
- [Dataset Description](#-dataset-description)
- [Model Architecture and Responsibilities](#-model-architecture-and-responsibilities)
- [Evaluation Results](#-evaluation-results)
- [Ablation Study Plan](#-ablation-study-plan)
- [Explainability and Clinical Safety](#-explainability-and-clinical-safety)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Model Setup](#-model-setup)
- [Training](#-training)
- [Building the Report Retrieval Index](#-building-the-report-retrieval-index)
- [Running the Application](#-running-the-application)
- [Example Outputs](#-example-outputs)
- [Deployment and Offline Capability](#-deployment-and-offline-capability)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Team Members](#-team-members)
- [References](#-references)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

Access to specialist radiological interpretation remains a challenge in many rural and low-resource healthcare environments. Even where chest X-ray equipment is available, delays in expert interpretation can affect the speed of preliminary clinical assessment.

This project explores an explainable multimodal AI framework designed to assist healthcare professionals by bringing several complementary capabilities into a single workflow.

Instead of relying exclusively on one image classifier or an unconstrained report-generation model, the proposed framework combines:

- **Visual classification** to estimate the likelihood of multiple chest radiograph abnormalities.
- **Vision-language cross-verification** to compare visual information with text-based disease descriptions.
- **Visual explanation** to highlight image regions influencing the classifier.
- **Image captioning** to provide supplementary descriptions of image content.
- **Retrieval-grounded clinical context** to find relevant examples from existing radiology reports.
- **Deterministic triage** to assign a preliminary urgency category using explicit rules.
- **Structured reporting** to consolidate the available evidence into a preliminary clinical-support report.

The principal research contribution is the **integration and orchestration of complementary AI components into an explainable, auditable, and deployment-conscious framework**.

The framework is intended to support—not replace—professional interpretation and clinical decision-making.

## 🎯 Research Problem

Conventional chest X-ray classification systems primarily focus on predicting abnormalities from images. Although useful for screening research, a standalone prediction may not provide the contextual information, visual evidence, uncertainty handling, and reporting support needed for practical clinical workflows.

Existing approaches may also face challenges involving:

1. Limited interpretability of model predictions.
2. Insufficient integration of visual and textual evidence.
3. Dependence on unconstrained generated text for reporting.
4. Limited transparency in urgency assignment.
5. Computational constraints in rural and low-resource settings.
6. The need for qualified human review and clinical validation.

This project investigates whether integrating classification, explainability, vision-language analysis, retrieval, and deterministic triage can provide a more transparent preliminary assessment workflow.

## 🎯 Research Objectives

- Develop a multi-label chest radiograph classification pipeline using a Vision Transformer.
- Integrate visual explanations to improve the interpretability of model predictions.
- Investigate CLIP-based image-text similarity as a supplementary cross-check.
- Use BLIP v1 to generate supplementary image descriptions.
- Retrieve relevant radiology reports using a TF-IDF-based information retrieval approach.
- Combine predictions and supporting evidence into a structured preliminary report.
- Implement deterministic urgency categories using auditable triage rules.
- Provide a simple Streamlit interface suitable for research and demonstration.
- Investigate local CPU inference and offline-capable deployment.
- Preserve human oversight and clearly communicate the limitations of AI-generated outputs.

## ✨ Key Features

| Feature | Purpose |
|---|---|
| ViT-B/16 classification | Estimates probabilities for multiple chest radiograph labels |
| Multi-label prediction | Supports simultaneous abnormality predictions |
| Attention-based heatmaps | Visualizes image regions receiving attention from the classifier |
| CLIP cross-verification | Compares image representations with text descriptions |
| BLIP v1 image captioning | Generates supplementary visual descriptions |
| TF-IDF report retrieval | Retrieves relevant reports from an indexed clinical-text corpus |
| Rule-based triage | Assigns an urgency category using explicit decision rules |
| Structured report generation | Consolidates predictions, explanations, and retrieved context |
| Streamlit interface | Provides an interactive application for image upload and analysis |
| PDF export | Enables downloading the generated preliminary report |
| CPU inference | Supports local inference without requiring a dedicated GPU, subject to hardware constraints |

---

## 🏗️ System Architecture

The proposed framework follows a modular architecture in which individual components have distinct responsibilities.

**Architecture diagram:** Replace the placeholder below with your actual architecture image.

<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/31188501-9c80-4b22-9879-8ec55c4451cb" />


*Figure 1. Proposed architecture of the explainable multimodal chest radiograph analysis framework.*

The framework consists of six logical stages:

1. **Image input and preprocessing:** Loads the uploaded radiograph and prepares it for model inference.
2. **Visual analysis:** The fine-tuned ViT-B/16 estimates probabilities for the supported classification labels.
3. **Evidence generation:** Attention-based visualization, CLIP similarity analysis, and BLIP captioning provide complementary evidence.
4. **Clinical-text retrieval:** A TF-IDF index retrieves relevant report text from the available IU/OpenI corpus.
5. **Evidence integration and triage:** The application consolidates outputs and applies deterministic rules to assign an urgency category.
6. **Report presentation:** A structured preliminary report is displayed through Streamlit and can be exported as a PDF.

The outputs of the individual models are not assumed to be equivalent or interchangeable. Classification probabilities, image-text similarity scores, generated captions, and retrieved report text have different meanings and should be interpreted separately.
## Comparatative Analysis
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/5a8c9345-124c-47c5-9823-c4b89f27a256" />

## 🔄 End-to-End Workflow

```text
Chest Radiograph Upload
          |
          v
Image Validation and Preprocessing
          |
          v
Fine-Tuned ViT-B/16 Classifier
          |
          +----------------------+
          |                      |
          v                      v
Attention-Based             Predicted Labels
Visual Explanation          and Probabilities
          |                      |
          +-----------+----------+
                      |
          +-----------+-----------+
          |                       |
          v                       v
    CLIP Similarity          BLIP v1 Caption
    Cross-Verification       Generation
          |                       |
          +-----------+-----------+
                      |
                      v
           TF-IDF Report Retrieval
                      |
                      v
           Evidence Consolidation
                      |
                      v
          Rule-Based Triage Engine
                      |
                      v
          Structured Preliminary Report
                      |
                      v
           Streamlit UI / PDF Export
                      |
                      v
             Qualified Human Review
                      |
                      v
             Clinical Decision-Making
```
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/46207ce9-e726-4cf8-aeba-4940a2d5082e" />
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/98b5966a-8b4a-4a7b-a6ba-cb6e3064de74" />
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/076cd1ab-64d2-429f-b483-4de2aedf8d22" />

*Figure 2. End-to-end workflow. The final clinical decision remains the responsibility of a qualified healthcare professional.*

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Primary visual model | Vision Transformer (ViT-B/16) |
| Vision-language similarity | CLIP |
| Image captioning | BLIP v1 |
| Explainability | Transformer attention visualization |
| Information retrieval | TF-IDF and text similarity |
| Application interface | Streamlit |
| Data processing | NumPy, pandas, and image-processing libraries as specified in `requirements.txt` |
| Reporting | PDF-generation utilities used by the application |
| Training environment | Kaggle GPU environment |
| Local inference | CPU-supported inference, subject to model and hardware requirements |

The exact package versions are defined by the project's dependency file.

## 📚 Dataset Description

### 1. NIH ChestX-ray14

The NIH ChestX-ray14 dataset is used to train and evaluate the primary image classifier.

- Approximately 112,120 frontal chest radiographs.
- Fourteen labeled thoracic disease categories.
- Commonly used for multi-label chest radiograph classification research.
- The project classifier additionally includes a **No Finding** output label, resulting in 15 classifier outputs.

The reported evaluation uses the project's patient-disjoint test split containing 25,596 images.

The original dataset labels are derived from associated reports and may contain uncertainty or label noise. They should not be treated as equivalent to definitive expert-adjudicated diagnoses.

### 2. Indiana University Chest X-ray Collection / OpenI

The IU/OpenI chest X-ray collection provides radiographs and associated radiology reports.

In this project, the available report corpus contains **7,426 reports** for text retrieval.

The retrieval module uses this corpus to identify relevant clinical-text examples. Retrieved reports provide contextual reference material and must not be interpreted as diagnoses of the currently uploaded patient.

### Dataset Summary

| Dataset | Project use |
|---|---|
| NIH ChestX-ray14 | Primary classifier training and evaluation |
| IU/OpenI reports | Retrieval-grounded clinical-text context |
| CLIP text descriptions | Supplementary image-text similarity analysis |

Dataset access and usage must comply with the relevant dataset terms, access conditions, and licensing requirements. The actual number of usable records may depend on preprocessing and missing or invalid entries.

## 🧠 Model Architecture and Responsibilities

The framework combines complementary components rather than treating them as competing versions of the same model.

### 1. Vision Transformer (ViT-B/16)

**Role:** Primary image classifier.

The fine-tuned ViT-B/16 processes the chest radiograph and predicts probabilities for the supported output labels.

For an input image \(x\), the classifier produces:

\[
\mathbf{p}=f_{\theta}(x)
\]

where \(\mathbf{p}\) is the vector of predicted probabilities and \(f_{\theta}\) denotes the trained model.

Because this is a multi-label task, each output represents an individual label probability. These values should not automatically be interpreted as calibrated probabilities of disease.

### 2. CLIP

**Role:** Supplementary image-text cross-verification.

CLIP maps images and text into a shared embedding space. The project uses zero-shot image-text similarity to compare the radiograph representation with selected textual descriptions.

A cosine similarity score can be expressed as:

\[
s(I,T)=
\frac{v(I)^\top t(T)}
{\|v(I)\|_2\|t(T)\|_2}
\]

where \(v(I)\) and \(t(T)\) are the image and text embeddings.

These scores indicate relative semantic similarity. They are not diagnostic probabilities, and agreement with the classifier does not establish clinical correctness.

### 3. BLIP v1

**Role:** Supplementary image description.

BLIP v1 generates a natural-language caption from the input image. Its output provides additional visual context for the application.

The captioning component is not fine-tuned specifically for this project unless separately documented. Its descriptions may omit abnormalities or generate clinically inaccurate content. Captions must therefore remain supplementary rather than authoritative diagnostic evidence.

### 4. Attention-Based Explainability

**Role:** Visualizing attention over image patches.

The application extracts attention information from the final transformer block and maps patch-level attention back onto the image to generate a heatmap and highlighted region.

This is an attention visualization technique, not conventional CNN Grad-CAM.

Attention maps can help users inspect which regions received greater model attention, but they do not prove that the highlighted regions are causal, medically meaningful, or sufficient to explain a prediction.

### 5. TF-IDF Report Retrieval

**Role:** Retrieval-grounded clinical-text context.

The retrieval module converts available reports into TF-IDF vectors and retrieves text with high lexical similarity to the query.

For a term \(t\) in document \(d\), the general TF-IDF weighting is:

\[
\operatorname{TFIDF}(t,d)
=
\operatorname{TF}(t,d)
\times
\operatorname{IDF}(t)
\]

The implementation's tokenization, normalization, ranking, and query construction determine the exact retrieval behavior.

Retrieved text is reference material from other cases. It is not evidence that the same finding exists in the current patient.

### 6. Deterministic Triage Engine

**Role:** Auditable urgency categorization.

The triage module combines configured classification thresholds and explicitly defined rules to assign one of four categories:

- **Routine**
- **Soon**
- **Urgent**
- **Immediate**

The categories are intended as preliminary workflow-support labels. They are not validated clinical urgency standards unless supported by a separate clinical protocol and validation study.

The rule definitions, threshold values, and behavior when components disagree should be documented and tested before any clinical use.

### 7. Structured Report Generation

**Role:** Consolidating the outputs.

The reporting module organizes available outputs into a preliminary clinical-support report, which may include:

- Predicted labels and their model scores.
- Attention-based visualization.
- CLIP image-text similarity findings.
- BLIP-generated caption.
- Retrieved report context.
- Rule-based triage category.
- Limitations and human-review notice.

The report should distinguish model predictions from generated descriptions, retrieved material, and deterministic rules.

---

## 📊 Evaluation Results

The following values are the project's reported ViT-B/16 evaluation results. They describe the primary classifier and should not be presented as evidence that the complete multimodal framework improves clinical outcomes.

### Primary classifier results

| Evaluation item | Reported value |
|---|---:|
| Training dataset | NIH ChestX-ray14 |
| Total dataset size | Approximately 112,120 images |
| Test split | 25,596 images |
| Classifier | Fine-tuned ViT-B/16 |
| Output labels | 15, including No Finding |
| Mean AUROC across 15 outputs | 0.7884 |
| Mean AUROC across 14 disease labels | 0.7926 |
| CheXNet published mean AUROC reference | 0.841 |

The CheXNet reference is included as historical context, not as a controlled head-to-head comparison. Differences in preprocessing, training, data splits, model implementation, and evaluation protocols can affect the results.

### Selected class-wise AUROC values

| Label | AUROC |
|---|---:|
| Cardiomegaly | 0.889 |
| Emphysema | 0.870 |
| Edema | 0.847 |
| Pneumothorax | 0.844 |
| Infiltration | 0.699 |
| Pneumonia | 0.713 |
| Nodule | 0.720 |
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/ba77e548-4404-4c8c-9081-7f114bd7f2f7" />
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/995e4ae1-2034-424e-8771-582d5b20142f" />
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/7a822264-d41e-44b3-bd7a-29d8e8e39b6d" />
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/b4ffcf8a-d005-4c5d-8bad-792c87f5edcb" />
<img width="520" height="450" alt="image" src="https://github.com/user-attachments/assets/77711640-c6c5-481f-ba80-8e8d1ce6d0a1" />

These values are the reported results for the project's classifier. Add the full class-wise results, confidence intervals, and evaluation script output when available.

**Important:** Precision, recall, F1-score, calibration, retrieval quality, caption quality, triage performance, inference latency, and end-to-end multimodal performance should be reported only when supported by actual experiments. They are not inferred from AUROC.

### Comparison with existing approaches

| Approach | Main capability | Important distinction |
|---|---|---|
| ResNet | Residual CNN image classification | Image classification backbone |
| DenseNet | Densely connected CNN classification | Image classification backbone |
| CheXNet | Chest X-ray pneumonia classification research | Established chest X-ray classification baseline |
| ConVIRT | Medical image-text representation learning | Contrastive visual-text representation learning |
| GLoRIA | Global-local image-text representation learning | Fine-grained multimodal representation learning |
| MedCLIP | Medical image-text contrastive learning | Uses unpaired medical images and text |
| MedKLIP | Medical knowledge-enhanced image-text learning | Incorporates medical knowledge |
| BLIP / BLIP-2 | Vision-language understanding and generation | Architecture depends on the selected version |
| Proposed framework | Classification, explanations, similarity checks, retrieval, triage, and reporting | Modular orchestration of complementary capabilities |

The table summarizes the primary focus of each approach. It does not establish a ranking of model performance. Fair quantitative comparisons require consistent datasets, splits, preprocessing, label definitions, and evaluation metrics.

## 🧪 Ablation Study Plan

The ablation study will investigate the individual contribution of each component. Unless results have been obtained and documented, the configurations below are **planned experiments**, not completed results.

| Configuration | Components included | Evaluation focus |
|---|---|---|
| A | ViT-B/16 classifier only | Baseline classification performance |
| B | Classifier + attention visualization | Explanation quality and overhead |
| C | Classifier + CLIP | Cross-verification behavior |
| D | Classifier + BLIP v1 | Caption usefulness and failure cases |
| E | Classifier + report retrieval | Retrieval relevance and contextual usefulness |
| F | Classifier + triage rules | Rule consistency and triage agreement |
| G | Full framework | Integrated behavior, latency, and usability |

### Recommended measurements

- Classification: AUROC, AUPRC, sensitivity, specificity, precision, recall, and F1-score.
- Calibration: Brier score or expected calibration error, where appropriate.
- Retrieval: Precision@k, Recall@k, and human relevance assessment.
- Explainability: Localization agreement with expert annotations where available, plus expert review.
- Triage: Per-category sensitivity, confusion matrix, and agreement with a predefined clinical reference standard.
- Runtime: Inference latency, peak memory use, and total end-to-end processing time.
- Usability: Structured evaluation by intended users under an appropriate study protocol.

Ablations should use consistent test data and clearly specify which components are disabled. Explainability and retrieval components may not change the classifier's AUROC when they do not feed information back into the classifier; their value should be measured using suitable component-specific outcomes.

## 🔎 Explainability and Clinical Safety

The framework aims to improve transparency by presenting several types of evidence separately.

- **Classification scores:** Indicate the model's predicted likelihood scores for supported labels.
- **Attention heatmaps:** Show the image regions receiving attention in the selected transformer layer.
- **CLIP scores:** Measure image-text semantic similarity.
- **BLIP captions:** Supply general-purpose generated image descriptions.
- **Retrieved reports:** Provide similar text from the indexed corpus.
- **Triage results:** Reflect the configured deterministic rules.

These evidence types are not interchangeable. Their agreement does not guarantee a correct result, and disagreement should not be hidden.

Before clinical deployment, the system requires external validation, evaluation across representative populations and imaging equipment, calibration assessment, bias analysis, and clinical workflow evaluation. Triage rules must be reviewed by appropriate clinical professionals.

## 📁 Repository Structure

The following structure reflects the project's current described organization. Adjust the tree to match the actual repository before publishing.

```text
xray-assistant/
│
├── app/
│   └── app.py
│
├── src/
│   ├── paths.py
│   ├── dataset.py
│   ├── model.py
│   ├── gradcam.py
│   ├── report_index.py
│   ├── report_generator.py
│   ├── triage.py
│   ├── clip_verify.py
│   ├── blip_caption.py
│   ├── report_pdf.py
│   └── inference.py
│
├── kaggle/
│   ├── train_vit_kaggle.py
│   ├── addendum_same_session.py
│   └── addendum_standalone.py
│
├── models/
│   ├── chest_classifier.pt
│   ├── thresholds.json
│   └── [retrieval index and other required artifacts]
│
├── data/
│   └── [locally prepared datasets; not necessarily committed]
│
├── outputs/
│   └── [generated reports and visualizations]
│
├── docs/
│   └── images/
│       ├── system_architecture.png
│       ├── app_home.png
│       ├── prediction_results.png
│       ├── attention_heatmap.png
│       ├── multimodal_analysis.png
│       └── pdf_report.png
│
├── requirements.txt
└── README.md
```

Model checkpoints, datasets, cached indexes, and generated outputs may be excluded from Git because of size, access restrictions, or privacy considerations. If required artifacts are not included in the repository, document how to obtain or generate them.

## ⚙️ Installation

### Prerequisites

- Python version compatible with the project's dependency file.
- Git.
- Sufficient RAM and disk space for the model weights and report index.
- Internet access for initial dependency and model downloads, unless all required artifacts are already available locally.
- A GPU is recommended for training; local inference may run on a CPU.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd xray-assistant
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the Python version and dependency versions tested by the project. If installation fails, check the compatibility of PyTorch, Transformers, Streamlit, and the other listed packages.

## 📦 Model Setup

The application requires the trained classifier checkpoint, its configured thresholds, the selected CLIP and BLIP model weights, and the report retrieval index.

### Classifier artifacts

Place the required trained artifacts in the expected `models/` directory, for example:

```text
models/
├── chest_classifier.pt
├── thresholds.json
└── [retrieval index artifacts]
```

Use the actual filenames expected by the project's configuration.

### Downloading pretrained models

If the repository contains a model-download script, run the documented command from the directory expected by that script. For example, if `download_models.py` is available in `src/`:

```bash
python src/download_models.py
```

Verify the script's actual location before running this command. If the project does not include that script, download or initialize the required model artifacts using the procedure implemented in the source code.

Initial downloads may require internet access. Once the required weights and artifacts have been cached locally, inference may be possible without an internet connection, provided the code is configured to load them locally.

## 🏋️ Training

The primary reported classifier was fine-tuned using the NIH ChestX-ray14 dataset in a Kaggle GPU environment.

Training on the full dataset is intended for a suitable GPU environment rather than a typical CPU-only laptop.

### Suggested training workflow

1. Create a Kaggle notebook.
2. Enable a GPU accelerator and allow internet access if required to retrieve dependencies.
3. Add the authorized NIH ChestX-ray14 dataset.
4. Upload or make available the training script.
5. Run the training pipeline and verify the data paths, split definitions, and output directory.
6. Save the trained checkpoint, threshold configuration, and evaluation artifacts.
7. Transfer the required artifacts into the local project's `models/` directory.
8. Reproduce the reported evaluation with the corresponding test split and evaluation code.

Example script location:

```text
kaggle/train_vit_kaggle.py
```

Run the script according to the notebook environment and its documented configuration.

**Reproducibility note:** Do not use a different training script or checkpoint and assume it reproduces the reported results. The architecture, data split, preprocessing, training configuration, and evaluation code must match the experiment being reported.

## 🔎 Building the Report Retrieval Index

The retrieval index is built from the available IU/OpenI report data.

If `src/report_index.py` supports the following command-line arguments, use:

```bash
python src/report_index.py \
  --reports path/to/indiana_reports.csv \
  --projections path/to/indiana_projections.csv
```

On Windows, the same command can be entered on one line:

```bash
python src/report_index.py --reports path\to\indiana_reports.csv --projections path\to\indiana_projections.csv
```

Replace the example paths with the actual dataset paths. Confirm the script's argument names and expected file format before running it.

The resulting index must be saved in the location configured by the application.

## 🚀 Running the Application

After installing dependencies and placing all required model and index artifacts in their expected locations, launch the Streamlit interface:

```bash
streamlit run app/app.py
```

Open the local URL printed by Streamlit in your browser.

### Typical user workflow

1. Upload a supported chest radiograph.
2. Review the predicted labels and associated scores.
3. Inspect the attention-based heatmap.
4. Review the CLIP similarity output and supplementary BLIP caption.
5. Examine retrieved report context.
6. Review the rule-based triage category and the rationale provided by the application.
7. Inspect the structured preliminary report.
8. Export the report as a PDF, if the export functionality is available and operational.

All results should be treated as AI-generated decision-support information pending qualified human review.
## 🎥 Project Demo

Watch the demonstration of the **Explainable Vision-Language Diagnostic Assistant for Chest Radiographs**.

https://github.com/user-attachments/assets/c12f6525-da72-4760-826e-fff92f95edae

## 🖼️ Example Outputs

Add real screenshots from your running application to the following locations, or replace these paths with the actual image URLs.

### 1. Application interface

<img width="1917" height="1020" alt="Screenshot 2026-10-06 185002" src="https://github.com/user-attachments/assets/c50b1e41-7c70-4bfc-850e-7f30686a4673" />


*Figure 3. Application interface for uploading a chest radiograph and viewing its analysis.*

### 2. Classification results
<img width="1917" height="991" alt="Screenshot 2026-10-06 185313" src="https://github.com/user-attachments/assets/b1065724-a658-4747-8fe9-67a20ad39c3f" />
<img width="1917" height="1018" alt="Screenshot 2026-10-06 185324" src="https://github.com/user-attachments/assets/d5ceb616-4a62-440a-961d-f6bc2d7594c7" />

*Figure 4. Predicted chest radiograph labels and corresponding model scores.*

### 3. Attention-based explanation

<img width="1917" height="1021" alt="Screenshot 2026-10-06 185257" src="https://github.com/user-attachments/assets/65faabf2-6027-4ac2-ad46-9c176877c53e" />

*Figure 5. Transformer attention visualization over the input radiograph.*

### 4. Multimodal outputs

<img width="1917" height="1018" alt="Screenshot 2026-10-06 185324" src="https://github.com/user-attachments/assets/b4617216-c704-4c6a-a5e8-bfb879548e91" />

*Figure 6. Supplementary image-text similarity, captioning, and retrieved report context.*

Use only appropriately de-identified, legally usable example images and reports. Do not upload identifiable patient information to a public repository.

## 💻 Deployment and Offline Capability

The framework investigates local inference for resource-constrained environments.

Potential deployment considerations include:

- CPU-based inference for environments without dedicated GPUs.
- Local storage of model weights and retrieval indexes.
- Cached pretrained models to avoid repeated downloads.
- Local processing of radiographs and generated reports.
- Reduced reliance on cloud inference services.
- A lightweight web interface accessible through a local browser.

**Offline capability depends on the implementation and setup.** Initial installation or model downloads may require internet access. The complete application should be tested with network access disabled to verify that all required models, tokenizers, configuration files, and retrieval artifacts load successfully.

CPU support does not guarantee acceptable processing speed on every device. Runtime and memory use should be measured on representative low-resource hardware before making deployment claims.

## ⚠️ Limitations

1. **Dataset limitations:** NIH ChestX-ray14 contains noisy report-derived labels and may not represent all patient populations or clinical environments.
2. **Generalization:** Performance on the selected test split does not establish generalization to other hospitals, imaging equipment, or populations.
3. **Classification uncertainty:** Model scores may be poorly calibrated and can produce false-positive or false-negative predictions.
4. **Attention visualization:** Highlighted regions do not establish causal explanations or diagnostic correctness.
5. **CLIP limitations:** Image-text similarity is a semantic comparison, not a validated diagnostic probability.
6. **Captioning limitations:** BLIP v1 may produce incomplete or inaccurate medical descriptions.
7. **Retrieval limitations:** Lexically similar reports may be clinically irrelevant or contain findings that do not apply to the current patient.
8. **Triage limitations:** Deterministic rules require appropriate clinical specification and validation before being used to guide real patient care.
9. **Clinical validation:** The reported classifier results do not establish the safety or effectiveness of the integrated framework in clinical practice.
10. **Resource constraints:** Model size, memory consumption, and CPU latency may limit use on lower-specification devices.

## 🔮 Future Work

- Conduct external evaluation using independent datasets and clinical sites.
- Evaluate calibration and uncertainty estimation.
- Assess classification performance across demographic and acquisition subgroups where appropriate data are available.
- Improve retrieval using clinically meaningful ranking and relevance evaluation.
- Evaluate generated descriptions and reports with qualified radiologists.
- Validate triage thresholds against an appropriate clinical reference standard.
- Measure component-wise and end-to-end inference latency.
- Test the complete offline workflow on representative low-resource devices.
- Conduct usability studies with intended healthcare users.
- Investigate privacy-preserving deployment and responsible data governance.
- Evaluate whether multimodal integration improves usefulness, transparency, or workflow efficiency over the classifier-only baseline.

## 👥 Team Members

**Institution:** KLH

**Project area:** Natural Language Processing / Applied Machine Learning for Text Analysis

| Team member | Student ID |
|---|---|
| P. Kavya Sai | 2420030322 |
| P. Meghna | 2420030401 |
| S. Anjana | 2420030499 |
| D. Poojitha | 2420030697 |

**Project supervisor:** Dr. K Swanthana

## 📖 References

1. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).

2. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993).

3. P. Rajpurkar et al., “CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,” 2017. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225).

4. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391).

5. J. Li, D. Li, C. Xiong, and S. Hoi, “BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation,” *Proceedings of the 39th International Conference on Machine Learning (ICML)*, 2022. [arXiv:2201.12086](https://arxiv.org/abs/2201.12086).

6. J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models,” *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597).

7. Z. Huang et al., “GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-Efficient Medical Image Recognition,” *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.

8. Y. Zhang et al., “Contrastive Learning of Medical Visual Representations from Paired Images and Text,” *Proceedings of the Machine Learning for Health (ML4H) Symposium*, 2022. [arXiv:2010.00747](https://arxiv.org/abs/2010.00747).

9. Z. Wang et al., “MedCLIP: Contrastive Learning from Unpaired Medical Images and Text,” *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2022. [arXiv:2210.10163](https://arxiv.org/abs/2210.10163).

10. X. Wang et al., “ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017. [arXiv:1705.02315](https://arxiv.org/abs/1705.02315).

11. A. Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale,” *International Conference on Learning Representations (ICLR)*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).

12. A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020).

13. J. Irvin et al., “CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison,” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2019. [arXiv:1901.07031](https://arxiv.org/abs/1901.07031).

14. D. Demner-Fushman et al., “Preparing a Collection of Radiology Examinations for Distribution and Retrieval,” *Journal of the American Medical Informatics Association*, 2016. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26525081/).

15. K. Clark, B. Vendt, K. Smith, J. Freymann, J. Kirby, P. Koppel, S. Moore, S. Phillips, D. Maffitt, M. Pringle, L. Tarbox, and F. Prior, “The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository,” *Journal of Digital Imaging*, 2013.

**Reference note:** Include only sources that are relevant to the methods or datasets actually discussed in the paper. Confirm bibliographic details and add the exact citations for any additional retrieval, dataset, or model components implemented in the final code.

## 📄 License

No license has been specified yet. Add a `LICENSE` file before describing the repository as open source. Ensure that any dataset, pretrained model, and third-party assets are used according to their respective terms.

## ⚖️ Disclaimer

This repository is intended for academic research and educational demonstration. It has not been established as a clinically validated medical device. The generated predictions, captions, retrieved text, visualizations, reports, and triage categories may be incorrect or incomplete.

Do not use this prototype as the sole basis for diagnosis, treatment, emergency decisions, or patient management. Qualified healthcare professionals must independently review all findings and make clinical decisions.

---

### 🚀 Project Summary

**From Image to Insight:** This project explores how visual classification, image-text similarity, image captioning, attention visualization, report retrieval, and deterministic triage can be orchestrated into a single explainable multimodal framework for preliminary chest radiograph assessment.

The main contribution is the integration of complementary capabilities, with explicit attention to transparency, reproducibility, low-resource deployment, and human oversight.



