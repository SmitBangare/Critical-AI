CriticalCare-AI Respiratory Emergency — Current Implementation and Next Steps

Overview
This repository contains the early implementation for the image-model track of the CriticalCare-AI project. It focuses on importing a strong pretrained medical vision backbone (BioViL‑T image tower), attaching a lightweight classifier head, and preparing training/evaluation/calibration scaffolds that will be used once the preprocessed dataset is ready.

Chronological log of what we implemented
1) Environment and dependencies
   - Added core dependencies for PyTorch, timm, and Hugging Face tooling.
   - File: Critical-AI/requirements.txt

2) BioViL‑T model import (local snapshot)
   - Downloaded Hugging Face model snapshot to a local folder for offline/reproducible loading.
   - Helper utilities to resolve local directory, download if missing, and load the image tower.
   - Files:
     - Critical-AI/src/models/biovil_t_loader.py
     - Local model files at Critical-AI/models/biovil_t/ (downloaded artifacts; not tracked in git)

3) Sanity verification script
   - Command-line script to verify that the local BioViL‑T model loads and (optionally) to compute an image embedding for a single CXR.
   - File: Critical-AI/scripts/sanity_biovil.py

4) Classifier head wrapper
   - A small module to attach a 1-unit linear head (binary) to the BioViL‑T image tower with helpers to freeze/unfreeze layers.
   - File: Critical-AI/src/models/classifier_head.py

5) Training/evaluation/calibration scaffolds
   - Training scaffold with mixed precision support, early stopping, and metric hooks.
   - Evaluation metrics (AUROC, AUPRC, sensitivity at fixed specificity) utilities.
   - Calibration stubs (Platt scaling, simple temperature scaling grid-search) to be fit on validation outputs.
   - Files:
     - Critical-AI/src/training/scaffold.py
     - Critical-AI/src/evaluation/metrics.py
     - Critical-AI/src/evaluation/calibration.py

What exists right now (modules and purpose)
- biovil_t_loader.py: Download/load local BioViL‑T snapshot and a quick programmatic sanity forward.
- sanity_biovil.py: Script to confirm import works; prints embedding shape if an image is provided.
- classifier_head.py: Wraps the BioViL‑T vision tower with a binary classifier head; includes freeze/unfreeze helpers.
- training/scaffold.py: Epoch loops (train/validate) with mixed precision and early stopping by AUROC.
- evaluation/metrics.py: AUROC, AUPRC, and sensitivity@specificity computation.
- evaluation/calibration.py: Platt scaling and temperature scaling interfaces.

How to run — quickstart
1) Install dependencies
   - pip install -r "Critical-AI/requirements.txt"

2) Download BioViL‑T locally (one-time)
   - PowerShell (Windows):
     - python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/BiomedVLP-BioViL-T', local_dir='D:/New folder/Critical-AI/models/biovil_t', local_dir_use_symlinks=False, revision='main')"

3) Verify the model loads
   - Without image:
     - python "D:/New folder/Critical-AI/scripts/sanity_biovil.py"
   - With one CXR image:
     - python "D:/New folder/Critical-AI/scripts/sanity_biovil.py" "D:/path/to/one_cxr.png"

Planned next steps (aligned to original project plan)
Immediate (as soon as data is ready)
- Dataloader wiring (requires friend’s outputs)
  - Expect: processed/ images and a metadata.csv with columns [filepath, patient_id, label (0/1), split (train/val/test)].
  - Implement a minimal Dataset that returns (image_tensor, label).

- Finetuning with classifier head
  - Freeze most of the backbone initially; unfreeze last N blocks later if needed.
  - Loss: weighted BCE (or focal). Optimizer: AdamW (lr ~1e-4–3e-4). Mixed precision.
  - Early stop on validation AUROC.

- Evaluation and calibration
  - Compute AUROC, AUPRC, sensitivity at 90% specificity (or target of choice).
  - Fit Platt or temperature scaling on validation predictions; save parameters.
  - Produce ROC/PR curves and a short latency measurement.

- Explainability
  - Add Grad-CAM utility for the image tower; generate overlays for a few validation examples.

Short-term deliverables for the demo
- Trained binary classifier (pneumothorax or pneumonia) with reported AUROC/AUPRC.
- Calibrated thresholds for alert tiers.
- Example Grad-CAM overlays for positive/negative cases.
- Saved artifacts: model weights, transforms, calibration params, and metrics.

Later phases (after week-1 demo)
- Segmentation for pneumothorax size estimation (if masks become available).
- Progression predictor (pneumonia, respiratory failure) and multi-modal fusion when clinical data arrives.
- Deployment interfaces (FastAPI service and dashboard) once the end-to-end imaging path is validated.

Assumptions and conventions
- Input size for BioViL‑T sanity and baseline finetune: 224×224 (pipeline: resize shorter side to 256 → center crop 224 → normalize with ImageNet stats).
- Keep image transforms consistent across train/val and log all preprocessing parameters in metadata/README.
- Patient-level splits must be used to prevent leakage.

Folder reference
- Critical-AI/models/biovil_t/      # local HF snapshot (large files, not tracked)
- Critical-AI/src/models/            # model code (loaders, heads)
- Critical-AI/src/training/          # training loop scaffold
- Critical-AI/src/evaluation/        # metrics and calibration utilities
- Critical-AI/scripts/               # sanity and helper scripts

What we need from data preprocessing (handoff contract)
- metadata.csv columns: filepath, patient_id, label (0/1), split
- Images: RGB PNG/JPG, standardized orientation, size ~any; we will transform to 224×224 internally for BioViL‑T
- Optional: record view (AP/PA) and original sizes for later analysis

Contact points / ownership
- Model import, head, training/eval: this codebase
- Preprocessing and splits: external handoff (friend), following the schema above

# CriticalCare-AI-Dual

A modular research project for dual-modality critical care AI. Goals:
- Build reproducible data pipelines for imaging and tabular EHR.
- Train baseline image models and integrate with structured features.
- Evaluate with clear metrics and experiment tracking.
- Provide a minimal deployment entrypoint for prototyping.

## Structure
- data/: raw and processed datasets (raw is git-ignored)
- src/: code modules (data processing, models, training, evaluation, deployment)
- 
otebooks/: exploratory analysis
- 	ests/: unit and integration tests

## Getting Started
1. Create and activate a venv
2. Install 
equirements.txt
3. Place source data under data/raw/

## License
MIT (add your chosen license)