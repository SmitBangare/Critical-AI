Critical-AI Code Overview (Simple, Detailed Explanations)

Purpose
This document explains every Python file currently in the repository, what it does, and how pieces fit together. It is written in simple language for quick onboarding and review.

Directory map (only relevant parts)
- Critical-AI/
  - models/biovil_t/                 ← local Hugging Face snapshot (large files, not tracked)
  - scripts/
    - sanity_biovil.py               ← script to verify model import and (optionally) compute one embedding
  - src/
    - evaluation/
      - calibration.py               ← probability calibration utilities (Platt, temperature scaling)
      - metrics.py                   ← evaluation metrics (AUROC, AUPRC, sensitivity@spec)
    - models/
      - biovil_t_loader.py           ← download & load BioViL‑T locally + quick sanity forward
      - classifier_head.py           ← attach a 1-unit classifier head to BioViL‑T image tower
    - training/
      - scaffold.py                  ← training loop (train/validate), early stopping, AMP support


scripts/sanity_biovil.py — What it does and how to use it
- Goal: Quickly prove that the local BioViL‑T model loads correctly on your machine.
- What it does:
  1) Loads the model from Critical-AI/models/biovil_t/.
  2) Prints a confirmation message (device and path).
  3) If you pass an image path, it computes a single image embedding and prints its shape.
- When to use it: Right after setting up the project, or to troubleshoot model-loading issues.
- Example:
  - python "Critical-AI/scripts/sanity_biovil.py"
  - python "Critical-AI/scripts/sanity_biovil.py" "D:/path/to/one_cxr.png"


src/models/biovil_t_loader.py — BioViL‑T model utilities
- Goal: Provide a clean way to download and load the BioViL‑T model locally.
- Key functions:
  - get_local_model_dir(project_root=None): returns a consistent path where we store the model snapshot, by default Critical-AI/models/biovil_t/.
  - ensure_local_snapshot(local_dir, revision="main"): downloads the Hugging Face snapshot if it’s not already present.
  - load_biovil_t(local_dir, device=None): loads the model (vision tower and processor) from the local folder for offline use.
  - quick_sanity_forward(image_path, local_dir=None): opens an image, runs a forward pass, and returns a probability (dummy head) — used only for environment validation.
- When to use it: In training or inference code where you need to guarantee the model files exist and can be loaded.


src/models/classifier_head.py — Classifier head on top of BioViL‑T
- Goal: Convert BioViL‑T’s image features into a single binary prediction (e.g., pneumothorax vs normal).
- Components:
  - ClassifierConfig: specifies input feature size (hidden size of the image tower), dropout, and bias.
  - BioViLTClassifier(nn.Module): wraps the vision model and adds a 1-unit Linear layer. Includes:
    - freeze_backbone()/unfreeze_backbone(): control trainable layers.
    - unfreeze_last_n_blocks(n): unfreezes only the last N transformer blocks (if exposed), otherwise unfreezes all.
    - forward(pixel_values): runs the vision model, pools features, and outputs logits [B, 1].
- When to use it: During finetuning to learn your task from the BioViL‑T image features.


src/training/scaffold.py — Training loop scaffold
- Goal: Provide a simple, reliable training loop that you can plug your DataLoader into.
- Main pieces:
  - TrainConfig: holds training hyperparameters (epochs, lr, weight decay, AMP, early stopping patience).
  - default_bce_loss: binary cross-entropy with logits (supports pos_weight for imbalance).
  - train_epoch(...): one pass over the training set; supports mixed precision.
  - validate_epoch(...): computes probabilities on val set and calls a user-supplied metric function.
  - fit(...): coordinates epochs, tracks best AUROC, restores best model based on validation.
- Expected DataLoader format: yields (images, labels) where images are [B,3,H,W] and labels are [B].
- When to use it: Once your friend provides processed images and a metadata.csv; you implement a matching Dataset and plug the DataLoader here.


src/evaluation/metrics.py — Evaluation utilities
- Goal: Compute standard classification metrics for the medical imaging demo.
- Functions:
  - compute_metrics(probs, labels, target_spec=0.9): returns a dict with:
    - auroc
    - auprc
    - sens_at_spec (sensitivity at the closest threshold achieving the target specificity)
    - threshold (the operating point threshold)
- When to use it: After each validation epoch or at the end of training to report results.


src/evaluation/calibration.py — Probability calibration
- Goal: Map raw model outputs to better-calibrated probabilities.
- Classes:
  - PlattCalibrator: logistic regression on probabilities. Fit on validation (probs, labels), then predict calibrated probs.
  - TemperatureScaler: finds a temperature T to rescale logits and reduce log-loss on validation; then converts logits→probs using that T.
- When to use it: After training, fit on validation set outputs, then apply to test/inference outputs for more trustworthy probabilities.


How these pieces fit together (simple flow)
1) Prepare data
   - Preprocessor (external): produce processed/ images and metadata.csv with filepath, patient_id, label, split.
2) Load backbone and model
   - ensure_local_snapshot → load_biovil_t → wrap with BioViLTClassifier.
3) Train
   - Build DataLoaders → call fit() with default_bce_loss and compute_metrics for AUROC early stopping.
4) Evaluate
   - compute_metrics for final AUROC/AUPRC and sensitivity@specificity.
5) Calibrate
   - Fit PlattCalibrator or TemperatureScaler on validation outputs; apply at inference.


FAQs / Notes
- Input size for BioViL‑T: use 224×224 (resize short side 256 → center crop 224 → ImageNet mean/std normalize).
- Mixed precision: enabled by default if CUDA is available; speeds up training.
- Class imbalance: use pos_weight in default_bce_loss or adopt focal loss (you can swap the loss_fn in fit()).
- Reproducibility: save model weights, transforms, and calibration parameters after training.


