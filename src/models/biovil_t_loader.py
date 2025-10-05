import os
from typing import Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoImageProcessor


HF_REPO_ID = "microsoft/BiomedVLP-BioViL-T"


def get_local_model_dir(project_root: str | None = None) -> str:
    """
    Resolve a writable local directory to store the BioViL-T snapshot.

    Defaults to "Critical-AI/models/biovil_t" relative to current working dir
    if project_root is not provided.
    """
    base = project_root or os.path.join(os.getcwd(), "Critical-AI")
    local_dir = os.path.join(base, "models", "biovil_t")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def ensure_local_snapshot(local_dir: str, revision: str = "main") -> str:
    """
    Download the Hugging Face model snapshot to local_dir if not already present.
    Returns the local directory path containing the snapshot files.
    """
    # If the directory already contains model files, assume it's ready.
    has_files = any(
        name.endswith((".bin", ".safetensors", "config.json")) for name in os.listdir(local_dir)
    ) if os.path.isdir(local_dir) else False
    if not has_files:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision=revision,
        )
    return local_dir


def load_biovil_t(local_dir: str, device: str | torch.device | None = None) -> Tuple[torch.nn.Module, AutoImageProcessor]:
    """
    Load the BioViL-T image tower and corresponding image processor from a local directory.

    Returns (vision_model, image_processor).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        local_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        local_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Extract the image tower for vision-only classification tasks
    vision_model = model.vision_model.to(device)
    return vision_model, image_processor


def quick_sanity_forward(image_path: str, local_dir: str | None = None) -> float:
    """
    Run a quick forward pass to produce a probability using a dummy linear head.
    This is meant for environment validation prior to training.
    """
    from PIL import Image

    local_dir = local_dir or get_local_model_dir()
    ensure_local_snapshot(local_dir)

    vision_model, image_processor = load_biovil_t(local_dir)

    img = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(next(vision_model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = vision_model(**{"pixel_values": inputs["pixel_values"]})
        pooled = out.pooler_output  # [B, hidden]

    classifier = torch.nn.Linear(pooled.shape[-1], 1).to(pooled.device)
    logit = classifier(pooled)
    prob = torch.sigmoid(logit).flatten().item()
    return prob


