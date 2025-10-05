from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ClassifierConfig:
    in_features: int  # hidden size from the BioViL-T vision tower
    dropout: float = 0.0
    bias: bool = True


class BioViLTClassifier(nn.Module):
    """
    Wraps a BioViL-T vision tower with a simple binary classifier head.

    Forward expects: pixel_values [B, 3, 224, 224]
    Returns: logits [B, 1]
    """

    def __init__(self, vision_model: nn.Module, config: ClassifierConfig):
        super().__init__()
        self.vision_model = vision_model

        layers = []
        if config.dropout and config.dropout > 0:
            layers.append(nn.Dropout(p=config.dropout))
        layers.append(nn.Linear(config.in_features, 1, bias=config.bias))
        self.classifier = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_backbone(self) -> None:
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.vision_model.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        Try to unfreeze the last n transformer blocks if available.
        If the model doesn't expose blocks, falls back to unfreezing all.
        """
        try:
            blocks = getattr(self.vision_model, "encoder", None)
            if blocks is None:
                blocks = getattr(self.vision_model, "vision_model", None)
            # common patterns: .encoder.layer or .encoder.blocks
            if hasattr(blocks, "layer"):
                block_list = list(blocks.layer)
            elif hasattr(blocks, "blocks"):
                block_list = list(blocks.blocks)
            else:
                block_list = []

            if block_list:
                # freeze all first
                self.freeze_backbone()
                # unfreeze last n
                for blk in block_list[-n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            else:
                # fallback
                self.unfreeze_backbone()
        except Exception:
            self.unfreeze_backbone()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output  # [B, hidden]
        else:
            # fallback: take CLS token if present in last_hidden_state
            feats = out.last_hidden_state[:, 0]
        logits = self.classifier(feats)
        return logits


