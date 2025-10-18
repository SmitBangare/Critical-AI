from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear
import math


@dataclass
class EnhancedClassifierConfig:
    hidden_size: int = 768
    num_heads: int = 12
    attention_dropout_rate: float = 0.2
    dropout: float = 0.0
    bias: bool = True


class MultiModalAttention(nn.Module):
    """
    IRENE's multi-modal attention mechanism adapted for BioViL-T + clinical data fusion.
    """
    def __init__(self, config: EnhancedClassifierConfig):
        super().__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Image attention projections
        self.query_image = Linear(config.hidden_size, self.all_head_size)
        self.key_image = Linear(config.hidden_size, self.all_head_size)
        self.value_image = Linear(config.hidden_size, self.all_head_size)

        # Clinical data attention projections
        self.query_clinical = Linear(config.hidden_size, self.all_head_size)
        self.key_clinical = Linear(config.hidden_size, self.all_head_size)
        self.value_clinical = Linear(config.hidden_size, self.all_head_size)

        # Output projections
        self.out_image = Linear(config.hidden_size, config.hidden_size)
        self.out_clinical = Linear(config.hidden_size, config.hidden_size)

        # Dropout layers
        self.attn_dropout_image = Dropout(config.attention_dropout_rate)
        self.attn_dropout_clinical = Dropout(config.attention_dropout_rate)
        self.attn_dropout_cross = Dropout(config.attention_dropout_rate)
        self.proj_dropout_image = Dropout(config.attention_dropout_rate)
        self.proj_dropout_clinical = Dropout(config.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # x is [B, hidden_size], we need to add sequence dimension for attention
        # Reshape to [B, 1, hidden_size] then to [B, 1, num_heads, head_size]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, hidden_size]
        
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, image_features: torch.Tensor, clinical_features: Optional[torch.Tensor] = None):
        """
        Args:
            image_features: [B, hidden_size] - BioViL-T pooled features
            clinical_features: [B, hidden_size] - Clinical data embeddings
        Returns:
            fused_features: [B, hidden_size] - Multi-modal fused features
        """
        # Self-attention for image features
        img_q = self.query_image(image_features)
        img_k = self.key_image(image_features)
        img_v = self.value_image(image_features)

        img_q = self.transpose_for_scores(img_q)
        img_k = self.transpose_for_scores(img_k)
        img_v = self.transpose_for_scores(img_v)

        img_attention_scores = torch.matmul(img_q, img_k.transpose(-1, -2))
        img_attention_scores = img_attention_scores / math.sqrt(self.attention_head_size)
        img_attention_probs = self.softmax(img_attention_scores)
        img_attention_probs = self.attn_dropout_image(img_attention_probs)

        img_context_layer = torch.matmul(img_attention_probs, img_v)
        img_context_layer = img_context_layer.permute(0, 2, 1, 3).contiguous()
        new_img_context_layer_shape = img_context_layer.size()[:-2] + (self.all_head_size,)
        img_context_layer = img_context_layer.view(*new_img_context_layer_shape)
        img_output = self.out_image(img_context_layer)
        img_output = self.proj_dropout_image(img_output)

        if clinical_features is not None:
            # Self-attention for clinical features
            clin_q = self.query_clinical(clinical_features)
            clin_k = self.key_clinical(clinical_features)
            clin_v = self.value_clinical(clinical_features)

            clin_q = self.transpose_for_scores(clin_q)
            clin_k = self.transpose_for_scores(clin_k)
            clin_v = self.transpose_for_scores(clin_v)

            clin_attention_scores = torch.matmul(clin_q, clin_k.transpose(-1, -2))
            clin_attention_scores = clin_attention_scores / math.sqrt(self.attention_head_size)
            clin_attention_probs = self.softmax(clin_attention_scores)
            clin_attention_probs = self.attn_dropout_clinical(clin_attention_probs)

            clin_context_layer = torch.matmul(clin_attention_probs, clin_v)
            clin_context_layer = clin_context_layer.permute(0, 2, 1, 3).contiguous()
            new_clin_context_layer_shape = clin_context_layer.size()[:-2] + (self.all_head_size,)
            clin_context_layer = clin_context_layer.view(*new_clin_context_layer_shape)
            clin_output = self.out_clinical(clin_context_layer)
            clin_output = self.proj_dropout_clinical(clin_output)

            # Cross-modal fusion: combine image and clinical features
            fused_features = img_output + clin_output
            return fused_features
        else:
            return img_output


class EnhancedBioViLTClassifier(nn.Module):
    """
    Enhanced BioViL-T classifier with IRENE's multi-modal attention capabilities.
    """
    def __init__(self, vision_model: nn.Module, config: EnhancedClassifierConfig):
        super().__init__()
        self.vision_model = vision_model
        self.config = config

        # Multi-modal attention mechanism
        self.attention = MultiModalAttention(config)

        # Classifier head
        layers = []
        if config.dropout and config.dropout > 0:
            layers.append(nn.Dropout(p=config.dropout))
        layers.append(nn.Linear(config.hidden_size, 1, bias=config.bias))
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

    def forward(self, pixel_values: torch.Tensor, clinical_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 224, 224] - Input images
            clinical_data: [B, hidden_size] - Clinical data embeddings (optional)
        Returns:
            logits: [B, 1] - Binary classification logits
        """
        # Extract image features using BioViL-T
        out = self.vision_model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            img_features = out.pooler_output  # [B, hidden_size]
        else:
            # Fallback: take CLS token if present in last_hidden_state
            img_features = out.last_hidden_state[:, 0]

        # Apply multi-modal attention if clinical data provided
        if clinical_data is not None:
            fused_features = self.attention(img_features, clinical_data)
        else:
            fused_features = self.attention(img_features, None)

        # Classification
        logits = self.classifier(fused_features)
        return logits
