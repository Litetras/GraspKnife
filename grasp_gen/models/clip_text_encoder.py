#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Frozen CLIP text encoder for language-conditioned grasp generation.

Usage::

    from grasp_gen.models.clip_text_encoder import TextEncoder

    enc = TextEncoder("ViT-B/32").to(device)
    feat = enc(["pick up the cup", "place the bottle upright"])
    # feat: torch.Tensor [B, 512], float32, no grad
"""

import torch
import torch.nn as nn


def _load_clip_to_cpu(backbone_name: str = "ViT-B/32"):
    """Load a CLIP model onto CPU and return ``(model, clip_module)``.

    Args:
        backbone_name (str): CLIP variant, e.g. ``'ViT-B/32'`` or ``'RN50'``.

    Returns:
        tuple: ``(clip_model, openai_clip)`` where ``openai_clip`` exposes
        the :func:`tokenize` helper used at inference time.

    Raises:
        ImportError: If the ``clip`` package is not installed.
    """
    try:
        import clip as openai_clip
    except ImportError:
        raise ImportError(
            "openai-clip is required for language conditioning. "
            "Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )
    model, _ = openai_clip.load(backbone_name, device="cpu")
    return model, openai_clip


class TextEncoder(nn.Module):
    """Frozen CLIP text encoder producing one feature vector per text prompt.

    All CLIP weights are frozen at construction time. Only the downstream
    ``text_projection`` linear layer inside :class:`GraspGenGenerator` is
    learned during training.

    Args:
        clip_backbone (str): CLIP backbone variant. Default: ``'ViT-B/32'``.

    Attributes:
        embed_dim (int): Width of the CLIP transformer hidden state
            (e.g. 512 for ViT-B/32). This is the output feature dimension.

    Example::

        encoder = TextEncoder("ViT-B/32").cuda()
        feats = encoder(["grasp from above", "side grasp"])
        # feats.shape == (2, 512)
    """

    def __init__(self, clip_backbone: str = "ViT-B/32"):
        super().__init__()
        clip_model, self._clip_module = _load_clip_to_cpu(clip_backbone)

        # Freeze every CLIP parameter — only self.text_projection (in the
        # parent generator) will be updated during training.
        for p in clip_model.parameters():
            p.requires_grad = False

        self.transformer         = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding     = clip_model.token_embedding
        self.ln_final            = clip_model.ln_final
        self.text_projection     = clip_model.text_projection
        self.dtype               = clip_model.dtype

        # Width of the transformer hidden state (e.g. 512 for ViT-B/32)
        self.embed_dim: int = self.transformer.width

    @torch.no_grad()
    def forward(self, text: list) -> torch.Tensor:
        """Encode a batch of text strings into fixed-size feature vectors.

        The encoder runs fully under ``torch.no_grad()``.

        Args:
            text (list[str]): Task description strings, one per object in the
                batch. Length must equal ``num_objects_in_batch``.

        Returns:
            torch.Tensor: Float32 tensor of shape ``[B, embed_dim]`` where
            ``B = len(text)``.
        """
        device = next(self.parameters()).device
        tokens = self._clip_module.tokenize(text, context_length=77).to(device)

        x = self.token_embedding(tokens).type(self.dtype)   # [B, 77, D]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                              # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                              # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Feature at the EOS token position, then project to embed_dim
        feat = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        feat = feat @ self.text_projection                   # [B, embed_dim]
        return feat.float()