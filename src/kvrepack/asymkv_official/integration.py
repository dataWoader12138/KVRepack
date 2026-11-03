from __future__ import annotations

from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

from .attention import qwen2_5_vl_asymkv_official_attention_forward
from .generation import asymkv_official_generate


def install_qwen2_5_vl_asymkv_official() -> None:
    """Install the released-code AsymKV path for one evaluation process."""
    conditional_class = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    if getattr(conditional_class, "_asymkv_official_installed", False):
        return

    attention_classes = (
        modeling_qwen2_5_vl.Qwen2_5_VLAttention,
        modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2,
        modeling_qwen2_5_vl.Qwen2_5_VLSdpaAttention,
    )
    for attention_class in attention_classes:
        if "_asymkv_official_original_forward" not in attention_class.__dict__:
            attention_class._asymkv_official_original_forward = attention_class.forward
        attention_class.forward = qwen2_5_vl_asymkv_official_attention_forward

    conditional_class._asymkv_official_original_generate = conditional_class.generate
    conditional_class.generate = asymkv_official_generate
    conditional_class._asymkv_official_installed = True
    print("Using released-code AsymKV (upstream commit 771ce4db0292).")
