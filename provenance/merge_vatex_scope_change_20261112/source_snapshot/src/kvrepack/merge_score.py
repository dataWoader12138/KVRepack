"""Merge-score variants shared by KVRepack ablation runs."""

from __future__ import annotations

import torch


MERGE_SCORE_MODES = {"full", "importance_only", "difference_only"}


def normalize_merge_score_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in MERGE_SCORE_MODES:
        choices = ", ".join(sorted(MERGE_SCORE_MODES))
        raise ValueError(f"merge score mode must be one of {choices}; got {value!r}")
    return mode


def merge_score_from_importance(
    importance: torch.Tensor,
    alpha: float,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Score adjacent entries; lower-scoring pairs are merged first."""
    left_imp = importance[:-1]
    right_imp = importance[1:]
    pair_sum = left_imp + right_imp
    pair_diff = torch.abs(right_imp - left_imp) / (pair_sum + eps)

    if mode == "full":
        # Preserve the original operation order for bitwise-compatible defaults.
        return pair_sum * (1.0 + float(alpha) * pair_diff)
    if mode == "importance_only":
        return pair_sum
    if mode == "difference_only":
        return pair_sum * pair_diff
    raise ValueError(f"unsupported normalized merge score mode: {mode!r}")
