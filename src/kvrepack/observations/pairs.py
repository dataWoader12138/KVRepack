"""CPU-only, outcome-independent nearest-neighbor plans for observation A.

The caller supplies one processed video's grid and tokenizer-derived text
segments after excluding system, special, example, and visual tokens. Visual
placeholders are mapped in their actual input order to the post-merge t/row/col
grid. Text adjacency always means native sequence positions differing by one
inside one supplied segment, never adjacency after filtering or concatenation.

Randomness only subsamples eligible nearest neighbors; there is no random or
distant-pair condition. A plan is built once per dataset row and reused across
all layers and KV heads. This module never deduplicates videos or dataset rows.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


_GROUPS = ("visual_spatial", "text_question", "text_instruction")
_TEXT_NAMES = {"question": "text_question", "instruction": "text_instruction",
               "text_question": "text_question", "text_instruction": "text_instruction"}


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        qualifier = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{qualifier} (not bool)")
    return value


def _integers(values: Sequence[int], name: str, *, minimum: int = 0) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    return [_integer(value, f"{name}[{index}]", minimum=minimum) for index, value in enumerate(values)]


def _select_indices(size: int, count: int, seed: int, group: str) -> list[int]:
    # Independent streams keep the main visual/text choices unchanged when an
    # unrelated supplementary segment is added. String seeding avoids hash().
    return sorted(random.Random(f"{seed}:{group}").sample(range(size), count))


def build_pair_plan(
    input_ids: Sequence[int],
    video_grid_thw: Sequence[int],
    spatial_merge_size: int,
    video_token_id: int,
    text_segments: Sequence[Mapping[str, object]],
    seed: int,
    max_pairs: int = 16,
    *,
    primary_text_group: str,
) -> dict:
    """Return native-position pairs, eligibility counts, and explicit coverage.

    ``text_segments`` contains ``{"name": "question" | "instruction",
    "positions": [native_sequence_positions]}``. Canonical names beginning
    with ``text_`` are also accepted. Each entry is its own block: even adjacent
    entries with the same name cannot form a cross-block pair. Segment positions
    may arrive unsorted but must be unique, disjoint, in range and nonvisual.

    The visual group and the primary text group each select exactly
    min(max_pairs, eligible_visual_pairs, eligible_primary_text_pairs). The
    other text group is an independently capped supplement, explicitly marked
    unmatched. Zero main pairs is valid and reported, rather than an exception.

    ``pairs`` are lists of [left, right] positions for direct tensor indexing;
    ``pair_ids`` and parallel ``pair_metadata`` identify every selected pair.
    Raw H/W must be divisible by spatial_merge_size. The number of placeholders
    must equal T*(H/merge)*(W/merge), including noncontiguous placeholders.
    """
    ids = _integers(input_ids, "input_ids")
    grid = _integers(video_grid_thw, "video_grid_thw", minimum=1)
    if len(grid) != 3:
        raise ValueError("video_grid_thw must contain exactly one [T, H, W] triple")
    merge = _integer(spatial_merge_size, "spatial_merge_size", minimum=1)
    video_id = _integer(video_token_id, "video_token_id", minimum=0)
    seed = _integer(seed, "seed")
    cap = _integer(max_pairs, "max_pairs", minimum=1)
    if primary_text_group not in _TEXT_NAMES:
        raise ValueError("primary_text_group must be question or instruction (optionally prefixed text_)")
    primary = _TEXT_NAMES[primary_text_group]
    temporal, height, width = grid
    if height % merge or width % merge:
        raise ValueError("Raw video grid H and W must be divisible by spatial_merge_size")
    rows, columns = height // merge, width // merge
    video_positions = [position for position, token in enumerate(ids) if token == video_id]
    expected_video_tokens = temporal * rows * columns
    if len(video_positions) != expected_video_tokens:
        raise ValueError(f"Video placeholder count {len(video_positions)} != grid token count {expected_video_tokens}")
    if not isinstance(text_segments, Sequence) or isinstance(text_segments, (str, bytes)):
        raise ValueError("text_segments must be a sequence of named position mappings")

    text_candidates = {name: [] for name in _GROUPS[1:]}
    text_positions = {name: [] for name in _GROUPS[1:]}
    blocks = {name: [] for name in _GROUPS[1:]}
    occupied = set()
    for block_id, segment in enumerate(text_segments):
        if not isinstance(segment, Mapping) or segment.get("name") not in _TEXT_NAMES:
            raise ValueError(f"text_segments[{block_id}] must name question or instruction")
        name = _TEXT_NAMES[segment["name"]]
        positions = _integers(segment.get("positions"), f"text_segments[{block_id}].positions")
        if len(set(positions)) != len(positions):
            raise ValueError(f"Duplicate text positions inside segment {block_id}")
        positions.sort()
        for position in positions:
            if position >= len(ids):
                raise ValueError(f"Text position {position} is outside input_ids")
            if ids[position] == video_id:
                raise ValueError(f"Text position {position} is a visual placeholder")
            if position in occupied:
                raise ValueError(f"Text position {position} belongs to more than one segment")
            occupied.add(position)
        pairs = [(left, right, block_id) for left, right in zip(positions, positions[1:]) if right == left + 1]
        text_candidates[name].extend(pairs)
        text_positions[name].extend(positions)
        blocks[name].append({"block_id": block_id, "name": segment["name"], "positions": positions,
                             "eligible_pair_count": len(pairs)})

    visual_count = temporal * rows * max(columns - 1, 0)
    primary_count = len(text_candidates[primary])
    matched = min(cap, visual_count, primary_count)
    if matched:
        main_status = "matched"
    elif not visual_count and not primary_count:
        main_status = "no_visual_or_primary_adjacent_pairs"
    elif not primary_count:
        main_status = "no_primary_text_adjacent_pairs"
    else:
        main_status = "no_visual_spatial_adjacent_pairs"

    groups = {}
    for name in _GROUPS:
        is_visual = name == "visual_spatial"
        eligible_count = visual_count if is_visual else len(text_candidates[name])
        is_matched = is_visual or name == primary
        count = matched if is_matched else min(cap, eligible_count)
        selected = _select_indices(eligible_count, count, seed, name)
        pairs, metadata = [], []
        for index in selected:
            if is_visual:
                time, remainder = divmod(index, rows * (columns - 1))
                row, column = divmod(remainder, columns - 1)
                offset = (time * rows + row) * columns + column
                left, right = video_positions[offset], video_positions[offset + 1]
                extra = {"coordinates": [[time, row, column], [time, row, column + 1]]}
            else:
                left, right, block_id = text_candidates[name][index]
                extra = {"block_id": block_id}
            pair_id = f"{name}:{left}:{right}"
            pairs.append([left, right])
            metadata.append({"pair_id": pair_id, "positions": [left, right], **extra})
        groups[name] = {
            "pairs": pairs, "pair_ids": [item["pair_id"] for item in metadata], "pair_metadata": metadata,
            "positions": video_positions.copy() if is_visual else sorted(text_positions[name]),
            "eligible_pair_count": eligible_count, "selected_pair_count": count,
            "matched_to_primary": is_matched,
            "selection_policy": "matched_main" if is_matched else "independent_supplement",
            "coverage_status": "selected" if count else ("no_eligible_adjacent_pairs" if not eligible_count else main_status),
        }
        if not is_visual:
            groups[name]["blocks"] = blocks[name]
    return {
        "schema_version": 1, "seed": seed, "max_pairs": cap, "input_length": len(ids),
        "primary_text_group": primary, "matched_pair_count": matched,
        "video_token_id": video_id, "video_token_count": len(video_positions), "video_positions": video_positions,
        "grid": {"raw_thw": grid, "spatial_merge_size": merge, "merged_thw": [temporal, rows, columns],
                 "placeholder_order": "time,row,column", "temporal_cells_are_not_claimed_to_be_single_frames": True},
        "groups": groups,
        "coverage": {"main_eligible": matched > 0, "status": main_status,
                     "eligible_visual_pair_count": visual_count, "eligible_primary_text_pair_count": primary_count,
                     "matched_pair_count": matched},
        "sampling": {"eligible_condition": "nearest_neighbors_only", "outcome_dependent": False,
                     "reuse": "same native positions across all layers and KV heads",
                     "dataset_rows_deduplicated": False},
    }
