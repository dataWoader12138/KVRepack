"""Prepare unchanged cached benchmark inputs and align natural text token spans.

Importing this module uses only the standard library. Dataset, video and tensor
dependencies are imported when inputs are actually prepared; models are never
loaded here. The task text functions execute from their real source AST without
importing task utilities that perform downloads or initialize scoring packages.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = {
    "nextqa_oe_test": ("nextqa", "nextqa_doc_to_text", "lmms-lab___n_ex_tqa/OE", "n_ex_tqa-test.arrow"),
    "vatex_test": ("vatex", "vatex_test_doc_to_text", "lmms-lab___vatex/vatex_test", "vatex-test.arrow"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tolist(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _task_prompt_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected exactly one source task function {name!r}: {path}")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_task(hf_home: Path, task: str, arrow_path: Path | None = None,
              video_root: Path | None = None, repo_root: Path | None = None) -> dict:
    """Load the cached test Arrow directly, preserving all original row indices."""
    if task not in TASKS:
        raise ValueError(f"Unsupported observation task: {task!r}")
    from datasets import Dataset
    import yaml

    hf_home = Path(hf_home).expanduser().resolve()
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    family, function_name, cache_name, arrow_name = TASKS[task]
    if arrow_path is None:
        matches = sorted((hf_home / "datasets" / cache_name).glob(f"**/{arrow_name}"))
        if len(matches) != 1:
            raise ValueError(f"Expected one cached {task} Arrow, found {len(matches)}; pass arrow_path explicitly")
        arrow = matches[0].resolve()
    else:
        arrow = Path(arrow_path).expanduser().resolve()
    if not arrow.is_file():
        raise FileNotFoundError(arrow)
    folder = root / "vendor/lmms-eval/lmms_eval/tasks" / family
    yaml_path = folder / f"{task}.yaml"
    utils_path = folder / "utils.py"
    # The original !function entries are irrelevant to data/prompt preparation.
    # Keep YAML's genuine defaults while avoiding module-import side effects.
    yaml_text = "\n".join(line for line in yaml_path.read_text(encoding="utf-8").splitlines()
                          if "!function" not in line)
    config = yaml.safe_load(yaml_text)
    kwargs = dict(config["lmms_eval_specific_kwargs"]["default"])
    if video_root is None:
        video_root = hf_home / ("nextqa/NExTVideo" if family == "nextqa" else "vatex_test")
    video_root = Path(video_root).expanduser().resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")
    data = {
        "task": task, "dataset": Dataset.from_file(str(arrow)),
        "arrow_path": str(arrow), "arrow_sha256": _sha256(arrow),
        "video_root": str(video_root), "hf_home": str(hf_home),
        "task_yaml_path": str(yaml_path), "task_yaml_sha256": _sha256(yaml_path),
        "task_utils_path": str(utils_path), "task_utils_sha256": _sha256(utils_path),
        "task_kwargs": kwargs, "lmms_eval_specific_kwargs": kwargs,
        "task_config": config, "_prompt_function": _task_prompt_function(utils_path, function_name),
        "row_policy": "Original Arrow row order; no video deduplication or filtering",
    }
    include = config.get("include")
    if isinstance(include, str) and (folder / include).is_file():
        data["included_yaml_path"] = str(folder / include)
        data["included_yaml_sha256"] = _sha256(folder / include)
    return data


def _video_path(root: Path, video_id: str) -> Path:
    if not video_id or Path(video_id).name != video_id or video_id in {".", ".."}:
        raise ValueError(f"Invalid benchmark video ID: {video_id!r}")
    base = video_id
    for suffix in (".mp4", ".MP4", ".mkv"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    candidates = [root / (base + suffix) for suffix in (".mp4", ".MP4", ".mkv")]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No video for {video_id!r}; tried {[str(path) for path in candidates]}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous video files for {video_id!r}: {[str(path) for path in matches]}")
    if matches[0].stat().st_size == 0:
        raise ValueError(f"Empty benchmark video: {matches[0]}")
    return matches[0]


def _frame_indices(original_frames: int, frames: int) -> list[int]:
    if type(frames) is not int or frames <= 0 or original_frames <= 0:
        raise ValueError("frames and decoded frame count must be positive")
    import numpy as np
    # Match the real Qwen adapter, including repeated frames for short clips.
    indices = np.linspace(0, original_frames - 1, frames, dtype=int)
    if original_frames - 1 not in indices:
        indices = np.append(indices, original_frames - 1)
    return [int(index) for index in indices]


def _expand_video_text(text: str, processor, grid: list[int]) -> str:
    token = processor.video_token
    if not token or text.count(token) != 1:
        raise ValueError("Observation input must contain exactly one video placeholder")
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        video_processor = processor.image_processor
    merge_size = int(video_processor.merge_size)
    count = int(grid[0]) * int(grid[1]) * int(grid[2])
    if merge_size <= 0 or count <= 0 or count % (merge_size ** 2):
        raise ValueError(f"Invalid video grid/merge size: {grid}, {merge_size}")
    return text.replace(token, token * (count // (merge_size ** 2)), 1)


def _natural_spans(task: str, doc: dict, prompt: str, kwargs: dict) -> list[dict]:
    """Character spans relative to the unchanged user prompt, not the chat shell."""
    if task == "nextqa_oe_test":
        question = doc["question"].strip()
        prefix, suffix = kwargs.get("pre_prompt", ""), kwargs.get("post_prompt", "")
        if prompt != prefix + question + suffix:
            raise ValueError("NextQA source prompt no longer matches its recorded question/instruction structure")
        result = [{"name": "question", "text": question, "char_span": [len(prefix), len(prefix) + len(question)]}]
        if prefix:
            result.append({"name": "instruction", "text": prefix, "char_span": [0, len(prefix)]})
        if suffix:
            start = len(prefix) + len(question)
            result.append({"name": "instruction", "text": suffix, "char_span": [start, len(prompt)]})
        return result
    instruction = kwargs["prompt"]
    if not instruction or not prompt.startswith(instruction + "\n"):
        raise ValueError("VATEX source prompt no longer begins with the configured instruction")
    # The four caption examples remain in model input, but not in the primary
    # natural-text comparison pool. They are identical across VATEX documents.
    return [{"name": "instruction", "text": instruction, "char_span": [0, len(instruction)]}]


def _align_segments(input_ids: list[int], offsets: list[list[int]], attention: list[int],
                    expanded_text: str, prompt: str, spans: list[dict], special_ids: set[int]) -> list[dict]:
    if len(input_ids) != len(offsets) or len(input_ids) != len(attention):
        raise ValueError("Input IDs, offsets and attention mask lengths differ")
    if not prompt or expanded_text.count(prompt) != 1:
        raise ValueError("The full task prompt must have one unambiguous span in expanded chat text")
    prompt_start = expanded_text.index(prompt)
    groups: dict[str, dict] = {}
    for span in spans:
        start, end = [prompt_start + int(value) for value in span["char_span"]]
        if not prompt_start <= start <= end <= prompt_start + len(prompt):
            raise ValueError("Natural text span lies outside the task prompt")
        group = groups.setdefault(span["name"], {"name": span["name"], "positions": [], "char_spans": [], "texts": []})
        group["char_spans"].append([start, end])
        group["texts"].append(span["text"])
        for position, (token_id, offset, valid) in enumerate(zip(input_ids, offsets, attention)):
            left, right = map(int, offset)
            if left < 0 or right < left or right > len(expanded_text):
                raise ValueError("Tokenizer returned an invalid character offset")
            if valid and token_id not in special_ids and right > left and start <= left and right <= end:
                group["positions"].append(position)
    for group in groups.values():
        group["positions"] = sorted(set(group["positions"]))
    return list(groups.values())


def prepare_sample(task_data: dict, doc_id: int, processor, frames: int = 64,
                   min_pixels: int = 200704, max_pixels: int = 1605632) -> tuple[dict, dict]:
    """Run the native benchmark video/processor pipeline, leaving all tensors on CPU."""
    import numpy as np
    from qwen_vl_utils import process_vision_info

    if type(doc_id) is not int or not 0 <= doc_id < len(task_data["dataset"]):
        raise ValueError("doc_id must be an existing original dataset row index")
    if not getattr(processor.tokenizer, "is_fast", False):
        raise ValueError("Reliable observation offsets require the real fast tokenizer")
    task = task_data["task"]
    doc = task_data["dataset"][doc_id]
    video_id = str(doc["video"] if task == "nextqa_oe_test" else doc["videoID"])
    video_path = _video_path(Path(task_data["video_root"]), video_id)
    prompt = task_data["_prompt_function"](doc, task_data["task_kwargs"])
    spans = _natural_spans(task, doc, prompt, task_data["task_kwargs"])
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "video", "video": str(video_path),
                                                "max_pixels": max_pixels, "min_pixels": min_pixels},
                                               {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info([messages])
    if images or videos is None or len(videos) != 1:
        raise ValueError("Expected exactly one decoded video and no images")
    original_frames = int(videos[0].shape[0])
    indices = _frame_indices(original_frames, frames)
    videos[0] = videos[0][np.asarray(indices, dtype=int)]
    batch = dict(processor(text=[text], images=images, videos=videos, padding=True,
                           return_tensors="pt", return_offsets_mapping=True))
    if "offset_mapping" not in batch:
        raise ValueError("Processor did not return fast-tokenizer offsets")
    offsets = _tolist(batch.pop("offset_mapping"))[0]
    ids = _tolist(batch["input_ids"])[0]
    attention = _tolist(batch["attention_mask"])[0]
    grids = _tolist(batch["video_grid_thw"])
    if len(grids) != 1 or len(grids[0]) != 3:
        raise ValueError("Expected a single video_grid_thw row")
    grid = [int(value) for value in grids[0]]
    expanded = _expand_video_text(text, processor, grid)
    repeated = processor.tokenizer([expanded], padding=True, return_tensors="pt", return_offsets_mapping=True)
    if _tolist(repeated["input_ids"]) != [ids] or _tolist(repeated["offset_mapping"])[0] != offsets:
        raise ValueError("Expanded chat re-tokenization differs from actual processor IDs/offsets")
    special_ids = set(int(value) for value in processor.tokenizer.all_special_ids)
    for name in ("video_token_id", "image_token_id"):
        value = getattr(processor, name, None)
        if value is not None:
            special_ids.add(int(value))
    segments = _align_segments(ids, offsets, attention, expanded, prompt, spans, special_ids)
    for name, tensor in batch.items():
        if hasattr(tensor, "device") and str(tensor.device) != "cpu":
            raise ValueError(f"Processor returned a non-CPU input: {name}")
    row_json = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    unique = len(set(indices))
    metadata = {
        "task": task, "doc_id": doc_id, "video_id": video_id,
        "source_video_id": video_id if task == "nextqa_oe_test" else video_id.rsplit("_", 2)[0],
        "video_path": str(video_path), "video_bytes": video_path.stat().st_size,
        "prompt": prompt, "text_segments": segments, "video_grid_thw": grid,
        "input_tokens": len(ids), "original_row_sha256": hashlib.sha256(row_json.encode()).hexdigest(),
        "expanded_text_sha256": hashlib.sha256(expanded.encode()).hexdigest(),
        "frame_sampling": {"original_frames": original_frames, "originalframes": original_frames,
                           "requested_frames": frames, "selected_indices": indices,
                           "selected_frames": len(indices), "unique_frames": unique,
                           "repeated_frames": len(indices) - unique},
        "text_scope": "NextQA question and instruction; VATEX instruction only; unchanged VATEX caption examples remain in model input",
        "alignment": "Actual processor offsets; exact expanded-chat re-tokenization; whole tokens inside natural spans; all special IDs excluded",
        "arrow_path": task_data["arrow_path"], "arrow_sha256": task_data["arrow_sha256"],
        "task_yaml_sha256": task_data["task_yaml_sha256"], "task_utils_sha256": task_data["task_utils_sha256"],
    }
    return batch, metadata
