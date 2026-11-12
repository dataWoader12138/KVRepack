"""Native benchmark inputs shared by the clean end-to-end video experiments.

This deliberately avoids observation-only token offsets and re-tokenization.
It changes neither the benchmark prompt nor the original 64-frame selection.
"""
from pathlib import Path

from .run_head_selection import FRAMES, MIN_PIXELS, MAX_PIXELS, json_sha


def prepare_input(data, doc_id, processor):
    import numpy as np
    from qwen_vl_utils import process_vision_info
    from ..observations.inputs import _video_path, _frame_indices

    task = data["task"]
    if task not in {"vatex_test", "nextqa_oe_test"}:
        raise ValueError("Only native VATEX and NExTQA video inputs are supported")
    if type(doc_id) is not int or not 0 <= doc_id < len(data["dataset"]):
        raise ValueError("Require an original dataset row index")
    doc = data["dataset"][doc_id]
    video_id = str(doc["video"] if task == "nextqa_oe_test" else doc["videoID"])
    video_path = _video_path(Path(data["video_root"]), video_id)
    prompt = data["_prompt_function"](doc, data["task_kwargs"])
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "video", "video": str(video_path),
                     "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
                    {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info([messages])
    if images or videos is None or len(videos) != 1:
        raise ValueError("Require one native video input")
    frame_count = int(videos[0].shape[0])
    indices = _frame_indices(frame_count, FRAMES)
    videos[0] = videos[0][np.asarray(indices, dtype=int)]
    batch = dict(processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors="pt"))
    metadata = dict(task=task, doc_id=doc_id, video_id=video_id,
        source_video_id=video_id if task == "nextqa_oe_test" else video_id.rsplit("_", 2)[0],
        video_path=str(video_path), input_tokens=int(batch["input_ids"].shape[-1]),
        frame_indices=indices, decoded_frames=frame_count,
        arrow_sha256=data["arrow_sha256"], original_row_sha256=json_sha(doc))
    return batch, metadata
