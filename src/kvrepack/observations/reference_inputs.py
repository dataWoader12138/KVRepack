"""Prepare five or 200 LongBench and OCRBench reference inputs per group on CPU.

No remote dataset script is executed. The ZIP is read as one whitelisted member;
OCR image bytes are preserved without conversion/resizing. The caller controls
processor pixel limits. The local output JSON contains full contexts and belongs
in an untracked experiment output directory, not in the published source tree.
Importing this module does not import torch, dataset frameworks, PIL or pyarrow.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import string
import tempfile
import urllib.request
import zipfile


LONG_BENCH = {
    "repo_id": "THUDM/LongBench", "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
    "filename": "data.zip", "size": 113932529,
}
OCR_BENCH = {
    "repo_id": "echo840/OCRBench", "revision": "92a54bd1384387c178d5a07140a2d85e0a3d12e1",
    "filename": "data/test-00000-of-00001.parquet", "size": 67576988,
}
TEXT_IDS = (0, 1, 2, 3, 4)
VISION_IDS = (0, 200, 400, 600, 800)
PROMPT_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
PROMPT_PATH = "LongBench/config/dataset2prompt.json"
PROMPT_URL = f"https://raw.githubusercontent.com/THUDM/LongBench/{PROMPT_COMMIT}/{PROMPT_PATH}"
PROMPT_SHA256 = "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"


def _sample_ids(sample_count):
    if type(sample_count) is not int or sample_count not in (5, 200):
        raise ValueError("sample_count must be exactly 5 or 200 (not bool)")
    return tuple(range(sample_count)), tuple(range(0, 1000, 1000 // sample_count))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace different existing input: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".reference-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Concurrent different input appeared: {path}")
    finally:
        os.unlink(temporary)


def _dataset_file(spec, hf_home, offline):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset",
                              revision=spec["revision"], filename=spec["filename"],
                              cache_dir=str(Path(hf_home) / "hub"), local_files_only=offline))
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Pinned dataset file has unexpected size: {path}")
    return path


def _prompt_file(hf_home, offline):
    path = Path(hf_home) / "hub" / "kvrepack-reference-prompts" / PROMPT_COMMIT / "dataset2prompt.json"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Pinned LongBench prompt is not cached: {path}")
        with urllib.request.urlopen(PROMPT_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != PROMPT_SHA256:
            raise ValueError("Official pinned LongBench prompt SHA256 mismatch")
        _write_once(path, payload)
    if _sha256(path) != PROMPT_SHA256:
        raise ValueError("Cached pinned LongBench prompt SHA256 mismatch")
    return path


def _qasper_template(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("qasper"), str):
        raise ValueError("Official prompt JSON requires the qasper string")
    template = data["qasper"]
    fields = set()
    for _, name, specification, conversion in string.Formatter().parse(template):
        if name is not None:
            if specification or conversion:
                raise ValueError("Unexpected formatting operation in qasper prompt")
            fields.add(name)
    if fields != {"context", "input"}:
        raise ValueError("Qasper prompt must contain exactly the context and input fields")
    return template


def _dataset_provenance(spec, path):
    return dict(repo_id=spec["repo_id"], revision=spec["revision"], filename=spec["filename"],
                local_file=str(Path(path).resolve()), file_sha256=_sha256(path),
                file_bytes=Path(path).stat().st_size)


def _text_samples(path, template, provenance, sample_count=5):
    text_ids, _ = _sample_ids(sample_count)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name in {"qasper.jsonl", "data/qasper.jsonl"}]
        if len(names) != 1:
            raise ValueError("ZIP must have exactly one qasper.jsonl or data/qasper.jsonl member")
        member = names[0]
        with archive.open(member) as stream:
            lines = list(itertools.islice(stream, sample_count))
    if len(lines) != sample_count:
        raise ValueError(f"Qasper member contains fewer than the {sample_count} required original rows")
    samples = []
    for doc_id, line in zip(text_ids, lines):
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("context"), str)
                or not row["context"] or not isinstance(row.get("input"), str) or not row["input"]):
            raise ValueError(f"Qasper row {doc_id}: missing original context/input strings")
        metadata = dict(**provenance, config="qasper", split="test", zip_member=member,
                        row_index=doc_id, original_row_id=row.get("_id"),
                        original_row_bytes_sha256=hashlib.sha256(line).hexdigest(),
                        context_sha256=hashlib.sha256(row["context"].encode()).hexdigest(),
                        question_sha256=hashlib.sha256(row["input"].encode()).hexdigest(),
                        question_field="input", source_field="dataset", source=row.get("dataset"),
                        question_type=row.get("question_type"), original_length=row.get("length"),
                        prompt_repo="THUDM/LongBench", prompt_commit=PROMPT_COMMIT,
                        prompt_path=PROMPT_PATH, prompt_file_sha256=PROMPT_SHA256)
        samples.append(dict(group="text", doc_id=doc_id, prompt=template.format(context=row["context"], input=row["input"]),
                            context=row["context"], source_metadata=metadata))
    return samples


def _vision_samples(path, output, provenance, sample_count=5):
    _, vision_ids = _sample_ids(sample_count)
    import pyarrow.parquet as pq
    from PIL import Image
    table = pq.read_table(path)
    required = {"question", "image", "dataset", "question_type"}
    if not required <= set(table.column_names):
        raise ValueError("OCRBench parquet lacks required columns")
    if table.num_rows != 1000:
        raise ValueError(f"Pinned OCRBench test must have exactly 1000 rows, found {table.num_rows}")
    samples = []
    for doc_id, row in zip(vision_ids, table.take(list(vision_ids)).to_pylist()):
        if not isinstance(row["question"], str) or not row["question"]:
            raise ValueError(f"OCRBench row {doc_id}: missing original question")
        image = row["image"]
        payload = image.get("bytes") if isinstance(image, dict) else None
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"OCRBench row {doc_id}: missing original encoded image bytes")
        with Image.open(io.BytesIO(payload)) as decoded:
            width, height, mode, image_format = decoded.width, decoded.height, decoded.mode, decoded.format
            decoded.verify()
        if width <= 0 or height <= 0:
            raise ValueError(f"OCRBench row {doc_id}: invalid image dimensions")
        suffix = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "TIFF": "tiff", "BMP": "bmp", "GIF": "gif"}.get(image_format)
        if suffix is None:
            raise ValueError(f"OCRBench row {doc_id}: unsupported encoded format {image_format}")
        image_path = Path(output).resolve() / "input_images" / f"ocrbench_{doc_id:04d}.{suffix}"
        _write_once(image_path, payload)  # Original encoded bytes, not PIL.save().
        digest = hashlib.sha256(payload).hexdigest()
        metadata = dict(**provenance, config="default", split="test", row_index=doc_id,
                        question_field="question", source_field="dataset", source=row["dataset"],
                        question_type=row["question_type"], original_image_field_path=image.get("path"),
                        original_image_sha256=digest, image_bytes=len(payload),
                        original_image_width=width, original_image_height=height,
                        original_image_mode=mode, original_image_format=image_format,
                        question_sha256=hashlib.sha256(row["question"].encode()).hexdigest(),
                        image_processing="Original encoded bytes preserved; resizing/pixel limits belong to the caller's processor")
        samples.append(dict(group="vision", doc_id=doc_id, prompt=row["question"],
                            image_path=str(image_path), source_metadata=metadata))
    return samples


def prepare_reference_samples(hf_home, output, offline=False, sample_count=5):
    """Return 2*N fixed samples and write local output/reference_samples.json.

    N is exactly 5 (the unchanged pilot default) or 200. LongBench uses qasper
    test rows0..N-1 with full original contexts and the official pinned prompt.
    OCRBench uses rows range(0,1000,1000//N), preserving images and questions.
    Selection never depends on token counts/scores; bad or long rows are not
    filtered, replaced or truncated. Model input-length checks belong to callers.
    """
    text_ids, vision_ids = _sample_ids(sample_count)
    if not isinstance(offline, bool):
        raise TypeError("offline must be bool")
    hf_home, output = Path(hf_home).expanduser(), Path(output).expanduser().resolve()
    text_file = _dataset_file(LONG_BENCH, hf_home, offline)
    vision_file = _dataset_file(OCR_BENCH, hf_home, offline)
    prompt_file = _prompt_file(hf_home, offline)
    template = _qasper_template(prompt_file)
    samples = _text_samples(text_file, template, _dataset_provenance(LONG_BENCH, text_file), sample_count)
    samples += _vision_samples(vision_file, output, _dataset_provenance(OCR_BENCH, vision_file), sample_count)
    manifest = dict(schema_version=1, sample_count=2 * sample_count,
                    selection={"text": list(text_ids), "vision": list(vision_ids),
                               "policy": "Fixed original dataset row indices declared before activations; no token sampling"},
                    prompt_source={"url": PROMPT_URL, "commit": PROMPT_COMMIT, "sha256": PROMPT_SHA256},
                    image_policy="Original full encoded image bytes; processor pixel settings supplied by the experiment runner",
                    samples=samples)
    if sample_count != 5:  # Preserve the published default pilot manifest bytes.
        manifest["sample_count_per_group"] = sample_count
        manifest["selection"]["text_policy"] = "Original qasper test rows range(0,N)"
        manifest["selection"]["vision_policy"] = "Original OCRBench 1000-row test range(0,1000,1000//N)"
    _write_once(output / "reference_samples.json", (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    return samples
