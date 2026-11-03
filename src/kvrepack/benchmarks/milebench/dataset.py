"""Read-only MileBench data checks and the source experiment's prompt protocol.

The dataset class is extracted from HybridKV-opensource/hybridkv/utils.py.
It implements the Python dataset protocol without importing torch or workers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sample_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"sample_id must be an integer, got {value!r}")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"sample_id must be an integer, got {value!r}") from exc
    return result


def annotation_path(data_root: str | os.PathLike, task: str) -> Path:
    if not task or task in {".", ".."} or "/" in task or "\\" in task:
        raise ValueError("task must be a single MileBench dataset name")
    return Path(data_root).expanduser().resolve() / task / f"{task}.json"


def _validate_annotation(annotation: dict) -> None:
    if not isinstance(annotation, dict) or not isinstance(annotation.get("data"), list):
        raise ValueError("annotation must contain a data list")
    metadata = annotation.get("meta_data")
    if not isinstance(metadata, dict):
        raise ValueError("annotation must contain meta_data")
    if metadata.get("question_type") not in {"open-ended", "multi-choice"}:
        raise ValueError("unsupported or missing meta_data.question_type")
    instructions = metadata.get("task_instruction")
    if not isinstance(instructions, (list, dict)):
        raise ValueError("meta_data.task_instruction must be a list or mapping")
    seen = set()
    for sample in annotation["data"]:
        if not isinstance(sample, dict) or "sample_id" not in sample:
            raise ValueError("each annotation sample must contain sample_id")
        sample_id = _sample_id(sample["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate annotation sample_id: {sample_id}")
        seen.add(sample_id)
        sample["sample_id"] = sample_id
        instance = sample.get("task_instance")
        if not isinstance(instance, dict) or not isinstance(instance.get("context"), str):
            raise ValueError(f"sample {sample_id}: missing task_instance.context")
        images = instance.get("images_path")
        if not isinstance(images, list) or not images or not all(isinstance(p, str) and p for p in images):
            raise ValueError(f"sample {sample_id}: images_path must be a nonempty list of paths")
        if "response" not in sample:
            raise ValueError(f"sample {sample_id}: missing response")
        try:
            instruction = instructions[sample["task_instruction_id"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"sample {sample_id}: invalid task_instruction_id") from exc
        if not isinstance(instruction, str):
            raise ValueError(f"sample {sample_id}: task instruction must be text")
        if metadata["question_type"] == "multi-choice":
            choices = instance.get("choice_list")
            if not isinstance(choices, list) or not choices or not all(isinstance(v, str) for v in choices):
                raise ValueError(f"sample {sample_id}: invalid choice_list")
            if str(sample["response"]) not in choices:
                raise ValueError(f"sample {sample_id}: response is absent from choice_list")


def load_annotation(data_root: str | os.PathLike, task: str, limit: int | None = None) -> dict:
    """Select the first ``limit`` original IDs before any image-count grouping.

    Metadata (including the original num_sample) remains intact. Integer string
    IDs are normalized to integers so source scoring's ID lookup is unambiguous.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    annotation = json.loads(annotation_path(data_root, task).read_text(encoding="utf-8"))
    _validate_annotation(annotation)
    if not annotation["data"]:
        raise ValueError("annotation contains no samples")
    if limit is not None:
        annotation["data"] = annotation["data"][:limit]
    return annotation


def _image_path(image_root: Path, relative: str) -> Path:
    path = (image_root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(image_root.resolve()):
        raise ValueError(f"image path escapes dataset images directory: {relative!r}")
    return path


def validate_data(data_root: str | os.PathLike, task: str, limit: int | None = None,
                  verify_images: bool = True) -> dict:
    """Check every selected image, never writing files or loading model packages.

    Existence/readability checks always run. ``verify_images=True`` additionally
    uses Pillow's image decoder verification. Errors are collected in the report;
    callers must require ``valid`` before generation.
    """
    report = {"schema_version": 1, "task": task, "valid": False, "errors": [],
              "sample_ids": [], "selected_samples": 0, "total_samples": 0,
              "image_references": 0, "unique_images": 0, "images_checked": 0,
              "verify_images": verify_images}
    try:
        source = annotation_path(data_root, task)
        report["annotation_path"] = str(source)
        raw = source.read_bytes()
        report["annotation_sha256"] = hashlib.sha256(raw).hexdigest()
        annotation = load_annotation(data_root, task, limit=limit)
        report["total_samples"] = len(json.loads(raw)["data"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report["errors"].append(f"annotation: {exc}")
        return report
    report["sample_ids"] = [sample["sample_id"] for sample in annotation["data"]]
    report["selected_samples"] = len(annotation["data"])
    image_root = source.parent / "images"
    paths = set()
    for sample in annotation["data"]:
        for relative in sample["task_instance"]["images_path"]:
            report["image_references"] += 1
            try:
                paths.add(_image_path(image_root, relative))
            except ValueError as exc:
                report["errors"].append(f"sample {sample['sample_id']}: {exc}")
    report["unique_images"] = len(paths)
    image_decoder = None
    if verify_images:
        try:
            from PIL import Image
            image_decoder = Image
        except ImportError:
            report["errors"].append("image verification requires Pillow; no image decoding was performed")
    for path in sorted(paths):
        report["images_checked"] += 1
        try:
            if not path.is_file():
                raise FileNotFoundError(f"missing image: {path}")
            with path.open("rb") as handle:
                if not handle.read(1):
                    raise ValueError("empty image")
            if image_decoder is not None:
                with image_decoder.open(path) as image:
                    image.verify()
        except (OSError, ValueError, SyntaxError) as exc:
            report["errors"].append(f"{path}: {exc}")
    report["valid"] = not report["errors"]
    return report


class MileBenchDataset:
    def __init__(
        self,
        annotation,
        task_instructions,
        img_dir,
        max_context_len,
        n_tokens_per_image,
        tokenizer,
        dataset_name,
        combine_image=None
    ):
        """
        Initialize the LongContextBenchmarkDataset class.

        Parameters:
            annotation (list): List of annotations.
            task_instructions (dict): Dictionary of task instructions.
            img_dir (str): Directory containing images.
            max_context_len (int): Maximum number of tokens the model can handle.
            tokenizer: Tokenizer class in Transformers.
            dataset_name: Name of the dataset.
            combine_image (int): Number of combined images.
        """
        self.img_dir = img_dir
        self.annotation = annotation
        self.task_instructions = task_instructions
        self.combine_image = combine_image
        self.max_context_len = max_context_len
        self.n_tokens_per_image = n_tokens_per_image
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        '''
        Get item by index from the dataset.
        If self.combine_image is not None, set different context prompt.

        Parameters:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing sample information.

        {
            'sample_id': 1,
            'raw_img_list': ['/path/to/image1',],
            'context': 'what is the image <ImageHere> about?',
            'response': '',
        }
        '''
        ann = self.annotation[index]

        # Set task instruction
        task_instruction = self.task_instructions[ann['task_instruction_id']] # at the very beginning
        context = ann['task_instance']['context'] # at the very end

        # Set choice_list for multi-choice QA
        if 'choice_list' in ann['task_instance'].keys():
            choice_str = '\nChoice list: \n'
            # GPR1200 has more than 26 options which cause bug in answer matching if we use normal alphabetical choices.
            # For this dataset, we simply concatenate all the options with "\n".
            choice_str += '\n'.join([(f'{chr(65+idx)}. ' if 'GPR1200' != self.dataset_name else '') + f'{item}'
                for idx, item in enumerate(ann['task_instance']['choice_list'])])
            choice_str += '\nYour answer is: '
            context += choice_str

        # Set prompt
        img_num = len(ann['task_instance']['images_path'])
        if self.combine_image:
            # set different context prompt for combined images
            for i in range(img_num):
                rmv_txt = '{image#%d}'% (i+1)
                rmv_tbl = '{table#%d}'% (i+1)
                context = context.replace(rmv_txt, f'<Image {i+1}> ')
                context = context.replace(rmv_tbl, f'<Image {i+1}> ')
            # context is without instruction here!
            # context = '<ImageHere>'*self.combine_image + '\n' + context # we do this later
        else:
            for i in range(img_num):
                rmv_txt = '{image#%d}'% (i+1)
                rmv_tbl = '{table#%d}'% (i+1)
                context = context.replace(rmv_txt, '<ImageHere>')
                context = context.replace(rmv_tbl, '<ImageHere>')

        # Set images paths
        raw_img_list = []
        if self.combine_image:
            combine_image_str = f'combined_{self.combine_image}_images'
            for p in ann['task_instance'][combine_image_str]:
                img_path = os.path.join(self.img_dir.replace(os.path.basename(self.img_dir), combine_image_str), p)
                raw_img_list.append(img_path)
        else:
            for p in ann['task_instance']['images_path']:
                img_path = os.path.join(self.img_dir, p)
                raw_img_list.append(img_path)


        '''
        For all datasets, we keep the instruction and question, and truncate the middle part from left.
        '''

        image_placeholder = '<ImageHere>'

        ret_img_list = []
        tokenized_instruction = self.tokenizer(task_instruction, add_special_tokens=False).input_ids
        tokenized_context = self.tokenizer(context, add_special_tokens=False).input_ids

        # needle datasets
        instruction_length = len(tokenized_instruction)
        length_for_context = self.max_context_len - instruction_length
        # break the question into fragments, then traverse the string in a reverse fashion
        context = context.split(image_placeholder)[::-1]

        past_total_len = 0
        context_id_chunks = []
        ret_img_list = []
        image_start = False # used for later, deciding we start with text string or image placeholder.

        for fragment in context:
            # add the text first
            cur_ids = self.tokenizer(fragment, add_special_tokens=False).input_ids
            cur_len = len(cur_ids)
            if cur_len + past_total_len > length_for_context:
                if len(context_id_chunks) == 0: # if there hasn't been any chunk, we want to truncate a piece from the current text.
                    context_id_chunks.insert(0, cur_ids[-length_for_context:])
                break # too long!
            image_start = False
            context_id_chunks.insert(0, cur_ids)
            past_total_len += cur_len

            # then concat images
            if not self.combine_image:
                if self.n_tokens_per_image + past_total_len > length_for_context:
                    break # too long!
                if len(raw_img_list) > 0:
                    image_start = True
                    ret_img_list.insert(0, raw_img_list.pop(-1))
                    past_total_len += self.n_tokens_per_image


        # concat everything together.
        # note that the ending of the input must be text, so we only need to take care of the start.
        # ret_context_str = image_placeholder if image_start else ''
        ret_context_str = ''
        for context_id_chunk in context_id_chunks[:-1]: # chunks are in correct order
            context_str = self.tokenizer.decode(context_id_chunk)
            ret_context_str += context_str
            ret_context_str += image_placeholder
        ret_context_str += self.tokenizer.decode(context_id_chunks[-1]) # add the last chunk without image placeholder appended

        if self.combine_image:
            assert len(raw_img_list) == 1, f"We only support 1 image for combined set, got {len(raw_img_list)} images."
            ret_img_list.insert(0, raw_img_list.pop(-1)) # bug for more than 1 image!!!!!
            ret_context_str = image_placeholder + '\n' + task_instruction + '\n' + ret_context_str
            pass
        else:
            if image_start:
                ret_context_str = image_placeholder + ret_context_str
            else:
                pass
            ret_context_str = task_instruction + '\n' + ret_context_str # prepend task instruction
        # done! We shall return `ret_context_str` and `ret_img_list`

        return {
            "sample_id": ann['sample_id'],
            "context": ret_context_str,
            "raw_img_list": ret_img_list, # a list of images
            "response": str(ann['response'])
        }

    def collate_fn(self, batch):
        """
        Custom collate function for batching samples.

        Parameters:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched data.
        """
        batch_data={}
        # Use the default key names
        batch_data['id'] = [sample['sample_id'] for sample in batch]
        batch_data['question'] = [sample['context'] for sample in batch]
        batch_data['image_path'] = [sample['raw_img_list'] for sample in batch]
        batch_data['gt_response'] = [sample['response'] for sample in batch]
        return batch_data
