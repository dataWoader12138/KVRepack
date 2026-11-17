"""Source-compatible MileBench metrics with strict sample coverage and offline CLI.

Eval is extracted from HybridKV-opensource/hybridkv/evaluate.py. Its scoring
algorithm is preserved; metric imports occur only when a metric is evaluated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re

from .dataset import _sample_id, _validate_annotation, load_annotation


class Eval:

    def __init__(self):
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index):
        if index < 26:
            return chr(index+65)
        elif index < 52:
            return 'A'+chr(index+65-26)
        else:
            return 'B'+chr(index+65-26-26)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip().lower()
        return answer

    def get_image_quantity_level(self, sample):
        # 2-5 6-31 32-109
        image_num = len(sample['image'])
        if image_num < 6:
            return 'Few'
        elif image_num > 31:
            return 'Many'
        else:
            return 'Medium'

    def evaluate_rouge(self, predictions, core_json):
        import numpy as np
        from rouge import Rouge
        # get image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent.')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'image_quantity_level' in pre.keys()

        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, res in enumerate(predictions):
            sample_id = res['sample_id']
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            assert gt_ans != ''
            if pred_ans == '':
                score = 0
            else:
                score = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(score)
            image_quantity_level_cnt[self.get_image_quantity_level(res)].append(score)
            eval_list.append({'id':str(sample_id),'score':str(round(score,3))})
        return {
            'Rouge-L f': np.mean(acc['f']),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list

    def match_choice(self, text, option):
        '''Return: A B C D...'''

        def preprocess_option_string(option_string):
            # First, preprocess the option text to normalize it
            processed_option = self.process(option_string)

            # Then, escape any special regex characters in the processed option text
            # List of regex special characters that need to be escaped
            special_chars = ["\\", ".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "|", "(", ")"]
            # Escape the special characters by prefixing them with a backslash
            for char in special_chars:
                if char in processed_option:
                    processed_option = processed_option.replace(char, "\\" + char)
            # escaped_option = escape_special_chars(processed_option)
            return processed_option

        if text == "":
            return 'C'
        try:
            # Maybe start from the head
            # 1. Char+Choice: `A. Blastomycosis`
            option_str = "|".join([preprocess_option_string(f"{k} {v}")for k,v in option.items()])
            option_pattern = rf'({option_str})'
            option_res = re.search(option_pattern, text, re.S)   # NOTE we dont use match_all
            if option_res:
                return (option_res.group(0)[0]).upper()

            # 2. Choice: `Blastomycosis`
            option_str = "|".join([preprocess_option_string(v).replace(' ', '') for k,v in option.items()])
            option_pattern = rf'({option_str})'
            option_res = re.search(option_pattern, text.replace(' ', ''), re.S)   # NOTE we dont use match_all
            if option_res:
                for k, v in option.items():
                    if option_res[0].strip() == preprocess_option_string(v).replace(' ', ''):
                        return k.upper()

            # 3. Char: `A` `AB`
            if len(text) in [1,2] and text.upper() in option.keys():
                return text.upper()

            # use gpt extract

        except Exception as e:
            print(f"something wrong during match_choice {text}: {e}")
            return text
        return "".join([i.upper() for i in text if i.upper() in option])

    def judge_multi_choice(self, sample):
        sample_id = sample['sample_id']
        gt_ans = sample["gt_response"]
        pred_ans = sample["pred_response"]
        choice_list = sample['choice_list']
        assert gt_ans in choice_list
        # Convert choice_list to a dictionary format expected by match_choice
        option_dict = {self.char(i): choice for i, choice in enumerate(choice_list)}

        # Use match_choice to determine the selected answer from pred_ans
        selected_answer = self.match_choice(pred_ans, option_dict)

        # Check if the selected answer matches the ground truth
        gt_ans_chr = self.char(choice_list.index(sample["gt_response"]))
        if selected_answer == gt_ans_chr:
            return 1, selected_answer
        else:
            return 0, selected_answer

    def process_sample(self, sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])
        for i in range(len(sample['choice_list'])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, predictions, core_json):
        import numpy as np
        '''
        predictions: raw prediction file output by models
        '''
        # get choice_list & image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent. {len(predictions)}!={len(core_json["data"])}')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            new_pres[int(sample['sample_id'])]['choice_list'] = sample['task_instance']['choice_list']
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'choice_list' in pre.keys()
            assert 'image_quantity_level' in pre.keys()

        correct = 0
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, sample in enumerate(predictions):
            # Process string
            self.process_sample(sample)
            # Score

            score, extracted_answer = self.judge_multi_choice(sample)
            sample['extracted'] = extracted_answer
            sample['result'] = score
            eval_list.append({'id':str(sample['sample_id']), 'score': str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(score)
        return predictions, {
            'Accuracy': correct/len(predictions),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list

    def evaluate_needle(self, predictions, core_json, needle=True):
        import numpy as np
        # get choice_list & image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent. {len(predictions)}!={len(core_json["data"])}')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'image_quantity_level' in pre.keys()

        correct = 0
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, sample in enumerate(predictions):
            # Process string
            sample_id = sample['sample_id']
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])

            # Score
            if needle:
                score = 1 if gt_ans in pred_ans.split() else 0
            else:
                score = 1 if gt_ans in pred_ans else 0

            sample['result'] = score
            eval_list.append({'id':str(sample['sample_id']), 'score': str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(score)
        return {
            'Accuracy': correct/len(predictions),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list


def evaluate_predictions(predictions: list[dict], annotation: dict, task: str) -> dict:
    """Score an exact selected ID set with authoritative annotation ground truth.

    Source metrics mutate their inputs. Deep copies preserve caller-owned data;
    ordering follows the selected annotation rather than prediction arrival order.
    Unlike the old count-only guard, duplicate, extra and missing IDs are errors.
    """
    core = deepcopy(annotation)
    _validate_annotation(core)
    if not core["data"]:
        raise ValueError("cannot score an empty annotation subset")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    by_id = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or "sample_id" not in prediction:
            raise ValueError("each prediction must contain sample_id")
        sample_id = _sample_id(prediction["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        if not isinstance(prediction.get("pred_response"), str):
            raise ValueError(f"sample {sample_id}: pred_response must be text")
        by_id[sample_id] = deepcopy(prediction)
    expected_ids = [sample["sample_id"] for sample in core["data"]]
    missing = sorted(set(expected_ids) - by_id.keys())
    extra = sorted(by_id.keys() - set(expected_ids))
    if missing or extra:
        raise ValueError(f"prediction sample_id coverage mismatch: missing={missing}, extra={extra}")
    selected = []
    for sample in core["data"]:
        sample_id = sample["sample_id"]
        prediction = by_id[sample_id]
        prediction["sample_id"] = sample_id
        ground_truth = str(sample["response"])
        if "gt_response" in prediction and str(prediction["gt_response"]) != ground_truth:
            raise ValueError(f"sample {sample_id}: prediction ground truth does not match annotation")
        prediction["gt_response"] = ground_truth
        if "image_quantity_level" not in sample:
            count = len(sample["task_instance"]["images_path"])
            sample["image_quantity_level"] = "Few" if count < 6 else "Many" if count > 31 else "Medium"
        selected.append(prediction)
    scorer = Eval()
    extracted = None
    if "NeedleInAHaystack" in task or "MMCoQA" in task:
        metrics, scores = scorer.evaluate_needle(selected, core, needle="NeedleInAHaystack" in task)
    elif core["meta_data"]["question_type"] == "open-ended":
        metrics, scores = scorer.evaluate_rouge(selected, core)
    else:
        extracted, metrics, scores = scorer.evaluate_multichoice(selected, core)
    original_count = core["meta_data"].get("num_sample")
    if not isinstance(original_count, int) or original_count < len(expected_ids):
        original_count = None
    result = {"task": task, "sample_ids": expected_ids, "sample_count": len(expected_ids),
              "annotation_total_samples": original_count,
              "full_task": len(expected_ids) == original_count if original_count is not None else None,
              "eval": metrics, "eval_score": scores}
    if extracted is not None:
        result["pred_with_extracted"] = extracted
    return result


def _output_path(output_dir: str | os.PathLike) -> Path:
    path = Path(output_dir).expanduser().resolve()
    package = Path(__file__).resolve().parents[2]
    if path.is_relative_to(package):
        raise ValueError("evaluation output must not be inside the package source tree")
    source_root = package.parent
    repository = source_root.parent
    if source_root.name == "src" and (repository / "pyproject.toml").is_file():
        if path.is_relative_to(repository) and not path.is_relative_to(repository / "outputs"):
            raise ValueError("output within the repository must be under outputs/, outside source and audit files")
    return path


def write_evaluation(result: dict, output_dir: str | os.PathLike) -> Path:
    """Write one scoring run to a new directory; never overwrite an old result."""
    output = _output_path(output_dir)
    payloads = {"eval.json": result["eval"], "eval_score.json": result["eval_score"],
                "evaluation_metadata.json": {key: result.get(key) for key in (
                    "task", "sample_ids", "sample_count", "annotation_total_samples", "full_task")}}
    if "pred_with_extracted" in result:
        payloads["pred_with_extracted.json"] = result["pred_with_extracted"]
    # Serialize before mkdir so malformed values cannot leave a misleading run.
    serialized = {name: json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                  for name, payload in payloads.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in serialized.items():
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score existing MileBench predictions without loading a model")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True, help="new scoring directory; existing directories are rejected")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        annotation = load_annotation(args.data_root, args.task, args.limit)
        predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        result = evaluate_predictions(predictions, annotation, args.task)
        write_evaluation(result, args.output)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"milebench evaluation failed: {exc}\n")
    print(json.dumps({"task": args.task, "sample_count": result["sample_count"],
                      "eval": result["eval"], "output": str(Path(args.output).resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
