"""Prepare a MileBench image input once and reuse it across head interventions."""

from __future__ import annotations

import importlib
import os
import random
import time

from ..benchmarks.milebench.workers import MileBenchWorker, cache_statistics, qwen_messages
from .head_selection import _tensor_record


def tensor_fingerprints(inputs):
    return {name: _tensor_record(value) for name, value in inputs.items()}


class PairedMileBenchWorker(MileBenchWorker):
    def __init__(self, config):
        if config["model"] != "qwen2_5_vl":
            raise ValueError("The referenced head-selection adapters support Qwen2.5-VL only")
        super().__init__(config)
        # The reference ablation fixes BF16 for both conditions; image resize and
        # prompt construction still follow MileBench rather than video inputs.
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.torch.backends.cudnn.allow_tf32 = False
        if self.model.dtype != self.torch.bfloat16:
            raise ValueError("Paired Random-head generation requires actual BF16 model weights")
        text = self.model.config.text_config
        self.geometry = {"layers": text.num_hidden_layers, "native_kv_heads": text.num_key_value_heads}
        self.method = config["public_method"]
        self.generation_metadata = {"effective_model_generation_config": self.model.generation_config.to_dict(),
                                    "explicit_generation_kwargs": self.gen_kwargs, "dtype": str(self.model.dtype),
                                    "resolved_model_revision": getattr(self.model.config, "_commit_hash", None)}

    def reset_rng(self):
        seed = self.config["seed"]
        random.seed(seed)
        importlib.import_module("numpy").random.seed(seed)
        self.torch.manual_seed(seed)

    def prepare_one(self, sample):
        self.reset_rng()
        messages = qwen_messages(sample["context"], sample["raw_img_list"], self.config["image"])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = importlib.import_module("qwen_vl_utils").process_vision_info([messages])
        try:
            batch = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        finally:
            for image in images or ():
                if hasattr(image, "close"):
                    image.close()
        os.environ["QUESTION"] = sample["context"].replace("<ImageHere>", "").strip()
        # This same GPU tensor mapping is reused for all arms, with before/after
        # fingerprints checked by the caller. Each generate gets a fresh cache.
        return batch.to(self.device)

    def fresh_prompt(self):
        # Match 6c65bf8: clear generation-local position/cache state without
        # changing any classifier, per-head budget or merge hyperparameter.
        for obj in (self.model, getattr(self.model, "model", None)):
            if obj is not None and hasattr(obj, "rope_deltas"):
                obj.rope_deltas = None
        for layer in self.model.model.language_model.layers:
            attention = layer.self_attn
            attention.kv_seq_len = 0
            if hasattr(attention.config, "_snapkv_kvrepack_records"):
                attention.config._snapkv_kvrepack_records = []
            if self.method == "hybridkv_kvrepack" and hasattr(attention, "kv_cluster"):
                delattr(attention, "kv_cluster")
        self.reset_rng()

    def selector(self, condition):
        if self.method == "hybridkv_kvrepack":
            from .hybrid_head_selection import HybridHeadSelectionAdapter
            return HybridHeadSelectionAdapter(strategy=condition["strategy"], seed=condition["mask_seed"])
        from .head_selection import SnapHeadSelectionAdapter
        module = importlib.import_module("kvrepack.qwen_model_snapkv_kvrepack")
        return SnapHeadSelectionAdapter(module, strategy=condition["strategy"], seed=condition["mask_seed"])

    def generate_prepared(self, sample, prepared):
        input_len = int(prepared["input_ids"].shape[-1])
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode(), self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
            generated = self.model.generate(
                **prepared, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                **self.gen_kwargs, use_cache=True, return_dict_in_generate=True,
            )
        self.torch.cuda.synchronize(self.device)
        ids = generated.sequences[0, input_len:].tolist()
        answer = self.processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        eos = self.tokenizer.eos_token_id
        maximum = self.gen_kwargs["max_new_tokens"]
        return {"answer": answer, "runtime": {
            "input_tokens": input_len, "generated_ids": ids, "generated_tokens": len(ids),
            "eos_token_id": eos, "eos_reached": bool(ids and ids[-1] == eos),
            "hit_max_new_tokens": len(ids) >= maximum,
            "truncated_without_eos": len(ids) >= maximum and not bool(ids and ids[-1] == eos),
            "generation_seconds": time.perf_counter() - started,
            "cache": cache_statistics(getattr(generated, "past_key_values", None)),
        }}
