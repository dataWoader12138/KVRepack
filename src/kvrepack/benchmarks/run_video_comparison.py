"""Stream an explicit original video prefix through one canonical method.

LookM uses a strict total prefill budget including its recent window. Other
methods retain native budget rules. Audits count active decode KV slots, not total VRAM or
HybridKV's full-history backing stores. No algorithm function is replaced by a
new implementation: the instrumentation calls and returns each original once.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from kvrepack.ablations.run_head_selection import (
    FRAMES, MIN_PIXELS, MAX_PIXELS, MODEL_SEED, atomic_json, cache_shapes,
    json_sha, sha256, tensor_fingerprints, timestamp,
)
from kvrepack.ablations.run_head_selection_full import output_lock, _fresh_prompt

METHODS = ('lookm', 'hybridkv_kvrepack', 'lookm_kvrepack')
TASKS = ('nextqa_oe_test', 'vatex_test')
NATIVE_HEADS, LAYERS = 4, 28
STRICT_LOOKM_POLICY = 'total_prefill_including_recent_v2'
LEGACY_LOOKM_POLICY = 'legacy_history_plus_recent_v1'
NATIVE_BUDGET_POLICY = 'native_nominal_budget_v1'
_OLD_POLICY_DESCRIPTION = 'Canonical nominal 10%; original window and per-head allocation unchanged; active KV slots audited separately from backing storage'


def normalize_budget_policy(method, policy=None):
    """Missing historical metadata is never promoted to the new strict policy."""
    if method == 'lookm':
        if policy == STRICT_LOOKM_POLICY: return STRICT_LOOKM_POLICY
        if policy in (None, LEGACY_LOOKM_POLICY, _OLD_POLICY_DESCRIPTION): return LEGACY_LOOKM_POLICY
        raise ValueError('Unknown LookM budget policy')
    if policy in (None, NATIVE_BUDGET_POLICY, _OLD_POLICY_DESCRIPTION): return NATIVE_BUDGET_POLICY
    raise ValueError('Unsupported budget policy for this method')


def validate_budget_audit(cache, *, method, budget_policy=None):
    policy = normalize_budget_policy(method, cache.get('budget_policy'))
    if budget_policy is not None and policy != normalize_budget_policy(method, budget_policy):
        raise ValueError('Cache budget policy differs from the declared contract/configuration')
    if policy == STRICT_LOOKM_POLICY:
        tokens = cache.get('input_tokens')
        if type(tokens) is not int or tokens < 1: raise ValueError('Missing actual strict-budget input length')
        capacity = math.floor(tokens*.1)
        window = min(32, capacity)
        if (capacity < 1 or cache.get('total_prefill_capacity') != capacity
                or cache.get('strict_budget_passed') is not True or cache.get('nominal_budget_ratio') != .1):
            raise ValueError('LookM total prefill capacity or strict-budget evidence is invalid')
        for layer in cache.get('layers', []):
            if (layer.get('before_tokens_per_head') != tokens or layer.get('head_lengths') != [capacity]*NATIVE_HEADS
                    or layer.get('total_prefill_capacity') != capacity or layer.get('configured_recent_window') != 32
                    or layer.get('recent_window') != window or layer.get('selected_history_tokens') != capacity-window):
                raise ValueError('LookM prefill exceeds or disagrees with total capacity including recent positions')
        if (cache.get('prefill_tokens_total') != LAYERS*NATIVE_HEADS*capacity
                or cache.get('uncompressed_tokens_total') != LAYERS*NATIVE_HEADS*tokens
                or cache.get('actual_retained_ratio', 1) > .1):
            raise ValueError('LookM total prefill cache violates the strict ten-percent cap')
    elif cache.get('strict_budget_passed') is True:
        raise ValueError('Legacy/native budget records cannot claim the strict v2 budget passed')
    return dict(budget_policy=policy, strict_budget_passed=True if policy==STRICT_LOOKM_POLICY else False if method=='lookm' else None,
                legacy_budget_result=policy==LEGACY_LOOKM_POLICY)


def validate_resume_contract(saved, current):
    if (current.get('method')=='lookm' and current.get('budget_policy')==STRICT_LOOKM_POLICY
            and saved.get('budget_policy')!=STRICT_LOOKM_POLICY):
        raise ValueError('Legacy LookM samples cannot resume into strict v2; use a fresh output directory')
    if saved != current: raise ValueError('Frozen run contract changed')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--limit', required=True, type=int, choices=(1, 200))
    for name in ('pretrained', 'hf-home', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def configure(args):
    from kvrepack.experiments.runner import build_plan
    overrides = ()
    if args.method == 'hybridkv_kvrepack':
        overrides = ('HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO=.50',
            'HYBRID_KVREPACK_COMPRESS_RATIO=.50', 'K_DIFF_ALPHA=2',
            'HYBRID_KVREPACK_SIMHEAD_SIM_SPACE=raw',
            'HYBRID_KVREPACK_K_MERGE_SPACE=raw_reapply')
    plan = build_plan(args.method, task=args.task, pretrained=str(args.pretrained),
        model_type='qwen2_5_vl-7b', frames=FRAMES, limit=args.limit,
        budget_ratio=.1, output=args.output, overrides=overrides)
    for name in list(os.environ):
        if name.startswith(('SNAPKV_', 'MADAKV_', 'HYBRID_', 'SPARSEMM_', 'LOOKM_')) or name in {
            'METHOD', 'MODEL_TYPE', 'BUDGET', 'BUDGET_RATIO', 'RATIO', 'K_DIFF_ALPHA',
            'HEAD_CLASSIFY', 'BUDGET_METHOD', 'K_MERGE_SPACE', 'KVREPACK_MERGE_SCORE_MODE'}:
            os.environ.pop(name)
    os.environ.update(plan.env)
    os.environ.update(HF_HOME=str(args.hf_home), HF_HUB_CACHE=str(args.hf_home/'hub'),
        HUGGINGFACE_HUB_CACHE=str(args.hf_home/'hub'), HF_DATASETS_CACHE=str(args.hf_home/'datasets'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return plan


def contract_for(args, data, plan):
    package = Path(__file__).resolve().parents[1]
    names = ('benchmarks/run_video_comparison.py', 'experiments/runner.py', 'experiments/registry.py',
        'observations/inputs.py', 'ablations/run_head_selection.py',
        'ablations/run_head_selection_full.py', 'baseline_utils.py', 'qwen_model.py',
        'qwen_model_croatt.py', 'monkeypatch.py', 'qwen_model_lookm_kvrepack.py',
        'merge_score.py', 'modeling_qwen2_5_vl.py', 'hybridkv_utils.py', 'hybridkv_cpu.py',
        'ablations/hybrid_head_selection.py', 'ablations/head_selection.py',
        'configs/param_config.json', 'data/head_score/qwen2_5.json')
    sources = {str(package/name): sha256(package/name) for name in names}
    contract = dict(schema_version=1, experiment='lookm_vs_kvrepack', method=args.method,
        task=args.task, target_docs=args.limit, target_generations=args.limit,
        dataset_rows=len(data['dataset']), model_snapshot=str(args.pretrained),
        model_metadata={p.name:sha256(p) for p in sorted(args.pretrained.iterdir())
            if p.is_file() and p.suffix in {'.json', '.txt'}},
        arrow_path=data['arrow_path'], arrow_sha256=data['arrow_sha256'],
        video_root=data['video_root'], task_yaml_sha256=data['task_yaml_sha256'],
        task_utils_sha256=data['task_utils_sha256'], included_yaml_sha256=data.get('included_yaml_sha256'),
        frames=FRAMES, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, budget_ratio=.1,
        algorithm_env=plan.resolved_config['algorithm_env'], model_seed=MODEL_SEED,
        dtype='bfloat16', attention='flash_attention_2', do_sample=False, use_cache=True,
        repetition_penalty=1.05, max_new_tokens=128 if args.task == 'nextqa_oe_test' else 64,
        sources_sha256=sources, versions={n:importlib.metadata.version(n) for n in
            ('torch', 'transformers', 'flash-attn', 'qwen-vl-utils', 'datasets', 'decord', 'numpy')},
        input_policy='Original Arrow prefix, no filtering or deduplication; fresh prompt per sample',
        budget_policy=STRICT_LOOKM_POLICY if args.method=='lookm' else NATIVE_BUDGET_POLICY)
    if args.method == 'hybridkv_kvrepack':
        import kvrepack_kernel
        contract['kernel'] = dict(path=str(Path(kvrepack_kernel.__file__).resolve()),
            sha256=sha256(kvrepack_kernel.__file__))
    return contract


class PrefillBudgetAudit:
    """Read-only capture around native prefill cache builders, one prompt at a time."""
    def __init__(self, method, *, module=None, budget_policy=None):
        self.method, self.module = method, module
        self.budget_policy = normalize_budget_policy(method, budget_policy if budget_policy is not None else STRICT_LOOKM_POLICY if method=='lookm' else NATIVE_BUDGET_POLICY)
        self.rows = []
        self.sample = None
        self.original = None

    def set_sample(self, task, doc_id, input_tokens):
        if self.sample is not None or type(input_tokens) is not int or input_tokens <= 0:
            raise ValueError('Finish the previous valid prefill audit first')
        self.sample = (task, doc_id, input_tokens)
        self.rows = []

    def _record(self, layer, key, value, lengths, **details):
        if self.sample is None or any(row['layer'] == int(layer) for row in self.rows):
            raise ValueError('Undeclared or repeated prefill layer')
        lengths = [int(n) for n in lengths]
        shape = list(key.shape)
        total = shape[0] if len(shape) == 2 else shape[0]*shape[1]*shape[2]
        if len(lengths) != NATIVE_HEADS or sum(lengths) != total or list(value.shape) != shape:
            raise ValueError('Native retained lengths disagree with actual prefill cache')
        self.rows.append(dict(layer=int(layer), before_tokens_per_head=self.sample[2],
            head_lengths=lengths, after_tokens_total=total, key_shape=shape,
            value_shape=list(value.shape), key_dtype=str(key.dtype), value_dtype=str(value.dtype), **details))

    def __enter__(self):
        if self.method == 'lookm':
            module = self.module or importlib.import_module('kvrepack.baseline_utils')
            self.owner, self.name = module.LookMCluster, 'update_kv'
            self.original = getattr(self.owner, self.name)
            def wrapped(obj, origin_key_states, query_states, origin_value_states):
                result = self.original(obj, origin_key_states, query_states, origin_value_states)
                details = dict(selected_history_tokens=int(obj.select_num),recent_window=int(obj.window_size),native_budget_ratio=float(obj.budget_ratio))
                if self.budget_policy == STRICT_LOOKM_POLICY:
                    capacity = math.floor(self.sample[2]*.1)
                    if (int(origin_key_states.shape[-2]) != self.sample[2] or float(obj.budget_ratio) != .1
                            or int(obj.window_size) != 32 or getattr(obj,'total_prefill_capacity',None) != capacity
                            or getattr(obj,'effective_window_size',None) != min(32,capacity)
                            or obj.select_num != capacity-min(32,capacity) or int(result[0].shape[-2]) != capacity):
                        raise ValueError('Original LookM output/effective window violates strict total prefill capacity')
                    details.update(total_prefill_capacity=capacity,configured_recent_window=int(obj.window_size),recent_window=int(obj.effective_window_size))
                self._record(obj.layer_idx, result[0], result[1], [result[0].shape[-2]]*NATIVE_HEADS,
                    **details)
                return result
        elif self.method == 'hybridkv_kvrepack':
            module = self.module or importlib.import_module('kvrepack.hybridkv_utils')
            self.owner, self.name = module.HybridKVCache, 'update'
            self.original = getattr(self.owner, self.name)
            def wrapped(obj, key_states, value_states, layer_idx, cache_kwargs=None):
                prefill = len(obj.key_cache) <= layer_idx
                result = self.original(obj, key_states, value_states, layer_idx, cache_kwargs)
                if prefill:
                    lengths = obj.layers[layer_idx].self_attn.kv_cluster.head_lens.detach().cpu().tolist()
                    budget = list(obj.sparse_budget_per_head[layer_idx][0])
                    if lengths != [int(n)+int(obj.local_len) for n in budget]:
                        raise ValueError('Hybrid native retained lengths drifted from allocated history plus local window')
                    self._record(layer_idx, obj.key_cache[layer_idx], obj.value_cache[layer_idx], lengths,
                        retained_history_budgets=budget, recent_window=int(obj.local_len))
                return result
        elif self.method == 'lookm_kvrepack':
            self.owner = self.module or importlib.import_module('kvrepack.qwen_model_lookm_kvrepack')
            self.name = '_finalize_prompt_cache'
            self.original = getattr(self.owner, self.name)
            def wrapped(config, cache):
                records = list(getattr(config, '_lookm_kvrepack_records', []))
                result = self.original(config, cache)
                for rec in records:
                    layer = int(rec['layer_idx'])
                    key, value = cache.key_cache[layer], cache.value_cache[layer]
                    self._record(layer, key, value, [key.shape[-2]]*NATIVE_HEADS,
                        final_history=rec.get('final_history'), candidate_history=rec.get('candidate_history'),
                        recent_window=int(config.lookm_kvrepack_window_size))
                return result
        else:
            raise ValueError('Unsupported audit method')
        setattr(self.owner, self.name, wrapped)
        return self

    def finish_sample(self, final_shapes, generated_tokens):
        if self.sample is None or sorted(row['layer'] for row in self.rows) != list(range(LAYERS)):
            raise ValueError('Missing native prefill layer coverage')
        rows = sorted(self.rows, key=lambda row: row['layer'])
        if len(final_shapes) != LAYERS or generated_tokens < 1:
            raise ValueError('Missing final cache or generated tokens')
        decode_steps = generated_tokens-1
        for row, final in zip(rows, final_shapes):
            shape = final['key_shape']
            final_total = shape[0] if len(shape) == 2 else shape[0]*shape[1]*shape[2]
            if (final.get('layer') != row['layer'] or final['value_shape'] != shape
                    or final_total-NATIVE_HEADS*decode_steps != row['after_tokens_total']):
                raise ValueError('Final cache minus decode steps does not reproduce captured prefill')
            row['final_tokens_total'] = final_total
            row['decode_added_tokens_per_head'] = decode_steps
        task, doc_id, tokens = self.sample
        total = sum(row['after_tokens_total'] for row in rows)
        denominator = LAYERS*NATIVE_HEADS*tokens
        audit = dict(status='completed', method=self.method, task=task, doc_id=doc_id,
            input_tokens=tokens, native_kv_heads=NATIVE_HEADS, layers=rows, layers_count=LAYERS,
            nominal_budget_ratio=.1, prefill_tokens_total=total,
            uncompressed_tokens_total=denominator, actual_retained_ratio=total/denominator,
            scope='Logical active decode KV slots; excludes full-history offload/backing stores and is not total VRAM')
        audit.update(budget_policy=self.budget_policy,strict_budget_passed=self.budget_policy==STRICT_LOOKM_POLICY if self.method=='lookm' else None)
        if self.budget_policy == STRICT_LOOKM_POLICY: audit['total_prefill_capacity'] = math.floor(tokens*.1)
        validate_budget_audit(audit,method=self.method,budget_policy=self.budget_policy)
        self.sample = None
        return audit

    def __exit__(self, *args):
        if self.original is not None:
            setattr(self.owner, self.name, self.original)


def validate_sample(row, *, doc_id, contract_sha256, method=None, task=None, budget_policy=None):
    """Validate a newly committed/resumed row without importing model packages."""
    method, task = method or row.get('method'), task or row.get('task')
    if (row.get('schema_version') != 1 or row.get('status') != 'completed'
            or row.get('doc_id') != doc_id or row.get('contract_sha256') != contract_sha256
            or row.get('method') != method or method not in METHODS
            or row.get('task') != task or task not in TASKS):
        raise ValueError('Sample identity/status/contract mismatch')
    meta, fp = row.get('reference_metadata', {}), row.get('input_fingerprints', {})
    if (meta.get('task') != task or meta.get('doc_id') != doc_id
            or row.get('source_video_id') != meta.get('source_video_id')
            or row.get('video_id') != meta.get('video_id') or not fp.get('input_hashes')
            or set(fp) != {'input_hashes', 'input_shapes', 'input_dtypes'}
            or any(row.get(k) != v for k, v in fp.items())
            or set(fp['input_hashes']) != set(fp['input_shapes'])
            or set(fp['input_hashes']) != set(fp['input_dtypes'])):
        raise ValueError('Input fingerprints or original source metadata mismatch')
    ids, config = row.get('generated_ids'), row.get('config', {})
    if (not isinstance(ids, list) or not ids or any(type(n) is not int or n < 0 for n in ids)
            or row.get('generated_tokens') != len(ids) or row.get('text') != row.get('prediction')
            or not isinstance(row.get('text'), str) or row.get('eos_reached') != (ids[-1] == row.get('eos_token_id'))
            or len(ids) > config.get('max_new_tokens', 0)
            or config.get('stream_contract_sha256') != contract_sha256
            or config.get('method') != method or config.get('task') != task
            or config.get('budget_ratio') != .1):
        raise ValueError('Generated tokens/configuration mismatch')
    audit, cache = row.get('audit', {}), row.get('cache_audit', {})
    policy = normalize_budget_policy(method,config.get('budget_policy'))
    if budget_policy is not None and policy != normalize_budget_policy(method,budget_policy):
        raise ValueError('Sample configuration budget policy differs from the frozen contract')
    validate_budget_audit(cache,method=method,budget_policy=policy)
    if policy==STRICT_LOOKM_POLICY:
        tokens=cache['input_tokens']
        if (row.get('input_tokens')!=tokens or meta.get('input_tokens')!=tokens
                or fp.get('input_shapes',{}).get('input_ids')!=[1,tokens]):
            raise ValueError('Strict budget denominator differs from actual prefill input length')
    layers = cache.get('layers', [])
    if (audit.get('status') != 'completed' or cache.get('status') != 'completed'
            or cache.get('method') != method or cache.get('task') != task or cache.get('doc_id') != doc_id
            or cache.get('native_kv_heads') != NATIVE_HEADS or cache.get('layers_count') != LAYERS
            or len(layers) != LAYERS or [r.get('layer') for r in layers] != list(range(LAYERS))
            or len(row.get('final_cache_shapes', [])) != LAYERS):
        raise ValueError('Incomplete native cache audit')
    total = 0
    for layer, final in zip(layers, row['final_cache_shapes']):
        lengths = layer.get('head_lengths', [])
        shape = layer.get('key_shape', [])
        slots = shape[0] if len(shape) == 2 else shape[0]*shape[1]*shape[2] if len(shape) == 4 else -1
        final_shape = final.get('key_shape', [])
        final_slots = final_shape[0] if len(final_shape) == 2 else final_shape[0]*final_shape[1]*final_shape[2] if len(final_shape) == 4 else -1
        if (len(lengths) != NATIVE_HEADS or any(type(n) is not int or n <= 0 for n in lengths)
                or sum(lengths) != slots or slots != layer.get('after_tokens_total')
                or layer.get('value_shape') != shape or final.get('value_shape') != final_shape
                or final.get('layer') != layer['layer']
                or final_slots-NATIVE_HEADS*(len(ids)-1) != slots):
            raise ValueError('Actual native cache sizes disagree with prefill/decode accounting')
        total += slots
    denominator = LAYERS*NATIVE_HEADS*cache.get('input_tokens', 0)
    if (cache.get('prefill_tokens_total') != total or cache.get('uncompressed_tokens_total') != denominator
            or denominator <= 0 or cache.get('actual_retained_ratio') != total/denominator):
        raise ValueError('Aggregate actual retained-token ratio mismatch')
    return row


def run_locked(args):
    plan = configure(args)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from kvrepack.observations import inputs
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    torch.set_num_threads(4)
    data = inputs.load_task(args.hf_home, args.task)
    if args.limit > len(data['dataset']):
        raise ValueError('Requested original prefix exceeds test split')
    contract = contract_for(args, data, plan)
    contract_path = args.output/'run_contract.json'
    if args.resume:
        validate_resume_contract(json.loads(contract_path.read_text()),contract)
    else:
        atomic_json(contract_path, contract)
    identity = json_sha(contract)
    paths = sorted((args.output/'samples').glob('*.json'))
    if len(paths) > args.limit or [p.name for p in paths] != [f'{i:06d}.json' for i in range(len(paths))]:
        raise ValueError('Resume requires a contiguous completed original prefix')
    for doc_id, path in enumerate(paths):
        validate_sample(json.loads(path.read_text()), doc_id=doc_id, contract_sha256=identity,
            method=args.method, task=args.task,budget_policy=contract['budget_policy'])
    initial = len(paths)
    state_path = args.output/'run_state.json'
    previous = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
    state = dict(status='loading', method=args.method, task=args.task, target_docs=args.limit,
        target_generations=args.limit, completed_docs=initial, completed_generations=initial,
        contract_sha256=identity, started_utc=previous.get('started_utc', timestamp()),
        updated_utc=timestamp(), estimated_remaining_seconds=None)
    atomic_json(state_path, state)
    started = time.monotonic()
    try:
        if initial == args.limit:
            state.update(status='complete', finished_utc=timestamp()); atomic_json(state_path, state)
            return 0
        processor = AutoProcessor.from_pretrained(str(args.pretrained), local_files_only=True,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        if args.method == 'hybridkv_kvrepack':
            from kvrepack.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        else:
            from kvrepack.monkeypatch import replace_qwen
            replace_qwen(args.method)
            from transformers import Qwen2_5_VLForConditionalGeneration
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),
            torch_dtype=torch.bfloat16, device_map=0, attn_implementation='flash_attention_2',
            local_files_only=True).eval()
        kwargs = dict(eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id, do_sample=False, temperature=None,
            top_p=None, num_beams=1, max_new_tokens=contract['max_new_tokens'],
            repetition_penalty=1.05, use_cache=True, return_dict_in_generate=True, output_scores=False)
        fields = ('method', 'task', 'model_snapshot', 'frames', 'budget_ratio', 'algorithm_env',
            'min_pixels', 'max_pixels', 'dtype', 'attention', 'model_seed', 'max_new_tokens',
            'do_sample', 'use_cache', 'repetition_penalty','budget_policy')
        config = {key:contract[key] for key in fields}
        config.update(stream_contract_sha256=identity, source_files=contract['sources_sha256'],
            effective_generation_config=model.generation_config.to_dict(), generation_overrides=kwargs)
        config_path = args.output/'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Effective generation configuration changed on resume')
        atomic_json(config_path, config)
        for doc_id in range(initial, args.limit):
            state.update(status='preparing', current_doc_id=doc_id, updated_utc=timestamp())
            atomic_json(state_path, state)
            random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
            batch, metadata = inputs.prepare_sample(data, doc_id, processor, FRAMES, MIN_PIXELS, MAX_PIXELS)
            fingerprints = tensor_fingerprints(batch)
            gpu = {k:v.to('cuda:0') for k,v in batch.items()}
            _fresh_prompt(model)
            for layer in model.model.language_model.layers:
                if hasattr(layer.self_attn, 'kv_cluster'):
                    delattr(layer.self_attn, 'kv_cluster')
                if hasattr(layer.self_attn.config, '_lookm_kvrepack_records'):
                    layer.self_attn.config._lookm_kvrepack_records = []
            state.update(status='running', updated_utc=timestamp()); atomic_json(state_path, state)
            with ExitStack() as stack:
                recorder = stack.enter_context(PrefillBudgetAudit(args.method,budget_policy=contract['budget_policy']))
                recorder.set_sample(args.task, doc_id, metadata['input_tokens'])
                selector = None
                if args.method == 'hybridkv_kvrepack':
                    from kvrepack.ablations.hybrid_head_selection import HybridHeadSelectionAdapter
                    selector = stack.enter_context(HybridHeadSelectionAdapter(strategy='similarity', seed=0))
                    selector.set_sample(args.task, doc_id)
                torch.cuda.reset_peak_memory_stats(); began = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(**gpu, **kwargs)
                torch.cuda.synchronize(); elapsed = time.monotonic()-began
                ids = result.sequences[0, batch['input_ids'].shape[-1]:].tolist()
                final_shapes = cache_shapes(result.past_key_values)
                cache_audit = recorder.finish_sample(final_shapes, len(ids))
                audit = selector.finish_sample() if selector else dict(cache_audit)
                if selector:
                    audit['prefill_cache_budget'] = cache_audit
            text = processor.batch_decode([ids], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            row = dict(schema_version=1, status='completed', method=args.method, task=args.task,
                doc_id=doc_id, contract_sha256=identity, source_video_id=metadata['source_video_id'],
                video_id=metadata['video_id'], reference_metadata=metadata, text=text, prediction=text,
                generated_ids=ids, generated_tokens=len(ids), input_fingerprints=fingerprints,
                **fingerprints, input_tokens=metadata['input_tokens'], config=config, audit=audit,
                cache_audit=cache_audit, max_new_tokens=contract['max_new_tokens'],
                eos_token_id=processor.tokenizer.eos_token_id,
                eos_reached=bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
                final_cache_shapes=final_shapes, runtime=dict(generation_seconds=elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30), completed_utc=timestamp())
            if tensor_fingerprints(batch) != fingerprints:
                raise ValueError('Generation mutated the shared CPU input')
            validate_sample(row, doc_id=doc_id, contract_sha256=identity, method=args.method, task=args.task,budget_policy=contract['budget_policy'])
            path = args.output/'samples'/f'{doc_id:06d}.json'
            if path.exists():
                raise ValueError('Refusing to overwrite a committed sample')
            atomic_json(path, row)
            completed = doc_id+1; elapsed_run = time.monotonic()-started
            state.update(completed_docs=completed, completed_generations=completed,
                updated_utc=timestamp(), elapsed_current_session_seconds=elapsed_run,
                estimated_remaining_seconds=elapsed_run/(completed-initial)*(args.limit-completed))
            atomic_json(state_path, state)
            print(json.dumps(dict(stage='sample_committed', method=args.method, task=args.task,
                doc_id=doc_id, completed_docs=completed, target_docs=args.limit,
                generated_tokens=len(ids), actual_retained_ratio=cache_audit['actual_retained_ratio'])), flush=True)
            del result, row, audit, cache_audit, gpu, batch
        state.update(status='complete', finished_utc=timestamp(), estimated_remaining_seconds=0)
        atomic_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


def main(argv=None):
    args = parse_args(argv)
    for name in ('pretrained', 'hf_home', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.pretrained.is_dir():
        raise FileNotFoundError(args.pretrained)
    with output_lock(args.output, resume=args.resume):
        return run_locked(args)


if __name__ == '__main__':
    raise SystemExit(main())
