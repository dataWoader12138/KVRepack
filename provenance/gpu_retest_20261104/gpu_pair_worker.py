"""Small real-video numerical comparison. No model/data download or external judge."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import sys

BASE = Path('/root/kvrepack-gpu-work')
REPO = BASE / 'KVRepack'
ARTIFACT = Path('/root/autodl-tmp/HybridKV-artifact/HybridKV-artifact')
INVENTORY = Path('/tmp/gpu_retest_data_inventory.json')


def digest_tensor(t):
    import torch
    raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def task_prompt(task, doc):
    import yaml
    family = 'nextqa' if task.startswith('nextqa') else 'vatex'
    folder = REPO / 'vendor/lmms-eval/lmms_eval/tasks' / family
    source = folder / 'utils.py'
    fn = 'nextqa_doc_to_text' if family == 'nextqa' else 'vatex_test_doc_to_text'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == fn)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    yaml_source = (folder / (task + '.yaml')).read_text()
    config = yaml.safe_load('\n'.join(line for line in yaml_source.splitlines() if '!function' not in line))
    return namespace[fn](doc, config['lmms_eval_specific_kwargs']['default'])


def prepare(count):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    inventory = json.loads(INVENTORY.read_text())
    model_path = inventory['models'][0]['snapshot_path']
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1605632, local_files_only=True)
    folder = BASE / 'inputs'
    folder.mkdir(exist_ok=True)
    records = []
    for data in inventory['datasets']:
        task = data['task']
        ds = Dataset.from_file(data['arrow_path'])
        for doc_id in range(count):
            doc = ds[doc_id]
            video = data['first_docs'][doc_id]['video_path']
            prompt = task_prompt(task, doc)
            messages = [{'role':'system','content':'You are a helpful assistant.'},
                        {'role':'user','content':[{'type':'video','video':video,'max_pixels':1605632,'min_pixels':256*28*28},
                                                  {'type':'text','text':prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            total = videos[0].shape[0]
            indices = np.linspace(0, total - 1, 64, dtype=int)
            if total-1 not in indices: indices = np.append(indices, total-1)
            videos[0] = videos[0][indices]
            batch = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt')
            inputs = dict(batch)
            hashes = {key:digest_tensor(value) for key,value in inputs.items()}
            path = folder / f'{task}_{doc_id}.pt'
            torch.save(inputs, path)
            record = {'id':f'{task}:{doc_id}','task':task,'doc_id':doc_id,'video_path':video,
                      'video_sha256':hashlib.sha256(Path(video).read_bytes()).hexdigest(),
                      'prompt':prompt,'input_path':str(path),'input_hashes':hashes,
                      'input_shapes':{k:list(v.shape) for k,v in inputs.items()},
                      'input_tokens':int(inputs['input_ids'].shape[-1]),'frames':64,
                      'source_dataset_arrow_sha256':data['arrow_sha256']}
            records.append(record)
            save_json(BASE/'inputs_manifest.json', {'model_path':model_path,'records':records})
            print(json.dumps({'prepared':record['id'],'tokens':record['input_tokens'],'input_MiB':round(path.stat().st_size/2**20,1)},ensure_ascii=False),flush=True)


def cache_snapshot(cache):
    if cache is None: return []
    output = []
    for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
        output.append({'layer':layer, 'key':list(key.shape), 'value':list(value.shape),
                       'key_sha256':digest_tensor(key), 'value_sha256':digest_tensor(value)})
    return output


def run(implementation, method):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((BASE/'inputs_manifest.json').read_text())
    public = method + '_kvrepack'
    common = {'BUDGET_RATIO':'0.10', 'RATIO':'0.10', 'K_DIFF_ALPHA':'2', 'MODEL_TYPE':'qwen2_5_vl-3b'}
    # Clear inherited algorithm flags, retaining only the infrastructure supplied by the driver.
    for key in list(os.environ):
        if any(part in key for part in ['DECOUPLE','KVREPACK','SIMHEAD']) or key in ['BUDGET','BUDGET_RATIO','METHOD','RATIO','K_DIFF_ALPHA']:
            os.environ.pop(key)
    os.environ.update(common)
    prefix = 'SNAPKV' if method == 'snapkv' else 'MADAKV'
    cr = .9 if method == 'snapkv' else .7
    gate = .7 if method == 'snapkv' else .5
    legacy_env = {f'{prefix}_SIMHEAD_MERGE_RATIO':str(gate), f'{prefix}_SIMHEAD_SIM_SPACE':'raw',
                  f'{prefix}_DECOUPLE_COMPRESS_RATIO':str(cr), f'{prefix}_DECOUPLE_SCORE_CORRECTION':'none',
                  f'{prefix}_DECOUPLE_ENABLE':'1', f'{prefix}_DECOUPLE_WINDOW_SIZE':'32',
                  f'{prefix}_DECOUPLE_GQA_FUNC':'mean', f'{prefix}_DECOUPLE_BUDGET_RATIO':'.1'}
    if method == 'snapkv':
        legacy_env.update(SNAPKV_DECOUPLE_KERNEL_SIZE='5', SNAPKV_DECOUPLE_POOLING='avgpool')
    else:
        legacy_env.update(BUDGET='1024',MADAKV_DECOUPLE_THRESHOLD='.8',MADAKV_DECOUPLE_K_DIFF_ALPHA='2')
    if implementation == 'original':
        sys.path.insert(0, str(ARTIFACT))
        os.environ.update(legacy_env, METHOD=method+'_decouplekv')
        patch = importlib.import_module('hybridkv.monkeypatch')
        patch.replace_qwen(method+'_decouplekv')
    else:
        sys.path.insert(0, str(REPO/'src'))
        os.environ.update({k.replace('DECOUPLE','KVREPACK'):v for k,v in legacy_env.items()}, METHOD=public)
        patch = importlib.import_module('kvrepack.monkeypatch')
        patch.replace_qwen(public)
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoProcessor
    module = importlib.import_module(('hybridkv.qwen_model_'+method+'_decouplekv') if implementation=='original'
                                     else ('kvrepack.qwen_model_'+method+'_kvrepack'))
    expected_root = ARTIFACT if implementation == 'original' else REPO
    assert Path(module.__file__).resolve().is_relative_to(expected_root)
    processor = AutoProcessor.from_pretrained(manifest['model_path'], local_files_only=True)
    print(json.dumps({'stage':'loading','implementation':implementation,'method':method,'source':module.__file__}),flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(manifest['model_path'],torch_dtype=torch.bfloat16,
                device_map=0,attn_implementation='flash_attention_2',local_files_only=True).eval()
    result_dir=BASE/'results';result_dir.mkdir(exist_ok=True)
    result_file=result_dir/f'{method}_{implementation}.json'
    scores_file=result_dir/f'{method}_{implementation}_scores.pt'
    records=[];saved_scores={};prefill=[]
    def snapshot(module, args, kwargs, output):
        if output.last_hidden_state.shape[1]>1:
            prefill.append(cache_snapshot(output.past_key_values))
    hook=model.model.language_model.register_forward_hook(snapshot,with_kwargs=True)
    for item in manifest['records']:
        torch.cuda.reset_peak_memory_stats()
        inputs=torch.load(item['input_path'],map_location='cpu',weights_only=True)
        hashes={k:digest_tensor(v) for k,v in inputs.items()}
        assert hashes==item['input_hashes']
        gpu_inputs={k:v.to('cuda') for k,v in inputs.items()}
        start=time.monotonic();prefill.clear()
        with torch.inference_mode():
            output=model.generate(**gpu_inputs,eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,do_sample=False,temperature=None,top_p=None,
                    num_beams=1,max_new_tokens=64,use_cache=True,return_dict_in_generate=True,
                    output_scores=False,output_logits=True)
        torch.cuda.synchronize()
        ids=output.sequences[0,item['input_tokens']:].tolist()
        saved_scores[item['id']]=torch.stack(output.logits).squeeze(1).detach().cpu()
        config={'model_snapshot':Path(manifest['model_path']).name,'model':'Qwen2.5-VL-3B-Instruct',
                'method':public,'simtop':gate,'compress_ratio':cr,'alpha':2,'budget_ratio':.1,
                'score_correction':'none','frames':64,'max_new_tokens':64,'dtype':'bfloat16','attention':'flash_attention_2'}
        record={'id':item['id'],'input_hashes':hashes,'input_shapes':item['input_shapes'],
                'generated_ids':ids,'text':processor.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                'prefill_cache_shapes':prefill[0] if prefill else [],
                'cache_shapes':cache_snapshot(output.past_key_values),'config':config,
                'seconds':time.monotonic()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
        records.append(record)
        save_json(result_file, {'implementation':implementation,'source_module':module.__file__,
                    'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                    'gpu':torch.cuda.get_device_name(0),'torch_version':torch.__version__,
                    'scores_kind':'unprocessed_generation_logits','records':records})
        torch.save(saved_scores,scores_file)
        print(json.dumps({'done':item['id'],'implementation':implementation,'method':method,'tokens':len(ids),
                          'text':record['text'],'seconds':record['seconds'],'peak_GiB':record['peak_allocated_GiB']},ensure_ascii=False),flush=True)
        del output,gpu_inputs,inputs
        torch.cuda.empty_cache()
    hook.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--count',type=int,default=2)
    parser.add_argument('--implementation',choices=['original','repack']);parser.add_argument('--method',choices=['snapkv','madakv'])
    args=parser.parse_args()
    prepare(args.count) if args.prepare else run(args.implementation,args.method)
