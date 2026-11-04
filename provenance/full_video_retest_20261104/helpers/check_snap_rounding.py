"""Three-doc causal probe only. Production sources and main matrix stay unchanged."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
REPO=HERE.parent/'KVRepack'


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounding',choices=['floor','ceil'],required=True)
    args=p.parse_args()
    inventory=json.loads(Path('/tmp/gpu_retest_data_inventory.json').read_text())
    data=next(d for d in inventory['datasets'] if d['task']=='nextqa_oe_test')
    contract=json.loads((HERE/'snapkv_nextqa_repack/trace_contract.json').read_text())
    for folder in [REPO/'src',REPO/'vendor/lmms-eval',REPO/'csrc']:
        sys.path.insert(0,str(folder))
    from kvrepack.experiments.runner import build_plan,child_environment
    os.environ.update(HF_HOME=inventory['recommended_HF_HOME'],HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    plan=build_plan('snapkv_kvrepack',task='nextqa_oe_test',pretrained=inventory['models'][0]['snapshot_path'],
                    model_type='qwen2_5_vl-3b',frames=64,limit=200,budget_ratio=.1,
                    overrides=['SNAPKV_SIMHEAD_MERGE_RATIO=.7','SNAPKV_KVREPACK_COMPRESS_RATIO=.9','K_DIFF_ALPHA=2'])
    environment=child_environment(plan)
    os.environ.clear();os.environ.update(environment)
    import numpy as np
    import torch
    from datasets import Dataset
    from lmms_eval.models.qwen2_5_vl import Qwen2_5_VL
    torch.set_num_threads(4);torch.manual_seed(1234);np.random.seed(1234);random.seed(0)
    adapter=Qwen2_5_VL(pretrained=inventory['models'][0]['snapshot_path'],max_num_frames=64,
                       attn_implementation='flash_attention_2',method='snapkv_kvrepack')
    import kvrepack.qwen_model_snapkv_kvrepack as core
    source_path=Path(core.__file__)
    source_before=hashlib.sha256(source_path.read_bytes()).hexdigest()
    original_budget=core._history_budget
    if args.rounding=='ceil':
        def ceil_budget(config,seq_len,history_len):
            if history_len<=0:return 0
            ratio=getattr(config,'snapkv_kvrepack_budget_ratio',None)
            if ratio is None:return original_budget(config,seq_len,history_len)
            return min(max(math.ceil(seq_len*float(ratio)),0),int(history_len))
        core._history_budget=ceil_budget
    ds=Dataset.from_file(data['arrow_path'])
    adapter.task_dict={'nextqa_oe_test':{'test':ds}}
    function_path=REPO/'vendor/lmms-eval/lmms_eval/tasks/nextqa/utils.py'
    node=next(n for n in ast.parse(function_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='nextqa_doc_to_text')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(function_path),'exec'),namespace)
    previous={r['doc_id']:r for r in map(json.loads,(HERE/'snapkv_nextqa_repack/trace.jsonl').read_text().splitlines())}
    old_text={61:'different bikes',117:'to catch it',131:'loading bar'}
    generated=[];current={}
    original_generate=adapter.model.generate
    def capture(**kwargs):
        input_hashes={key:digest(value) for key,value in kwargs.items() if torch.is_tensor(value)}
        result=original_generate(**kwargs,return_dict_in_generate=True,output_logits=True)
        length=kwargs['input_ids'].shape[-1]
        ids=result.sequences[0,length:].tolist()
        row={'doc_id':current['doc_id'],'rounding':args.rounding,'input_hashes':input_hashes,
             'input_matches_production':input_hashes==previous[current['doc_id']]['input_hashes'],
             'input_tokens':length,'generated_ids':ids,'prefill_key_lengths':current.get('prefill_lengths'),
             'logits_sha256':digest(torch.stack(result.logits).squeeze(1))}
        generated.append(row)
        return result.sequences
    adapter.model.generate=capture
    def prefill(module, call_args, call_kwargs, result):
        if result.last_hidden_state.shape[1]>1:
            current['prefill_lengths']=[int(x.shape[2]) for x in result.past_key_values.key_cache]
    hook=adapter.model.model.language_model.register_forward_hook(prefill,with_kwargs=True)
    for doc_id in [61,117,131]:
        current.update(doc_id=doc_id,prefill_lengths=None)
        doc=ds[doc_id]
        prompt=namespace['nextqa_doc_to_text'](doc,{'pre_prompt':'','post_prompt':'\nAnswer a question using a short phrase or sentence.'})
        def visual(doc):return [str(Path(data['video_root'])/(str(doc['video'])+'.mp4'))]
        request=SimpleNamespace(args=(prompt,{'temperature':0},visual,doc_id,'nextqa_oe_test','test'))
        answer=adapter.generate_until([request])[0]
        # Forward hook runs before capture returns, so use its final current snapshot.
        generated[-1].update(text=answer,prefill_key_lengths=current['prefill_lengths'],
                              production_text=previous[doc_id]['text'],historical_text=old_text[doc_id],
                              matches_production_text=answer==previous[doc_id]['text'],matches_historical_text=answer==old_text[doc_id])
        assert generated[-1]['input_matches_production']
        print(json.dumps(generated[-1],ensure_ascii=False),flush=True)
    hook.remove()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==source_before
    report={'diagnostic_only':True,'production_source_modified':False,'rounding':args.rounding,
            'source':str(source_path),'source_sha256':source_before,'profile':contract['config'],
            'runtime_intervention':'Only _history_budget ratio conversion floor->ceil' if args.rounding=='ceil' else 'Unmodified production helper',
            'records':generated,'limitations':['Three previously differing historical records only; this is not a full 200-row ceil replay.']}
    (HERE/f'snap_rounding_{args.rounding}_3docs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
