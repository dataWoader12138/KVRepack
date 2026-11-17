"""Full adjacent-K observations on fixed additional LongBench text subsets."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

from .adjacent import append_json,atomic_json,sha,timestamp
from .reference import MODELS,prepare_input
from .reference_inputs import LONG_BENCH,_dataset_file,_prompt_file

KIND='longbench_full_adjacent_key_controls_v1'
SUBSETS=('hotpotqa','triviaqa','lcc','2wikimqa','musique','qmsum')


def load_samples(dataset_zip,prompt_file,subsets,count):
    """Original prefix and official task template, without truncating content."""
    prompts=json.loads(Path(prompt_file).read_text())
    result=[]
    with zipfile.ZipFile(dataset_zip) as z:
        for subset in subsets:
            names=[n for n in z.namelist() if n in (subset+'.jsonl','data/'+subset+'.jsonl')]
            if len(names)!=1:
                raise ValueError('Missing or ambiguous original LongBench member: '+subset)
            with z.open(names[0]) as f:
                rows=list(itertools.islice(f,count))
            if len(rows)!=count:
                raise ValueError(f'{subset} has fewer than {count} original rows; do not fill/substitute')
            for doc_id,line in enumerate(rows):
                row=json.loads(line)
                if not isinstance(row.get('context'),str) or not row['context']:
                    raise ValueError('Missing complete original context')
                prompt=prompts[subset].format(**row)
                metadata=dict(dataset='THUDM/LongBench',revision=LONG_BENCH['revision'],
                              subset=subset,split='test',zip_member=names[0],row_index=doc_id,
                              original_row_id=row.get('_id'),original_row_bytes_sha256=hashlib.sha256(line).hexdigest(),
                              context_sha256=hashlib.sha256(row['context'].encode()).hexdigest())
                result.append(dict(group='text',subset=subset,doc_id=doc_id,prompt=prompt,source_metadata=metadata))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hf-home',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--subsets',nargs='+',choices=SUBSETS,default=['hotpotqa','triviaqa','lcc'])
    p.add_argument('--sample-count',type=int,choices=(2,200),default=200)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if len(set(args.subsets))!=len(args.subsets):p.error('Do not repeat subsets')
    output=args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error('Use a fresh output directory or explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.device,HF_HOME=str(args.hf_home.resolve()),
                      HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    import torch
    from transformers import AutoTokenizer,Qwen2ForCausalLM
    from .native_key_diagnostics import NativeKeyDiagnosticsObserver
    if not torch.cuda.is_available():raise RuntimeError('GPU required for native model observations')
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model_id,revision,family=MODELS['text'];pretrained=args.pretrained.resolve()
    config=json.loads((pretrained/'config.json').read_text())
    if pretrained.name!=revision or config['model_type']!='qwen2' or config['num_hidden_layers']!=28 or config['num_key_value_heads']!=4:
        raise ValueError('Use pinned Qwen2.5-7B-Instruct original snapshot')
    dataset=_dataset_file(LONG_BENCH,args.hf_home,True);prompt_file=_prompt_file(args.hf_home,True)
    samples=load_samples(dataset,prompt_file,args.subsets,args.sample_count)
    source_names=('longbench_controls.py','native_key_diagnostics.py','native_adjacent.py','native_all_pairs.py',
                  'all_pairs.py','reference.py','reference_inputs.py','adjacent.py')
    contract=dict(schema_version=1,statistics_kind=KIND,model_id=model_id,model_revision=revision,
                  pretrained=str(pretrained),subsets=args.subsets,sample_count=args.sample_count,
                  doc_ids=list(range(args.sample_count)),row_policy='Original firstN per subset, full native input, no filtering/substitution/truncation',
                  dataset_sha256=sha(dataset),dataset_revision=LONG_BENCH['revision'],prompt_file_sha256=sha(prompt_file),
                  dtype='bfloat16',attention='flash_attention_2',use_cache=False,generation=False,compression=False,
                  primary='Every original consecutive pair (i,i+1), before current-layer RoPE; equal inputs/layers/native heads',
                  diagnostics='Raw all-pairs; sequence-mean-subtracted adjacent; projection-bias-subtracted adjacent; post-hoc only',
                  sources_sha256={str(Path(__file__).parent/n):sha(Path(__file__).parent/n) for n in source_names},
                  model_metadata_sha256={f.name:sha(f) for f in sorted(pretrained.iterdir()) if f.is_file() and f.suffix in ('.json','.txt')},
                  versions={n:importlib.metadata.version(n) for n in ('torch','transformers','flash-attn','numpy')})
    cp=output/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Refusing mixed-protocol resume')
    atomic_json(cp,contract);contract_sha=sha(cp)
    expected={(s['subset'],s['doc_id']) for s in samples};completed=set()
    if (output/'records.jsonl').exists():
        for line in (output/'records.jsonl').open():
            row=json.loads(line);key=(row['subset'],row['doc_id'])
            if (not line.endswith('\n') or row['status']!='completed' or row['statistics_kind']!=KIND
                    or row['contract_sha256']!=contract_sha or key not in expected or key in completed):
                raise ValueError('Incomplete/duplicate/incompatible observation record')
            completed.add(key)
    started=time.monotonic()
    state=dict(status='loading',pid=os.getpid(),started_utc=timestamp(),completed=len(completed),target=len(samples),
               subsets=args.subsets,gpu=torch.cuda.get_device_name(0),statistics_kind=KIND)
    atomic_json(output/'run_state.json',state)
    try:
        tokenizer=AutoTokenizer.from_pretrained(pretrained,local_files_only=True)
        model=Qwen2ForCausalLM.from_pretrained(pretrained,local_files_only=True,torch_dtype=torch.bfloat16,
                                             attn_implementation='flash_attention_2',device_map='cuda:0').eval()
        state.update(status='running',model_load_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json',state)
        for sample in samples:
            key=(sample['subset'],sample['doc_id'])
            if key in completed:continue
            before=time.monotonic();cpu,info=prepare_input(sample,tokenizer)
            if info['input_tokens']>config['max_position_embeddings']:
                raise ValueError(f'Complete context exceeds native window: {key}; no truncation')
            hashes={name:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for name,t in cpu.items() if torch.is_tensor(t)}
            batch={name:t.to('cuda:0') if torch.is_tensor(t) else t for name,t in cpu.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(),NativeKeyDiagnosticsObserver(model,family=family,input_length=info['input_tokens']) as observer:
                result=model.model(**batch,use_cache=False,output_attentions=False,output_hidden_states=False,return_dict=True)
                torch.cuda.synchronize();stats=observer.finish()
            row=dict(schema_version=1,status='completed',statistics_kind=KIND,contract_sha256=contract_sha,
                     subset=sample['subset'],doc_id=sample['doc_id'],model_id=model_id,model_revision=revision,
                     **info,adjacent_pair_count=info['input_tokens']-1,
                     source_metadata=sample['source_metadata'],input_hashes=hashes,layer_stats=stats,
                     seconds=time.monotonic()-before,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,completed_utc=timestamp())
            append_json(output/'records.jsonl',row);completed.add(key)
            state.update(completed=len(completed),counts_by_subset=dict(Counter(k[0] for k in completed)),
                         current_subset=key[0],current_doc_id=key[1],elapsed_seconds=time.monotonic()-started,updated_utc=timestamp())
            atomic_json(output/'run_state.json',state)
            print(json.dumps(dict(subset=key[0],doc_id=key[1],completed=len(completed),target=len(samples),
                                  tokens=info['input_tokens'],seconds=row['seconds'])),flush=True)
            del result,batch,cpu;torch.cuda.empty_cache()
        del model;torch.cuda.empty_cache()
        state.update(status='complete',finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json',state)
        return 0
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json',state);traceback.print_exc();return 1


if __name__=='__main__':
    raise SystemExit(main())
