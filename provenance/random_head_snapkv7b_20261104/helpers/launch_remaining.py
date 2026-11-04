from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
root=Path('outputs/ablations/random_head_snapkv7b_20261104').resolve()
repo=Path.cwd()
hf=repo.parent/'HybridKV-artifact/scripts/data/huggingface_cache'
model=hf/'hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'src/kvrepack/ablations/run_head_selection.py',repo/'src/kvrepack/ablations/head_selection.py',repo/'src/kvrepack/qwen_model_snapkv_kvrepack.py',repo/'src/kvrepack/observations/inputs.py']}
state={'status':'waiting_first_condition','started_utc':stamp(),'target':160,'seeds':[0,1,2],'tasks':['nextqa_oe_test','vatex_test'],'source_sha256':sources,'completed_conditions':[]}
def save():
 p=root/'experiment_state.json'; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(state,indent=2)+'\n'); tmp.replace(p)
def frozen():
 for p,v in sources.items():
  if hashlib.sha256((repo/p).read_bytes()).hexdigest()!=v: raise RuntimeError('Frozen source changed: '+p)
save()
try:
 first=root/'nextqa_oe_test/similarity/run_state.json'
 while True:
  frozen()
  value=json.loads(first.read_text()) if first.exists() else {}
  if value.get('status')=='complete': break
  if value.get('status')=='failed': raise RuntimeError('First condition failed: '+str(value))
  time.sleep(2)
 state['completed_conditions'].append('nextqa_oe_test/similarity')
 for task in state['tasks']:
  conditions=[('similarity',0)]+[('random_global',seed) for seed in state['seeds']]
  for strategy,seed in conditions:
   label='similarity' if strategy=='similarity' else f'{strategy}_seed{seed}'
   if task=='nextqa_oe_test' and label=='similarity': continue
   frozen()
   output=root/task/label
   if output.exists(): raise RuntimeError('Refusing to overwrite '+str(output))
   state.update(status='running',current=f'{task}/{label}',updated_utc=stamp()); save()
   command=[sys.executable,'-B','-m','kvrepack.ablations.run_head_selection','--task',task,'--strategy',strategy,'--seed',str(seed),'--limit','20','--pretrained',str(model),'--hf-home',str(hf),'--cache',str(root/'cache'),'--output',str(output)]
   print(json.dumps({'starting':state['current'],'utc':stamp()}),flush=True)
   with (root/f'{task}_{label}.log').open('w') as stream: subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
   run_state=json.loads((output/'run_state.json').read_text())
   if run_state.get('status')!='complete' or run_state.get('completed')!=20: raise RuntimeError('Incomplete condition')
   frozen();state['completed_conditions'].append(state['current']);save()
 state.update(status='complete',finished_utc=stamp());save();print('ALL 160 GENERATIONS COMPLETE',flush=True)
except BaseException as error:
 state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=stamp());save();raise
