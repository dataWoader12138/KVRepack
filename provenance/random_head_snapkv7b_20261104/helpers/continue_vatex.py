from pathlib import Path
import datetime,hashlib,json,subprocess,sys
repo=Path.cwd();root=(repo/'outputs/ablations/random_head_snapkv7b_20261104')
hf=repo.parent/'HybridKV-artifact/scripts/data/huggingface_cache';model=hf/'hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
archive=root/'failed_attempts/vatex_similarity_preparation_20261104T131901'
state=json.loads((archive/'experiment_state.json').read_text())
state.pop('error',None);state.pop('finished_utc',None)
state.update(status='continuing_vatex',continued_utc=stamp(),preparation_retry={'reason':'Two CPU preparation workers accidentally overlapped; incomplete doc15 companion-file check stopped first VATEX worker before any generation. Duplicate worker terminated. All 16 retained inputs revalidated before serial continuation.','generated_rows_before_retry':0,'failed_state':str(archive/'experiment_state.json')})
def frozen():
 for p,v in state['source_sha256'].items():
  if hashlib.sha256((repo/p).read_bytes()).hexdigest()!=v:raise RuntimeError('Frozen source changed: '+p)
def save():
 p=root/'experiment_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
frozen();save()
try:
 for strategy,seed in [('similarity',0)]+[('random_global',i) for i in range(3)]:
  label='similarity' if strategy=='similarity' else f'{strategy}_seed{seed}';out=root/'vatex_test'/label
  if out.exists():raise RuntimeError('Output must be fresh: '+str(out))
  frozen();state.update(status='running',current=f'vatex_test/{label}',updated_utc=stamp());save()
  command=[sys.executable,'-B','-m','kvrepack.ablations.run_head_selection','--task','vatex_test','--strategy',strategy,'--seed',str(seed),'--limit','20','--pretrained',str(model),'--hf-home',str(hf),'--cache',str(root/'cache'),'--output',str(out)]
  print(json.dumps({'starting':state['current'],'utc':stamp()}),flush=True)
  with (root/f'vatex_test_{label}.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
  r=json.loads((out/'run_state.json').read_text())
  if r.get('status')!='complete' or r.get('completed')!=20:raise RuntimeError('Incomplete generation')
  frozen();state['completed_conditions'].append(state['current']);save()
 state.update(status='complete',finished_utc=stamp());save();print('ALL 160 GENERATIONS COMPLETE',flush=True)
except BaseException as error:
 state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=stamp());save();raise
