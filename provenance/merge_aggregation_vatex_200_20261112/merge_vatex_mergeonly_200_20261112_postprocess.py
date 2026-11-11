"""Only score a successfully completed frozen VATEX merge matrix; never resume."""
from pathlib import Path
import datetime,json,os,subprocess,time
ROOT=Path('/root/autodl-tmp/HybridKV-artifact/KVRepack')
RUN=ROOT/'outputs/ablations/merge_vatex_mergeonly_200_20261112'
STATE=RUN/'postprocess_state.json'
def save(**fields):
 fields['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(fields,indent=2)+'\n');temp.replace(STATE)
save(status='waiting_for_complete_matrix',pid=os.getpid())
while True:
 try:state=json.loads((RUN/'experiment_state.json').read_text())
 except (OSError,ValueError):state={}
 status=state.get('status')
 if status=='complete':break
 if status in ('failed','stopped_by_user','paused','superseded_by_user_scope') or state.get('resume_authorized') is False:
  save(status='not_scored',reason='generation_not_complete',generation_state=state);raise SystemExit(1)
 time.sleep(15)
env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ('src','vendor/lmms-eval','csrc')),CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH='/root/miniconda3/envs/hybridkv/lib'+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
command=['/root/miniconda3/envs/hybridkv/bin/python','-B','-m','kvrepack.ablations.merge_comparison_report','--run-root',str(RUN),'--output',str(RUN/'report'),'--score-parent-prefix']
save(status='scoring',pid=os.getpid(),command=command)
with (RUN/'scoring.log').open('x') as log:
 result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
save(status='complete' if result.returncode==0 else 'failed',pid=os.getpid(),returncode=result.returncode,report=str(RUN/'report'))
raise SystemExit(result.returncode)
