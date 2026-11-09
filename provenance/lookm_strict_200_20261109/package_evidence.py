"""Package only the completed strict-v2 LookM runs; preserve every numeric field."""
from pathlib import Path
import gzip,hashlib,io,json,shutil,tarfile
from kvrepack.benchmarks.run_video_comparison import validate_sample,STRICT_LOOKM_POLICY
repo=Path.cwd();src=repo/'outputs/benchmarks/lookm_strict_200_20261109';dst=repo/'provenance/lookm_strict_200_20261109'
assert json.loads((src/'lookm_state.json').read_text())['status']=='complete'
for name in ['independent_execution_check.json','independent_scoring_check.json']:
 assert json.loads((src/name).read_text())['status']=='passed',name
assert not dst.exists();dst.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for name in ['lookm_state.json','source_freeze.json','independent_execution_check.json','independent_scoring_check.json']:
 copy(src/name,dst/name)
copy(Path(__file__).resolve(),dst/'package_evidence.py')
for name in ['independent_lookm_strict_200_audit_20261109.py','independent_lookm_strict_200_scoring_20261109.py']:
 copy(src.parent/name,dst/name)
for p in src.glob('*.log'):copy(p,dst/'execution'/p.name)
for name in ['launch.json','controller.log']:
 p=src.with_name(src.name+'_'+name)
 if p.is_file():copy(p,dst/'execution'/p.name)
manifest=[];omit={'prompt','text_segments','video_path','arrow_path','text_scope','alignment'}
def clean(record):
 record['reference_metadata']={k:v for k,v in record['reference_metadata'].items() if k not in omit};return record
for task in ['nextqa_oe_test','vatex_test']:
 case=src/task;state=json.loads((case/'run_state.json').read_text());summary=json.loads((case/'report/summary.json').read_text())
 assert state['status']=='complete' and state['completed_docs']==state['completed_generations']==200
 assert summary['status']=='complete' and summary['strict_budget_passed'] and summary['formal_result']
 for name in ['run_contract.json','run_config.json','run_state.json']:copy(case/name,dst/task/name)
 for p in (case/'report').iterdir():
  if p.is_file():copy(p,dst/task/'report'/p.name)
with (dst/'samples.tar.gz').open('wb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
 for task in ['nextqa_oe_test','vatex_test']:
  case=src/task;identity=json.loads((case/'run_state.json').read_text())['contract_sha256']
  for doc in range(200):
   original=case/'samples'/f'{doc:06}.json';raw=original.read_bytes();record=clean(json.loads(raw))
   validate_sample(record,doc_id=doc,contract_sha256=identity,method='lookm',task=task,budget_policy=STRICT_LOOKM_POLICY)
   assert record['cache_audit']['actual_retained_ratio']<=.1
   data=(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
   member=f'{task}/samples/{doc:06}.json';info=tarfile.TarInfo(member);info.size=len(data);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(data))
   manifest.append(dict(task=task,doc_id=doc,member=member,contract_sha256=identity,original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw),public_sha256=hashlib.sha256(data).hexdigest(),public_bytes=len(data)))
with tarfile.open(dst/'samples.tar.gz','r:gz') as archive:
 assert archive.getnames()==[r['member'] for r in manifest]
 for row in manifest:
  data=archive.extractfile(row['member']).read();assert hashlib.sha256(data).hexdigest()==row['public_sha256'];public=json.loads(data)
  assert public==clean(json.loads((src/row['member']).read_text()))
  validate_sample(public,doc_id=row['doc_id'],contract_sha256=row['contract_sha256'],method='lookm',task=row['task'],budget_policy=STRICT_LOOKM_POLICY)
(dst/'sample_manifest.json').write_text(json.dumps(dict(transform='JSON compacted. Only six declared free-input/local-path reference_metadata fields removed. All generated predictions/token IDs, numeric audits/configs and fingerprints unchanged.',removed_metadata_fields=sorted(omit),samples=manifest),indent=2)+'\n')
(dst/'archive_validation.json').write_text(json.dumps(dict(status='passed',samples=400,generations=400,source_public_full_object_comparisons=400,validator_calls=800,strict_budget_v2_validated=True,gpu_or_model_execution=False),indent=2)+'\n')
(dst/'README.md').write_text('''# Strict LookM:400 fresh video generations,2026-11-09\n\nQwen2.5-VL-7B-Instruct,64frames,NextQA/VATEX original rows0..199 each. OnlyLookM was rerun. Budget policy total_prefill_including_recent_v2:all28x4 native KV heads retain floor(.1N) prefill positions including the recent window. Decoded tokens append under the original decoding rule and are accounted separately.\n\nSee [formal results](../../docs/LOOKM_STRICT_RESULTS_20261109.md) and [source correction/validation](../../docs/LOOKM_STRICT_BUDGET.md). The corrected implementation also fixes head-local drop selection and numerical/boundary behavior; differences from the legacy400 cannot be attributed solely to the32-position budget correction. Hybrid/RoPE stopped runs remain stopped.\n\nEach task has the frozen contract/config/state and native corpus reports. Independent checks audit all400 executions and recompute native task scores while preserving report SHA. samples.tar.gz contains all400 compact prediction/audit records, with declared metadata omissions recorded in sample_manifest.json. archive_validation.json checks the full original/public objects after only that transformation. No videos, weights, tensor caches or credentials are included.\n\nTo rescore, extract the archive into a fresh replay folder, copy each task contract/config/state tree there, then invoke kvrepack.benchmarks.single_video_report --run-root TASK_CASE --hf-home MATCHING_LOCAL_CACHE --output TASK_CASE/report --limit 200 for each task. Matching original Arrow/NLTK/Java resources are required; scoring does not use a model/GPU. Original local paths remain historical provenance.\n''')
assert len(manifest)==400
files=[p for p in dst.rglob('*') if p.is_file()];assert max(p.stat().st_size for p in files)<100_000_000
(dst/'index.json').write_text(json.dumps(dict(schema_version=1,status='complete',method='lookm',budget_policy=STRICT_LOOKM_POLICY,samples=400,generations=400,files_sha256={str(p.relative_to(dst)):sha(p) for p in sorted(files)}),indent=2)+'\n')
print('Packaged strict400:',len(files),'files,',sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()),'bytes')
