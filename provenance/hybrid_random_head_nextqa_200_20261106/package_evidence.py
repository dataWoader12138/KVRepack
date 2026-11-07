from pathlib import Path
import gzip,hashlib,io,json,shutil,tarfile
from kvrepack.ablations.run_hybrid_head_selection import validate_bundle
repo=Path.cwd();src=repo/'outputs/ablations/hybrid_random_head_nextqa_200_20261106';dst=repo/'provenance/hybrid_random_head_nextqa_200_20261106'
required=['independent_execution_check.json','independent_scoring_check.json']
for name in required:
 if not (src/name).is_file():raise FileNotFoundError(name)
assert json.loads((src/'run_state.json').read_text())['completed_generations']==800
assert json.loads((src/'report/summary.json').read_text())['status']=='complete'
for name in required:
 assert json.loads((src/name).read_text())['status']=='passed',name
assert not dst.exists();dst.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
copy(Path(__file__).resolve(),dst/'package_evidence.py')
for name in ['run_contract.json','run_config.json','run_state.json','first20_baseline_equivalence.json',*required]:copy(src/name,dst/name)
for f in (src/'report').iterdir():
 if f.is_file():copy(f,dst/'report'/f.name)
for label in ['controller_state.json','launch.json','worker.log','scoring.log','controller.log']:
 p=src.with_name(src.name+'_'+label)
 if p.exists():copy(p,dst/'execution'/p.name)
log=Path('/tmp/kvrepack_hybrid_random_cpu_suite_20261106.log')
if log.is_file():copy(log,dst/'execution/cpu_suite_20261106.log')
identity=json.loads((src/'run_state.json').read_text())['contract_sha256'];rows=[]
omit={'prompt','text_segments','video_path','arrow_path','text_scope','alignment'}
with (dst/'samples.tar.gz').open('wb') as stream, gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
 for doc in range(200):
  p=src/'samples'/f'{doc:06}.json';raw=p.read_bytes();bundle=json.loads(raw)
  metadata={k:v for k,v in bundle['reference_metadata'].items() if k not in omit};bundle['reference_metadata']=metadata
  for row in bundle['records'].values():row['reference_metadata']=metadata
  validate_bundle(bundle,doc_id=doc,contract_sha256=identity)
  data=(json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
  name=f'samples/{doc:06}.json';info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
  archive.addfile(info,io.BytesIO(data))
  rows.append({'doc_id':doc,'original_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw),'archive_member':name,'public_sha256':hashlib.sha256(data).hexdigest(),'public_bytes':len(data)})
(dst/'sample_manifest.json').write_text(json.dumps({'samples':rows,'transform':'JSON compacted. Only reference_metadata prompt,text_segments,video_path,arrow_path,text_scope,alignment removed at bundle and record levels. All prediction/token IDs, numerical audits, hashes/configs preserved.'},indent=2)+'\n')
with tarfile.open(dst/'samples.tar.gz','r:gz') as archive:
 assert archive.getnames()==[r['archive_member'] for r in rows]
 for row in rows:
  data=archive.extractfile(row['archive_member']).read();assert hashlib.sha256(data).hexdigest()==row['public_sha256']
  public=json.loads(data)
  validate_bundle(public,doc_id=row['doc_id'],contract_sha256=identity)
  expected=json.loads((src/'samples'/f"{row['doc_id']:06}.json").read_text())
  metadata={k:v for k,v in expected['reference_metadata'].items() if k not in omit}
  expected['reference_metadata']=metadata
  for record in expected['records'].values():record['reference_metadata']=metadata
  assert public==expected, row['doc_id']
(dst/'archive_validation.json').write_text(json.dumps(dict(status='passed',samples=200,generations=800,archive_member_hashes_checked=200,source_to_public_full_object_comparisons=200,validator_calls=400,differences_outside_declared_metadata_removals=0,scoring_inputs_and_generated_predictions_unchanged=True,gpu_or_model_execution=False),indent=2)+'\n')
(dst/'README.md').write_text('''# HybridKV-KVRepack NExTQA 200-input Random-head ablation\n\nGenerated on 2026-11-06, independently checked on 2026-11-07. Qwen2.5-VL-7B + HybridKV-KVRepack; original rows0..199; 4 conditions, 800 completed generations, 184 source videos. No VATEX, All-head rerun, or extension to9178 rows. All conditions select and effectively merge56 of112 heads with unchanged native per-head retained budgets. Heterogeneous budgets and global identity/layer changes mean total removed-token work is not controlled equal; the report preserves these differences.\n\nSee [Chinese result and interpretation](../../docs/HYBRID_RANDOM_HEAD_NEXTQA_200_RESULTS_20261106.md) and [protocol](../../docs/HYBRID_RANDOM_HEAD_NEXTQA_200.md). `report/` contains official WUPS scores, seed SD, paired cluster intervals and PNG/PDF/SVG plots. Two independent checks audit execution and recalculate original-task WUPS and source-video intervals.\n\n`samples.tar.gz` contains all200 four-condition bundles. All numerical fields and generated token IDs are preserved; free-text input prompts and local video/Arrow paths are removed from reference_metadata. The exact transformation and original/public hashes are in sample_manifest.json. Video/image data, model weights, tensor caches and credentials are excluded. Unabridged originals remain in local outputs.\n\nTo rescore with matching local NextQA reference Arrow and NLTK resources, create a fresh replay directory, copy run_contract.json/run_config.json/run_state.json into it, extract samples.tar.gz there, then run:\n\n```bash\nPYTHONPATH=src python -m kvrepack.ablations.hybrid_head_selection_report \\\n  --run-root /path/to/replay --hf-home /path/to/huggingface_cache \\\n  --output /path/to/replay/report --limit 200\n```\n\nArchived configurations preserve original local paths as provenance. The scorer reads reference data from the HF_HOME argument; no model/GPU is loaded. index.json covers the published files.\n''')
index={'schema_version':1,'status':'complete','samples':200,'generations':800,'source_sha256':json.loads((src/'run_contract.json').read_text())['sources_sha256'],'original_sample_sha256':{str((src/'samples'/f'{i:06}.json').relative_to(repo)):r['original_sha256'] for i,r in enumerate(rows)},'files_sha256':{str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*')) if p.is_file()}}
(dst/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print('Packaged',len(index['files_sha256']),'files; bytes',sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()))
