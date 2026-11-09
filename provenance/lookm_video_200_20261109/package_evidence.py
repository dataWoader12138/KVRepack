from pathlib import Path
import json,hashlib,shutil,tarfile,gzip,io
from kvrepack.benchmarks.run_video_comparison import validate_sample
repo=Path.cwd();src=repo/'outputs/benchmarks/lookm_vs_kvrepack_20261109';dst=repo/'provenance/lookm_video_200_20261109'
assert json.loads((src/'experiment_state.json').read_text())['status']=='stopped_by_user'
assert json.loads((src/'lookm_only_state.json').read_text())['status']=='complete'
assert not dst.exists();dst.mkdir()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
copy(Path(__file__).resolve(),dst/'package_evidence.py')
for name in ['experiment_state.json','lookm_only_state.json']:
 copy(src/name,dst/'execution'/name)
copy(src/'hybridkv_kvrepack/nextqa_oe_test/run_state.json',dst/'execution/cancelled_hybrid_nextqa_state.json')
for name in ['lookm_nextqa_oe_test.log','lookm_vatex_test.log','lookm_nextqa_scoring.log','lookm_vatex_scoring.log']:
 p=src/name
 if p.is_file():copy(p,dst/'execution'/name)
for name in ['lookm_vs_kvrepack_20261109_source_audit.json','lookm_existing_hybrid_reference_20261109.json']:
 copy(src.parent/name,dst/'sources'/name)
for p in src.with_name(src.name+'_preflight').iterdir():
 if p.is_file():copy(p,dst/'preflight'/p.name)
manifest=[];omit={'prompt','text_segments','video_path','arrow_path','text_scope','alignment'}
def clean(row):
 row['reference_metadata']={k:v for k,v in row['reference_metadata'].items() if k not in omit};return row
for task in ['nextqa_oe_test','vatex_test']:
 case=src/'lookm'/task;state=json.loads((case/'run_state.json').read_text());assert state['status']=='complete' and state['completed_docs']==200
 summary=json.loads((case/'report/summary.json').read_text());assert summary['status']=='complete'
 for name in ['run_contract.json','run_config.json','run_state.json']:
  copy(case/name,dst/task/name)
 for p in (case/'report').iterdir():
  if p.is_file():copy(p,dst/task/'report'/p.name)
 if (case/'independent_nextqa_score.json').exists():copy(case/'independent_nextqa_score.json',dst/task/'independent_nextqa_score.json')
with (dst/'samples.tar.gz').open('wb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
 for task in ['nextqa_oe_test','vatex_test']:
  case=src/'lookm'/task;identity=json.loads((case/'run_state.json').read_text())['contract_sha256']
  for doc in range(200):
   p=case/'samples'/f'{doc:06}.json';raw=p.read_bytes();row=clean(json.loads(raw))
   validate_sample(row,doc_id=doc,contract_sha256=identity,method='lookm',task=task)
   n=row['input_tokens'];expected=int(n*.1)+32
   for layer in row['cache_audit']['layers']:assert layer['head_lengths']==[expected]*4
   data=(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
   member=f'{task}/samples/{doc:06}.json';info=tarfile.TarInfo(member);info.size=len(data);info.mode=0o644;info.mtime=0
   archive.addfile(info,io.BytesIO(data));manifest.append(dict(task=task,doc_id=doc,member=member,contract_sha256=identity,original_sha256=hashlib.sha256(raw).hexdigest(),public_sha256=hashlib.sha256(data).hexdigest(),public_bytes=len(data)))
with tarfile.open(dst/'samples.tar.gz','r:gz') as archive:
 assert archive.getnames()==[x['member'] for x in manifest]
 for row in manifest:
  data=archive.extractfile(row['member']).read();assert hashlib.sha256(data).hexdigest()==row['public_sha256']
  value=json.loads(data);assert value==clean(json.loads((src/'lookm'/row['member']).read_text()))
  validate_sample(value,doc_id=row['doc_id'],contract_sha256=row['contract_sha256'],method='lookm',task=row['task'])
(dst/'sample_manifest.json').write_text(json.dumps(dict(transform='Remove only six declared free-input/local-path reference_metadata fields; JSON compacted. All predictions, token IDs, numeric audits/configs/fingerprints are unchanged.',removed_metadata_fields=sorted(omit),samples=manifest),indent=2)+'\n')
(dst/'archive_validation.json').write_text(json.dumps(dict(status='passed',samples=400,generations=400,full_source_to_public_object_comparisons=400,validator_calls=800,per_layer_native_lookm_budget_formula_checked=True,gpu_or_model_execution=False),indent=2)+'\n')
(dst/'README.md').write_text('''# LookM-only video results,2026-11-09\n\nQwen2.5-VL-7B-Instruct,64frames,nominal10% budget; original first200 rows each of NextQA and VATEX,400 completed LookM generations. See [results](../../docs/LOOKM_RESULTS_20261109.md) and [protocol](../../docs/LOOKM_COMPARISON.md). Hybrid rerun stopped at119 records on user request; that stopped state is preserved but its predictions are excluded from this LookM archive. No stopped experiment is resumed.\n\nNative LookM implementation is preserved. Per-head active prefill slots=floor(.1N)+32; active-cache payload is not total VRAM. Sources/preflight preserve original AST and regression evidence. NextQA independent native WUPS matches the formal single-case report; VATEX uses original corpus BLEU4/METEOR/ROUGE-L/CIDEr. Historical Hybrid reference is marked historical, with missing checkpoint/input-byte evidence disclosed.\n\nExtract samples.tar.gz into a fresh replay root and copy each task's contract/config/state files into the corresponding folders. Then run the single_video_report CLI for each task with a local matching HF_HOME reference Arrow cache and NLTK/Java resources. No model/GPU is loaded by that scorer. The archive strips only declared metadata fields; all400 records were compared against originals after that transformation. Videos, weights, tensors and credentials are excluded. Original local paths in contracts remain historical provenance.\n''')
files=[p for p in dst.rglob('*') if p.is_file()]
assert len(manifest)==400 and max(p.stat().st_size for p in files)<100_000_000
(dst/'index.json').write_text(json.dumps(dict(schema_version=1,status='complete',method='lookm',samples=400,files_sha256={str(p.relative_to(dst)):sha(p) for p in sorted(files)}),indent=2)+'\n')
print('Packaged LookM only:',len(files),'files, bytes',sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()))
