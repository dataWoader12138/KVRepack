from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import math
import tarfile
import traceback

REPO=Path('/root/autodl-tmp/KVRepack-styled-images')
RELEASE=REPO/'provenance/styled_milebench_20261117/final'
EVIDENCE=Path('/tmp/styled_milebench_release_audit_20261117.json')
sys.path.insert(0,str(REPO/'src'))
ARMS=('hybridkv_asymkv','hybridkv_lookm','hybridkv_kvrepack','snapkv_asymkv','snapkv_lookm','snapkv_kvrepack','asymkv','lookm')
TASKS=('Spot-the-Diff','MMCoQA')
PAIRS={(a,t) for a in ARMS for t in TASKS}
result={'schema_version':1,'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'release_directory':str(RELEASE),'gpu_execution':False,'release_directory_write_operations':False}

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def relative(name):
    p=PurePosixPath(name)
    require(isinstance(name,str) and name==p.as_posix() and not p.is_absolute()
            and all(part not in ('','.','..') for part in p.parts) and '\\' not in name,'Unsafe archive/index member')
    return Path(*p.parts)

def capture():
    paths=list(RELEASE.rglob('*'))
    require(not any(p.is_symlink() for p in paths),'Published directory contains a symlink')
    return {p.relative_to(RELEASE).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
            for p in paths if p.is_file()}

def close(a,b,label):
    require(math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),label)

def compare_csv(path,expected,columns,keys):
    with path.open(newline='') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==columns,'CSV header mismatch: '+path.name);rows=list(reader)
    actual={tuple(row[k] for k in keys):row for row in rows}
    wanted=[{key:'' if row[key] is None else str(row[key]) for key in columns} for row in expected]
    wanted_map={tuple(row[k] for k in keys):row for row in wanted}
    require(len(actual)==len(rows) and actual==wanted_map,'CSV raw-derived values mismatch: '+path.name)
    return len(rows)

try:
    print('AUDIT_START',result['started_utc'],flush=True)
    before=capture();manifest=read(RELEASE/'release_manifest.json')
    require(manifest['status']=='complete_and_independently_verified' and manifest['source_commit']=='b9104d8'
            and manifest['condition_count']==16 and manifest['samples_per_condition']==200
            and manifest['formal_generations']==3200 and manifest['warmup_generations_excluded']==16,'Not a complete 16x200 release')
    require(manifest['arms']==list(ARMS) and manifest['tasks']==list(TASKS),'Release matrix differs')
    for name,key in [('recorded_files_sha256.json','recorded_files_index_sha256'),
                     ('release_files_sha256.json','release_files_index_sha256'),
                     ('source_files_sha256.json','source_files_index_sha256')]:
        require(before[name]['sha256']==manifest[key],'Manifest index SHA mismatch: '+name)
    release_index=read(RELEASE/'release_files_sha256.json')
    require(release_index['schema_version']==1 and set(release_index['excludes'])=={'release_files_sha256.json','release_manifest.json'},'Release index schema differs')
    require(set(before)==set(release_index['files'])|set(release_index['excludes']),'Release file coverage differs')
    for name,entry in release_index['files'].items():
        relative(name);require(before[name]==entry,'Release file SHA/size differs: '+name)
    archive=RELEASE/manifest['archive_file']
    require(archive.name=='recorded_runs.tar.gz' and archive.stat().st_size==manifest['archive_bytes']
            and archive.stat().st_size<=90_000_000 and sha(archive)==manifest['archive_sha256'],'Archive SHA/size differs')
    raw_index=read(RELEASE/'recorded_files_sha256.json')
    entries={entry['path']:entry for entry in raw_index['files']}
    require(len(entries)==len(raw_index['files'])==3279 and raw_index['condition_count']==16
            and raw_index['formal_samples']==3200 and raw_index['warmup_samples']==16,'Raw index coverage differs')
    require(manifest['raw_source_files']==len(entries) and manifest['archive_files']==len(entries)+1
            and sum(entry['bytes'] for entry in entries.values())==manifest['raw_source_uncompressed_bytes'],'Manifest raw counts differ')
    spec=importlib.util.spec_from_file_location('audit_reporter',REPO/'scripts/summarize_styled_milebench.py')
    reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)
    from kvrepack.benchmarks.milebench.dataset import load_annotation,annotation_path
    from kvrepack.benchmarks.milebench.evaluate import evaluate_predictions
    contracts={};states={};predictions={pair:[] for pair in PAIRS};per_input={pair:[] for pair in PAIRS}
    warmups={pair:[] for pair in PAIRS};seen_samples={pair:[] for pair in PAIRS};metadata_names={pair:set() for pair in PAIRS}
    boundary_ids={pair:[] for pair in PAIRS};controller=None
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        require(len(members)==3280 and len({m.name for m in members})==len(members)
                and {m.name for m in members}==set(entries)|{'recorded_files_sha256.json'},'Tar file coverage differs')
        for member in members:
            path=relative(member.name);require(member.isfile(),'Tar contains non-regular file')
            data=tar.extractfile(member).read()
            if member.name=='recorded_files_sha256.json':
                require(data==(RELEASE/member.name).read_bytes(),'Embedded raw index differs');continue
            entry=entries[member.name]
            require(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],'Tar member SHA/size differs: '+member.name)
            value=json.loads(data)
            parts=path.parts
            if member.name=='experiment_state.json': controller=value;continue
            require(len(parts) in (3,4) and (parts[0],parts[1]) in PAIRS,'Unknown tar condition')
            pair=(parts[0],parts[1]);arm,task=pair
            if len(parts)==3:
                metadata_names[pair].add(parts[2])
                if parts[2]=='run_contract.json':
                    contracts[pair]=value
                    require(value['arm']==arm and value['task']==task and value['all_samples'] is True
                            and value['limit']==value['annotation_total_samples']==200
                            and len(set(value['selected_sample_ids']))==200 and value['warmup_per_arm']==1,'Archived full contract differs')
                elif parts[2]=='run_state.json': states[pair]=value
            else:
                require(parts[2] in ('samples','warmup'),'Tar contains non-allowlisted raw folder')
                contract=contracts[pair]
                if parts[2]=='samples':
                    index=int(Path(parts[3]).stem)
                    require(parts[3]==f'{index:06d}.json' and 0<=index<200,'Bad sample filename')
                    derived=reporter.validate_sample(value,contract,index,contract['generation_sample_ids'][index])
                    predictions[pair].append(value['prediction']);per_input[pair].append(derived);seen_samples[pair].append(index)
                    if derived['asymkv_boundary_affected']:boundary_ids[pair].append(value['sample_id'])
                else:
                    require(parts[3]=='000000.json','Unexpected warmup filename')
                    reporter.validate_sample(value,contract,0,contract['generation_sample_ids'][0],warmup=True)
                    warmups[pair].append(parts[3])
    require(controller['status']=='complete' and controller['completed_runs']==16 and controller['target_generations']==3200,'Archived controller is incomplete')
    require({(v['arm'],v['task']) for v in controller['runs']}==PAIRS and all(v['status']=='complete' for v in controller['runs']),'Archived controller conditions differ')
    require(set(contracts)==set(states)==PAIRS,'Missing archived contracts/states')
    expected_raw={'experiment_state.json'}
    for pair in PAIRS:
        arm,task=pair;contract=contracts[pair];state=states[pair]
        expected_meta={'run_contract.json','run_state.json','environment.json'}
        if arm not in ('asymkv','lookm'):expected_meta.add('adapter_transform.json')
        elif arm=='lookm':expected_meta.add('legacy_lookm_transform.json')
        require(metadata_names[pair]==expected_meta,'Unexpected metadata allowlist')
        require(sorted(seen_samples[pair])==list(range(200)) and warmups[pair]==['000000.json'],'Not exact 200 formal + 1 warmup')
        require(state['status']=='complete' and state['contract_sha256']==reporter.json_sha(contract)
                and all(state[k]==200 for k in ('target_samples','completed_samples','target_generations','completed_generations')),'Archived run state mismatch')
        expected_raw.update(f'{arm}/{task}/{name}' for name in expected_meta)
        expected_raw.update(f'{arm}/{task}/samples/{i:06d}.json' for i in range(200))
        expected_raw.add(f'{arm}/{task}/warmup/000000.json')
    require(set(entries)==expected_raw,'Archive contains forbidden/non-allowlisted files')
    summary=read(RELEASE/'report/summary.json')
    require(summary['status']=='passed' and summary['full_experiment'] is True and summary['formal_generations']==3200
            and summary['completed_runs']==16 and summary['warmup_generations_excluded']==16,'Published report is not full')
    require(before['report/summary.json']['sha256']==manifest['report_summary_sha256'],'Report summary manifest SHA differs')
    summary_map={(v['arm'],v['task']):v for v in summary['results']}
    require(len(summary['results'])==16 and set(summary_map)==PAIRS,'Summary matrix differs')
    source=Path(manifest['source_run']);conditions=[];raw_runs={};all_latency=[];all_summary=[]
    for arm in ARMS:
        for task in TASKS:
            pair=(arm,task);contract=contracts[pair]
            annotation=load_annotation(contract['data_root'],task,limit=None)
            require(len(annotation['data'])==200 and [row['sample_id'] for row in annotation['data']]==contract['selected_sample_ids']
                    and sha(annotation_path(contract['data_root'],task))==contract['annotation_sha256'],'Original annotation SHA/IDs differ')
            evaluation=evaluate_predictions(predictions[pair],annotation,task)
            metric='Accuracy' if task=='MMCoQA' else 'Rouge-L f';score=float(evaluation['eval'][metric])*100
            close(score,summary_map[pair]['score_percent'],'Raw-tar score differs from published summary')
            reporter.equal_number(score,sum(row['end_to_end_seconds'] for row in per_input[pair])*0+score,'finite score')
            condition=RELEASE/'report/conditions'/arm/task
            require(read(condition/'eval.json')==evaluation['eval'] and read(condition/'eval_score.json')==evaluation['eval_score'],'Published condition scores differ from raw-tar independent rescore')
            require(read(condition/'evaluation_metadata.json')=={key:evaluation.get(key) for key in ('task','sample_ids','sample_count','annotation_total_samples','full_task')},'Published condition metadata differs')
            # Source raw is byte-identical to every archive entry (checked below).
            # The original reporter recomputes all stage/cache/EOS aggregates and
            # unrounded singleton scores, independently of published CSV/JSON.
            record=reporter.read_run(source/arm/task,arm,task);raw_runs[pair]=record
            require(record[0]==contract,'Archived contract differs from source')
            require(record[2]==summary_map[pair],'Full raw-derived summary differs from published summary')
            all_summary.append(record[2]);all_latency.extend(record[3])
            evidence=next(v for v in summary['source_and_sample_evidence'] if (v['arm'],v['task'])==pair)
            require(record[5]==evidence,'Published sample/contract evidence differs from raw')
            conditions.append({'arm':arm,'task':task,'formal_samples':200,'warmup_samples_excluded':1,
                               'metric':metric,'score_percent':score,'e2e_mean_seconds':record[2]['e2e_mean_seconds'],
                               'contract_file_sha256':entries[f'{arm}/{task}/run_contract.json']['sha256'],
                               'asymkv_boundary_sample_ids':boundary_ids[pair]})
    require(summary['pairing']==reporter.validate_pairing(raw_runs),'Pairing proof differs')
    comparison_rows=compare_csv(RELEASE/'report/comparison.csv',all_summary,reporter.SUMMARY_COLUMNS,('arm','task'))
    latency_rows=compare_csv(RELEASE/'report/per_input_latency.csv',all_latency,reporter.PER_INPUT_COLUMNS,('arm','task','sample_id'))
    for name,entry in entries.items():
        p=source/relative(name)
        require(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Archived raw differs from actual source: '+name)
    origins=read(RELEASE/'source_files_sha256.json')['files'];origin_changed=[]
    for entry in origins:
        p=Path(entry['source'])
        if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:origin_changed.append(str(p))
    require(not origin_changed,'Live export origins changed: '+str(origin_changed[:8]))
    require(before==capture(),'Published release changed during read-only audit')
    require('torch' not in sys.modules,'CPU release audit unexpectedly imported torch')
    result.update(status='passed',ended_utc=datetime.now(timezone.utc).isoformat(),
                  published_file_count=len(before),release_indexed_files=len(release_index['files']),
                  source_index_files_checked=len(origins),raw_indexed_files=len(entries),tar_members_checked=len(members),
                  archive_bytes=archive.stat().st_size,archive_megabytes_decimal=archive.stat().st_size/1e6,
                  archive_mebibytes=archive.stat().st_size/(1024**2),archive_sha256=sha(archive),archive_limit_bytes=90_000_000,
                  condition_count=16,formal_samples=3200,warmup_samples_excluded=16,
                  comparison_csv_rows=comparison_rows,per_input_latency_csv_rows=latency_rows,
                  every_release_file_sha_and_size_verified=True,every_tar_member_sha_and_size_verified=True,
                  every_archived_raw_matches_original_source=True,all_source_index_hashes_still_match=True,
                  archive_whitelist_verified=True,exact_full_annotation_ids_and_hashes_verified=True,
                  independent_raw_tar_rescore_matches_summary_and_condition_json=True,
                  all_summary_and_per_input_csv_fields_match_recomputed_raw=True,native_pairing_rechecked=True,
                  release_unchanged_throughout_audit=True,torch_imported=False,
                  release_manifest_sha256=before['release_manifest.json']['sha256'],
                  release_files_index_sha256=before['release_files_sha256.json']['sha256'],
                  recorded_files_index_sha256=before['recorded_files_sha256.json']['sha256'],conditions=conditions)
except Exception as error:
    result.update(status='failed',ended_utc=datetime.now(timezone.utc).isoformat(),error=str(error),traceback=traceback.format_exc())
finally:
    EVIDENCE.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','started_utc','ended_utc','archive_bytes','archive_sha256','condition_count','formal_samples','error')},ensure_ascii=False),flush=True)
    print('EVIDENCE',str(EVIDENCE),flush=True)
if result['status']!='passed': raise SystemExit(1)
