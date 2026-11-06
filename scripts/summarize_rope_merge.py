#!/usr/bin/env python3
"""Combine completed paired reports; optionally wait for their matrix worker."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kvrepack.ablations.run_head_selection import atomic_json,timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--limit',type=int,choices=(1,200),required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();root=args.run_root.resolve()
    state_path=root/'aggregation_state.json'
    state=dict(status='waiting' if args.wait else 'checking',started_utc=timestamp(),limit=args.limit)
    atomic_json(state_path,state)
    try:
        if args.wait:
            while True:
                path=root/'experiment_state.json'
                matrix=json.loads(path.read_text()) if path.exists() else {}
                if matrix.get('status')=='complete':break
                if matrix.get('status')=='failed':raise RuntimeError('Matrix failed: '+matrix.get('error','unknown'))
                time.sleep(10)
        rows=[];sources={}
        for task in ['nextqa_oe_test','vatex_test']:
            for method in ['snapkv_kvrepack','hybridkv_kvrepack']:
                run=root/method/task
                generation=json.loads((run/'run_state.json').read_text())
                path=run/'report/summary.json';summary=json.loads(path.read_text())
                if generation.get('status')!='complete' or generation.get('completed_docs')!=args.limit:
                    raise ValueError('Incomplete generations: '+str(run))
                if summary.get('status')!='complete' or summary.get('limit')!=args.limit:
                    raise ValueError('Incomplete metrics: '+str(path))
                sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
                for metric,data in summary['comparisons'].items():
                    raw,post=data['raw_reapply'],data['post_rope']
                    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (raw,post)):
                        raise ValueError('Invalid metric')
                    if abs(raw-post-data['raw_minus_post'])>1e-10:raise ValueError('Recorded difference disagrees')
                    interval=data['ci95']
                    rows.append(dict(task=task,method=method,metric=metric,raw_reapply=raw,post_rope=post,
                        post_minus_raw=post-raw,post_minus_raw_ci95=None if interval is None else [-interval[1],-interval[0]],
                        n_inputs=args.limit))
        atomic_json(root/'comparison.json',dict(status='complete',limit=args.limit,
            total_generations=8*args.limit,comparisons=rows,source_reports_sha256=sources,
            difference='post_rope minus raw_reapply; positive favors post-RoPE aggregation',
            units='NextQA WUPS: 0-100 points; VATEX: original raw corpus units',completed_utc=timestamp()))
        text=['# Paired K merge-space comparison','',f'Completed: {args.limit} original inputs per task, {8*args.limit} generations.',
            '', 'Positive differences favor post-RoPE K aggregation. WUPS uses 0-100 points; VATEX uses raw corpus scores.',
            '', '| Task | Method | Metric | Raw-reapply | Post-RoPE | Post − raw | Paired 95% interval |',
            '|---|---|---|---:|---:|---:|---|']
        for row in rows:
            ci=row['post_minus_raw_ci95'];label='—' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'
            text.append(f"| {row['task']} | {row['method']} | {row['metric']} | {row['raw_reapply']:.6f} | {row['post_rope']:.6f} | {row['post_minus_raw']:+.6f} | {label} |")
        (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (root/'comparison.csv').write_text(stream.getvalue())
        state.update(status='complete',completed_utc=timestamp());atomic_json(state_path,state)
        print(json.dumps(dict(status='complete',output=str(root/'RESULTS.md'))))
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=timestamp())
        atomic_json(state_path,state);raise


if __name__=='__main__':main()
