"""Small CPU stream fixtures; no model, GPU, download, or real inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.ablations.full_head_selection_report import (
    TASK, VARIANTS, _canonical, _json_sha, progress, score_stream, write_report,
)
from test_head_selection_report import row as pilot_row


def fixture(root, count=2, complete=True):
    root=Path(root);(root/"samples").mkdir()
    docs=[dict(video=f"video-{i//2}",answer="yes",type="CW",additional_ref_answer="",question=f"q{i}") for i in range(count)]
    reference=dict(sha256="arrow",split_rows=9178,doc_ids=list(range(count)))
    contract=dict(schema_version=1,task=TASK,target_docs=count,target_generations=4*count,arrow_sha256="arrow",dataset_rows=9178)
    identity=_json_sha(contract)
    (root/"run_contract.json").write_text(json.dumps(contract))
    (root/"run_state.json").write_text(json.dumps(dict(status="complete" if complete else "running",target_docs=count,target_generations=4*count,
                                                     completed_docs=count if complete else 0,completed_generations=4*count if complete else 0)))
    import hashlib
    for doc_id,doc in enumerate(docs):
        meta=dict(task=TASK,doc_id=doc_id,video_id=doc["video"],source_video_id=doc["video"],arrow_sha256="arrow",
                  original_row_sha256=hashlib.sha256(_canonical(doc).encode()).hexdigest())
        fingerprints=dict(input_hashes={"input_ids":f"hash-{doc_id}"},input_shapes={"input_ids":[1,100]},input_dtypes={"input_ids":"torch.int64"})
        rows={}
        for variant in VARIANTS:
            row=pilot_row(TASK,variant,doc_id)
            row.update(reference_metadata=meta,**fingerprints,input_fingerprint_sha256=_json_sha(fingerprints),text="yes",prediction="yes",
                       generated_tokens=2,eos_token_id=2,eos_reached=True,final_cache_shapes=row["audit"]["prefill_cache_shapes"])
            row["config"].pop("mask_strategy")
            row["config"].update(strategy=row["audit"]["strategy"],stream_contract_sha256=identity)
            # Use the actual frozen selector ranking, not a test-only ordering.
            if variant!="similarity":
                from kvrepack.ablations.head_selection import _ranking
                chosen=set(_ranking(int(variant[-1]),"global",28,4,0).sample(range(112),112)[:34])
                row["audit"]["selected_masks"]=[[[4*l+h in chosen for h in range(4)] for l in range(28)]]
                row["audit"]["selected_layer_counts"]=[[sum(layer) for layer in row["audit"]["selected_masks"][0]]]
            row["audit"].update(task=TASK,doc_id=doc_id,merge_calls=[dict(layer=l,batch=0,head=h,before=4,after=1)
                for l,layer in enumerate(row["audit"]["selected_masks"][0]) for h,selected in enumerate(layer) if selected])
            rows[variant]=row
        bundle=dict(schema_version=1,status="completed",task=TASK,doc_id=doc_id,contract_sha256=identity,
                    input_fingerprints=fingerprints,reference_metadata=meta,records=rows,fairness={"status":"passed"})
        (root/"samples"/f"{doc_id:06d}.json").write_text(json.dumps(bundle))
    return docs,reference


def scorer(doc,predictions):
    return {"WUPS":{"0":1. if predictions[0]==doc["answer"] else 0.,"0.9":1.,"qtype":"CW"}}


def aggregate(values):return 100.*sum(row["0"] for row in values)/len(values)


class FullHeadSelectionReportTests(unittest.TestCase):
    def test_stream_scores_memoizes_exact_predictions_and_preserves_four_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,ref=fixture(directory)
            summary=score_stream(directory,docs,ref,scorer=scorer,aggregate=aggregate,limit=2,bootstrap_samples=20)
            self.assertEqual(summary["status"],"complete")
            self.assertEqual(summary["n_generations"],8)
            self.assertEqual(summary["scores"],dict.fromkeys(VARIANTS,100.))
            self.assertEqual(summary["scorer_cache"],dict(unique_prediction_reference_evaluations=1,exact_cache_hits=7))
            self.assertEqual(len(summary["scores_table"]),8)
            self.assertEqual(summary["n_source_videos"],1)
            self.assertIsNone(summary["paired_WUPS"]["random_mean"]["ci95"])
            output=Path(directory)/"report"
            write_report(summary,output,plots=False)
            for name in ("scores.csv","paired_scores.csv","selector_audit.csv","sample_files.csv","run_scores.csv","summary.json","RESULTS.md"):
                self.assertGreater((output/name).stat().st_size,20)
            with self.assertRaises(FileExistsError):write_report(summary,output,plots=False)

    def test_inprogress_only_reports_progress_and_cannot_produce_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,ref=fixture(directory,complete=False)
            status=progress(directory,2)
            self.assertTrue(status["progress_only"])
            self.assertFalse(status["scoring_performed"])
            self.assertIsNone(status["metrics"])
            with self.assertRaises(ValueError):score_stream(directory,docs,ref,scorer=scorer,aggregate=aggregate,limit=2)

    def test_reference_source_mismatch_missing_bundle_and_wrong_target_are_rejected(self):
        for mode in ("row", "arrow", "source", "missing", "target"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                docs,ref=fixture(directory)
                path=Path(directory)/"samples/000000.json"
                bundle=json.loads(path.read_text())
                if mode=="row":docs[0]["answer"]="different"
                elif mode=="arrow":ref["sha256"]="different"
                elif mode=="source":docs[0]["video"]="different"
                elif mode=="missing":path.unlink()
                else:
                    state=Path(directory)/"run_state.json";value=json.loads(state.read_text());value["target_docs"]=200;state.write_text(json.dumps(value))
                with self.assertRaises(ValueError):score_stream(directory,docs,ref,scorer=scorer,aggregate=aggregate,limit=2)

    def test_candidate_or_configuration_fairness_tamper_rejected(self):
        for mode in ("candidate","config"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                docs,ref=fixture(directory)
                path=Path(directory)/"samples/000000.json";bundle=json.loads(path.read_text())
                row=bundle["records"][VARIANTS[1]]
                if mode=="candidate":row["audit"]["candidate_hashes"][0]["candidate_raw"]["sha256"]="changed"
                else:row["config"]["max_new_tokens"]=64
                path.write_text(json.dumps(bundle))
                with self.assertRaises(ValueError):score_stream(directory,docs,ref,scorer=scorer,aggregate=aggregate,limit=2)


if __name__=="__main__":unittest.main()
