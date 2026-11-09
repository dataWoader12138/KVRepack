"""A completed legacy run must not bypass strict-budget checks on resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class LookMLauncherTests(unittest.TestCase):
    def test_complete_legacy_case_is_rejected_before_model_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model';hf=root/'hf';model.mkdir();hf.mkdir()
            output=root/'run';case=output/'nextqa_oe_test';(case/'report').mkdir(parents=True)
            contract=case/'run_contract.json';contract.write_text(json.dumps(dict(method='lookm')))
            (case/'run_state.json').write_text(json.dumps(dict(status='complete',completed_docs=200,completed_generations=200)))
            report=case/'report/summary.json';report.write_text(json.dumps(dict(status='complete')))
            before=(contract.read_bytes(),report.read_bytes())
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_lookm_video.py'),'--pretrained',str(model),'--hf-home',str(hf),'--output',str(output),'--tasks','nextqa_oe_test','--resume'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Legacy LookM used an extra recent window',result.stderr)
            self.assertEqual((contract.read_bytes(),report.read_bytes()),before)
            self.assertNotIn('Loading checkpoint',result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
