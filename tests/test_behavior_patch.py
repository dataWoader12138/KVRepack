"""Declared behavioral changes must remain reviewable against the original code."""
import ast
import copy
import difflib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_patch_source_checker',ROOT/'scripts/check_sources.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


class BehavioralPatchTests(unittest.TestCase):
    def record(self,base,target,root):
        path=root/'change.patch'
        text=''.join(difflib.unified_diff(base.splitlines(keepends=True),target.splitlines(keepends=True),fromfile='a/test.py',tofile='b/test.py'))
        path.write_text(text)
        before=checker.qualified_definition(ast.parse(base),'changed')[2]
        after=checker.qualified_definition(ast.parse(target),'changed')[2]
        return dict(previous_target_sha256=checker.digest(base.encode()),previous_target_ast_sha256=checker.ast_digest(ast.parse(base)),
                    reversible_edits=[dict(target_start=0,target_end=len(target),before=base,after=target)],
                    function_changes=[dict(name='changed',before_ast_sha256=checker.ast_digest(before),after_ast_sha256=checker.ast_digest(after))],
                    patch_path='change.patch',patch_sha256=checker.digest(text.encode()),patch_from='a/test.py',patch_to='b/test.py')

    def test_real_patch_recovers_original_integration(self):
        manifest=json.loads((ROOT/'provenance/core_sources.json').read_text())
        entry=next(e for e in manifest['entries'] if e['target_path']=='src/kvrepack/baseline_utils.py')
        self.assertEqual(entry['source_sha256'],'e59c37edd0e267199496b958d671d26c60784e0335abf9d2222d0b02f46fe0b4')
        tree=checker.verify_behavior_patch((ROOT/entry['target_path']).read_text(),entry['behavior_patch'])
        self.assertEqual(checker.ast_digest(tree),entry['renamed_source_ast_sha256'])

    def test_undeclared_function_change_is_rejected_even_with_fresh_patch_hashes(self):
        base='def changed():\n    return 1\ndef untouched():\n    return 2\n'
        target=base.replace('return 1','return 3').replace('return 2','return 4')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);record=self.record(base,target,root)
            with patch.object(checker,'ROOT',root),self.assertRaisesRegex(AssertionError,'Undeclared algorithm changes'):
                checker.verify_behavior_patch(target,record)

    def test_patch_file_and_reverse_spans_are_not_ignored(self):
        base='def changed():\n    return 1\n';target=base.replace('return 1','return 3')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);record=self.record(base,target,root)
            with patch.object(checker,'ROOT',root):
                checker.verify_behavior_patch(target,record)
                altered=copy.deepcopy(record);altered['reversible_edits'][0]['before']=base.replace('return 1','return 9')
                with self.assertRaisesRegex(AssertionError,'Pre-fix integration changed'):checker.verify_behavior_patch(target,altered)
                (root/'change.patch').write_text('unrecorded patch content')
                with self.assertRaises(AssertionError):checker.verify_behavior_patch(target,record)


if __name__=='__main__':unittest.main()
