import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from kvrepack.observations.longbench_controls import load_samples


class LongBenchControlInputTests(unittest.TestCase):
    def fixture(self, folder, rows):
        archive=folder/'data.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for subset,records in rows.items():
                z.writestr('data/'+subset+'.jsonl',''.join(json.dumps(r)+'\n' for r in records))
        prompt=folder/'prompts.json'
        prompt.write_text(json.dumps({'hotpotqa':'{context}\nQuestion: {input}', 'lcc':'{context}'}))
        return archive,prompt

    def test_original_prefix_complete_context_and_context_only_template(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);context='  full {literal} context\n'*500
            archive,prompt=self.fixture(folder,{'hotpotqa':[{'context':context,'input':'Q0'},
                {'context':'second','input':'Q1'},{'context':'do not include','input':'Q2'}],
                'lcc':[{'context':'def f():\n    return 2'},{'context':'  // code'}]})
            rows=load_samples(archive,prompt,['hotpotqa','lcc'],2)
            self.assertEqual([(r['subset'],r['doc_id']) for r in rows],[('hotpotqa',0),('hotpotqa',1),('lcc',0),('lcc',1)])
            self.assertEqual(rows[0]['prompt'],context+'\nQuestion: Q0')
            self.assertEqual(rows[-1]['prompt'],'  // code')
            self.assertEqual(rows[0]['source_metadata']['row_index'],0)
            self.assertEqual(len(rows[0]['source_metadata']['context_sha256']),64)

    def test_short_source_is_not_filled_or_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            archive,prompt=self.fixture(Path(d),{'hotpotqa':[{'context':'A','input':'Q'}]})
            with self.assertRaisesRegex(ValueError,'fewer than 2'):
                load_samples(archive,prompt,['hotpotqa'],2)

    def test_missing_or_empty_original_context_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive,prompt=self.fixture(Path(d),{'hotpotqa':[{'context':'','input':'Q'}]})
            with self.assertRaisesRegex(ValueError,'context'):
                load_samples(archive,prompt,['hotpotqa'],1)
            with self.assertRaisesRegex(ValueError,'member'):
                load_samples(archive,prompt,['lcc'],1)


if __name__=='__main__':unittest.main()
