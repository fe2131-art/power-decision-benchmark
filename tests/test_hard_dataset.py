"""New difficult-set tests. API keys and GPU are not required."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import benchmark as b
from scripts.make_hard_dataset import build_records, render, OUT
from scripts.paired_comparison import analyze


class HardDatasetTests(unittest.TestCase):
    def test_reproducible_bytes(self):
        a, _, _ = render()
        self.assertEqual(OUT.read_text('utf-8'), a)

    def test_expected_size_uniqueness(self):
        rows = b.data()
        self.assertEqual(len(rows), 474)
        self.assertEqual(len({r['id'] for r in rows}), 474)
        self.assertEqual(len({r['text'] for r in rows}), 474)
        self.assertEqual(Counter(r['difficulty'] for r in rows), {'hard':432,'very_hard':42})
        self.assertEqual(len({r['family_id'] for r in rows}), 47)

    def test_label_support_and_explicit_normals(self):
        rows = b.data()
        for f in b.FIELDS:
            self.assertEqual({r['gold'][f] for r in rows}, set(b.QUESTIONS[f]['criteria']))
        for r in rows:
            if r['gold']['observation'] == 'normal':
                self.assertEqual(r['gold']['cause_status'], 'unknown')
                self.assertTrue(any(term in r['text'] for term in ('異常なし','正常','異常所見なし','異常は認められなかった','認めず')))

    def test_stratified_sampling_is_repeatable_and_keeps_gold(self):
        rows = b.data()
        a=b.stratified_sample(rows,64,20261109)
        aa=b.stratified_sample(rows,64,20261109)
        other=b.stratified_sample(rows,64,20261110)
        self.assertEqual([r['id'] for r in a],[r['id'] for r in aa])
        self.assertNotEqual([r['id'] for r in a],[r['id'] for r in other])
        self.assertEqual(len(a),64)
        self.assertEqual(Counter(r['difficulty'] for r in a), {'hard':58,'very_hard':6})
        self.assertEqual(a, [r for r in rows if r['id'] in {v['id'] for v in a}])
        self.assertEqual(len(rows),len(b.stratified_sample(rows,len(rows),20261109)))

    def test_invalid_sample_sizes(self):
        for size in [-1,0,475]:
            with self.assertRaises(ValueError):
                b.stratified_sample(b.data(),size,1)

    def test_target_current_event_pairing(self):
        rows=b.data()
        # Each controlled scenario has cause-confirmed/suspected/unknown variants.
        scenarios={r['family_id'] for r in rows if r['variant'].startswith('template_')}
        for fam in scenarios:
            matching=[r for r in rows if r['family_id']==fam and r['variant'].startswith('template_')]
            self.assertEqual(len(matching),12)
            self.assertEqual(Counter(r['gold']['cause_status'] for r in matching),
                             {'confirmed':4,'suspected':4,'unknown':4})

    def test_gold_is_never_sent_in_the_request_payload(self):
        r=b.data()[0]
        serialized=json.dumps(b.llm_body(r['text'],'none'),ensure_ascii=False)
        self.assertIn(r['text'], serialized)
        self.assertNotIn('"gold"',serialized)
        self.assertNotIn('"rationale"',serialized)

    def test_paired_cluster_comparison(self):
        gold=b.data()[0]['gold']
        data=[]
        for i in range(6):
            fid='fam1' if i<3 else 'fam2'
            incorrect={**gold,'cause_status':'unknown' if gold['cause_status']!='unknown' else 'confirmed'}
            for be in ['luna-llm','openai-decisions']:
                data.append({'backend':be,'id':f'id{i}','family_id':fid,'error':None,
                             'gold':gold,'predicted':gold if be=='openai-decisions' else incorrect})
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'predictions.jsonl'
            f.write_text(''.join(json.dumps(r)+'\n' for r in data),'utf-8')
            out=analyze(f,n_boot=100,seed=1)
            self.assertEqual(len(out),1)
            self.assertEqual(out[0]['n_families'],2)
            self.assertEqual(out[0]['delta_b_minus_a'],1.0)
            self.assertEqual(out[0]['cluster_bootstrap_95_low'],1.0)

    def test_paired_comparison_rejects_partial(self):
        gold=b.data()[0]['gold']
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'predictions.jsonl'
            f.write_text('\n'.join(json.dumps(x) for x in [
                {'backend':'a','id':'1','gold':gold,'predicted':gold,'family_id':'1','error':None},
                {'backend':'b','id':'1','gold':gold,'predicted':None,'family_id':'1','error':'server offline'},
            ]),'utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete prediction'):
                analyze(f,n_boot=100)


if __name__ == '__main__':
    unittest.main()
