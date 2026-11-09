"""Confidence/calibration regression tests; no API keys or GPU."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import benchmark as b
from scripts.make_calibration_split import build, CALIBRATION, TEST, MANIFEST
from scripts.probability_audit import (
    CLAMP_EPS, analyze, calculate_metrics, choose_threshold, coverage_risk, ece_bins,
    index_predictions, normalize_probabilities, score_prediction,
)


class ProbabilityAuditTests(unittest.TestCase):
    def test_family_disjoint_manifest_reproducible(self):
        a,t,manifest=build()
        self.assertEqual(a,CALIBRATION.read_bytes())
        self.assertEqual(t,TEST.read_bytes())
        self.assertEqual(manifest,json.loads(MANIFEST.read_text('utf-8')))
        self.assertEqual(manifest['n_calibration']+manifest['n_test'],474)
        self.assertEqual(len(set(manifest['calibration_case_ids']) & set(manifest['test_case_ids'])),0)
        self.assertFalse(set(manifest['calibration_family_ids']) & set(manifest['test_family_ids']))
        self.assertGreaterEqual(manifest['n_test'],250)

    def test_probability_schema_for_llm(self):
        schema=b.json_schema(include_probabilities=True)
        self.assertEqual(set(schema['required']),set(b.FIELDS)|{'probabilities'})
        self.assertEqual(set(schema['properties']['probabilities']['properties']),set(b.FIELDS))
        self.assertEqual(b.llm_body('test','none',include_probabilities=True)['max_output_tokens'],1500)
        self.assertEqual(b.llm_body('test','none')['max_output_tokens'],300)

    def test_verbalized_llm_probability_parsing(self):
        labels=b.data()[0]['gold']
        probabilities={f:{label:float(label==labels[f]) for label in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
        output={**labels,'probabilities':probabilities}
        raw={'status':'completed','output_text':json.dumps(output)}
        preds,extra=b.normalize_llm(raw,expect_probabilities=True)
        self.assertEqual(preds,labels)
        self.assertEqual(extra['probability_source'],'verbalized_llm')
        self.assertEqual(extra['probabilities'],probabilities)
        bad={**labels,'probabilities':{**probabilities,'equipment':{'other':1}}}
        with self.assertRaisesRegex(ValueError,'incomplete'):
            b.normalize_llm({'status':'completed','output_text':json.dumps(bad)},True)

    def test_mock_llm_api_requests_probabilities_when_opted_in(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        gold=b.data()[0]['gold']
        probs={f:{k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
        args=SimpleNamespace(reasoning='none',timeout=1,openai_url='http://127.0.0.1:8801/v1',
                             llm_model='example-llm',llm_probabilities=True)
        observed=[]
        def fake_post(url,body,key=None,timeout=None):
            observed.append(body)
            return {'status':'completed','output_text':json.dumps({**gold,'probabilities':probs})}
        with patch.dict('os.environ',{'OPENAI_API_KEY':'DUMMY'}),patch.object(b,'post_json',side_effect=fake_post):
            pred,info=b.call_backend('luna-llm','点検記録',args)
        self.assertEqual(pred,gold)
        self.assertEqual(info['probability_source'],'verbalized_llm')
        self.assertIn('probabilities',observed[0]['text']['format']['schema']['properties'])

    def test_normalization_rejects_invalids_and_tracks_rounding(self):
        f='cause_status'
        self.assertEqual(normalize_probabilities({},f)[1],'missing')
        self.assertEqual(normalize_probabilities({'confirmed':.5},f)[1],'wrong_labels')
        p={'confirmed':.5,'suspected':.3,'unknown':.195}
        v,status=normalize_probabilities(p,f)
        self.assertEqual(status,'renormalized')
        self.assertAlmostEqual(sum(v.values()),1.0)
        self.assertEqual(normalize_probabilities({**p,'confirmed':.7},f)[1],'invalid_sum')
        self.assertEqual(normalize_probabilities({**p,'confirmed':float('nan')},f)[1],'invalid_values')

    def test_brier_nll_ece_tiny_exactly(self):
        p, bins=ece_bins([(.8,1),(.8,0)])
        self.assertAlmostEqual(p,.3)
        self.assertEqual(sum(z['n'] for z in bins),2)
        gold=b.data()[0]['gold']
        probs={f:{k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
        row={'id':'x','backend':'native','family_id':'x','gold':gold,'predicted':gold,
             'probabilities':probs,'error':None,'probability_source':'native_decision'}
        s=score_prediction(row)
        m,_=calculate_metrics([s],bootstrap=100)
        self.assertEqual(m['exact_accuracy_over_planned'],1)
        for field in b.FIELDS:
            self.assertEqual(m['per_field'][field]['multiclass_brier'],0)
            self.assertEqual(m['per_field'][field]['negative_log_likelihood_nats'],0)
            self.assertEqual(m['per_field'][field]['ece_10bins'],0)

    def test_provider_confidence_not_treated_as_probability(self):
        gold=b.data()[0]['gold']
        probs={f:{k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
        row={'id':'x','backend':'native','family_id':'x','gold':gold,'predicted':gold,
             'probabilities':probs,'provider_confidence':{f:0.01 for f in b.FIELDS},'error':None}
        m,_=calculate_metrics([score_prediction(row)],bootstrap=100)
        self.assertEqual(m['per_field']['equipment']['ece_10bins'],0)
        self.assertEqual(m['per_field']['equipment']['provider_confidence_mean'],0.01)

    def test_decision_selection_mismatch_blocks_acceptance(self):
        gold=b.data()[0]['gold']
        probs={f:{k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
        bad=dict(probs['cause_status'])
        wrong=next(k for k in bad if k!=gold['cause_status'])
        bad[gold['cause_status']]=0.2
        bad[wrong]=0.8
        probs['cause_status']=bad
        row={'id':'x','backend':'m','gold':gold,'predicted':gold,'probabilities':probs,'error':None}
        item=score_prediction(row)
        self.assertIsNone(item['case_p'])
        self.assertTrue(item['fields']['cause_status']['choice_probability_mismatch'])
        self.assertEqual(coverage_risk([item],0)['accepted'],0)

    def test_threshold_development_only(self):
        gold=b.data()[0]['gold']
        def sample(name, p, exact):
            wrong={**gold,'cause_status':'confirmed' if gold['cause_status']!='confirmed' else 'unknown'}
            return {'id':name,'backend':'a','family_id':name,'gold':gold,
                    'predicted':gold if exact else wrong,
                    'probabilities':{f:({k:(p if k==(gold if exact else wrong)[f] else (1-p)/(len(b.QUESTIONS[f]['criteria'])-1))
                                         for k in b.QUESTIONS[f]['criteria']}) for f in b.FIELDS},
                    'error':None}
        dev=[score_prediction(sample(f'd{i}',.99,True)) for i in range(10)]
        dev+=[score_prediction(sample('low',.5,False))]
        gate=choose_threshold(dev,risk_budget=0,min_accepted=5)
        self.assertEqual(gate['dev_accepted'],10)
        self.assertGreater(gate['threshold'],.5)
        # A test failure doesn't change threshold, only the observed held-out risk.
        test=[score_prediction(sample('t1',.99,False))]
        self.assertEqual(coverage_risk(test,gate['threshold'])['empirical_risk'],1.0)

    def test_incomplete_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            index_predictions([{'backend':'a','id':'1'}],['1','2'],'test')

    def test_no_overlap_and_stable_comparison(self):
        # Small manifest with two disjoint families, correct native vectors.
        labels=b.data()[0]['gold']
        def row(id,fam,backend,correct=True):
            pred=dict(labels)
            if not correct:pred['cause_status']='confirmed' if labels['cause_status']!='confirmed' else 'unknown'
            vec={f:{x:(.96 if x==pred[f] else .04/(len(b.QUESTIONS[f]['criteria'])-1)) for x in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
            return {'id':id,'backend':backend,'family_id':fam,'gold':labels,'predicted':pred,'probabilities':vec,
                    'probability_source':'native_decision','error':None}
        manifest={'calibration_case_ids':[str(i) for i in range(10)],'test_case_ids':['t1','t2'],
                  'calibration_family_ids':['dev_fam'],'test_family_ids':['test_fam']}
        dev=[row(str(i),'dev_fam','native') for i in range(10)]
        test=[row('t1','test_fam','native'),row('t2','test_fam','native',False)]
        r,tab=analyze(dev,test,manifest,risk_budget=0,bootstrap=100,seed=1)
        self.assertEqual(r['models']['native']['test']['exact_accuracy_over_planned'],.5)
        self.assertEqual(r['models']['native']['test']['test_selective_policy']['coverage'],1)
        self.assertEqual(r['models']['native']['test']['test_selective_policy']['empirical_risk'],.5)
        self.assertEqual(len(tab['case_probability']),2)


if __name__=='__main__':
    unittest.main()
