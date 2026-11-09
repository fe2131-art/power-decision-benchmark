"""No network/keys required. python -m unittest -v"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import benchmark as b

class BenchmarkTests(unittest.TestCase):
    def test_llm_and_decisions_model_ids_are_independently_configurable(self):
        gold = b.data()[0]['gold']
        from types import SimpleNamespace
        args=SimpleNamespace(reasoning='none',timeout=1,openai_url='http://local.test/v1',
                             llm_model='example-llm-id',decisions_model='example-decisions-id')
        seen=[]
        def fake_post(url,body,key=None,timeout=None):
            seen.append((url,body['model']))
            if url.endswith('/responses'):
                return {'output':[{'content':[{'type':'output_text','text':json.dumps(gold)}]}]}
            return {'answers':[{'type':'choice','name':f,'choice':gold[f],
                     'probabilities':[{'value':k,'probability':float(k==gold[f])}
                        for k in b.QUESTIONS[f]['criteria']]}
                     for f in b.FIELDS]}
        with patch.dict('os.environ',{'OPENAI_API_KEY':'TEST_KEY'}),patch.object(b,'post_json',side_effect=fake_post):
            b.call_backend('luna-llm',b.data()[0]['text'],args)
            b.call_backend('openai-decisions',b.data()[0]['text'],args)
        self.assertEqual(seen,[('http://local.test/v1/responses','example-llm-id'),
                               ('http://local.test/v1/decisions','example-decisions-id')])

    def test_dataset(self):
        cases = b.data()
        self.assertEqual(len(cases), 474)
        self.assertEqual(len(set(c['id'] for c in cases)), 474)
        self.assertGreaterEqual(len(set(c['group'] for c in cases)), 8)
        self.assertGreaterEqual(len(set(c['family_id'] for c in cases)), 30)
        self.assertEqual(len(set(c['text'] for c in cases)), 474)
        demo = b.data(b.HERE / 'data' / 'legacy' / 'demo_10_cases.jsonl')
        self.assertEqual(len(demo), 10)
        self.assertEqual({c['group'] for c in demo}, {'clear','negation','hypothesis','mixed_status','resolved','out_of_scope'})
        self.assertEqual(demo[0]['id'], 'PS-009')
        self.assertEqual(demo[-1]['id'], 'PS-012')
        full = b.data(b.HERE / 'data' / 'legacy' / 'synthetic_inspection_cases.jsonl')
        self.assertEqual(len(full), 48)

    def test_demo_10_covers_all_answer_classes(self):
        cases = b.data(b.HERE / 'data' / 'legacy' / 'demo_10_cases.jsonl')
        self.assertEqual({r['gold']['equipment'] for r in cases}, set(b.QUESTIONS['equipment']['criteria']))
        self.assertEqual({r['gold']['observation'] for r in cases}, set(b.QUESTIONS['observation']['criteria']))
        self.assertEqual({r['gold']['cause_status'] for r in cases}, set(b.QUESTIONS['cause_status']['criteria']))

    def test_schema(self):
        schema = b.json_schema()
        self.assertEqual(schema['required'], ['equipment','observation','cause_status'])
        self.assertEqual(schema['additionalProperties'],False)
        self.assertEqual(len(b.decision_questions()), 3)
        self.assertEqual(len(b.systemone_questions()), 3)

    def test_llm_parser(self):
        example = {'equipment':'transformer','observation':'heat','cause_status':'suspected'}
        raw = {'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(example)}]}], 'usage':{'input_tokens':100,'output_tokens':12}}
        labels,extra = b.normalize_llm(raw)
        self.assertEqual(labels,example)
        self.assertEqual(extra['usage']['input_tokens'],100)
        with self.assertRaises(ValueError):
            b.normalize_llm({'output':[{'content':[{'type':'output_text','text':'{}'}]}]})

    def test_decisions_parser(self):
        gold = b.data()[0]['gold']
        ans = []
        for f in b.FIELDS:
            opts=list(b.QUESTIONS[f]['criteria'])
            probs={x:(1.0 if x==gold[f] else 0.0) for x in opts}
            ans.append({'type':'choice','name':f,'choice':gold[f],
                        'probabilities':[{'value':x,'probability':v} for x,v in probs.items()],
                        'confidence':0.9})
        labels,extra = b.normalize_decisions({'answers':ans,'usage':{'input_tokens':91,'output_tokens':0}})
        self.assertEqual(labels,gold)
        self.assertEqual(extra['probabilities']['equipment'][gold['equipment']],1.0)

    def test_systemone_parser(self):
        gold = b.data()[0]['gold']
        ans={f:{'choice':gold[f],'probabilities':{k:(1.0 if k==gold[f] else 0.0) for k in b.QUESTIONS[f]['criteria']}} for f in b.FIELDS}
        labels,_ = b.normalize_systemone({'answers':ans})
        self.assertEqual(labels,gold)

    def test_urls_and_request_bodies_without_network(self):
        from os import environ
        rec=b.data()[0]['text']
        args=SimpleNamespace(reasoning='none',timeout=1,clef_url='http://127.0.0.1:30000',clef_model='Cloudflare/clef-flash', d1_url='http://127.0.0.1:8080', d1_model='', qwen_url='http://127.0.0.1:8082', kev_url='http://127.0.0.1:11434')
        gold=b.data()[0]['gold']
        def fake_post(url,body,key=None,timeout=None):
            if url.endswith('/chat/completions'):
                self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
                self.assertEqual(body['logprobs'], True)
                return {'choices':[{'message':{'content':json.dumps(gold)},'finish_reason':'stop'}],'usage':{'prompt_tokens':7,'completion_tokens':3}}
            if url.endswith('/responses'):
                self.assertEqual(body['model'],'gpt-6-luna')
                self.assertEqual(body['reasoning']['effort'],'none')
                return {'output':[{'content':[{'type':'output_text','text':json.dumps(gold)}]}]}
            if url.endswith('/decisions'):
                self.assertEqual(len(body['questions']),3)
                return {'answers':[{'type':'choice','name':f,'choice':gold[f],'probabilities':[{'value':k,'probability':float(k==gold[f])} for k in b.QUESTIONS[f]['criteria']]} for f in b.FIELDS]}
            self.assertTrue(url.startswith('http://127.0.0.1:'))
            self.assertIn('systemone',url)
            return {'answers':{f:{'choice':gold[f], 'probabilities':{k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']}} for f in b.FIELDS}}
        with patch.dict(environ,{'OPENAI_API_KEY':'FAKE_TEST_KEY'}),patch.object(b,'post_json',side_effect=fake_post):
            for name in b.BACKENDS:
                pred,_ = b.call_backend(name,rec,args)
                self.assertEqual(pred,gold)

    def test_metrics(self):
        gold=b.data()[0]['gold']
        probs = {f:{k:(1.0 if k == gold[f] else 0.0) for k in b.QUESTIONS[f]['criteria']} for f in b.FIELDS}
        records=[{'backend':'openai-decisions','id':'a','group':'clear','error':None,'gold':gold,'predicted':gold,
                  'probabilities':probs,'elapsed_ms':11,'usage':{'input_tokens':500,'output_tokens':0}},
                 {'backend':'openai-decisions','id':'b','group':'negation','error':'timeout','gold':gold,'predicted':None,
                  'probabilities':{},'elapsed_ms':300,'usage':{}}]
        result=b.evaluate(records,0.85)['openai-decisions']
        self.assertEqual(result['exact_all_fields'],1.0)
        self.assertEqual(result['macro_f1_over_fields'],1.0)
        self.assertEqual(result['errors'],1)
        self.assertEqual(result['selective_coverage'],0.5)
        self.assertEqual(result['multiclass_brier_mean'],0.0)
        self.assertIsNone(result['approx_openai_usd_uncached'])  # failed request: cannot report full-run cost
        self.assertEqual(result['exact_all_fields_over_planned'],0.5)
        self.assertEqual(result['probability_complete_cases'],1)

    def test_probability_validation(self):
        gold=b.data()[0]['gold']
        for f in b.FIELDS:
            dist={k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']}
            self.assertTrue(b.valid_distribution(f, dist))
            bad=dict(dist);bad[gold[f]]=1.5
            with self.assertRaises(ValueError): b.valid_distribution(f,bad)
            incomplete={gold[f]:1.0}
            self.assertFalse(b.valid_distribution(f,incomplete))

    def test_missing_probabilities_not_counted_as_accepted(self):
        gold=b.data()[0]['gold']
        rows=[{'backend':'openai-decisions','id':str(i),'group':'clear','error':None,'gold':gold,
               'predicted':gold,'probabilities':{},'elapsed_ms':10,'usage':{'input_tokens':100}} for i in range(2)]
        result=b.evaluate(rows,0.85,planned_cases=2)['openai-decisions']
        self.assertEqual(result['exact_all_fields'],1)
        self.assertIsNone(result['selective_coverage'])
        self.assertIsNone(result['ece_10bins'])
        self.assertEqual(result['probability_complete_cases'],0)
        self.assertEqual(result['approx_openai_usd_uncached'],0.00002)

    def test_missing_usage_does_not_look_free(self):
        gold=b.data()[0]['gold']
        row={'backend':'openai-decisions','id':'1','group':'clear','error':None,'gold':gold,
             'predicted':gold,'probabilities':{},'elapsed_ms':10,'usage':{}}
        result=b.evaluate([row],0.85)['openai-decisions']
        self.assertIsNone(result['approx_openai_usd_uncached'])

    def test_absent_field_and_duplicate_response(self):
        opts=b.decision_questions()
        labels=b.data()[0]['gold']
        answers=[]
        for q in opts:
            name=q['name']
            answers.append({'type':'choice','name':name,'choice':labels[name],
                            'probabilities':[{'value':it['value'],'probability':1.0 if it['value']==labels[name] else 0.0}
                                             for it in q['choices']]})
        with self.assertRaises(ValueError): b.normalize_decisions({'answers':answers+[answers[0]]})
        with self.assertRaises(ValueError): b.normalize_decisions({'answers':answers[:-1]})

    def test_shared_input_same_for_all(self):
        rec=b.data()[0]['text']
        self.assertIn(rec,b.shared_input(rec))
        self.assertIn(b.SYSTEM,b.shared_input(rec))
        self.assertIn(b.shared_input(rec),b.llm_body(rec,'none')['input'][0]['content'])

    def test_incomplete_response_is_failure(self):
        with self.assertRaises(ValueError):
            b.normalize_llm({'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'},'output':[]})


    def test_d1_systemone_request_follows_official_example_without_model_by_default(self):
        gold=b.data()[0]['gold']
        args=SimpleNamespace(timeout=2,d1_url='http://127.0.0.1:8080',d1_model='')
        observed=[]
        def mock_post(url,body,key=None,timeout=None):
            observed.append((url,body))
            return {'answers':{f:{'type':'choice','choice':gold[f],
                        'probabilities':{k:float(k==gold[f]) for k in b.QUESTIONS[f]['criteria']}} for f in b.FIELDS}}
        with patch.object(b,'post_json',side_effect=mock_post):
            result,_=b.call_backend('d1-2b',b.data()[0]['text'],args)
        self.assertEqual(result,gold)
        self.assertNotIn('model',observed[0][1])
        self.assertEqual(observed[0][0],'http://127.0.0.1:8080/v1/systemone')
        args.d1_model='custom-server-registered-model'
        with patch.object(b,'post_json',side_effect=mock_post):
            b.call_backend('d1-2b',b.data()[0]['text'],args)
        self.assertEqual(observed[-1][1]['model'],'custom-server-registered-model')

    def test_duplicate_probabilities_raise_instead_of_silent_overwrite(self):
        gold=b.data()[0]['gold']
        answers=[]
        for f in b.FIELDS:
            values=[{'value':x,'probability':float(x==gold[f])} for x in b.QUESTIONS[f]['criteria']]
            answers.append({'name':f,'type':'choice','choice':gold[f],'probabilities':values})
        answers[0]['probabilities'].append(answers[0]['probabilities'][0])
        with self.assertRaisesRegex(ValueError,'duplicate probability'):
            b.normalize_decisions({'answers':answers})

    def test_invalid_probability_types_raise(self):
        f='equipment'
        dist={k:0.0 for k in b.QUESTIONS[f]['criteria']}
        dist['transformer']='1.0'
        with self.assertRaisesRegex(ValueError,'must be numeric'):
            b.valid_distribution(f,dist)
        dist['transformer']=True
        with self.assertRaisesRegex(ValueError,'must be numeric'):
            b.valid_distribution(f,dist)

    def test_unfinished_llm_response_not_accepted(self):
        gold=b.data()[0]['gold']
        with self.assertRaisesRegex(ValueError,'not completed'):
            b.normalize_llm({'status':'in_progress','output_text':json.dumps(gold)})

    def test_reject_wrong_systemone_answer_type_or_extra_question(self):
        gold=b.data()[0]['gold']
        ans={f:{'type':'choice','choice':gold[f]} for f in b.FIELDS}
        with self.assertRaises(ValueError):
            b.normalize_systemone({'answers':{**ans, 'unexpected':{'choice':'anything'}}})
        ans['equipment']['type']='noul'
        with self.assertRaises(ValueError):
            b.normalize_systemone({'answers':ans})

    def test_csv_formula_like_inputs_are_escaped(self):
        self.assertEqual(b.safe_csv_text('=HYPERLINK("x")'), "'=HYPERLINK(\"x\")")
        self.assertEqual(b.safe_csv_text('  @unknown'), "'  @unknown")
        self.assertEqual(b.safe_csv_text('通常の設備点検'), '通常の設備点検')

    def test_all_failed_does_not_invent_macro_f1_or_error_counts(self):
        gold=b.data()[0]['gold']
        rows=[{'backend':'clef-flash','id':'a','error':'connection failed','gold':gold,
               'predicted':None,'probabilities':{},'elapsed_ms':4,'usage':{}}]
        summary=b.evaluate(rows,0.85,planned_cases=10)['clef-flash']
        self.assertIsNone(summary['macro_f1_over_fields'])
        self.assertIsNone(summary['unsupported_confirmed'])
        self.assertEqual(summary['exact_all_fields_over_planned'],0.0)
        self.assertEqual(summary['not_attempted'],9)

    def test_reject_invalid_planned_count(self):
        gold=b.data()[0]['gold']
        records=[{'backend':'openai-decisions','id':'a','error':None,'gold':gold,
                  'predicted':gold,'probabilities':{},'elapsed_ms':1,'usage':{}},
                 {'backend':'openai-decisions','id':'b','error':None,'gold':gold,
                  'predicted':gold,'probabilities':{},'elapsed_ms':1,'usage':{}}]
        with self.assertRaisesRegex(ValueError,'planned_cases'):
            b.evaluate(records,0.85,planned_cases=1)

    def test_gold_and_text_unchanged_from_source_48(self):
        full={r['id']:r for r in b.data(b.HERE/'data'/'legacy'/'synthetic_inspection_cases.jsonl')}
        from scripts.make_demo_dataset import DEMO_GROUP
        for row in b.data(b.HERE / 'data' / 'legacy' / 'demo_10_cases.jsonl'):
            self.assertEqual(row['text'],full[row['id']]['text'])
            self.assertEqual(row['gold'],full[row['id']]['gold'])
            self.assertEqual(row['group'],DEMO_GROUP[row['id']])


if __name__=='__main__':
    unittest.main()
