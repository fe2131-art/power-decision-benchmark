"""Offline tests for the official logprob path of luna-llm (--llm-logprobs). No network/keys required."""
import math
import unittest

import benchmark as b


def _tok(token, logprob, alts=()):
    return {'token': token, 'logprob': logprob, 'bytes': [],
            'top_logprobs': [{'token': t, 'logprob': lp, 'bytes': []} for t, lp in alts]}


def _raw(stream, text=None):
    text = text if text is not None else ''.join(t['token'] for t in stream)
    return {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text, 'logprobs': stream}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 5}}


class LogprobPathTests(unittest.TestCase):
    def test_request_asks_for_output_logprobs_only_when_enabled(self):
        plain = b.llm_body('x', 'none')
        self.assertNotIn('include', plain)
        self.assertNotIn('top_logprobs', plain)
        lp = b.llm_body('x', 'none', include_logprobs=True)
        self.assertEqual(lp['include'], ['message.output_text.logprobs'])
        self.assertEqual(lp['top_logprobs'], 20)
        self.assertEqual(lp['text']['format']['schema'], b.json_schema())  # labels-only schema unchanged

    def test_first_token_distribution_is_renormalized(self):
        # {"equipment":"breaker", ...}; alternatives at the value token of equipment
        stream = [
            _tok('{"', 0.0), _tok('equipment', 0.0), _tok('":"', 0.0),
            _tok('breaker', math.log(0.7), [('breaker', math.log(0.7)), ('switch', math.log(0.2)), ('meter', math.log(0.1))]),
            _tok('","', 0.0), _tok('observation', 0.0), _tok('":"', 0.0), _tok('heat', 0.0),
            _tok('","', 0.0), _tok('cause_status', 0.0), _tok('":"', 0.0), _tok('suspected', 0.0),
            _tok('"}', 0.0),
        ]
        labels, extra = b.normalize_llm_logprobs(_raw(stream))
        self.assertEqual(labels, {'equipment': 'breaker', 'observation': 'heat', 'cause_status': 'suspected'})
        self.assertEqual(extra['probability_source'], 'logprobs_first_token_approx')
        dist = extra['probabilities']['equipment']
        self.assertAlmostEqual(dist['breaker'], 0.7, places=6)
        self.assertAlmostEqual(dist['switch'], 0.2, places=6)
        self.assertAlmostEqual(dist['meter'], 0.1, places=6)
        self.assertEqual(dist['transformer'], 0.0)
        self.assertTrue(b.valid_distribution('equipment', dist))

    def test_multi_token_candidate_matches_longest_prefix(self):
        # 'overhead_line' is produced as 'overhead' + '_line'; first token 'overhead' carries its mass
        stream = [
            _tok('{"', 0.0), _tok('equipment', 0.0), _tok('":"', 0.0),
            _tok('overhead', math.log(0.6), [('overhead', math.log(0.6)), ('other', math.log(0.4))]),
            _tok('_line', 0.0), _tok('","', 0.0), _tok('observation', 0.0), _tok('":"', 0.0), _tok('heat', 0.0),
            _tok('","', 0.0), _tok('cause_status', 0.0), _tok('":"', 0.0), _tok('suspected', 0.0), _tok('"}', 0.0),
        ]
        _, extra = b.normalize_llm_logprobs(_raw(stream))
        dist = extra['probabilities']['equipment']
        self.assertAlmostEqual(dist['overhead_line'], 0.6, places=6)
        self.assertAlmostEqual(dist['other'], 0.4, places=6)

    def test_missing_logprobs_yield_no_distribution_not_zero(self):
        raw = {'output': [{'type': 'message', 'content': [{'type': 'output_text',
               'text': '{"equipment":"breaker","observation":"heat","cause_status":"suspected"}'}]}], 'usage': {}}
        labels, extra = b.normalize_llm_logprobs(raw)
        self.assertEqual(labels['equipment'], 'breaker')
        self.assertTrue(all(extra['probabilities'][f] == {} for f in b.FIELDS))


if __name__ == '__main__':
    unittest.main()
