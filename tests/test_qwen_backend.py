"""Offline tests for the qwen3.8-27b llama-server backend (chat completions + token logprobs)."""
import json
import math
import unittest

import benchmark as b


def _chat_tok(token, logprob, alts=()):
    return {'token': token, 'logprob': logprob, 'bytes': [],
            'top_logprobs': [{'token': t, 'logprob': lp, 'bytes': []} for t, lp in alts]}


class QwenBackendTests(unittest.TestCase):
    def test_request_disables_thinking_and_requests_token_logprobs(self):
        body = b.qwen_body('x')
        self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(body['logprobs'])
        self.assertEqual(body['top_logprobs'], 20)
        self.assertEqual(body['temperature'], 0)
        self.assertEqual(body['response_format']['json_schema']['schema'], b.json_schema())

    def test_chat_logprobs_give_first_token_distribution(self):
        stream = [
            _chat_tok('{"', 0.0), _chat_tok('equipment', 0.0), _chat_tok('":"', 0.0),
            _chat_tok('breaker', math.log(0.6), [('breaker', math.log(0.6)), ('switch', math.log(0.3)), ('meter', math.log(0.1))]),
            _chat_tok('","', 0.0), _chat_tok('observation', 0.0), _chat_tok('":"', 0.0), _chat_tok('heat', 0.0),
            _chat_tok('","', 0.0), _chat_tok('cause_status', 0.0), _chat_tok('":"', 0.0), _chat_tok('suspected', 0.0),
            _chat_tok('"}', 0.0),
        ]
        content = ''.join(t['token'] for t in stream)
        raw = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop',
                            'logprobs': {'content': stream}}],
               'usage': {'prompt_tokens': 900, 'completion_tokens': 14}}
        labels, extra = b.normalize_qwen(raw)
        self.assertEqual(labels, {'equipment': 'breaker', 'observation': 'heat', 'cause_status': 'suspected'})
        self.assertEqual(extra['probability_source'], 'logprobs_first_token_approx')
        self.assertEqual(extra['usage'], {'input_tokens': 900, 'output_tokens': 14})
        dist = extra['probabilities']['equipment']
        self.assertAlmostEqual(dist['breaker'], 0.6, places=6)
        self.assertAlmostEqual(dist['switch'], 0.3, places=6)
        self.assertTrue(b.valid_distribution('equipment', dist))

    def test_truncated_generation_is_an_error(self):
        raw = {'choices': [{'message': {'content': '{"equipment":"bre'}, 'finish_reason': 'length'}], 'usage': {}}
        with self.assertRaises(ValueError):
            b.normalize_qwen(raw)



class PrettyJsonLogprobTests(unittest.TestCase):
    def test_pretty_printed_output_is_located_by_regex(self):
        # llama.cpp emits pretty-printed JSON; tokens split as ' "', 'equipment', '":', ' "', 'breaker', ...
        toks = [('{', 0.0, []), ('\n', 0.0, []), ('  ', 0.0, []), ('"', 0.0, []), ('equipment', 0.0, []), ('":', 0.0, []),
                (' "', 0.0, []), ('breaker', math.log(0.8), [('breaker', math.log(0.8)), ('switch', math.log(0.2))]),
                ('",', 0.0, []), ('\n  "observation": "heat",', 0.0, []),
                ('\n  "cause_status": "suspected"\n}', 0.0, [])]
        stream = [_chat_tok(t, lp, alts) for t, lp, alts in toks]
        content = ''.join(t for t, _, _ in toks)
        raw = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop', 'logprobs': {'content': stream}}], 'usage': {}}
        labels, extra = b.normalize_qwen(raw)
        self.assertEqual(labels['equipment'], 'breaker')
        self.assertAlmostEqual(extra['probabilities']['equipment']['breaker'], 0.8, places=6)


if __name__ == '__main__':
    unittest.main()
