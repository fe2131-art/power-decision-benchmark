#!/usr/bin/env python3
"""Common benchmark for power-equipment maintenance records: generative LLMs, the Decisions API, and local decision models.

外部依存ゼロ。OpenAI API key or local /v1/systemone servers are needed for live runs.
Inputs are fully synthetic inspection records. Gold labels are used for scoring only and are never sent to the models.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'data' / 'hard_inspection_cases.jsonl'  # 474 hard cases; 10-case demo kept separately
OPENAI_URL = 'https://api.openai.com/v1'
# Current official spec: the model ID accepted by /v1/decisions is gpt-6-luna.
# The same ID does not guarantee identical internal implementations.
MODEL = 'gpt-6-luna'

# The specification is shared by all models. The same Japanese instructions are converted to each provider's format.
QUESTIONS = {
    'equipment': {
        'instructions': '記録の主たる点検対象設備を選んでください。明記された対象のみ選び、推測で補完しない。複数なら不明を選ぶ。',
        'criteria': {
            'transformer': '変圧器。配電用変圧器、柱上変圧器を含む。',
            'breaker': '遮断器。VCB、遮断器の主接点等を含む。',
            'switch': '開閉器。負荷開閉器、区分開閉器、LBS等を含む。',
            'overhead_line': '架空線。架空電線、電線接続部、架空導体を含む。',
            'meter': '電力量計、スマートメーター。',
            'other': '上記以外、対象なし、または対象の特定不可。',
        },
    },
    'observation': {
        'instructions': '現在、実際に観測・記録された主要事象を選んでください。否定・改善済みの事象は選ばない。仮説上の事象は観測済みとして扱わない。',
        'criteria': {
            'heat': '発熱、温度上昇、赤外線点検でのホットスポット。',
            'oil_leak': '油漏れ、絶縁油のにじみ、油滴が観測された。',
            'operation_failure': '開閉・遮断動作不良、動作不能、トリップ失敗。',
            'corrosion': '腐食、錆の発生、金属部材の著しい錆。',
            'communication': '通信断、データ通信エラー、遠隔検針の通信障害。',
            'damage': '断線、亀裂、破損、ひび割れなどの物理損傷。',
            'normal': '特に異常なし。観測済み異常が明示的に否定される。',
            'other': '上記以外または観測事象が十分に特定できない。',
        },
    },
    'cause_status': {
        'instructions': '記録に記述された「異常原因の説明」の根拠レベルを選んでください。原因と断定された場合のみconfirmed。原因が推定・疑い・可能性ならsuspected。原因が未記載・調査中・原因の記述なしならunknown。対象は観測事象の有無ではない。',
        'criteria': {
            'confirmed': '原因が調査・試験・分解等で確定したと記載されている。',
            'suspected': '原因として疑われる、推定される、可能性があると記載されている。',
            'unknown': '原因記載なし、原因未特定、調査中、または異常なし。',
        },
    },
}
FIELDS = tuple(QUESTIONS)
SYSTEM = 'あなたは電力設備の点検記録をデータ分析用にラベル化する担当者です。原文で確認できることだけを分類し、原因を推測で断定しないでください。判断は設備操作や保安判定ではなく、分析用データの整理です。'
BACKENDS = ['luna-llm', 'openai-decisions', 'clef-flash', 'd1-2b', 'qwen3.8-27b',
            'kev-0.8b', 'kev-9b', 'kev-0.8b-bf16', 'kev-9b-bf16']
KEV_MODELS = {'kev-0.8b': 'kev:0.8b-q8_0', 'kev-9b': 'kev:9b-q8_0',
              'kev-0.8b-bf16': 'kev:0.8b-bf16', 'kev-9b-bf16': 'kev:9b-bf16'}  # Ollama tags
PRICES = {'luna-llm': (0.10, 0.50), 'openai-decisions': (0.10, 0.00)}  # 1M token, 非キャッシュ概算


def data(path: Path = DATA_PATH) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate IDs in dataset')
    for r in rows:
        if not isinstance(r.get('text'), str) or not r['text'].strip():
            raise ValueError(f'missing text in {r.get("id")}')
        if not isinstance(r.get('gold'), dict) or set(r['gold']) != set(FIELDS):
            raise ValueError(f'missing gold fields in {r["id"]}')
        for f in FIELDS:
            if r['gold'][f] not in QUESTIONS[f]['criteria']:
                raise ValueError(f'unknown label {r["id"]}:{f}')
    return rows


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Deterministic sampling stratified by difficulty (not by gold labels).

    Preserves the dataset's original order after selection so all models receive
    the same sequence. For fair inference, do not treat templated records as
    independent observations; see scripts/paired_comparison.py.
    """
    if size < 1 or size > len(rows):
        raise ValueError('sample-size must be between 1 and dataset size')
    if size == len(rows):
        return list(rows)
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r.get('difficulty', 'unspecified')].append(i)
    levels = sorted(buckets)
    quotas = {level: size*len(buckets[level])/len(rows) for level in levels}
    counts = {level: int(quotas[level]) for level in levels}
    remainder = size-sum(counts.values())
    for level in sorted(levels, key=lambda l: (- (quotas[l]-counts[l]), l))[:remainder]:
        counts[level] += 1
    rng = random.Random(seed)
    chosen = set()
    for level in levels:
        chosen.update(rng.sample(buckets[level], counts[level]))
    return [r for i,r in enumerate(rows) if i in chosen]


def post_json(url: str, body: dict, api_key: str | None = None, timeout: float = 90) -> dict:
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read(1500).decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {e.code} on {url}: {detail}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'Connection failure to {url}: {e.reason}') from e


def decision_questions() -> list[dict]:
    return [
        {'type': 'choice', 'name': f, 'instructions': spec['instructions'],
         'choices': [{'value': key, 'description': description} for key, description in spec['criteria'].items()]}
        for f, spec in QUESTIONS.items()
    ]


def systemone_questions() -> dict:
    return {
        f: {'type': 'choice', 'instructions': spec['instructions'], 'criteria': spec['criteria']}
        for f, spec in QUESTIONS.items()
    }


def json_schema(include_probabilities: bool = False) -> dict:
    properties = {f: {'type': 'string', 'enum': list(spec['criteria'])} for f, spec in QUESTIONS.items()}
    if include_probabilities:
        properties['probabilities'] = {
            'type': 'object',
            'properties': {
                f: {
                    'type': 'object',
                    'properties': {label: {'type': 'number'} for label in spec['criteria']},
                    'required': list(spec['criteria']),
                    'additionalProperties': False,
                } for f, spec in QUESTIONS.items()
            },
            'required': list(FIELDS),
            'additionalProperties': False,
        }
    return {
        'type': 'object', 'properties': properties, 'required': list(properties),
        'additionalProperties': False,
    }


def shared_input(record: str) -> str:
    """All backends receive the same primary evidence and common safety criteria."""
    return f'{SYSTEM}\n\n点検記録:\n{record}'


def safe_csv_text(value: str) -> str:
    """Escape spreadsheet formula-leading strings when exporting untrusted records.

    JSONL remains byte-faithful; only the human-readable CSV text column is protected.
    """
    if value.lstrip(' \t\r\n').startswith(('=', '+', '-', '@')):
        return "'" + value
    return value

LOGPROB_TOP_K = 20  # Responses API: top_logprobs accepts an integer 0-20


def _dists_from_tokens(toks: list) -> dict:
    """Core of the first-token logprob method. toks: [{'token','logprob','top_logprobs':[...]}]."""
    if not toks:
        return {}
    text = ''.join(t['token'] for t in toks)
    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t['token'])
    dists = {}
    for f in FIELDS:
        # Whitespace-tolerant: pretty-printed JSON (llama.cpp) and compact JSON (Responses) both match.
        m = re.search(r'"' + re.escape(f) + r'"\s*:\s*"', text)
        if not m:
            continue
        vs = m.end()
        i = max(j for j, s_ in enumerate(starts) if s_ <= vs)
        alts = {}
        for e in list(toks[i].get('top_logprobs', [])) + [{'token': toks[i]['token'], 'logprob': toks[i]['logprob']}]:
            alts.setdefault(e['token'], e['logprob'])
        mass = {}
        for c in QUESTIONS[f]['criteria']:
            matches = [t for t in alts if t and c.startswith(t)]
            mass[c] = math.exp(alts[max(matches, key=len)]) if matches else 0.0
        total = sum(mass.values())
        dists[f] = {c: v / total for c, v in mass.items()} if total > 0 else {}
    return dists


def logprob_distributions(raw: dict) -> dict:
    """Responses API: candidate distributions from output-token logprobs (official include option)."""
    out = [c for it in raw.get('output', []) for c in it.get('content', []) if c.get('type') == 'output_text']
    if not out or not out[0].get('logprobs'):
        return {}
    return _dists_from_tokens(out[0]['logprobs'])


def qwen_body(record: str) -> dict:
    """llama-server chat completion: grammar-constrained JSON labels, token logprobs, thinking disabled."""
    instructions = json.dumps(QUESTIONS, ensure_ascii=False, separators=(',', ':'))
    return {
        'messages': [{'role': 'user', 'content': f'{shared_input(record)}\n\n次の各項目について、指定候補から1つ選択してください:\n{instructions}'}],
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'inspection_labels', 'strict': True, 'schema': json_schema()}},
        'logprobs': True, 'top_logprobs': LOGPROB_TOP_K,
        'max_tokens': 300, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
    }


def normalize_qwen(raw: dict) -> tuple[dict, dict]:
    choice = raw['choices'][0]
    if choice.get('finish_reason') not in (None, 'stop'):
        raise ValueError(f'generation not completed: finish_reason={choice.get("finish_reason")}')
    labels = json.loads(choice['message']['content'])
    if set(labels) != set(FIELDS):
        raise ValueError(f'fields mismatch: {list(labels)}')
    validate_labels(labels)
    lp = (choice.get('logprobs') or {}).get('content') or []
    dists = _dists_from_tokens(lp)
    probabilities = {f: complete_probabilities(f, dists.get(f, {})) for f in FIELDS}
    u = raw.get('usage', {}) or {}
    usage = {'input_tokens': u.get('prompt_tokens'), 'output_tokens': u.get('completion_tokens')}
    return labels, {'probabilities': probabilities, 'usage': usage, 'provider_confidence': {},
                    'probability_source': 'logprobs_first_token_approx'}


def llm_body(record: str, reasoning: str, model: str = MODEL,
             include_probabilities: bool = False, include_logprobs: bool = False) -> dict:
    instructions = json.dumps(QUESTIONS, ensure_ascii=False, separators=(',', ':'))
    prob_instruction = (
        '\n各分類のすべての候補について、0〜1の自己推定確率をprobabilitiesに記載してください。'
        '分類項目ごとに合計1.0とし、確率最大の候補を選んでください。'
        '確率は「モデルが自己申告した値」であり、内部のlogitやAPIネイティブの確率ではありません。'
        '根拠が薄い場合は無理に極端な確率にしないでください。'
        if include_probabilities else ''
    )
    return {
        'model': model,
        'reasoning': {'effort': reasoning},
        'input': [{'role': 'user', 'content': (
            f'{shared_input(record)}\n\n次の各項目について、指定候補から1つ選択してください:\n'
            f'{instructions}{prob_instruction}')}],
        'text': {'format': {'type': 'json_schema', 'name': 'inspection_labels',
                            'strict': True, 'schema': json_schema(include_probabilities)}},
        'max_output_tokens': 1500 if include_probabilities else 300,
        **({'include': ['message.output_text.logprobs'], 'top_logprobs': LOGPROB_TOP_K} if include_logprobs else {}),
    }


def response_text(raw: dict) -> str:
    if raw.get('status') not in (None, 'completed'):
        raise ValueError(f'Response not completed: status={raw.get("status")} details={raw.get("incomplete_details")}')
    if isinstance(raw.get('output_text'), str) and raw['output_text'].strip():
        return raw['output_text']
    pieces = []
    for item in raw.get('output', []):
        for content in item.get('content', []):
            if content.get('type') == 'output_text' and isinstance(content.get('text'), str):
                pieces.append(content['text'])
    if pieces:
        return ''.join(pieces)
    raise ValueError(f'No output_text; status={raw.get("status")}, output={str(raw.get("output"))[:450]}')


def normalize_llm_logprobs(raw: dict) -> tuple[dict, dict]:
    labels = json.loads(response_text(raw))
    if set(labels) != set(FIELDS):
        raise ValueError(f'fields mismatch: {list(labels)}')
    validate_labels(labels)
    dists = logprob_distributions(raw)
    probabilities = {f: complete_probabilities(f, dists.get(f, {})) for f in FIELDS}
    return labels, {'probabilities': probabilities, 'usage': raw.get('usage', {}),
                    'provider_confidence': {}, 'probability_source': 'logprobs_first_token_approx'}


def normalize_llm(raw: dict, expect_probabilities: bool = False) -> tuple[dict, dict]:
    output = json.loads(response_text(raw))
    required = set(FIELDS) | ({'probabilities'} if expect_probabilities else set())
    if set(output) != required:
        raise ValueError(f'fields mismatch: {list(output)}')
    labels = {f: output[f] for f in FIELDS}
    validate_labels(labels)
    probabilities = {}
    if expect_probabilities:
        all_probs = output['probabilities']
        if not isinstance(all_probs, dict) or set(all_probs) != set(FIELDS):
            raise ValueError('LLM probability fields missing or unexpected')
        for f in FIELDS:
            p = all_probs[f]
            if not isinstance(p, dict) or not valid_distribution(f, p):
                raise ValueError(f'LLM {f} returned incomplete probability distribution')
            probabilities[f] = p
    return labels, {
        'probabilities': probabilities, 'usage': raw.get('usage', {}),
        'provider_confidence': {},
        'probability_source': 'verbalized_llm' if expect_probabilities else 'not_available',
    }


def valid_distribution(field: str, probs: dict) -> bool:
    """Return True only for complete probability distributions; reject malformed values.

    An absent/incomplete distribution can still yield a valid label, but MUST NOT
    contribute to Brier, ECE, or auto-acceptance coverage.
    """
    if not probs:
        return False
    expected = set(QUESTIONS[field]['criteria'])
    if set(probs) != expected:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in probs.values()):
        raise ValueError(f'{field}: probabilities must be numeric (not boolean/string)')
    values = [float(x) for x in probs.values()]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
        raise ValueError(f'{field}: invalid probability range or nonfinite value')
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=0.02):
        raise ValueError(f'{field}: probability total = {sum(values):.4f}, expected ~1')
    return True


def complete_probabilities(field: str, probs: dict) -> dict:
    if not isinstance(probs, dict) or not probs:
        return {}
    return probs if valid_distribution(field, probs) else {}


def probability_list_to_dict(entries: list, field: str) -> dict:
    """Convert provider probability pairs without silently overwriting duplicates."""
    output = {}
    for entry in entries:
        if not isinstance(entry, dict) or 'value' not in entry or 'probability' not in entry:
            raise ValueError(f'{field}: malformed probability entry')
        key = entry['value']
        if not isinstance(key, str):
            raise ValueError(f'{field}: expected a string choice value')
        if key in output:
            raise ValueError(f'{field}: duplicate probability for choice {key}')
        output[key] = entry['probability']
    return output


def normalize_decisions(raw: dict) -> tuple[dict, dict]:
    answers = raw.get('answers')
    if not isinstance(answers, list):
        raise ValueError('decisions.answers must be list')
    if any(not isinstance(a, dict) or 'name' not in a for a in answers):
        raise ValueError('decisions.answers entries must be named objects')
    by_name = {a['name']: a for a in answers}
    if len(by_name) != len(answers) or set(by_name) != set(FIELDS):
        raise ValueError('decisions.answers duplicate, missing, or unexpected names')
    labels, probabilities, confidence = {}, {}, {}
    for f in FIELDS:
        a = by_name[f]
        if a.get('type') != 'choice':
            raise ValueError(f'{f} was refused / unexpected type: {a.get("type")}')
        labels[f] = a['choice']
        plist = a.get('probabilities', [])
        if not isinstance(plist, list):
            raise ValueError('decisions probabilities must be list')
        probabilities[f] = complete_probabilities(f, probability_list_to_dict(plist, f))
        confidence[f] = a.get('confidence')
    validate_labels(labels)
    return labels, {'probabilities': probabilities, 'usage': raw.get('usage', {}), 'provider_confidence': confidence, 'probability_source': 'native_decision'}


def normalize_systemone(raw: dict) -> tuple[dict, dict]:
    answers = raw.get('answers')
    if not isinstance(answers, dict):
        raise ValueError('systemone.answers must be keyed object')
    labels, probabilities, confidence = {}, {}, {}
    if set(answers) != set(FIELDS):
        raise ValueError('systemone.answers missing or unexpected question IDs')
    for f in FIELDS:
        a = answers[f]
        if not isinstance(a, dict) or a.get('type', 'choice') != 'choice':
            raise ValueError(f'{f}: unexpected answer type')
        labels[f] = a['choice']
        p = a.get('probabilities', {})
        if isinstance(p, dict):
            probabilities[f] = complete_probabilities(f, p)
        elif isinstance(p, list):
            probabilities[f] = complete_probabilities(f, probability_list_to_dict(p, f))
        else:
            probabilities[f] = {}
        confidence[f] = a.get('confidence')
    validate_labels(labels)
    return labels, {'probabilities': probabilities, 'usage': raw.get('usage', {}), 'provider_confidence': confidence, 'probability_source': 'native_decision'}


def validate_labels(labels):
    for f in FIELDS:
        if labels.get(f) not in QUESTIONS[f]['criteria']:
            raise ValueError(f'invalid {f}: {labels.get(f)}')


def call_backend(name: str, record: str, args) -> tuple[dict, dict]:
    if name in ('luna-llm', 'openai-decisions'):
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY is not set')
        if name == 'luna-llm':
            use_lp = getattr(args, 'llm_logprobs', False)
            raw = post_json(getattr(args, 'openai_url', OPENAI_URL).rstrip('/') + '/responses',
                            llm_body(record, args.reasoning, getattr(args, 'llm_model', MODEL),
                                     getattr(args, 'llm_probabilities', False) and not use_lp, use_lp), key, args.timeout)
            if use_lp:
                return normalize_llm_logprobs(raw)
            return normalize_llm(raw, expect_probabilities=getattr(args, 'llm_probabilities', False))
        return normalize_decisions(post_json(getattr(args, 'openai_url', OPENAI_URL).rstrip('/') + '/decisions', {
            'model': getattr(args, 'decisions_model', MODEL), 'input': shared_input(record), 'questions': decision_questions(),
        }, key, args.timeout))
    if name == 'qwen3.8-27b':
        return normalize_qwen(post_json(args.qwen_url.rstrip('/') + '/v1/chat/completions', qwen_body(record), timeout=args.timeout))
    # Both are native decision-model endpoints, not text-generation/chat routes.
    body = {'state': shared_input(record), 'questions': systemone_questions()}
    if name == 'clef-flash':
        url = args.clef_url
        body['model'] = args.clef_model  # required in official SGLang example
    elif name in KEV_MODELS:
        # Ollama serves kev on its own /v1/systemone endpoint; the tag is passed as `model`.
        url = args.kev_url
        body['model'] = KEV_MODELS[name]
    elif name == 'd1-2b':
        url = args.d1_url
        # LiquidAI GGUF's llama.cpp model-card example omits `model`.
        # Some servers may check a loaded model ID; opt in if needed.
        if getattr(args, 'd1_model', ''):
            body['model'] = args.d1_model
    else:
        raise ValueError(name)
    return normalize_systemone(post_json(url.rstrip('/') + '/v1/systemone', body,
                                         timeout=args.timeout))


def percentile(nums: list[float], p: float) -> float | None:
    if not nums:
        return None
    v = sorted(nums)
    if len(v) == 1:
        return v[0]
    z = (len(v)-1)*p/100
    low, high = math.floor(z), math.ceil(z)
    return v[low]*(high-z) + v[high]*(z-low) if low != high else v[low]


def max_prob(probs: dict, chosen: str) -> float | None:
    if not isinstance(probs, dict) or chosen not in probs:
        return None
    try:
        p = float(probs[chosen])
        return p if math.isfinite(p) and 0 <= p <= 1 else None
    except (ValueError, TypeError):
        return None


def ece(rows: list[tuple[float, int]], bins: int = 10):
    if not rows:
        return None
    score = 0.0
    for i in range(bins):
        sub = [(p,y) for p,y in rows if ((i/bins <= p < (i+1)/bins) or (i==bins-1 and p==1.0))]
        if sub:
            acc = sum(y for _,y in sub)/len(sub)
            conf = sum(p for p,_ in sub)/len(sub)
            score += len(sub)/len(rows)*abs(acc-conf)
    return score


def evaluate(rows: list[dict], threshold: float, planned_cases: int | None = None):
    summaries = {}
    by_backend = defaultdict(list)
    for r in rows:
        by_backend[r['backend']].append(r)
    for b, rr in by_backend.items():
        ok = [r for r in rr if r['error'] is None]
        n_attempts = len(rr)
        n_planned = planned_cases if planned_cases is not None else n_attempts
        if n_planned <= 0 or n_attempts > n_planned:
            raise ValueError(f'{b}: planned_cases must cover attempts')
        counts = Counter()
        all_confs, brier_values = [], []
        group_cases = defaultdict(list)
        difficulty_cases = defaultdict(list)
        field_macro = {}
        accepted = 0
        wrong_accepted = 0
        cases_with_full_prob = 0
        probability_argmax_mismatches = 0
        for r in ok:
            group_cases[r.get('group', '')].append(r)
            difficulty_cases[r.get('difficulty', 'unspecified')].append(r)
            complete_case = True
            for f in FIELDS:
                pred, gold = r['predicted'][f], r['gold'][f]
                counts[f + '_correct'] += int(pred == gold)
                counts['field_total'] += 1
                counts['field_correct'] += int(pred == gold)
                p = r.get('probabilities', {}).get(f, {})
                if valid_distribution(f, p):
                    # The full probability vector remains valid for Brier, but
                    # the selected answer must be the maximum-probability label
                    # before it can be used as a top-label ECE or gating score.
                    brier_values.append(sum((float(p[k]) - float(k == gold))**2 for k in QUESTIONS[f]['criteria']))
                    if float(p[pred]) + 1e-9 >= max(float(v) for v in p.values()):
                        all_confs.append((max_prob(p, pred), int(pred == gold)))
                    else:
                        probability_argmax_mismatches += 1
                        complete_case = False
                else:
                    complete_case = False
            if r['predicted']['cause_status'] == 'confirmed' and r['gold']['cause_status'] != 'confirmed':
                counts['unsupported_confirmed'] += 1
            if complete_case:
                cases_with_full_prob += 1
                confvals = [max_prob(r['probabilities'][f], r['predicted'][f]) for f in FIELDS]
                if all(v is not None for v in confvals) and min(confvals) >= threshold:
                    accepted += 1
                    if not all(r['predicted'][f] == r['gold'][f] for f in FIELDS):
                        wrong_accepted += 1
        for f in FIELDS:
            # Each backend is evaluated against exactly the same fixed label vocabulary.
            # With tiny subsets, zero-support categories are excluded from the macro mean.
            present = set(r['gold'][f] for r in ok) | set(r['predicted'][f] for r in ok)
            scores = []
            for lab in present:
                tp = sum(r['gold'][f] == lab and r['predicted'][f] == lab for r in ok)
                fp = sum(r['gold'][f] != lab and r['predicted'][f] == lab for r in ok)
                fn = sum(r['gold'][f] == lab and r['predicted'][f] != lab for r in ok)
                scores.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0)
            field_macro[f] = sum(scores)/len(scores) if scores else None
        exact = sum(all(r['predicted'][f] == r['gold'][f] for f in FIELDS) for r in ok)
        group_exact = {
            g: {'n': len(rs), 'exact': round(sum(all(r['gold'][f]==r['predicted'][f] for f in FIELDS) for r in rs)/len(rs),4)}
            for g,rs in group_cases.items()
        }
        difficulty_exact = {
            d: {'n':len(rs), 'exact':round(sum(all(r['gold'][f]==r['predicted'][f] for f in FIELDS) for r in rs)/len(rs),4)}
            for d,rs in difficulty_cases.items()
        }
        times_ok = [r['elapsed_ms'] for r in ok]
        times_attempted = [r['elapsed_ms'] for r in rr if r.get('elapsed_ms') is not None]
        usage_rows = [r for r in ok if isinstance(r.get('usage'), dict) and
                      isinstance(r['usage'].get('input_tokens'), (int,float)) and
                      (b == 'openai-decisions' or isinstance(r['usage'].get('output_tokens'), (int,float)))]
        input_tokens = sum(r['usage'].get('input_tokens', 0) or 0 for r in usage_rows)
        output_tokens = sum(r['usage'].get('output_tokens', 0) or 0 for r in usage_rows)
        cost = None
        if b in PRICES and len(usage_rows) == len(ok) and len(ok) == n_planned:
            pi, po = PRICES[b]
            cost = (input_tokens*pi + output_tokens*po)/1_000_000
        summaries[b] = {
            'planned': n_planned, 'attempted': n_attempts, 'success': len(ok),
            'errors': n_attempts-len(ok), 'not_attempted': n_planned - n_attempts,
            'complete_run': len(ok) == n_planned,
            'exact_all_fields': round(exact/len(ok),4) if ok else None,
            'exact_all_fields_over_planned': round(exact/n_planned,4),
            'micro_accuracy': round(counts['field_correct']/counts['field_total'],4) if ok else None,
            'macro_f1_by_field': {f:round(v,4) if v is not None else None for f,v in field_macro.items()},
            'macro_f1_over_fields': (round(sum(valid_f1)/len(valid_f1),4) if (valid_f1 := [v for v in field_macro.values() if v is not None]) else None),
            'exact_by_group': group_exact,
            'exact_by_difficulty': difficulty_exact,
            'accuracy_by_field': {f:round(counts[f+'_correct']/len(ok),4) if ok else None for f in FIELDS},
            'unsupported_confirmed': counts['unsupported_confirmed'] if ok else None,
            'p50_ms': round(percentile(times_ok,50),1) if times_ok else None,
            'p95_ms': round(percentile(times_ok,95),1) if times_ok else None,
            'p95_attempted_ms': round(percentile(times_attempted,95),1) if times_attempted else None,
            'ece_10bins': round(ece(all_confs),4) if all_confs else None,
            'multiclass_brier_mean': round(sum(brier_values)/len(brier_values),4) if brier_values else None,
            'probability_complete_cases': cases_with_full_prob,
            'probability_argmax_mismatches': probability_argmax_mismatches,
            'threshold': threshold,
            'selective_coverage': round(accepted/n_planned,4) if cases_with_full_prob else None,
            'selective_risk': round(wrong_accepted/accepted,4) if accepted else None,
            'selective_wrong_count': wrong_accepted if cases_with_full_prob else None,
            'selective_accepted_count': accepted if cases_with_full_prob else None,
            'usage_available_cases': len(usage_rows),
            'input_tokens': input_tokens if usage_rows else None,
            'output_tokens': output_tokens if usage_rows else None,
            'approx_openai_usd_uncached': round(cost,6) if cost is not None else None,
        }
    return summaries

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backends', default='luna-llm,openai-decisions,d1-2b,clef-flash',
                    help='comma-separated: '+','.join(BACKENDS))
    ap.add_argument('--dataset', default=str(DATA_PATH), help='JSONL with id,text,gold')
    ap.add_argument('--limit', type=int, default=0, help='number of rows from the top; default 0 = whole dataset')
    ap.add_argument('--sample-size', type=int, default=0, help='deterministic difficulty-stratified sample; 0 = whole dataset')
    ap.add_argument('--seed', type=int, default=20261109, help='random seed for --sample-size')
    ap.add_argument('--timeout', type=float, default=90)
    ap.add_argument('--llm-model', default=MODEL,
                    help='model ID for Responses API (independent from Decisions API)')
    ap.add_argument('--decisions-model', default=MODEL,
                    help='model ID for Decisions API; check current official supported IDs')
    ap.add_argument('--openai-url', default=OPENAI_URL,
                    help='OpenAI-compatible base URL, default official /v1')
    ap.add_argument('--reasoning', choices=['none','low','medium','high'], default='none',
                    help='Responses API only; none for low-latency comparable classification')
    ap.add_argument('--llm-probabilities', action='store_true',
                    help='Ask the LLM to VERBALIZE probability vectors; unlike native Decisions probabilities. More tokens/cost.')
    ap.add_argument('--llm-logprobs', action='store_true',
                    help='luna-llm: official Responses output-token logprobs (include=message.output_text.logprobs, top_logprobs=20); candidate probabilities are first-token approximations.')
    ap.add_argument('--qwen-url', default='http://127.0.0.1:8082',
                    help='llama-server (chat completions) for Qwen3.8-27B, logprob-based probabilities')
    ap.add_argument('--kev-url', default='http://127.0.0.1:11434',
                    help='Ollama server hosting kev (POST /v1/systemone)')
    ap.add_argument('--clef-url', default='http://127.0.0.1:30000')
    ap.add_argument('--clef-model', default='Cloudflare/clef-flash')
    ap.add_argument('--d1-url', default='http://127.0.0.1:8080')
    ap.add_argument('--d1-model', default='',
                    help='optional model ID; omitted by default per LiquidAI llama.cpp example')
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--allow-partial', action='store_true', help='permit missing/errors; otherwise exit code 2')
    ap.add_argument('--dry-run', action='store_true', help='validate dataset & output request samples without contacting APIs')
    args=ap.parse_args()
    names = [s.strip().replace('luna-decisions','openai-decisions') for s in args.backends.split(',') if s.strip()]
    if len(names) != len(set(names)):
        ap.error('duplicate backends')
    if not names or any(n not in BACKENDS for n in names):
        ap.error('Invalid --backends: '+','.join(names))
    if not (0 < args.threshold <= 1):
        ap.error('--threshold must be in (0,1]')
    all_records = data(Path(args.dataset))
    if args.limit < 0 or args.sample_size < 0:
        ap.error('--limit and --sample-size must be nonnegative')
    if args.limit > 0 and args.sample_size > 0:
        ap.error('Use only one of --limit and --sample-size')
    if args.limit > len(all_records):
        ap.error('--limit exceeds number of cases')
    if args.sample_size > len(all_records):
        ap.error('--sample-size exceeds number of cases')
    records = (stratified_sample(all_records, args.sample_size, args.seed) if args.sample_size > 0
               else all_records[:args.limit] if args.limit > 0 else all_records)
    if not records:
        ap.error('No cases')
    if args.dry_run:
        print(json.dumps({
            'cases': len(records), 'dataset':str(Path(args.dataset)),
            'sample_size':args.sample_size, 'seed':args.seed if args.sample_size else None,
            'difficulty_distribution':dict(Counter(r.get('difficulty','unspecified') for r in records)),
            'backends': names, 'fields': {k:list(v['criteria']) for k,v in QUESTIONS.items()},
            'example_case': records[0], 'example_decisions_request': {
                'model':args.decisions_model, 'input': shared_input(records[0]['text']), 'questions': decision_questions()},
            'example_systemone_request': {
            **({'model': args.d1_model} if args.d1_model else {}),
            'state': shared_input(records[0]['text']), 'questions': systemone_questions()},
            'example_llm_request':llm_body(records[0]['text'], args.reasoning, args.llm_model, args.llm_probabilities)
        }, ensure_ascii=False, indent=2))
        return 0

    out = Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in names:
        print(f'== {name} ==', flush=True)
        for i,r in enumerate(records,1):
            item={'backend':name,'id':r['id'],'group':r.get('group',''),
                  'difficulty':r.get('difficulty',''), 'family_id':r.get('family_id',r['id']),
                  'variant':r.get('variant',''), 'gold_origin':r.get('gold_origin',''),
                  'text':r['text'],'gold':r['gold'],'predicted':None,
                  'probabilities':{},'provider_confidence':{},'probability_source':('logprobs_first_token_approx' if getattr(args, 'llm_logprobs', False) else 'verbalized_llm' if args.llm_probabilities else 'not_available') if name == 'luna-llm' else ('logprobs_first_token_approx' if name == 'qwen3.8-27b' else 'native_decision'),'elapsed_ms':None,'usage':{},'error':None}
            tic = time.perf_counter()
            try:
                labels, info = call_backend(name, r['text'], args)
                item.update({'predicted':labels, **info})
            except Exception as e:
                item['error'] = f'{type(e).__name__}: {e}'
            item['elapsed_ms'] = round((time.perf_counter()-tic)*1000,1)
            results.append(item)
            state='OK' if item['error'] is None else 'ERR'
            print(f' {i:>2}/{len(records)} {r["id"]} {state} {item["elapsed_ms"]:.0f}ms', flush=True)
            if item['error']:
                print('    ', item['error'][:250], flush=True)
                # Avoid repeated costs/timeouts on broken connection or missing key.
                if i == 1:
                    print('  Backend skipped after first failure', flush=True)
                    break
    meta={'generated_utc':datetime.now(timezone.utc).isoformat(),
          'llm_model': args.llm_model, 'decisions_model': args.decisions_model, 'synthetic_only':True,
          'notes':'model calls are not performed in --dry-run; these are sequential end-to-end wall times; cache discount not measured',
          'backends': names, 'threshold': args.threshold,
          'd1_model': args.d1_model or None, 'clef_model': args.clef_model,
          'dataset_filename': Path(args.dataset).name,
          'llm_probabilities': args.llm_probabilities,
          'dataset_sha256': __import__('hashlib').sha256(Path(args.dataset).read_bytes()).hexdigest(),
          'case_ids': [r['id'] for r in records], 'sample_size': args.sample_size,
          'seed': args.seed if args.sample_size else None,
          'difficulty_distribution':dict(Counter(r.get('difficulty','unspecified') for r in records))}
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as fp:
        for r in results:
            fp.write(json.dumps(r,ensure_ascii=False) + '\n')
    summary=evaluate(results,args.threshold,planned_cases=len(records))
    (out/'summary.json').write_text(json.dumps({'metadata':meta,'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    # Per-case table is easier to inspect in a spreadsheet than nested predictions JSONL.
    with (out/'case_comparison.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        columns=['backend','id','group','difficulty','family_id','variant','gold_origin','text','probability_source']
        for f in FIELDS:
            columns += ['gold_'+f,'pred_'+f,'correct_'+f,'chosen_probability_'+f,'provider_confidence_'+f]
        columns += ['all_correct','elapsed_ms','error']
        writer=csv.DictWriter(fp,fieldnames=columns)
        writer.writeheader()
        for r in results:
            line={'backend':r['backend'],'id':r['id'],'group':r.get('group',''),
                  'difficulty':r.get('difficulty',''), 'family_id':r.get('family_id',''),
                  'variant':r.get('variant',''), 'gold_origin':r.get('gold_origin',''),
                  'text':safe_csv_text(r['text']),'elapsed_ms':r['elapsed_ms'],'error':r['error'] or '',
                  'probability_source':r.get('probability_source','')}
            for f in FIELDS:
                line['gold_'+f]=r['gold'][f]
                pred=(r['predicted'] or {}).get(f)
                line['pred_'+f]=pred if pred is not None else ''
                line['correct_'+f]=int(pred==r['gold'][f]) if pred is not None else ''
                line['chosen_probability_'+f]=(max_prob(r.get('probabilities',{}).get(f,{}),pred) if pred else '')
                line['provider_confidence_'+f]=r.get('provider_confidence',{}).get(f,'')
            line['all_correct']=int(all((r['predicted'] or {}).get(f)==r['gold'][f] for f in FIELDS)) if r['predicted'] else ''
            writer.writerow(line)
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        fields=['backend','planned','attempted','success','errors','not_attempted','complete_run','exact_all_fields','exact_all_fields_over_planned','micro_accuracy','macro_f1_over_fields','unsupported_confirmed','p50_ms','p95_ms','p95_attempted_ms','ece_10bins','multiclass_brier_mean','probability_complete_cases','selective_coverage','selective_risk','selective_wrong_count','usage_available_cases','input_tokens','output_tokens','approx_openai_usd_uncached']
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for b,s in summary.items():
            writer.writerow({'backend':b, **{f:s.get(f) for f in fields if f!='backend'}})
    print('\n=== Summary ===')
    for b,s in summary.items():
        print(f'{b:17s} success={s["success"]:>3}  exact={str(s["exact_all_fields"]):>6}  '
              f'p50={str(s["p50_ms"]):>7}ms  p95={str(s["p95_ms"]):>7}ms  '\
              f'false-confirmed={s["unsupported_confirmed"]}')
    print(f'Written to {out.resolve()}')
    incomplete = [b for b in names if summary.get(b, {}).get('success', 0) != len(records)]
    if incomplete and not args.allow_partial:
        print(f'INCOMPLETE: {incomplete}; not a valid four-way benchmark. See summary.json', file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
