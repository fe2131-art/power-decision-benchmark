"""Build publishable summary tables from raw predictions and the probability audit report.

Reads (kept local, not published):
  results/raw/calibration/predictions.jsonl
  results/raw/test/predictions.jsonl
  results/probability_audit/probability_report.json   (from scripts.probability_audit)
Writes (published):
  results/SUMMARY.md          human-readable tables
  results/summary_table.csv   one row per method, test split

Stdlib only. Latency and token columns are measured per request by benchmark.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = ('equipment', 'observation', 'cause_status')
ORDER = ['luna-llm', 'openai-decisions', 'qwen3.8-27b', 'kev-9b', 'kev-9b-bf16', 'kev-0.8b', 'kev-0.8b-bf16',
         'clef-flash', 'd1-2b']
LABEL = {
    'luna-llm': 'GPT-6 Luna (Responses, logprobs)',
    'openai-decisions': 'GPT-6 Luna (Decisions API)',
    'qwen3.8-27b': 'Qwen3.8-27B (UD-Q4_K_M, llama.cpp, logprobs)',
    'clef-flash': 'Clef-Flash (Q8_0, llama.cpp)',
    'd1-2b': 'd1-2B (Q8_0, llama.cpp)',
    'kev-9b': 'kev 9B (Q8_0, Ollama)',
    'kev-9b-bf16': 'kev 9B (BF16, Ollama)',
    'kev-0.8b': 'kev 0.8B (Q8_0, Ollama)',
    'kev-0.8b-bf16': 'kev 0.8B (BF16, Ollama)',
}
DEPLOY = {
    'luna-llm': 'OpenAI API',
    'openai-decisions': 'OpenAI API',
    'qwen3.8-27b': 'local V100 32GB',
    'clef-flash': 'local V100 32GB',
    'd1-2b': 'local V100 32GB',
    'kev-9b': 'local V100 32GB (Ollama)',
    'kev-9b-bf16': 'local V100 32GB (Ollama)',
    'kev-0.8b': 'local V100 32GB (Ollama)',
    'kev-0.8b-bf16': 'local V100 32GB (Ollama)',
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def nearest_rank(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = max(1, round(p / 100 * len(xs)))
    return xs[min(k, len(xs)) - 1]


def split_stats(rows: list[dict]) -> dict:
    """Per-backend counts, exact accuracy, latency and token usage on one split."""
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r['backend']].append(r)
    for name, rs in by.items():
        planned = len(rs)
        ok = [r for r in rs if not r.get('error') and r.get('predicted')]
        correct = [r for r in ok if all(r['predicted'][f] == r['gold'][f] for f in FIELDS)]
        lat = [float(r['elapsed_ms']) for r in ok if r.get('elapsed_ms') is not None]
        usage_in = [r['usage'].get('input_tokens') for r in ok if isinstance(r.get('usage', {}).get('input_tokens'), (int, float))]
        usage_out = [r['usage'].get('output_tokens') for r in ok if isinstance(r.get('usage', {}).get('output_tokens'), (int, float))]
        out[name] = {
            'planned': planned,
            'success': len(ok),
            'errors': planned - len(ok),
            'exact_over_planned': len(correct) / planned if planned else None,
            'latency_p50_ms': nearest_rank(lat, 50),
            'latency_p95_ms': nearest_rank(lat, 95),
            'latency_mean_ms': statistics.fmean(lat) if lat else None,
            'input_tokens_mean': statistics.fmean(usage_in) if usage_in else None,
            'output_tokens_mean': statistics.fmean(usage_out) if usage_out else None,
        }
    return out


def fmt(x, nd=3):
    if x is None:
        return '—'
    if isinstance(x, float):
        return f'{x:.{nd}f}'
    return str(x)


def fmt_ms(x):
    return '—' if x is None else f'{x:,.0f}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, default=Path('results/raw'))
    ap.add_argument('--audit', type=Path, default=Path('results/probability_audit/probability_report.json'))
    ap.add_argument('--outdir', type=Path, default=Path('results'))
    args = ap.parse_args()

    cal = split_stats(load(args.raw / 'calibration' / 'predictions.jsonl'))
    test = split_stats(load(args.raw / 'test' / 'predictions.jsonl'))
    audit = json.loads(args.audit.read_text(encoding='utf-8'))['models']
    names = [n for n in ORDER if n in test]

    csv_rows = []
    lines = []
    lines.append('# Results summary (test split, 350 cases)\n')
    lines.append('Comparison on 474 synthetic power-equipment inspection records (calibration 124, test 350; '
                 'split by scenario family). Raw per-case predictions (`results/raw/`) are not published. '
                 'Values in parentheses for Exact are 95% cluster-bootstrap confidence intervals over scenario families. '
                 'Probabilities are first-token approximations of output-token logprobs (luna-llm, Qwen) or native '
                 'probabilities of the decision models (all others). Qwen was run with temperature=0; the other methods '
                 'use each service\'s default sampling settings.\n')

    lines.append('## Accuracy and probability calibration (test split, 350 cases)\n')
    lines.append('| Method | Deployment | Exact (95% CI) | ECE equipment | ECE observation | ECE cause | Brier equipment | Brier observation | Brier cause | Coverage @ 5% risk budget | Error rate of accepted | Cause-contrast triplets |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in names:
        a = audit.get(n, {}).get('test', {})
        ci = a.get('exact_accuracy_cluster_bootstrap_95_ci')
        ci_s = f'{ci[0]:.3f}–{ci[1]:.3f}' if isinstance(ci, list) and len(ci) == 2 else '—'
        pf = a.get('per_field', {})
        pol = a.get('test_selective_policy', {}) or {}
        tr = a.get('cause_status_contrast_triplets', {}) or {}
        exact = a.get('exact_accuracy_over_planned')
        row = [
            LABEL[n], DEPLOY[n],
            f'{fmt(exact)} ({ci_s})',
            fmt(pf.get('equipment', {}).get('ece_10bins')), fmt(pf.get('observation', {}).get('ece_10bins')), fmt(pf.get('cause_status', {}).get('ece_10bins')),
            fmt(pf.get('equipment', {}).get('multiclass_brier')), fmt(pf.get('observation', {}).get('multiclass_brier')), fmt(pf.get('cause_status', {}).get('multiclass_brier')),
            fmt(pol.get('coverage')), fmt(pol.get('empirical_risk')),
            fmt(tr.get('triplet_accuracy')),
        ]
        lines.append('| ' + ' | '.join(row) + ' |')
        csv_rows.append({
            'method': LABEL[n], 'deployment': DEPLOY[n], 'exact_accuracy': exact,
            'exact_ci95_low': ci[0] if isinstance(ci, list) else None, 'exact_ci95_high': ci[1] if isinstance(ci, list) else None,
            'ece_equipment': pf.get('equipment', {}).get('ece_10bins'), 'ece_observation': pf.get('observation', {}).get('ece_10bins'),
            'ece_cause_status': pf.get('cause_status', {}).get('ece_10bins'),
            'brier_equipment': pf.get('equipment', {}).get('multiclass_brier'), 'brier_observation': pf.get('observation', {}).get('multiclass_brier'),
            'brier_cause_status': pf.get('cause_status', {}).get('multiclass_brier'),
            'selective_coverage_at_5pct_budget': pol.get('coverage'), 'selective_risk': pol.get('empirical_risk'),
            'cause_triplet_accuracy': tr.get('triplet_accuracy'),
        })
    lines.append('')
    lines.append('ECE and Brier are reported only for methods that output probabilities. Coverage and error rate use the '
                 'threshold selected on the 124 calibration cases and applied unchanged to the test split. The 5% budget is an '
                 'empirical evaluation condition, not a safety guarantee.\n')

    lines.append('## Latency and tokens per case (test split)\n')
    lines.append('| Method | Deployment | Cases | Failed | p50 (ms) | p95 (ms) | Mean (ms) | Mean input tokens | Mean output tokens |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for n in names:
        s = test[n]
        lines.append('| ' + ' | '.join([
            LABEL[n], DEPLOY[n], str(s['planned']), str(s['errors']),
            fmt_ms(s['latency_p50_ms']), fmt_ms(s['latency_p95_ms']), fmt_ms(s['latency_mean_ms']),
            fmt(s['input_tokens_mean'], 0) if s['input_tokens_mean'] is not None else '—',
            fmt(s['output_tokens_mean'], 1) if s['output_tokens_mean'] is not None else '—',
        ]) + ' |')
        csv_rows[names.index(n)].update({
            'latency_p50_ms': s['latency_p50_ms'], 'latency_p95_ms': s['latency_p95_ms'], 'latency_mean_ms': s['latency_mean_ms'],
            'input_tokens_mean': s['input_tokens_mean'], 'output_tokens_mean': s['output_tokens_mean'], 'errors': s['errors'],
        })
    lines.append('')
    lines.append('Latency is wall-clock time measured per request. Each method was run on its own, one after another. '
                 'OpenAI API timings include network latency; local timings were measured on a single V100 32GB. '
                 'These numbers are not a like-for-like comparison of raw model speed.\n')

    lines.append('## Calibration split (124 cases, used for threshold selection)\n')
    lines.append('| Method | Exact | Cases | Failed |')
    lines.append('|---|---|---|---|')
    for n in names:
        s = cal.get(n)
        if s:
            lines.append(f"| {LABEL[n]} | {fmt(s['exact_over_planned'])} | {s['planned']} | {s['errors']} |")
    lines.append('')

    lines.append('## Reproduction\n')
    lines.append('```bash\n'
                 'python -m scripts.probability_audit --calibration results/raw/calibration/predictions.jsonl '
                 '--test results/raw/test/predictions.jsonl --manifest data/hard_split_manifest.json --outdir results/raw/probability_audit\n'
                 'python -m scripts.make_summary\n```\n')

    (args.outdir / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    with (args.outdir / 'summary_table.csv').open('w', encoding='utf-8', newline='') as fh:
        cols = list(csv_rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f'wrote {args.outdir / "SUMMARY.md"} and {args.outdir / "summary_table.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
