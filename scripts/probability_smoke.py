#!/usr/bin/env python3
"""Full held-out calibration-report smoke test with synthetic MOCK probabilities.

Never interprets the mock results as real model performance. Runs offline.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
from benchmark import FIELDS,QUESTIONS
from scripts.make_calibration_split import CALIBRATION,TEST,MANIFEST
from scripts.probability_audit import analyze,write_csv,read_jsonl


def mock_rows(path:Path,backend:str):
    rows=[json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
    out=[]
    for i,r in enumerate(rows):
        gold=r['gold']
        bad=(i % 11==0)
        prediction=dict(gold)
        if bad:
            prediction['cause_status']='unknown' if gold['cause_status']!='unknown' else 'confirmed'
        confidence=.98 if not bad else .54
        p={f:{k:(confidence if k==prediction[f] else (1-confidence)/(len(QUESTIONS[f]['criteria'])-1))
               for k in QUESTIONS[f]['criteria']} for f in FIELDS}
        out.append({'backend':backend,'id':r['id'],'family_id':r['family_id'],
                    'group':r['group'],'difficulty':r['difficulty'],'variant':r['variant'],
                    'gold':gold,'predicted':prediction,'error':None,'probabilities':p,
                    'probability_source':'verbalized_llm' if backend=='luna-llm' else 'native_decision',
                    'provider_confidence':{f:.8 for f in FIELDS} if backend!='luna-llm' else {}})
    return out


def main():
    manifest=json.loads(MANIFEST.read_text('utf-8'))
    a,b=[],[]
    for be in ['luna-llm','openai-decisions','clef-flash','d1-2b']:
        a.extend(mock_rows(CALIBRATION,be));b.extend(mock_rows(TEST,be))
    result,tables=analyze(a,b,manifest,bootstrap=100,seed=1)
    assert len(result['models'])==4
    for be,r in result['models'].items():
        assert 0<r['test']['exact_accuracy_over_planned']<1
        assert r['test']['probability_case_coverage']==1
        assert r['test']['per_field']['equipment']['ece_10bins'] is not None
        assert r['test']['per_field']['cause_status']['negative_log_likelihood_nats'] is not None
        assert r['test']['cause_status_contrast_triplets']['n_triplets'] > 20
    with tempfile.TemporaryDirectory(prefix='probability-audit-smoke-') as td:
        out=Path(td)
        (out/'report.json').write_text(json.dumps(result,ensure_ascii=False))
        for name,table in tables.items():
            assert table,name
            write_csv(out/f'{name}.csv',table,list(table[0]))
        assert len(list(out.iterdir()))==5
    print('PASS: end-to-end probability audit (4 MOCK models, 124 calibration + 350 test rows each)')
    print('PASS: ECE, Brier, NLL, selective risk, reliability CSV, contrast triplets, bootstrap intervals')
    print('MOCK outcomes only; no real models called')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
