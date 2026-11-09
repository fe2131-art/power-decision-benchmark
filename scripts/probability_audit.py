#!/usr/bin/env python3
"""Held-out, probability-aware evaluation of typed decisions and LLM verbalized probabilities.

Mirrors the *evaluation dimensions* of independent Jev benchmarks (accuracy,
Brier, NLL, ECE, selective risk/coverage, risk-coverage curve, CIs) rather than
copying their datasets or claiming comparability of their numbers.

Important distinction: model-native probability vectors and LLM-self-reported
probabilities are different measurement modes. Provider `confidence` is a
concentration statistic, not P(correct), and is NEVER used as a probability.

No API or model needed; consumes benchmark.py's predictions.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from benchmark import FIELDS, QUESTIONS
from scripts.make_calibration_split import MANIFEST

CLAMP_EPS = 1e-12
GRID = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]


def read_jsonl(path: Path) -> list[dict]:
    rows=[json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
    if not rows: raise ValueError(f'Empty prediction file: {path}')
    return rows


def index_predictions(rows: list[dict], expected_ids: list[str], split: str) -> dict[str, dict[str, dict]]:
    expected=set(expected_ids)
    by_backend=defaultdict(dict)
    for row in rows:
        b, id = row['backend'],row['id']
        if id in by_backend[b]: raise ValueError(f'Duplicate {split} prediction: {b}/{id}')
        if id not in expected: raise ValueError(f'Unknown {split} ID: {id}')
        by_backend[b][id]=row
    if not by_backend: raise ValueError(f'No {split} backends')
    for b, mapped in by_backend.items():
        if set(mapped)!=expected:
            raise ValueError(f'{b}: {split} predictions are incomplete: {len(mapped)}/{len(expected)}')
    return by_backend


def normalize_probabilities(probs: dict, field: str) -> tuple[dict | None, str]:
    """Return validated normalized vector. Reject bad sums, NaN and bad schema.

    Per-response model outputs may be rounded; a deviation <= .02 is tolerated
    and re-normalized, with the raw-sum flag surfaced in reports.
    """
    if not isinstance(probs,dict) or not probs:
        return None,'missing'
    expected=set(QUESTIONS[field]['criteria'])
    if set(probs)!=expected:
        return None,'wrong_labels'
    if any(type(p) not in (int,float) or not math.isfinite(p) or p<0 or p>1 for p in probs.values()):
        return None,'invalid_values'
    s=sum(probs.values())
    if not (0.98 <= s <= 1.02) or s<=0:
        return None,'invalid_sum'
    return {k:float(v)/s for k,v in probs.items()}, 'renormalized' if abs(s-1)>1e-9 else 'valid'


def ece_bins(pairs: list[tuple[float,int]], nbins:int=10) -> tuple[float | None,list[dict]]:
    if not pairs:return None,[]
    result=[]
    err=0.0
    for i in range(nbins):
        lo,hi=i/nbins,(i+1)/nbins
        section=[(p,ok) for p,ok in pairs if lo <= p < hi or (i==nbins-1 and p==1)]
        if not section:continue
        prob=mean(p for p,_ in section);correct=mean(ok for _,ok in section)
        result.append({'bin':i,'bin_lower':lo,'bin_upper':hi,'n':len(section),
                       'mean_predicted_probability':round(prob,8),'observed_correct_rate':round(correct,8),
                       'abs_gap':round(abs(prob-correct),8)})
        err+=len(section)/len(pairs)*abs(prob-correct)
    return err,result


def score_prediction(row:dict) -> dict:
    """Compute an auditable record. Failed calls are never accepted."""
    gold=row['gold']
    predicted=row.get('predicted')
    success=not row.get('error') and isinstance(predicted,dict) and all(f in predicted for f in FIELDS)
    exact=success and all(predicted[f]==gold[f] for f in FIELDS)
    per_field={}
    for f in FIELDS:
        pred=predicted.get(f) if success else None
        candidate=row.get('probabilities',{}).get(f,{})
        vector,status=normalize_probabilities(candidate,f)
        correct=bool(success and pred==gold[f])
        mismatch=bool(vector and pred is not None and vector[pred]+1e-9 < max(vector.values())) if pred in (vector or {}) else False
        p_pred=vector.get(pred) if vector is not None and pred in vector else None
        p_gold=vector.get(gold[f]) if vector is not None else None
        per_field[f]={
            'correct':correct,'pred':pred,'gold':gold[f],
            'probability_status':status,'vector':vector,'p_selected':p_pred,'p_gold':p_gold,
            'choice_probability_mismatch':mismatch,
            'provider_confidence':row.get('provider_confidence',{}).get(f),
        }
    valid_all=success and all(per_field[f]['vector'] is not None and not per_field[f]['choice_probability_mismatch'] for f in FIELDS)
    case_p=min(per_field[f]['p_selected'] for f in FIELDS) if valid_all else None
    return {
        'id':row['id'],'backend':row['backend'],'family_id':row.get('family_id') or row['id'],
        'group':row.get('group',''),'difficulty':row.get('difficulty',''),
        'variant':row.get('variant',''),
        'probability_source':row.get('probability_source','unspecified'),
        'success':success,'exact':bool(exact), 'case_p':case_p,
        'fields':per_field,'invalid_or_missing_probability':not bool(valid_all),
    }


def coverage_risk(cases:list[dict],threshold:float) -> dict:
    selected=[case for case in cases if case['case_p'] is not None and case['case_p']>=threshold]
    n=len(selected);fail=sum(not case['exact'] for case in selected)
    return {'accepted':n,'accepted_errors':fail,'coverage':n/len(cases) if cases else None,
            'empirical_risk':fail/n if n else None,'accepted_accuracy':(n-fail)/n if n else None}


def choose_threshold(dev:list[dict],*,risk_budget:float=.05,min_accepted:int=10) -> dict:
    """Choose threshold using calibration data ONLY; select max accepted cases.

    Threshold=1+epsilon is explicit abstain-all when no feasible threshold.
    Empirical dev constraint is NOT a deployment safety guarantee.
    """
    unique=sorted(set([0.0]+[case['case_p'] for case in dev if case['case_p'] is not None]))
    feasible=[]
    for t in unique:
        detail=coverage_risk(dev,t)
        if detail['accepted']>=min_accepted and detail['empirical_risk']<=risk_budget:
            feasible.append((detail['accepted'],t,detail))
    if not feasible:
        return {'threshold':None,'status':'no_feasible_threshold','dev_accepted':0,
                'dev_coverage':0.0,'dev_empirical_risk':None,'risk_budget':risk_budget,
                'min_accepted':min_accepted}
    # Most accepted, then highest threshold if acceptance ties.
    n,t,detail=max(feasible,key=lambda x:(x[0],x[1]))
    return {'threshold':t,'status':'selected_on_calibration','dev_accepted':n,
            'dev_coverage':detail['coverage'],'dev_empirical_risk':detail['empirical_risk'],
            'risk_budget':risk_budget,'min_accepted':min_accepted}


def cluster_ci(cases:list[dict], *, reps:int=1000,seed:int=20261109,score='exact') -> list[float] | None:
    groups=defaultdict(list)
    for item in cases: groups[item['family_id']].append(float(bool(item[score])))
    if len(groups)<2:return None
    units=list(groups.values())
    rng=random.Random(seed)
    estimates=[]
    for _ in range(reps):
        pick=[units[rng.randrange(len(units))] for _ in units]
        merged=[val for cluster in pick for val in cluster]
        estimates.append(sum(merged)/len(merged))
    estimates.sort()
    return [round(estimates[int(.025*(reps-1))],6),round(estimates[int(.975*(reps-1))],6)]


def auc_risk_coverage(cases: list[dict]):
    """AURC among cases with 3 valid probability vectors, not missing rows.

    This conditional measure MUST be accompanied by probability coverage.
    """
    filtered=sorted((c for c in cases if c['case_p'] is not None),key=lambda x:(-x['case_p'],x['id']))
    if not filtered:return None
    wrong=0; total=0
    for i,c in enumerate(filtered,1):
        wrong+=int(not c['exact'])
        total+=wrong/i
    return total/len(filtered)


def calculate_metrics(cases:list[dict],*,bootstrap:int=1000,seed:int=20261109) -> tuple[dict,list[dict]]:
    if not cases:raise ValueError('empty evaluation cases')
    total=len(cases)
    success=sum(c['success'] for c in cases)
    exact=sum(c['exact'] for c in cases)
    per_field={}; reliability=[]
    for f in FIELDS:
        records=[c['fields'][f] for c in cases]
        valid=[r for r in records if r['vector'] is not None]
        valid_aligned=[r for r in valid if not r['choice_probability_mismatch'] and r['p_selected'] is not None]
        brier=[];nll=[];top=[]
        for r in valid:
            v=r['vector'];g=r['gold']
            brier.append(sum((p-float(k==g))**2 for k,p in v.items()))
            nll.append(-math.log(max(v[g],CLAMP_EPS)))
        for r in valid_aligned:
            top.append((r['p_selected'],int(r['correct'])))
        error,bins=ece_bins(top)
        for stat in bins:
            reliability.append({'field':f,**stat})
        per_field[f]={
            'accuracy_over_planned':sum(r['correct'] for r in records)/total,
            'n_valid_distribution':len(valid),'n_probability_missing_or_invalid':total-len(valid),
            'n_pred_probability_argmax_mismatch':len(valid)-len(valid_aligned),
            'n_valid_ece':len(top),
            'ece_10bins':error,'multiclass_brier':mean(brier) if brier else None,
            'negative_log_likelihood_nats':mean(nll) if nll else None,
            'n_gold_probability_zero':sum(r['p_gold']==0 for r in valid),
            'provider_confidence_mean':(mean(v['provider_confidence'] for v in records
                                           if type(v['provider_confidence']) in (float,int) and math.isfinite(v['provider_confidence'])
                                           and 0<=v['provider_confidence']<=1)
                                        if any(type(v['provider_confidence']) in (float,int) and math.isfinite(v['provider_confidence'])
                                               and 0<=v['provider_confidence']<=1 for v in records) else None),
        }
    n_complete=sum(c['case_p'] is not None for c in cases)
    # Minimal contrast sets: only cause-status evidence changes, while equipment,
    # observation, and template style are fixed. These triplets expose shortcuts
    # that inflate accuracy when labels follow superficial words.
    triples=defaultdict(list)
    for c in cases:
        if c['variant'].startswith('template_'):
            triples[(c['family_id'],c['variant'])].append(c)
    eligible_triples=[]
    for (fam,style),items in triples.items():
        if (len(items)==3 and
            {v['fields']['cause_status']['gold'] for v in items} == {'confirmed','suspected','unknown'}):
            eligible_triples.append(items)
    perfect_triplets=sum(all(item['fields']['cause_status']['correct'] for item in triple) for triple in eligible_triples)
    by_group={}
    for g in sorted(set(c['group'] for c in cases)):
        part=[c for c in cases if c['group']==g]
        by_group[g]={'n':len(part),'exact':sum(c['exact'] for c in part)/len(part)}
    by_difficulty={}
    for g in sorted(set(c['difficulty'] for c in cases)):
        part=[c for c in cases if c['difficulty']==g]
        by_difficulty[g]={'n':len(part),'exact':sum(c['exact'] for c in part)/len(part)}
    out={
        'planned':total,'successful':success,'failed_or_not_completed':total-success,
        'exact_accuracy_over_planned':exact/total,'exact_correct':exact,
        'exact_accuracy_cluster_bootstrap_95_ci':cluster_ci(cases,reps=bootstrap,seed=seed),
        'probability_complete_cases':n_complete,
        'probability_case_coverage':n_complete/total,
        'aurc_conditional_on_probability_complete_cases':auc_risk_coverage(cases),
        'source_counts':dict(Counter(c['probability_source'] for c in cases)),
        'per_field':per_field,
        'cause_status_contrast_triplets':{'n_triplets':len(eligible_triples),
                 'all_three_correct':perfect_triplets,
                 'triplet_accuracy':perfect_triplets/len(eligible_triples) if eligible_triples else None},
        'by_group':by_group,'by_difficulty':by_difficulty,
    }
    return out,reliability


def write_csv(path:Path,rows:list[dict],fields:list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(dev_records:list[dict],test_records:list[dict],manifest:dict,*,risk_budget:float=.05,
            min_accepted:int=10,bootstrap:int=1000,seed:int=20261109) -> tuple[dict,dict[str,list[dict]]]:
    dev_index=index_predictions(dev_records,manifest['calibration_case_ids'],'calibration')
    test_index=index_predictions(test_records,manifest['test_case_ids'],'test')
    if set(dev_index)!=set(test_index):
        raise ValueError('calibration and test have different backends')
    if set(manifest['calibration_family_ids']) & set(manifest['test_family_ids']):
        raise ValueError('family leakage in split manifest')
    if set(manifest['calibration_case_ids']) & set(manifest['test_case_ids']):
        raise ValueError('case leakage in split manifest')
    tables={'reliability':[],'risk_curve':[],'case_probability':[],'group_accuracy':[]}
    output={'meta':{'method':'synthetic family-disjoint held-out evaluation',
                    'risk_budget_is_empirical_not_a_safety_guarantee':True,
                    'llm_probabilities_are_verbalized_not_native':True,
                    'seed':seed,'bootstrap':bootstrap,
                    'n_calibration':len(manifest['calibration_case_ids']),
                    'n_test':len(manifest['test_case_ids']),
                    'family_disjoint':True}, 'models':{}}
    for backend in sorted(dev_index):
        dev=[score_prediction(dev_index[backend][id]) for id in manifest['calibration_case_ids']]
        test=[score_prediction(test_index[backend][id]) for id in manifest['test_case_ids']]
        if any(c['family_id'] not in set(manifest['calibration_family_ids']) for c in dev):
            raise ValueError(f'{backend}: calibration family mismatch')
        if any(c['family_id'] not in set(manifest['test_family_ids']) for c in test):
            raise ValueError(f'{backend}: test family mismatch')
        decision=choose_threshold(dev,risk_budget=risk_budget,min_accepted=min_accepted)
        dmetrics,_=calculate_metrics(dev,bootstrap=bootstrap,seed=seed)
        tmetrics,bins=calculate_metrics(test,bootstrap=bootstrap,seed=seed)
        threshold=decision['threshold']
        selected=coverage_risk(test,threshold) if threshold is not None else {
            'accepted':0,'accepted_errors':0,'coverage':0.0,'empirical_risk':None,'accepted_accuracy':None}
        tmetrics['selected_threshold_from_calibration']=decision
        tmetrics['test_selective_policy']=selected
        output['models'][backend]={'calibration':dmetrics,'test':tmetrics}
        for row in bins:
            tables['reliability'].append({'backend':backend,'split':'test',**row})
        for t in GRID:
            metrics=coverage_risk(test,t)
            tables['risk_curve'].append({'backend':backend,'threshold':t,**metrics,
                                         'n_planned':len(test)})
        if threshold is not None and threshold not in GRID:
            metrics=coverage_risk(test,threshold)
            tables['risk_curve'].append({'backend':backend,'threshold':threshold,**metrics,
                                         'n_planned':len(test)})
        for c in test:
            row={'backend':backend,'id':c['id'],'family_id':c['family_id'],
                 'group':c['group'],'difficulty':c['difficulty'],'success':c['success'],
                 'exact':c['exact'],'case_chosen_min_probability':c['case_p'],
                 'probability_source':c['probability_source'],
                 'accepted_by_dev_threshold':bool(threshold is not None and c['case_p'] is not None and c['case_p']>=threshold)}
            for f in FIELDS:
                v=c['fields'][f]
                row[f'{f}_gold']=v['gold'];row[f'{f}_prediction']=v['pred']
                row[f'{f}_correct']=v['correct'];row[f'{f}_selected_probability']=v['p_selected']
                row[f'{f}_provider_confidence']=v['provider_confidence']
                row[f'{f}_probability_status']=v['probability_status']
                row[f'{f}_argmax_mismatch']=v['choice_probability_mismatch']
            tables['case_probability'].append(row)
        for group, values in tmetrics['by_group'].items():
            tables['group_accuracy'].append({'backend':backend,'group':group,**values})
    return output,tables


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibration',type=Path,required=True,help='calibration predictions.jsonl')
    ap.add_argument('--test',type=Path,required=True,help='test predictions.jsonl')
    ap.add_argument('--manifest',type=Path,default=MANIFEST)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--risk-budget',type=float,default=.05)
    ap.add_argument('--min-accepted-dev',type=int,default=10)
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261109)
    args=ap.parse_args()
    if not (0 <= args.risk_budget < 1):ap.error('--risk-budget must be in [0,1)')
    if args.bootstrap<100 or args.min_accepted_dev<1:ap.error('bootstrap>=100; min-accepted-dev>=1')
    manifest=json.loads(args.manifest.read_text('utf-8'))
    result,tables=analyze(read_jsonl(args.calibration),read_jsonl(args.test),manifest,
                          risk_budget=args.risk_budget,min_accepted=args.min_accepted_dev,
                          bootstrap=args.bootstrap,seed=args.seed)
    out=args.outdir;out.mkdir(exist_ok=True,parents=True)
    (out/'probability_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    for name,table in tables.items():
        if table:
            write_csv(out/f'{name}.csv',table,list(table[0]))
    for b,d in result['models'].items():
        ts=d['test']; gate=ts['test_selective_policy']
        print(f'{b}: exact={ts["exact_accuracy_over_planned"]:.4f}, '
              f'prob_coverage={ts["probability_case_coverage"]:.3f}, '
              f'gate_cov={gate["coverage"]:.3f}, gate_risk={gate["empirical_risk"]}')
    print(f'Wrote {out.resolve()}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
