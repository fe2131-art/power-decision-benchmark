#!/usr/bin/env python3
"""Cluster-aware paired comparison of completed runs from one predictions.jsonl.

Bootstrap resamples scenario families, not near-duplicate examples. This does not
make a synthetic collection representative of real maintenance operations.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def bootstrap_ci(values: list[tuple[int,float]], n_boot: int, seed: int) -> tuple[float,float]:
    """Values are cluster weights and average per-cluster signed score deltas."""
    rng=random.Random(seed)
    n=len(values)
    if n<2:
        return (float('nan'),float('nan'))
    diffs=[]
    for _ in range(n_boot):
        picks=[values[rng.randrange(n)] for _ in range(n)]
        weight=sum(w for w,_ in picks)
        diffs.append(sum(w*d for w,d in picks)/weight)
    diffs.sort()
    return (diffs[int(0.025*(n_boot-1))], diffs[int(0.975*(n_boot-1))])


def analyze(path: Path, *, n_boot: int=2000, seed: int=20261109):
    records=[json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('predictions file is empty')
    by_model=defaultdict(dict)
    all_ids={}
    for r in records:
        b,id=r['backend'],r['id']
        if id in by_model[b]:
            raise ValueError(f'duplicate result {b}/{id}')
        by_model[b][id]=r
        if id in all_ids and all_ids[id]['gold'] != r['gold']:
            raise ValueError(f'gold mismatch for case {id}')
        all_ids[id]=r
    names=sorted(by_model)
    output=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            aa,bb=by_model[a],by_model[b]
            if set(aa)!=set(bb):
                raise ValueError(f'incomplete or mismatched IDs for {a} vs {b}; rerun before comparison')
            pairs=[]
            clusters=defaultdict(list)
            for id in sorted(aa):
                x,y=aa[id],bb[id]
                if x.get('error') or y.get('error') or not x.get('predicted') or not y.get('predicted'):
                    raise ValueError(f'incomplete prediction in pair {a} vs {b} case {id}')
                if x.get('family_id') != y.get('family_id'):
                    raise ValueError(f'inconsistent family_id for {id}')
                correct_a=int(all(x['predicted'][f]==x['gold'][f] for f in ('equipment','observation','cause_status')))
                correct_b=int(all(y['predicted'][f]==y['gold'][f] for f in ('equipment','observation','cause_status')))
                pairs.append((correct_a,correct_b))
                clusters[x.get('family_id') or id].append(correct_b-correct_a)
            grouped=[(len(ds),sum(ds)/len(ds)) for ds in clusters.values()]
            lo,hi=bootstrap_ci(grouped,n_boot,seed)
            delta=sum(y-x for x,y in pairs)/len(pairs)
            output.append({
                'backend_a':a,'backend_b':b,'n_cases':len(pairs),
                'n_families':len(clusters),'accuracy_a':round(sum(x for x,_ in pairs)/len(pairs),6),
                'accuracy_b':round(sum(y for _,y in pairs)/len(pairs),6),
                'delta_b_minus_a':round(delta,6),'cluster_bootstrap_95_low':round(lo,6),
                'cluster_bootstrap_95_high':round(hi,6),'resamples':n_boot,'seed':seed,
                'note':'Descriptive for this synthetic benchmark; scenario families are not statistically independent real-world sites.',
            })
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions',type=Path)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--resamples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261109)
    args=p.parse_args()
    if args.resamples < 100: p.error('--resamples must be at least 100')
    result=analyze(args.predictions,n_boot=args.resamples,seed=args.seed)
    dest=args.out or args.predictions.with_name('paired_comparison.csv')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if result:
        with dest.open('w',encoding='utf-8-sig',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(result[0]))
            writer.writeheader();writer.writerows(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('written:',dest)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
