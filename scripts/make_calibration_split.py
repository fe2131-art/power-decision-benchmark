#!/usr/bin/env python3
"""Reproducible family-disjoint calibration/test split for the hard synthetic corpus.

No model output or confidence is used for the split. This is a *synthetic* test,
not independent real-world prevalence. Unit of splitting is scenario family.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/hard_inspection_cases.jsonl'
CALIBRATION = ROOT / 'data/hard_calibration.jsonl'
TEST = ROOT / 'data/hard_test.jsonl'
MANIFEST = ROOT / 'data/hard_split_manifest.json'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize(rows: list[dict]) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode('utf-8')


def build(*, seed: int = 20261109, fraction: float = .30, source: Path = SOURCE):
    if not (0 < fraction < 1):
        raise ValueError('calibration fraction must be in (0,1)')
    rows = [json.loads(line) for line in source.read_text('utf-8').splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('dataset empty or duplicate IDs')
    families = defaultdict(list)
    for row in rows:
        fam = row.get('family_id')
        if not isinstance(fam, str) or not fam:
            raise ValueError(f'missing family_id in {row["id"]}')
        families[fam].append(row)
    difficulty_families = defaultdict(list)
    for fam, samples in families.items():
        difficulties = set(r['difficulty'] for r in samples)
        if len(difficulties) != 1:
            raise ValueError(f'mixed difficulty in family {fam}')
        difficulty_families[next(iter(difficulties))].append(fam)
    rng = random.Random(seed)
    dev_families = set()
    for difficulty, family_ids in sorted(difficulty_families.items()):
        ordered = sorted(family_ids)
        rng.shuffle(ordered)
        n_dev = max(1, min(len(ordered)-1, round(len(ordered)*fraction)))
        if len(ordered) < 2:
            raise ValueError(f'cannot hold out {difficulty} with a single family')
        dev_families.update(ordered[:n_dev])
    calibration = [r for r in rows if r['family_id'] in dev_families]
    test = [r for r in rows if r['family_id'] not in dev_families]
    if not calibration or not test:
        raise AssertionError('empty calibration/test split')
    a,b=serialize(calibration),serialize(test)
    assert len(calibration)+len(test)==len(rows)
    manifest = {
        'dataset_type': 'author_created_synthetic_not_field_validated',
        'source': source.name,
        'source_sha256': sha256(source.read_bytes()),
        'seed': seed, 'fraction_target': fraction,
        'strategy': 'disjoint scenario family IDs, stratified at family level by difficulty',
        'calibration_sha256': sha256(a), 'test_sha256': sha256(b),
        'n_total': len(rows), 'n_calibration': len(calibration), 'n_test': len(test),
        'n_families_total': len(families),
        'n_calibration_families': len(dev_families),
        'n_test_families': len(families)-len(dev_families),
        'calibration_family_ids': sorted(dev_families),
        'test_family_ids': sorted(set(families)-dev_families),
        'calibration_case_ids': [r['id'] for r in calibration],
        'test_case_ids': [r['id'] for r in test],
        'calibration_difficulty': dict(Counter(r['difficulty'] for r in calibration)),
        'test_difficulty': dict(Counter(r['difficulty'] for r in test)),
    }
    return a,b,manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--fraction', type=float, default=.30)
    args=ap.parse_args()
    a,b,metadata=build(seed=args.seed,fraction=args.fraction)
    CALIBRATION.write_bytes(a); TEST.write_bytes(b)
    MANIFEST.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(f'calibration={metadata["n_calibration"]} cases, test={metadata["n_test"]} cases; disjoint family IDs')
    print(f'calibration sha256={metadata["calibration_sha256"]}, test sha256={metadata["test_sha256"]}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
