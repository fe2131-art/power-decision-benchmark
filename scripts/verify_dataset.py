"""Verify that the bundled JSONL exactly matches a fresh deterministic generation."""
from pathlib import Path
import subprocess
import hashlib
import json
import tempfile
import shutil
from scripts.make_demo_dataset import render_demo, OUTPUT as DEMO_DATASET, DEMO_IDS
import sys
from scripts.make_hard_dataset import render as hard_render, OUT as HARD_DATASET, AUDIT as HARD_AUDIT, STATS as HARD_STATS
from scripts.make_calibration_split import build as split_build, CALIBRATION as CALIBRATION_DATASET, TEST as TEST_DATASET, MANIFEST as SPLIT_MANIFEST

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts' / 'make_legacy_dataset.py'
BUNDLED = ROOT / 'data' / 'legacy' / 'synthetic_inspection_cases.jsonl'

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    with tempfile.TemporaryDirectory(prefix='decision-dataset-') as name:
        temp=Path(name)
        shutil.copy2(SOURCE,temp/'make_legacy_dataset.py')
        subprocess.run([sys.executable,str(temp/'make_legacy_dataset.py'),str(temp/BUNDLED.name)], check=True, capture_output=True, text=True)
        generated=temp/BUNDLED.name
        actual, expected=sha256(BUNDLED),sha256(generated)
        records=[json.loads(x) for x in BUNDLED.read_text('utf-8').splitlines() if x.strip()]
        print('dataset cases:',len(records))
        print('bundled  SHA256:',actual)
        print('generated SHA256:',expected)
        if actual != expected:
            print('FAILED: bundled JSONL differs from source generator',file=sys.stderr)
            return 1
        if len(records) != 48:
            print('FAILED: expected 48 records',file=sys.stderr)
            return 1
        print('PASS: 48 records and byte-for-byte reproducible')
        actual_demo = DEMO_DATASET.read_text('utf-8')
        expected_demo = render_demo()
        if actual_demo != expected_demo:
            print('FAILED: curated 10-case JSONL differs from deterministic selection',file=sys.stderr)
            return 1
        demos = [json.loads(x) for x in actual_demo.splitlines() if x.strip()]
        if len(demos) != 10 or [x['id'] for x in demos] != list(DEMO_IDS):
            print('FAILED: wrong curated 10 case ordering',file=sys.stderr)
            return 1
        print('PASS: curated 10 records, exact source gold labels and deterministic order')
        expected_hard, expected_audit, expected_stats = hard_render()
        if HARD_DATASET.read_text('utf-8') != expected_hard:
            print('FAILED: hard dataset changed from its generator',file=sys.stderr)
            return 1
        if HARD_AUDIT.read_text('utf-8-sig') != expected_audit:
            print('FAILED: audit CSV differs from its generator',file=sys.stderr)
            return 1
        if HARD_STATS.read_text('utf-8') != expected_stats:
            print('FAILED: hard dataset stats differ from generator',file=sys.stderr)
            return 1
        hard_records = [json.loads(x) for x in expected_hard.splitlines()]
        if len(hard_records) != 474 or len({r['id'] for r in hard_records}) != 474 or len({r['text'] for r in hard_records}) != 474:
            print('FAILED: hard dataset size/unique constraints',file=sys.stderr)
            return 1
        import benchmark
        benchmark.data(HARD_DATASET)
        supported = {f: set(spec['criteria']) for f,spec in benchmark.QUESTIONS.items()}
        for f in supported:
            if {r['gold'][f] for r in hard_records} != supported[f]:
                print(f'FAILED: hard dataset missing label of {f}',file=sys.stderr)
                return 1
        if len({r['family_id'] for r in hard_records}) < 30 or len({r['group'] for r in hard_records}) < 8:
            print('FAILED: insufficient diversity',file=sys.stderr)
            return 1
        print('PASS: 474 hard synthetic cases, exact source/audit/stats match, unique IDs/texts, full label coverage')
        calib_bytes,test_bytes,split_meta=split_build()
        if (CALIBRATION_DATASET.read_bytes()!=calib_bytes or TEST_DATASET.read_bytes()!=test_bytes
                or json.loads(SPLIT_MANIFEST.read_text('utf-8')) != split_meta):
            print('FAILED: holdout split does not match frozen split manifest',file=sys.stderr)
            return 1
        if set(split_meta['calibration_family_ids']) & set(split_meta['test_family_ids']):
            print('FAILED: template family leakage between calibration and test',file=sys.stderr)
            return 1
        print(f'PASS: family-disjoint frozen holdout, calibration={split_meta["n_calibration"]}, test={split_meta["n_test"]}')
    return 0

if __name__=='__main__':
    sys.exit(main())
