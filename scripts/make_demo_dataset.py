"""Rebuild the 10-case demo selection from the unchanged 48-case source dataset."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'legacy' / 'synthetic_inspection_cases.jsonl'
OUTPUT = ROOT / 'data' / 'legacy' / 'demo_10_cases.jsonl'
# Only the demo-specific *primary challenge group* is relabeled. Text and gold stay identical.
DEMO_GROUP = {
    'PS-009': 'clear', 'PS-001': 'hypothesis', 'PS-002': 'negation',
    'PS-018': 'mixed_status', 'PS-031': 'clear', 'PS-014': 'negation',
    'PS-010': 'resolved', 'PS-037': 'resolved', 'PS-044': 'negation',
    'PS-012': 'out_of_scope',
}
DEMO_IDS = (
    'PS-009', 'PS-001', 'PS-002', 'PS-018', 'PS-031',
    'PS-014', 'PS-010', 'PS-037', 'PS-044', 'PS-012',
)


def render_demo() -> str:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate case IDs in 48-case source')
    if len(set(DEMO_IDS)) != len(DEMO_IDS):
        raise ValueError('duplicate selected case IDs')
    missing = set(DEMO_IDS) - set(by_id)
    if missing:
        raise ValueError(f'not found in 48-case source: {missing}')
    chosen = [{**by_id[i], 'group': DEMO_GROUP[i]} for i in DEMO_IDS]
    return ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in chosen)


def main() -> int:
    OUTPUT.write_text(render_demo(), encoding='utf-8')
    print(f'Wrote {len(DEMO_IDS)} curated cases to {OUTPUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
