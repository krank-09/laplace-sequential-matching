"""Summarise evaluate.py JSON outputs into a per-variant funnel table (markdown)."""
import json
import statistics
import sys
from pathlib import Path

KEYS = ['assignments', 'mutual_acceptances', 'dates', 'mutual_second_meeting_intention',
        'msmi_per_100_arrived_members', 'coverage', 'unserved_members', 'ask_cost',
        'missing_feedback', 'inference_seconds']
SHORT = ['assign', 'mutual', 'dates', 'MSMI', 'MSMI/100', 'cover', 'unserved', 'ask', 'missing', 'inf_s']


def fmt(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{mean:.2f}±{sd:.2f}' if mean < 10 else f'{mean:.1f}±{sd:.1f}'


def main(paths):
    print('| policy | variant | ' + ' | '.join(SHORT) + ' | median wait |')
    print('|' + '---|' * (len(SHORT) + 3))
    for path in paths:
        data = json.loads(Path(path).read_text())
        name = Path(path).stem
        variants = sorted({e['variant'] for e in data['episodes']}, key=lambda v: [e['variant'] for e in data['episodes']].index(v))
        for v in variants:
            rows = [e for e in data['episodes'] if e['variant'] == v and e.get('valid')]
            waits = [statistics.median(r['first_introduction_wait_days']) for r in rows if r['first_introduction_wait_days']]
            cells = [fmt([r[k] for r in rows]) for k in KEYS]
            print(f'| {name} | {v} (n={len(rows)}) | ' + ' | '.join(cells) + f' | {statistics.mean(waits):.1f} |')
        s = data['summary']
        print(f'| **{name}** | **primary** | | | | | **{s["primary_score"]:.3f}** | {s["overall"]["coverage"]:.3f} | | {s["overall"]["ask_cost"]:.0f} | | | |')


if __name__ == '__main__':
    main(sys.argv[1:])
