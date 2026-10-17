"""Allocation-value gap of the shipped matcher vs an exact general-graph solver.

A. Density sweep: random graphs, 120 vertices, edge probability 0.01-0.5, log-normal weights.
B. Real decision graphs: every daily graph the Laplace policy faced on validation seeds 2000-2001
   (all six variants), recording greedy / local-improvement / exact total weight.
Outcome impact (exact matcher inside the policy) is measured separately by run_ablations.py.

usage: python3 scripts/matching_gap.py --output results/matching_gap.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from exact_matching import exact_matching  # noqa: E402
from laplace.matching import max_weight_matching  # noqa: E402
from evaluate import VARIANTS  # noqa: E402


def weight(pairs, edges):
    return sum(edges[tuple(p)] for p in pairs)


def compare(edges):
    t = time.perf_counter()
    loc = max_weight_matching(edges, improve=True)
    t_loc = time.perf_counter() - t
    gre = max_weight_matching(edges, improve=False)
    ex = exact_matching(edges)
    w_ex = weight(ex, edges)
    return {'greedy': weight(gre, edges), 'local': weight(loc, edges), 'exact': w_ex,
            'local_seconds': t_loc, 'edges': len(edges),
            'nodes': len({v for e in edges for v in e})}


def sweep(n=120, densities=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), reps=20, seed=7):
    rng = random.Random(seed)
    out = []
    for p in densities:
        rows = []
        for _ in range(reps):
            edges = {}
            for a in range(n):
                for b in range(a + 1, n):
                    if rng.random() < p:
                        edges[(f'v{a:03d}', f'v{b:03d}')] = rng.lognormvariate(-4, 0.8)
            rows.append(compare(edges))
        out.append({'density': p, 'rows': rows})
    return out


def episode_graphs(seeds=(2000, 2001)):
    from fast_eval import run_episode
    records = []

    def matcher(edges):
        if edges:
            r = compare(edges)
            records.append(r)
        return max_weight_matching(edges, improve=True)

    for v in VARIANTS:
        for s in seeds:
            start = len(records)
            run_episode((s, v, 'laplace', {'matcher': matcher}, False))
            for r in records[start:]:
                r.update(variant=v, seed=s)
    return records


def describe(rows):
    ratio = lambda k: [r[k] / r['exact'] if r['exact'] > 0 else 1.0 for r in rows]
    return {'n': len(rows), 'edges_mean': statistics.mean(r['edges'] for r in rows),
            'greedy_ratio_mean': statistics.mean(ratio('greedy')), 'greedy_ratio_min': min(ratio('greedy')),
            'local_ratio_mean': statistics.mean(ratio('local')), 'local_ratio_min': min(ratio('local')),
            'local_optimal_share': statistics.mean(r['local'] >= r['exact'] - 1e-12 for r in rows),
            'local_seconds_max': max(r['local_seconds'] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/matching_gap.json')
    a = p.parse_args()
    sw = sweep()
    eg = episode_graphs()
    summary = {'sweep': [dict(density=d['density'], **describe(d['rows'])) for d in sw],
               'episode_graphs': describe(eg),
               'episode_graphs_by_variant': {v: describe([r for r in eg if r['variant'] == v]) for v in VARIANTS}}
    Path(a.output).write_text(json.dumps({'summary': summary, 'episode_rows': eg}, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
