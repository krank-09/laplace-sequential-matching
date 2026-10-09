"""Bootstrap CIs (resampling seeds within variant) for primary score and paired greedy-random difference."""
import json
import random
import statistics as st

V = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']
K = 'msmi_per_100_arrived_members'
d = {b: json.load(open(f'results/{b}.json'))['episodes'] for b in ['greedy', 'random', 'no_asks']}


def primary(eps):
    return st.mean(st.mean(e[K] for e in eps if e['variant'] == v) for v in V)


def boot(b, B=20000, seed=0):
    r = random.Random(seed)
    out = sorted(st.mean(st.mean(r.choice([e for e in d[b] if e['variant'] == v])[K] for _ in range(3)) for v in V)
                 for _ in range(B))
    return out[int(.025 * B)], out[int(.975 * B)]


for b in d:
    lo, hi = boot(b)
    print(f'{b}: primary {primary(d[b]):.3f}, 95% seed-bootstrap CI [{lo:.3f}, {hi:.3f}], '
          f'MSMI events {sum(e["mutual_second_meeting_intention"] for e in d[b])}')
g = {(e['variant'], e['seed']): e[K] for e in d['greedy']}
ra = {(e['variant'], e['seed']): e[K] for e in d['random']}
diffs = {v: [g[(v, s)] - ra[(v, s)] for s in (101, 102, 103)] for v in V}
r = random.Random(1)
out = sorted(st.mean(st.mean(r.choice(diffs[v]) for _ in range(3)) for v in V) for _ in range(20000))
print(f'paired greedy-random: {st.mean(st.mean(x) for x in diffs.values()):.3f}, 95% CI [{out[500]:.3f}, {out[19500]:.3f}]')
