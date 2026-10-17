"""Tables and paired comparisons for one seed set: results/<set>/*.json -> results/<set>/analysis.{md,json}.

Uncertainty: 95% bootstrap intervals that resample seeds within each variant (20 seeds x 6 variants),
mirroring the primary score's equal weighting of families. Comparisons are paired by (variant, seed).

usage: python3 scripts/analyse.py --set reporting [--ref full]
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import VARIANTS  # noqa: E402

ORDER = ['submitted', 'final', 'full', 'greedy', 'random', 'no_asks', 'online_off', 'final_fixed', 'zero_prior',
         'thompson_on', 'thompson_off', 'soft_asks_on', 'soft_asks_off', 'soft_partner_known', 'discount_on', 'no_discount',
         'baseline_asks', 'scarcity_off', 'wait_on', 'cardinality_first', 'greedy_matcher', 'exact_matcher',
         'round1_design', 'reference_full_info', 'reference_full_info_greedy']
B = 4000


def metric(e, k):
    n = max(1, e['arrived_members'])
    if k == 'msmi':
        return e['msmi_per_100_arrived_members']
    if k == 'mutual':
        return e['mutual_acceptances_per_100']
    if k == 'dates':
        return 100 * e['dates'] / n
    if k == 'assign':
        return 100 * e['assignments'] / n
    if k == 'coverage':
        return e['coverage']
    if k == 'unserved':
        return e['unserved_members']
    if k == 'ask':
        return e['ask_cost']
    if k == 'missing':
        return e['missing_feedback']
    if k == 'wait':
        w = e['first_introduction_wait_days']
        return st.median(w) if w else 0
    if k == 'inf':
        return e['inference_seconds']
    raise KeyError(k)


def family_mean(rows, k):
    return st.mean(st.mean(metric(e, k) for e in rows if e['variant'] == v) for v in VARIANTS)


def boot(by_variant, rng):
    """by_variant: variant -> list of values (one per seed). Returns 95% CI of the family-mean."""
    sims = []
    for _ in range(B):
        sims.append(st.mean(st.mean(rng.choice(xs) for _ in xs) for xs in by_variant.values()))
    sims.sort()
    return sims[int(.025 * B)], sims[int(.975 * B)]


def load(setname, d=''):
    runs = {}
    for f in sorted((Path(d) if d else ROOT / 'results' / setname).glob('*.json')):
        if f.name.startswith('analysis'):
            continue
        d = json.loads(f.read_text())
        if 'episodes' in d and 'name' in d:
            runs[d['name']] = d
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', default='reporting')
    p.add_argument('--ref', default='submitted')
    p.add_argument('--dir', default='', help='read runs from this directory instead of results/<set>')
    a = p.parse_args()
    runs = load(a.set, a.dir)
    names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]
    rng = random.Random(11)
    out, md = {}, []
    md.append(f'## Seed set: {a.set} ({len(next(iter(runs.values()))["seeds"])} seeds x 6 variants)\n')
    md.append('| run | MSMI/100 [95% CI] | mutual/100 | dates/100 | assign/100 | coverage | unserved | median 1st wait | ask cost | missing fb | infer s/ep | valid |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in names:
        rows = runs[n]['episodes']
        bv = {v: [metric(e, 'msmi') for e in rows if e['variant'] == v] for v in VARIANTS}
        ci = boot(bv, rng)
        r = {k: family_mean(rows, k) for k in ['msmi', 'mutual', 'dates', 'assign', 'coverage', 'unserved', 'wait', 'ask', 'missing', 'inf']}
        r['msmi_ci'] = ci
        r['valid'] = sum(e['valid'] for e in rows)
        r['episodes'] = len(rows)
        out[n] = r
        md.append(f"| {n} | {r['msmi']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['mutual']:.2f} | {r['dates']:.2f} | {r['assign']:.1f} | "
                  f"{r['coverage']:.3f} | {r['unserved']:.1f} | {r['wait']:.1f} | {r['ask']:.0f} | {r['missing']:.1f} | {r['inf']:.1f} | {r['valid']}/{r['episodes']} |")
    md.append('\n### MSMI per 100 arrived members by variant (mean of 20 seeds)\n')
    md.append('| run | ' + ' | '.join(VARIANTS) + ' |')
    md.append('|---|' + '---|' * len(VARIANTS))
    for n in names:
        rows = runs[n]['episodes']
        md.append(f'| {n} | ' + ' | '.join(f"{st.mean(metric(e, 'msmi') for e in rows if e['variant'] == v):.3f}" for v in VARIANTS) + ' |')
    md.append('\n### Mutual acceptances per 100 by variant\n')
    md.append('| run | ' + ' | '.join(VARIANTS) + ' |')
    md.append('|---|' + '---|' * len(VARIANTS))
    for n in names:
        rows = runs[n]['episodes']
        md.append(f'| {n} | ' + ' | '.join(f"{st.mean(metric(e, 'mutual') for e in rows if e['variant'] == v):.2f}" for v in VARIANTS) + ' |')
    if a.ref in runs:
        md.append(f'\n### Paired differences: {a.ref} minus run (positive = {a.ref} better), 95% CI\n')
        md.append('| run | ΔMSMI/100 | Δmutual/100 | Δdates/100 | Δcoverage |')
        md.append('|---|---|---|---|---|')
        ref = {(e['variant'], e['seed']): e for e in runs[a.ref]['episodes']}
        out['paired'] = {}
        for n in names:
            if n == a.ref:
                continue
            cells, rec = [], {}
            for k in ['msmi', 'mutual', 'dates', 'coverage']:
                bv = {v: [metric(ref[(v, e['seed'])], k) - metric(e, k) for e in runs[n]['episodes'] if e['variant'] == v]
                      for v in VARIANTS}
                d = st.mean(st.mean(x) for x in bv.values())
                lo, hi = boot(bv, rng)
                rec[k] = (d, lo, hi)
                star = '*' if lo > 0 or hi < 0 else ''
                cells.append(f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}')
            out['paired'][n] = rec
            md.append(f'| {n} | ' + ' | '.join(cells) + ' |')
        md.append('\n`*` interval excludes zero.')
    full = runs.get(a.ref)
    if full:
        md.append(f'\n### Outcome funnel for {a.ref} by variant (means per episode)\n')
        md.append('| variant | arrived | assignments | mutual | dates | MSMI | coverage | unserved | missing fb |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for v in VARIANTS:
            rs = [e for e in full['episodes'] if e['variant'] == v]
            f = lambda k: st.mean(e[k] for e in rs)
            md.append(f"| {v} | {f('arrived_members'):.1f} | {f('assignments'):.1f} | {f('mutual_acceptances'):.2f} | {f('dates'):.2f} | "
                      f"{f('mutual_second_meeting_intention'):.2f} | {f('coverage'):.3f} | {f('unserved_members'):.1f} | {f('missing_feedback'):.1f} |")
    (ROOT / 'results' / a.set / 'analysis.md').write_text('\n'.join(md) + '\n')
    (ROOT / 'results' / a.set / 'analysis.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
