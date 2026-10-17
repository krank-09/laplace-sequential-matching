"""Calibration, weight movement, waiting and failure taxonomy from a --keep run (observable data only).

Targets (defined at decision time, matured by day 100):
  directional:  P(actor answers Yes | actor responded by the 7-day deadline)
  joint:        P(both Yes | both responded), predicted by the product p_ab * p_ba
  reciprocal:   P(MSMI) for the introduction, predicted by the pair value v
                (= rho_a^2 rho_b^2 p_ab p_ba q_ab, includes response propensities)

usage: python3 scripts/calibration.py --run results/reporting/full.json [--prior-run results/reporting/online_off.json]
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import VARIANTS  # noqa: E402
from kit import HARD  # noqa: E402
from laplace.observe import labels, pair_key  # noqa: E402


def bins(pairs, k=10):
    pairs = sorted(pairs)
    n = len(pairs)
    out = []
    for i in range(k):
        chunk = pairs[i * n // k:(i + 1) * n // k]
        if chunk:
            out.append((st.mean(p for p, _ in chunk), st.mean(y for _, y in chunk), len(chunk)))
    return out


def scores(pairs):
    n = len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    base = st.mean(y for _, y in pairs)
    brier0 = sum((base - y) ** 2 for _, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)) for p, y in pairs) / n
    ece = sum(c * abs(mp - my) for mp, my, c in bins(pairs)) / n
    return {'n': n, 'mean_pred': st.mean(p for p, _ in pairs), 'observed': base, 'brier': brier,
            'brier_constant': brier0, 'log_loss': ll, 'ece': ece}


def collect(run):
    d, a, j, r, w, rows = [], [], [], [], {}, []
    for e in run['episodes']:
        f = e['final']
        mem = f['memory'] or {}
        pred = mem.get('pred', {})
        state = {'day': 100, 'introductions': f['introductions'], 'feedback': f['feedback']}
        accept, respond, success = labels(state)
        intro_pair = {i['introduction_id']: (i['user_a'], i['user_b']) for i in f['introductions']}
        yes = {}
        for iid, actor, target, t, _ in accept:
            pk = pair_key(actor, target)
            if pk in pred:
                d.append((pred[pk][actor], t, e['variant']))
                yes.setdefault(iid, {})[actor] = (pred[pk][actor], t)
        for iid, m in yes.items():
            if len(m) == 2:
                (pa, ya), (pb, yb) = m.values()
                j.append((pa * pb, ya * yb, e['variant'], ya - pa, yb - pb))
        succ = {iid: t for iid, t, _ in success}
        for iid, (ua, ub) in intro_pair.items():
            pk = pair_key(ua, ub)
            if pk in pred:
                r.append((pred[pk]['v'], succ.get(iid, 0), e['variant']))
        if mem.get('w'):
            w.setdefault(e['variant'], []).append(mem['w'])
        rows.append(e)
    return d, j, r, w, rows


def table(pairs, title):
    s = scores([(p, y) for p, y, *_ in pairs])
    lines = [f'\n**{title}**: n={s["n"]}, mean predicted {s["mean_pred"]:.3f}, observed {s["observed"]:.3f}, '
             f'Brier {s["brier"]:.4f} (constant-rate {s["brier_constant"]:.4f}), log loss {s["log_loss"]:.3f}, ECE {s["ece"]:.3f}\n',
             '| bin mean predicted | observed rate | n |', '|---|---|---|']
    for mp, my, c in bins([(p, y) for p, y, *_ in pairs]):
        lines.append(f'| {mp:.3f} | {my:.3f} | {c} |')
    return lines, s


def failures(rows):
    unserved = {'blocked_declined': 0, 'hard_fields_never_clarified': 0, 'complete_but_never_introduced': 0}
    exits = {'no_response': 0, 'declined_intro': 0, 'no_date_or_date_failed': 0, 'date_after_30_days': 0,
             'second_answer_missing_late_or_no': 0, 'msmi': 0}
    newcomer = {'day0': [0, 0, []], 'later': [0, 0, []]}
    for e in rows:
        f = e['final']
        served = {}
        for i in f['introductions']:
            for m in (i['user_a'], i['user_b']):
                served.setdefault(m, i['assigned_day'])
        for m in f['members']:
            if m['member_id'] not in f['arrived']:
                continue
            grp = 'day0' if m['arrived_day'] == 0 else 'later'
            newcomer[grp][1] += 1
            if m['member_id'] in served:
                newcomer[grp][0] += 1
                newcomer[grp][2].append(served[m['member_id']] - m['arrived_day'])
                continue
            stt = m['field_status']
            if any(stt[k] == 'declined' for k in HARD):
                unserved['blocked_declined'] += 1
            elif any(stt[k] != 'observed' for k in HARD):
                unserved['hard_fields_never_clarified'] += 1
            else:
                unserved['complete_but_never_introduced'] += 1
        by = {}
        for ev in f['feedback']:
            by.setdefault(ev['introduction_id'], []).append(ev)
        for i in f['introductions']:
            es = by.get(i['introduction_id'], [])
            rs = [x for x in es if x['event'] == 'introduction_response']
            if any(x['value'] is None for x in rs):
                exits['no_response'] += 1
            elif any(x['value'] == 'no' for x in rs):
                exits['declined_intro'] += 1
            else:
                ds = [x for x in es if x['event'] == 'date_happened']
                if not ds or ds[0]['value'] is not True:
                    exits['no_date_or_date_failed'] += 1
                elif ds[0]['occurred_day'] - i['assigned_day'] > 30:
                    exits['date_after_30_days'] += 1
                else:
                    ss = [x for x in es if x['event'] == 'second_meeting_intention']
                    ok = len(ss) == 2 and all(x['value'] == 'yes' and x['occurred_day'] - ds[0]['occurred_day'] <= 3 for x in ss)
                    exits['msmi' if ok else 'second_answer_missing_late_or_no'] += 1
    n = len(rows)
    return ({k: v / n for k, v in unserved.items()}, {k: v / n for k, v in exits.items()},
            {g: {'coverage': c[0] / max(1, c[1]), 'median_wait': st.median(c[2]) if c[2] else None,
                 'p90_wait': sorted(c[2])[int(.9 * len(c[2]))] if c[2] else None, 'members_per_ep': c[1] / n}
             for g, c in newcomer.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='results/reporting/full.json')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    run = json.loads(Path(a.run).read_text())
    prior = json.loads((ROOT / 'laplace' / 'priors.json').read_text())
    d, j, r, w, rows = collect(run)
    md = [f'## Calibration and diagnostics: {run["name"]} on {run["seed_set"]} seeds ({len(rows)} episodes)']
    out = {}
    lines, out['directional'] = table(d, 'Directional acceptance P(Yes | responded)')
    md += lines
    lines, out['joint_product'] = table(j, 'Joint acceptance P(both Yes | both responded) predicted by p_ab*p_ba')
    md += lines
    ra = [x[3] for x in j]
    rb = [x[4] for x in j]
    corr = st.correlation(ra, rb) if len(j) > 2 else float('nan')
    out['residual_correlation'] = corr
    md.append(f'\nCorrelation of the two directional residuals within a pair: {corr:+.3f} (n={len(j)}). '
              'A positive value means a shared pair component; the product then under-predicts the joint.')
    lines, out['reciprocal'] = table(r, 'Reciprocal outcome P(MSMI) predicted by pair value v')
    md += lines
    md.append('\n**Per variant**: predicted vs observed totals\n')
    md.append('| variant | directional pred / obs | joint pred / obs | MSMI pred / obs per episode |')
    md.append('|---|---|---|---|')
    for v in VARIANTS:
        dv = [x for x in d if x[2] == v]
        jv = [x for x in j if x[2] == v]
        rv = [x for x in r if x[2] == v]
        neps = sum(1 for e in rows if e['variant'] == v)
        md.append(f'| {v} | {st.mean(x[0] for x in dv):.3f} / {st.mean(x[1] for x in dv):.3f} | '
                  f'{st.mean(x[0] for x in jv):.3f} / {st.mean(x[1] for x in jv):.3f} | '
                  f'{sum(x[0] for x in rv) / neps:.2f} / {sum(x[1] for x in rv) / neps:.2f} |')
    names = prior['feature_names']
    md.append('\n**Online weight movement**: end-of-decision-horizon acceptance weights (mean over seeds) vs prior mean\n')
    md.append('| feature | prior | ' + ' | '.join(VARIANTS) + ' |')
    md.append('|---|---|' + '---|' * len(VARIANTS))
    out['weights'] = {}
    for k, name in enumerate(names):
        cells = []
        for v in VARIANTS:
            vals = [x[k] for x in w.get(v, [])]
            cells.append(f'{st.mean(vals):+.3f}' if vals else 'n/a')
            out['weights'].setdefault(v, []).append(st.mean(vals) if vals else None)
        md.append(f"| {name} | {prior['accept_mean'][k]:+.3f} | " + ' | '.join(cells) + ' |')
    unserved, exits, newcomer = failures(rows)
    out.update(unserved=unserved, exits=exits, newcomer=newcomer)
    md.append('\n**Who is left waiting** (per episode)\n')
    md.append('| group | members | coverage | median first wait (days) | 90th pct wait |')
    md.append('|---|---|---|---|---|')
    for g, s in newcomer.items():
        md.append(f"| arrived {'day 0' if g == 'day0' else 'days 1-20'} | {s['members_per_ep']:.1f} | {s['coverage']:.3f} | {s['median_wait']} | {s['p90_wait']} |")
    md.append('\n**Unserved members by cause** (per episode): ' + ', '.join(f'{k.replace("_", " ")} {v:.1f}' for k, v in unserved.items()))
    md.append('\n**Introduction exits** (per episode): ' + ', '.join(f'{k.replace("_", " ")} {v:.2f}' for k, v in exits.items()))
    target = Path(a.output or Path(a.run).with_name(Path(a.run).stem + '_calibration.md'))
    target.write_text('\n'.join(md) + '\n')
    target.with_suffix('.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
