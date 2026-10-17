"""Fit the weak-prior training asset laplace/priors.json from our own simulator rollouts.

Declared training seeds: 1000-1059, all six public variants (360 episodes). Rollouts use the Laplace
policy with zero-mean priors and Thompson sampling, so logged introductions are diverse. Training rows
use only what the policy observed: soft-field agreement *at assignment time* (kept in policy memory)
and matured feedback events (laplace.observe.labels, the same censoring rules used online).
No hidden simulator fields are read. The output only sets prior means; online learning refits them.

usage: python3 scripts/fit_priors.py [--seeds 1000-1059] [--output laplace/priors.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate import VARIANTS  # noqa: E402
from fast_eval import run, parse_seeds  # noqa: E402
from laplace.observe import labels, pair_key  # noqa: E402
from laplace.model import map_logistic  # noqa: E402
from kit import SOFT  # noqa: E402


def dataset(rows):
    X_a, y_a, X_s, y_s, resp = [], [], [], [], [0, 0]
    for r in rows:
        f = r['final']
        feats = (f['memory'] or {}).get('feat', {})
        state = {'day': 100, 'introductions': f['introductions'], 'feedback': f['feedback']}
        accept, respond, success = labels(state)
        pairs = {i['introduction_id']: (i['user_a'], i['user_b']) for i in f['introductions']}
        for _, a, b, t, _ in accept:
            X_a.append([1] + feats[pair_key(a, b)])
            y_a.append(t)
        for iid, t, _ in success:
            X_s.append([1] + feats[pair_key(*pairs[iid])])
            y_s.append(t)
        for k, n in respond.values():
            resp[0] += k
            resp[1] += n
    return X_a, y_a, X_s, y_s, resp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1000-1059')
    p.add_argument('--output', default=str(ROOT / 'laplace' / 'priors.json'))
    a = p.parse_args()
    cfg = {'log_predictions': False}
    import os
    os.environ['LAPLACE_PRIORS'] = 'none'  # rollouts start from zero-mean defaults (inherited by workers)
    rows = run('laplace', parse_seeds(a.seeds), list(VARIANTS), cfg, keep=True)
    X_a, y_a, X_s, y_s, resp = dataset(rows)
    wide = [3.0] * 8
    w_a, v_a = map_logistic(X_a, y_a, [1.0] * len(X_a), [0.0] * len(X_a), [0.0] * 8, wide)
    w_s, v_s = map_logistic(X_s, y_s, [1.0] * len(X_s), [0.0] * len(X_s), [-1.5] + [0.0] * 7, wide)
    names = ['intercept'] + SOFT
    out = {
        'description': 'Weak-prior means fitted on declared training rollouts; refit online every call.',
        'training_seeds': a.seeds, 'variants': list(VARIANTS), 'episodes': len(rows),
        'accept_rows': len(y_a), 'success_rows': len(y_s), 'success_events': sum(y_s),
        'feature_names': names,
        'accept_mean': [round(x, 4) for x in w_a], 'accept_sd': [round(x ** .5, 4) for x in v_a],
        'success_mean': [round(x, 4) for x in w_s], 'success_sd': [round(x ** .5, 4) for x in v_s],
        'response_rate': round(resp[0] / max(1, resp[1]), 4),
    }
    Path(a.output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
