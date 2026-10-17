"""Team Laplace policy: executable JSON adapter, never reads simulator hidden state.

Default (no arguments, as in the container entrypoint): the Laplace policy in laplace/.
`--baseline greedy|no_asks|random` keeps the supplied reference baselines runnable through evaluate.py.
`--baseline laplace` selects the Laplace policy explicitly. LAPLACE_CFG='{"online": false}' overrides
configuration keys for ablations (see laplace/core.py DEFAULT).
"""
import argparse
import itertools
import json
import os
import random
import sys
from kit import baseline_asks, baseline_match, eligibility


def decide_baseline(request, mode='greedy'):
    state = request['state']
    memory = request.get('memory') or {}
    if request['phase'] == 'ask':
        return {'asks': [] if mode == 'no_asks' else baseline_asks(state), 'memory': memory}
    if mode != 'random':
        pairs = baseline_match(state)
    else:
        candidates = [m for m in state['members'] if m['available']]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = [tuple(sorted((a['member_id'], b['member_id'])))
                 for a, b in itertools.combinations(candidates, 2)
                 if eligibility(a, b)['status'] == 'feasible'
                 and tuple(sorted((a['member_id'], b['member_id']))) not in past]
        random.Random(17 + state['day']).shuffle(edges)
        pairs, used = [], set()
        for pair in edges:
            if not used.intersection(pair):
                pairs.append(list(pair))
                used.update(pair)
    return {'pairs': pairs, 'memory': memory}


def decide(request, mode='laplace', cfg=None):
    if mode in ('greedy', 'no_asks', 'random'):
        return decide_baseline(request, mode)
    from laplace.core import decide as laplace_decide
    return laplace_decide(request, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=['greedy', 'no_asks', 'random', 'laplace'], default='laplace')
    args = parser.parse_args()
    cfg = json.loads(os.environ.get('LAPLACE_CFG') or '{}')
    request = json.load(sys.stdin)
    print(json.dumps(decide(request, args.baseline, cfg), allow_nan=False))
