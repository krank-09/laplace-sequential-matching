"""Two-phase decision entry point and the final action validator."""
import sys
import traceback

from kit import SOFT, eligibility
from .observe import View, agreement, pair_key
from .model import Model
from .asks import choose_asks
from .matching import max_weight_matching

DEFAULT = {
    'online': True,          # refit weights / member effects / response rates from feedback each call
    'thompson': False,       # True: act on one posterior draw (validation: no gain, slight harm)
    'gamma': 1.0,            # per-day discount on old evidence; <1 tested, no reliable gain
    'prior_sd': 0.6,         # weak prior around the training-asset means
    'tau': 0.6,              # member-effect prior s.d.
    'rho_strength': 4.0,     # pseudo-counts for response-rate shrinkage
    'rank_asks': True,       # unlock-value bundle ranking (False: baseline list order)
    'askable_partner_weight': 0.4,
    'soft_asks': False,      # True: spend leftover budget on soft fields (validation: no gain, +360 ask cost)
    'soft_mode': 'broad',    # 'broad' or 'partner_known' (first design)
    'scarcity': 1.0,         # boost edges of members with few feasible options (0 disables)
    'cardinality': 0.0,      # constant added to every edge weight: >0 favours matching more pairs today
    'wait_kappa': 0.0,       # waiting threshold as a fraction of a member's best alternatives (0 disables)
    'matcher': 'local',      # 'local' (greedy + local improvement), 'greedy', or a callable (offline tests)
    'log_predictions': True, # keep decision-time predictions in memory for calibration reports
}


def log(msg):
    print(msg, file=sys.stderr)


def build_edges(view, model, cfg):
    """Return (weights dict, prediction dict) for today's feasible edges."""
    w, u = model.weights()
    raw, preds = {}, {}
    for a, b in view.feasible_edges():
        a, b = sorted((a, b))
        g = agreement(view.by_id[a]['fields'], view.by_id[b]['fields'])
        p_ab, p_ba, v = model.pair_value(a, b, g, w, u)
        raw[(a, b)] = v
        preds[(a, b)] = (g, p_ab, p_ba, v)
    deg = {}
    for a, b in raw:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    if cfg['wait_kappa'] > 0 and view.days_left > 10:
        top = {}
        for (a, b), v in raw.items():
            top.setdefault(a, []).append(v)
            top.setdefault(b, []).append(v)
        bar = {i: cfg['wait_kappa'] * sum(sorted(vs)[-3:]) / len(sorted(vs)[-3:]) for i, vs in top.items()}
        raw = {(a, b): v for (a, b), v in raw.items()
               if (deg[a] < 3 or v >= bar[a]) and (deg[b] < 3 or v >= bar[b])}
    s = cfg['scarcity']
    k = cfg['cardinality']
    weights = {(a, b): k + v * (1 + s * (1 / deg[a] + 1 / deg[b]) / 2) for (a, b), v in raw.items()}
    return weights, preds


def validate_pairs(view, pairs):
    used, out = set(), []
    for a, b in pairs:
        ma, mb = view.by_id.get(a), view.by_id.get(b)
        if (not ma or not mb or a == b or a in used or b in used or not ma['available']
                or not mb['available'] or pair_key(a, b) in view.past
                or eligibility(ma, mb)['status'] != 'feasible'):
            log(f'validator dropped pair {a} {b}')
            continue
        used.update((a, b))
        out.append([a, b])
    return out


def validate_asks(view, asks):
    budget, seen, out = view.state['ask_budget_remaining'], set(), []
    for q in asks:
        m = view.by_id.get(q['member_id'])
        cost = 3 if q['field'] == 'constraints' else 1
        if (m is None or not m['available'] or q['field'] not in ['constraints'] + SOFT
                or (q['member_id'], q['field']) in seen or cost > budget):
            continue
        seen.add((q['member_id'], q['field']))
        budget -= cost
        out.append({'member_id': q['member_id'], 'field': q['field']})
    return out


def decide(request, cfg=None):
    cfg = {**DEFAULT, **(cfg or {})}
    memory = request.get('memory') or {}
    phase = request['phase']
    try:
        view = View(request['state'])
        model = Model(request['state'], memory, cfg)
        if phase == 'ask':
            return {'asks': validate_asks(view, choose_asks(view, cfg, model.w)), 'memory': memory}
        weights, preds = build_edges(view, model, cfg)
        matcher = cfg['matcher']
        if callable(matcher):
            pairs = matcher(weights)
        else:
            pairs = max_weight_matching(weights, improve=(matcher == 'local'))
        pairs = validate_pairs(view, pairs)
        feat = dict(memory.get('feat', {}))
        pred = dict(memory.get('pred', {}))
        for a, b in pairs:
            key = pair_key(a, b)
            g, p_ab, p_ba, v = preds[tuple(sorted((a, b)))]
            feat[key] = g
            if cfg['log_predictions']:
                # directional predictions keyed by actor, plus the pair value, at decision time
                pred[key] = {a: round(p_ab, 4), b: round(p_ba, 4), 'v': round(v, 5), 'day': view.day}
        memory = {k: v for k, v in (('feat', feat), ('pred', pred)) if v}
        if feat:
            memory['w'] = [round(x, 4) for x in model.w]  # learned weights, for the movement diagnostic
        return {'pairs': pairs, 'memory': memory}
    except Exception:  # never emit an invalid action: waiting / not asking is always valid
        log(traceback.format_exc())
        return {'asks' if phase == 'ask' else 'pairs': [], 'memory': memory}
