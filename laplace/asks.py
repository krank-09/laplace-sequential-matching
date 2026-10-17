"""Clarification: constraint bundles ranked by unlock value, then value-of-information soft asks."""
from kit import SOFT, eligibility
from .observe import hard_state, pair_key

BUNDLE, SOFT_COST = 3, 1


def rank_bundles(view, cfg):
    """Askable available members ordered by the expected partners a clarification would unlock."""
    status = {m['member_id']: hard_state(m) for m in view.members}
    scored = []
    for m in view.available:
        i = m['member_id']
        if status[i] != 'askable':
            continue  # complete needs no bundle; blocked (declined) can never become feasible
        unlock = 0.0
        for c in view.members:
            j = c['member_id']
            if j == i or status[j] == 'blocked' or pair_key(i, j) in view.past:
                continue
            if eligibility(m, c)['status'] == 'infeasible':
                continue  # a known hard failure: clarification cannot help this pair
            unlock += 1.0 if status[j] == 'complete' else cfg['askable_partner_weight']
        if unlock > 0:
            scored.append((-unlock, i))
    return [i for _, i in sorted(scored)]


def baseline_order(view):
    return [m['member_id'] for m in view.available if hard_state(m) == 'askable']


def soft_asks(view, model_w, budget, edges, mode='broad'):
    """Spend leftover budget on soft fields whose agreement drives the learned acceptance model.

    mode 'broad' (default): any available member who has, or may soon have, candidate edges; the score is
      |learned weight| x (1 + current candidate degree), so high-weight fields of busy candidates first.
    mode 'partner_known' (first design, kept as an ablation): only fields a candidate partner already
      shows. It rarely fires early because most fields are unknown on both sides.
    """
    by_id = view.by_id
    nbrs = {}
    for a, b in edges:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    scored = []
    for m in view.available:
        i = m['member_id']
        ps = nbrs.get(i, [])
        if mode == 'partner_known' and not ps:
            continue
        if mode == 'broad' and not ps and hard_state(m) == 'blocked':
            continue  # can never be introduced, so soft answers cannot change a decision
        for k_idx, k in enumerate(SOFT):
            if m['fields'].get(k) is not None or m['field_status'].get(k) == 'declined':
                continue
            if mode == 'partner_known':
                known = sum(1 for p in ps if by_id[p]['fields'].get(k) is not None)
                score = abs(model_w[1 + k_idx]) * known
            else:
                score = abs(model_w[1 + k_idx]) * (1 + len(ps))
            if score > 0:
                scored.append((-score, i, k))
    out = []
    for score, i, k in sorted(scored):
        if budget < SOFT_COST:
            break
        out.append({'member_id': i, 'field': k})
        budget -= SOFT_COST
    return out


def choose_asks(view, cfg, model_w):
    budget = view.state['ask_budget_remaining']
    order = rank_bundles(view, cfg) if cfg['rank_asks'] else baseline_order(view)
    asks = []
    for i in order[:budget // BUNDLE]:
        asks.append({'member_id': i, 'field': 'constraints'})
    budget -= BUNDLE * len(asks)
    if cfg['soft_asks'] and budget >= SOFT_COST:
        asks += soft_asks(view, model_w, budget, view.feasible_edges(), cfg['soft_mode'])
    return asks
