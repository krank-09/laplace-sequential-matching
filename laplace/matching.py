"""Maximum-weight matching on a general graph: greedy start, then local improvement.

Moves (each applied only if it strictly increases total weight):
  * augment:  free f adjacent to matched a (a-b):  swap a-b for a-f, then match b to a free g if possible
  * swap:     matched a-b, c-d  ->  a-c, b-d   or   a-d, b-c
Greedy alone is a 1/2-approximation; the result is compared offline with an exact MILP solver
(scripts/matching_gap.py), which is never shipped in the policy image.
"""


def greedy(edges):
    """edges: dict (a,b)->w with a<b. Returns mate dict."""
    mate = {}
    for (a, b), w in sorted(edges.items(), key=lambda kv: (-kv[1], kv[0])):
        if w > 0 and a not in mate and b not in mate:
            mate[a], mate[b] = b, a
    return mate


def total(mate, wt):
    return sum(wt(a, b) for a, b in mate.items() if a < b)


def local_improve(edges, mate, max_rounds=50):
    adj = {}
    for (a, b), w in edges.items():
        if w > 0:
            adj.setdefault(a, {})[b] = w
            adj.setdefault(b, {})[a] = w

    def wt(a, b):
        return adj.get(a, {}).get(b, 0.0)

    eps = 1e-12
    for _ in range(max_rounds):
        improved = False
        # augmenting moves through free vertices
        for f in sorted(adj):
            if f in mate:
                continue
            best = None
            for a, w_fa in adj[f].items():
                if a in mate:
                    b = mate[a]
                    gain = w_fa - wt(a, b)
                    g_best, g_node = 0.0, None
                    for g, w_bg in adj.get(b, {}).items():
                        if g not in mate and g != f and w_bg > g_best:
                            g_best, g_node = w_bg, g
                    gain += g_best
                    if gain > eps and (best is None or gain > best[0]):
                        best = (gain, a, b, g_node)
                elif w_fa > eps and (best is None or w_fa > best[0]):
                    best = (w_fa, a, None, None)
            if best:
                _, a, b, g = best
                if b is not None:
                    del mate[b]
                mate[a], mate[f] = f, a
                if g is not None:
                    mate[b], mate[g] = g, b
                improved = True
        # 2-edge swaps
        pairs = sorted((a, b) for a, b in mate.items() if a < b)
        used = set()
        for i in range(len(pairs)):
            a, b = pairs[i]
            if a in used:
                continue
            for j in range(i + 1, len(pairs)):
                c, d = pairs[j]
                if c in used or a in used:
                    continue
                cur = wt(a, b) + wt(c, d)
                options = [((a, c), (b, d)), ((a, d), (b, c))]
                best = max(options, key=lambda o: wt(*o[0]) + wt(*o[1]))
                new = wt(*best[0]) + wt(*best[1])
                if new > cur + eps:
                    for x in (a, b, c, d):
                        mate.pop(x, None)
                    for x, y in best:
                        if wt(x, y) > 0:
                            mate[x], mate[y] = y, x
                    used.update((a, b, c, d))
                    improved = True
        if not improved:
            break
    return mate


def max_weight_matching(edges, improve=True):
    """Return a list of [a, b] pairs (a < b) maximising total positive weight (approximately)."""
    mate = greedy(edges)
    if improve:
        mate = local_improve(edges, mate)
    return sorted([a, b] for a, b in mate.items() if a < b)
