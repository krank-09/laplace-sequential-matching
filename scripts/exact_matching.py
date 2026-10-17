"""Exact maximum-weight general-graph matching as a 0/1 integer program (SciPy HiGHS). Offline only.

max sum_e w_e x_e  s.t.  sum_{e incident to v} x_e <= 1 for every vertex v,  x_e in {0, 1}.
The integer program is exact for general (non-bipartite) graphs; it is the reference for measuring
the optimality gap of laplace.matching. SciPy is a development dependency, never in the policy image.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def exact_matching(edges):
    """edges: dict (a,b)->w. Returns sorted list of [a, b] pairs of an optimal matching."""
    items = [(k, w) for k, w in sorted(edges.items()) if w > 0]
    if not items:
        return []
    nodes = sorted({v for (a, b), _ in items for v in (a, b)})
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for j, ((a, b), _) in enumerate(items):
        rows += [idx[a], idx[b]]
        cols += [j, j]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(items)))
    c = -np.array([w for _, w in items])
    res = milp(c, constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(len(items)),
               bounds=Bounds(0, 1), options={'mip_rel_gap': 0})
    if not res.success:
        raise RuntimeError(res.message)
    return sorted([list(k) for (k, _), x in zip(items, res.x) if x > 0.5])
