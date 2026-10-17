"""Pair-value model fitted online from matured, observable feedback.

Directional acceptance:  logit P(a says Yes to b) = w0 + sum_k w_k g_k(a,b) + u_a
Post-acceptance success: logit P(MSMI | mutual Yes) = c0 + sum_k c_k g_k(a,b)
Response propensity:     rho_a, Beta-binomial shrinkage towards the episode's pooled rate.

g_k is the 3-state soft-field agreement (+1 / -1 / 0 unknown). Priors come from the declared training
asset (priors.json) and are deliberately weak; every call refits from the current episode's feedback.
"""
import json
import math
import os
import random
from pathlib import Path

from .observe import labels, agreement, pair_key

PRIORS_PATH = Path(__file__).with_name('priors.json')
DEFAULT_PRIORS = {'accept_mean': [0.0] * 8, 'success_mean': [-1.5] + [0.0] * 7, 'response_rate': 0.75}


def load_priors():
    override = os.environ.get('LAPLACE_PRIORS')  # offline experiments only; 'none' = zero-mean defaults
    if override == 'none':
        return DEFAULT_PRIORS
    try:
        return json.loads(Path(override or PRIORS_PATH).read_text())
    except (OSError, ValueError):
        return DEFAULT_PRIORS


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def solve(A, b):
    """Gaussian elimination with partial pivoting for small dense systems."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                if f:
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def map_logistic(X, y, wts, offsets, mu, sd, iters=12):
    """MAP logistic regression with independent Gaussian priors N(mu, sd^2); returns (w, posterior var)."""
    d = len(mu)
    w = list(mu)
    prec = [1 / (s * s) for s in sd]
    H = None
    for _ in range(iters):
        g = [-(w[j] - mu[j]) * prec[j] for j in range(d)]
        H = [[0.0] * d for _ in range(d)]
        for j in range(d):
            H[j][j] = prec[j]
        for x, t, s, o in zip(X, y, wts, offsets):
            p = sigmoid(sum(a * b for a, b in zip(w, x)) + o)
            r = s * (t - p)
            v = s * p * (1 - p)
            for j in range(d):
                if x[j]:
                    g[j] += r * x[j]
                    for k in range(d):
                        if x[k]:
                            H[j][k] += v * x[j] * x[k]
        step = solve(H, g)
        w = [a + b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    var = [1 / H[j][j] if H and H[j][j] > 0 else s * s for j, s in enumerate(sd)]
    return w, var


def member_effect(rows, w, tau):
    """1-D MAP random effect for one actor given fixed weights; returns (u, var)."""
    u, h = 0.0, 1 / (tau * tau)
    for _ in range(8):
        g, h = -u / (tau * tau), 1 / (tau * tau)
        for x, t, s in rows:
            p = sigmoid(sum(a * b for a, b in zip(w, x)) + u)
            g += s * (t - p)
            h += s * p * (1 - p)
        u += g / h
    return u, 1 / h


class Model:
    def __init__(self, state, memory, cfg):
        self.cfg = cfg
        pri = load_priors()
        self.day = state['day']
        feats = (memory or {}).get('feat', {})
        by_id = {m['member_id']: m for m in state['members']}
        intro_pair = {i['introduction_id']: (i['user_a'], i['user_b']) for i in state['introductions']}

        def x_for(a, b):
            key = pair_key(a, b)
            if key in feats:          # features as observed when the pair was assigned
                return [1] + feats[key]
            fa, fb = by_id.get(a), by_id.get(b)
            return [1] + (agreement(fa['fields'], fb['fields']) if fa and fb else [0] * 7)

        accept, respond, success = labels(state)
        gamma = cfg['gamma']
        disc = lambda od: gamma ** max(0, self.day - od)
        self.mu_a = list(pri['accept_mean'])
        self.mu_s = list(pri['success_mean'])
        sd_a = [cfg['prior_sd']] * 8
        if cfg['online']:
            X = [x_for(a, b) for _, a, b, _, _ in accept]
            y = [t for *_, t, _ in accept]
            wts = [disc(od) for *_, od in accept]
            w, var = map_logistic(X, y, wts, [0.0] * len(X), self.mu_a, sd_a)
            # member effects, then one refit of the shared weights with those offsets
            per = {}
            for x, (_, a, _, t, od) in zip(X, accept):
                per.setdefault(a, []).append((x, t, disc(od)))
            self.u = {a: member_effect(rows, w, cfg['tau']) for a, rows in per.items()}
            offs = [self.u[a][0] for _, a, *_ in accept]
            w, var = map_logistic(X, y, wts, offs, self.mu_a, sd_a)
            self.w, self.w_var = w, var
            Xs = [x_for(*intro_pair[i]) for i, _, _ in success]
            ys = [t for _, t, _ in success]
            # success weights: only the intercept is updated online (few events per episode)
            sd_s = [cfg['prior_sd']] + [1e-3] * 7
            self.c, _ = map_logistic(Xs, ys, [1.0] * len(Xs), [0.0] * len(Xs), self.mu_s, sd_s)
            n = sum(c[1] for c in respond.values())
            k = sum(c[0] for c in respond.values())
            r0 = pri['response_rate']
            self.pooled_rho = (r0 * 10 + k) / (10 + n)
            self.respond = respond
        else:
            self.w, self.w_var, self.u, self.c = self.mu_a, [cfg['prior_sd'] ** 2] * 8, {}, self.mu_s
            self.pooled_rho, self.respond = pri['response_rate'], {}
        self.n_accept, self.n_success = len(accept), len(success)
        self.sample = None
        if cfg['thompson'] and cfg['online']:
            rng = random.Random(1000003 * self.day + 17)
            self.sample = rng

    def rho(self, a):
        k, n = self.respond.get(a, (0, 0))
        m = self.cfg['rho_strength']
        return (self.pooled_rho * m + k) / (m + n)

    def weights(self):
        """Posterior mean, or one Thompson draw for today's decisions."""
        if not self.sample:
            return self.w, {a: u for a, (u, _) in self.u.items()}
        rng = self.sample
        w = [m + rng.gauss(0, math.sqrt(v)) for m, v in zip(self.w, self.w_var)]
        u = {a: m + rng.gauss(0, math.sqrt(v)) for a, (m, v) in sorted(self.u.items())}
        return w, u

    def pair_value(self, a, b, g, w, u):
        """Return (p_ab, p_ba, value) for members a,b with agreement vector g."""
        x = [1] + g
        z = sum(p * q for p, q in zip(w, x))
        p_ab = sigmoid(z + u.get(a, 0.0))
        p_ba = sigmoid(z + u.get(b, 0.0))
        q = sigmoid(sum(p * q for p, q in zip(self.c, x)))
        ra, rb = self.rho(a), self.rho(b)
        # Both must respond and accept, then both must answer the second-meeting question.
        value = (ra * rb) ** 2 * p_ab * p_ba * q
        return p_ab, p_ba, value
