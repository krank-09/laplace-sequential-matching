"""Run the full policy, its ablations and the three baselines on one seed set (fast in-process harness).

Seed sets (declared): training 1000-1059 (priors only), validation 2000-2019 (design choices),
reporting 101-120 (fresh; final tables). Each run writes results/<set>/<name>.json.

usage: python3 scripts/run_ablations.py --set validation [--only full,online_off]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate import VARIANTS, summarise  # noqa: E402
from fast_eval import run  # noqa: E402

SEEDS = {'validation': list(range(2000, 2020)), 'reporting': list(range(101, 121))}

# name -> (policy mode, cfg overrides, env overrides)
RUNS = {
    # Ablations are relative to the submitted defaults in laplace/core.py (decided on validation seeds):
    # online learning on, posterior mean (no Thompson), no soft asks, no discount, scarcity 1, local matcher.
    'greedy': ('greedy', {}, {}),
    'no_asks': ('no_asks', {}, {}),
    'random': ('random', {}, {}),
    'submitted': ('laplace', {}, {}),
    'online_off': ('laplace', {'online': False}, {}),               # fixed weak prior only
    'zero_prior': ('laplace', {}, {'LAPLACE_PRIORS': 'none'}),      # online learning from zero-mean prior
    'thompson_on': ('laplace', {'thompson': True}, {}),
    'soft_asks_on': ('laplace', {'soft_asks': True}, {}),
    'discount_on': ('laplace', {'gamma': 0.97}, {}),
    'baseline_asks': ('laplace', {'rank_asks': False}, {}),
    'scarcity_off': ('laplace', {'scarcity': 0.0}, {}),
    'wait_on': ('laplace', {'wait_kappa': 0.5}, {}),
    'greedy_matcher': ('laplace', {'matcher': 'greedy'}, {}),
    'exact_matcher': ('laplace', {'matcher': 'EXACT'}, {}),
    'round1_design': ('laplace', {'thompson': True, 'soft_asks': True, 'gamma': 0.97}, {}),
    # analysis-only references: the harness reveals all disclosed fields at arrival (not a policy)
    'reference_full_info': ('laplace', {}, {'LAPLACE_REVEAL': '1'}),
    'reference_full_info_greedy': ('greedy', {}, {'LAPLACE_REVEAL': '1'}),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', choices=list(SEEDS), default='validation')
    p.add_argument('--only', default='')
    p.add_argument('--keep', action='store_true')
    p.add_argument('--outdir', default='', help='write here instead of results/<set> (e.g. outside a synced folder)')
    a = p.parse_args()
    names = a.only.split(',') if a.only else list(RUNS)
    outdir = Path(a.outdir) if a.outdir else ROOT / 'results' / a.set
    outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        mode, cfg, env = RUNS[name]
        if cfg.get('matcher') == 'EXACT':
            from exact_matching import exact_matching
            cfg = {**cfg, 'matcher': exact_matching}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        t = time.time()
        rows = run(mode, SEEDS[a.set], list(VARIANTS), cfg, keep=a.keep)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cfg_out = {k: (v if not callable(v) else 'exact_milp') for k, v in cfg.items()}
        out = {'name': name, 'policy': mode, 'cfg': cfg_out, 'env': env, 'seed_set': a.set,
               'seeds': SEEDS[a.set], 'mode': 'fast_in_process', 'episodes': rows,
               'summary': summarise(rows), 'wall_seconds': round(time.time() - t, 1)}
        (outdir / f'{name}.json').write_text(json.dumps(out, allow_nan=False) + '\n')
        s = out['summary']
        print(json.dumps({'run': name, 'primary': round(s['primary_score'], 4),
                          'coverage': round(s['overall']['coverage'], 4),
                          'mutual/100': round(s['overall']['mutual_acceptances_per_100'], 3),
                          'ask_cost': round(s['overall']['ask_cost'], 1), 'wall_s': out['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
