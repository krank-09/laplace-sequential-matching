"""In-process replica of evaluate.episode() for fast, parallel experiments.

Same simulator, same 60 + 40 day protocol, same metrics and the same action digest as evaluate.py, but
the policy is called as a function (memory still round-trips through JSON). Because the policy is
deterministic, digests match the subprocess harness; scripts/check_fast_eval.py verifies this.
Final reported numbers are re-run through evaluate.py.

usage: python3 scripts/fast_eval.py --policy laplace --seeds 2000-2019 --variants all \
           --cfg '{"online": false}' --output results/x.json [--workers 8]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import VARIANTS, summarise, VERSION  # noqa: E402
from kit import Simulator, generate  # noqa: E402
import policy  # noqa: E402


def run_episode(args):
    seed, variant, mode, cfg, keep = args
    world = generate(seed, 200, 'evaluation', variant)
    if os.environ.get('LAPLACE_REVEAL') == '1':
        # ANALYSIS-ONLY full-information reference (harness side, never in the policy): every answer the
        # member would disclose is observable from arrival; declined answers stay unknown.
        for m in world['members']:
            for k, v in m['truth'].items():
                if m['field_status'][k] != 'declined':
                    m['fields'][k] = v
                    m['field_status'][k] = 'observed'
    sim = Simulator(world)
    memory, digest, inference = None, hashlib.sha256(), 0.0
    for _ in range(60):
        for phase in ('ask', 'match'):
            request = {'schema_version': VERSION, 'phase': phase, 'state': sim.observe(), 'memory': memory}
            started = time.perf_counter()
            response = policy.decide(json.loads(json.dumps(request)), mode, cfg)
            inference += time.perf_counter() - started
            memory = json.loads(json.dumps(response['memory'], allow_nan=False))
            action = response['asks' if phase == 'ask' else 'pairs']
            digest.update(json.dumps([phase, sim.day, action], sort_keys=True).encode())
            if phase == 'ask':
                sim.resolve_asks(action)
            else:
                sim.advance(action)
    arrived = {m['member_id']: m['arrived_day'] for m in sim.observe()['members'] if m['arrived_day'] <= 59}
    for _ in range(40):
        sim.advance([])
    result = sim.metrics()
    first = {}
    for intro in sim.introductions:
        for member in (intro['user_a'], intro['user_b']):
            first.setdefault(member, intro['assigned_day'])
    waits = sorted(first[i] - arrived[i] for i in first)
    result.update(valid=True, arrived_members=len(arrived),
                  msmi_per_100_arrived_members=100 * result['mutual_second_meeting_intention'] / max(1, len(arrived)),
                  served_members=len(first), unserved_members=len(arrived) - len(first),
                  coverage=len(first) / max(1, len(arrived)),
                  mutual_acceptances_per_100=100 * result['mutual_acceptances'] / max(1, len(arrived)),
                  first_introduction_wait_days=waits, inference_seconds=inference,
                  action_sha256=digest.hexdigest(), seed=seed, variant=variant)
    if keep:
        # Observable end-of-episode data for calibration / failure analysis (never fed back to the policy).
        final = sim.observe()
        result['final'] = {'memory': memory, 'introductions': final['introductions'],
                           'feedback': final['feedback'], 'arrived': arrived,
                           'members': [{k: m[k] for k in ('member_id', 'arrived_day', 'field_status')}
                                       for m in final['members']]}
    return result


def parse_seeds(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            out += list(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def run(mode, seeds, variants, cfg=None, workers=8, keep=False):
    jobs = [(s, v, mode, cfg or {}, keep) for v in variants for s in seeds]
    with Pool(workers) as pool:
        rows = pool.map(run_episode, jobs, chunksize=1)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='laplace', choices=['laplace', 'greedy', 'no_asks', 'random'])
    p.add_argument('--seeds', default='2000-2019')
    p.add_argument('--variants', default='all')
    p.add_argument('--cfg', default='{}')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--keep', action='store_true', help='store end-of-episode observable data for analysis')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    variants = list(VARIANTS) if a.variants == 'all' else a.variants.split(',')
    cfg = json.loads(a.cfg)
    t = time.time()
    rows = run(a.policy, parse_seeds(a.seeds), variants, cfg, a.workers, a.keep)
    out = {'release': VERSION, 'mode': 'fast_in_process', 'policy': a.policy, 'cfg': cfg,
           'decision_days': 60, 'follow_up_days': 40, 'members_per_episode': 200,
           'episodes': rows, 'summary': summarise(rows), 'wall_seconds': round(time.time() - t, 1)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, allow_nan=False) + '\n')
    s = out['summary']
    print(json.dumps({'policy': a.policy, 'cfg': cfg, 'primary': s['primary_score'],
                      'coverage': s['overall']['coverage'], 'mutual/100': s['overall']['mutual_acceptances_per_100'],
                      'wall_s': out['wall_seconds']}))


if __name__ == '__main__':
    main()
