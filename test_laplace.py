"""Checks for the Team Laplace policy: validity, censoring, matching, memory and timing."""
import json
import random
import sys
import time
import unittest

from evaluate import invoke
from kit import Simulator, generate, eligibility
from laplace.core import decide
from laplace.matching import max_weight_matching
from laplace.observe import labels


class LaplaceTests(unittest.TestCase):
    def test_empty_states(self):
        state = Simulator(generate(42, 0)).observe()
        for phase in ('ask', 'match'):
            out = decide({'phase': phase, 'state': state, 'memory': None})
            self.assertEqual(out[('asks' if phase == 'ask' else 'pairs')], [])

    def test_episode_actions_valid_all_variants(self):
        for variant in ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift'):
            sim, memory = Simulator(generate(7, 60, 'test', variant)), None
            for _ in range(25):
                r = decide({'phase': 'ask', 'state': sim.observe(), 'memory': memory})
                sim.resolve_asks(r['asks'])        # raises on any invalid ask
                memory = r['memory']
                r = decide({'phase': 'match', 'state': sim.observe(), 'memory': memory})
                state = sim.observe()
                by = {m['member_id']: m for m in state['members']}
                for a, b in r['pairs']:
                    self.assertEqual(eligibility(by[a], by[b])['status'], 'feasible')
                sim.advance(r['pairs'])            # raises on any invalid batch
                memory = json.loads(json.dumps(r['memory'], allow_nan=False))
                self.assertLess(len(json.dumps(memory)), 1024 * 1024)

    def test_matching_is_a_matching_and_beats_nothing(self):
        rng = random.Random(3)
        for _ in range(50):
            n = rng.randint(2, 40)
            edges = {(f'{a:02d}', f'{b:02d}'): rng.random() for a in range(n) for b in range(a + 1, n) if rng.random() < .3}
            pairs = max_weight_matching(edges)
            used = [v for p in pairs for v in p]
            self.assertEqual(len(used), len(set(used)))
            self.assertTrue(all(tuple(p) in edges for p in pairs))
            if edges:
                self.assertGreaterEqual(sum(edges[tuple(p)] for p in pairs), max(edges.values()) - 1e-12)

    def test_pending_outcomes_are_censored_and_no_response_is_not_a_label(self):
        intro = {'introduction_id': 'i1', 'user_a': 'a', 'user_b': 'b', 'assigned_day': 10, 'response_deadline_day': 17}
        fb = [{'introduction_id': 'i1', 'member_id': 'a', 'event': 'introduction_response', 'value': 'yes',
               'occurred_day': 12, 'observed_day': 12}]
        accept, respond, success = labels({'day': 13, 'introductions': [intro], 'feedback': fb})
        self.assertEqual([x[1:4] for x in accept], [('a', 'b', 1)])
        self.assertNotIn('b', respond)              # still pending: censored
        self.assertEqual(success, [])
        fb.append({'introduction_id': 'i1', 'member_id': 'b', 'event': 'introduction_response', 'value': None,
                   'missing_reason': 'no_response', 'occurred_day': None, 'observed_day': 17})
        accept, respond, success = labels({'day': 17, 'introductions': [intro], 'feedback': fb})
        self.assertEqual(len(accept), 1)            # a no-response is not a No
        self.assertEqual(respond['b'], [0, 1])

    def test_container_style_call_is_fast(self):
        sim = Simulator(generate(101, 200, 'evaluation', 'cold_start'))
        state = sim.observe()
        started = time.perf_counter()
        invoke([sys.executable, 'policy.py'], {'schema_version': '1.0.0', 'phase': 'ask', 'state': state, 'memory': None})
        self.assertLess(time.perf_counter() - started, 5)


if __name__ == '__main__':
    unittest.main()
