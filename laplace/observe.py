"""Normalise one observable state: candidate graph, soft-field agreement and matured feedback labels."""
import itertools
from kit import HARD, SOFT, eligibility

HORIZON = 60          # decision days in the published protocol
RESPONSE_WINDOW = 7   # introduction response deadline (PS section 6)
DATE_WINDOW = 30      # date must happen within 30 days of assignment (PS section 8)
SECOND_WINDOW = 3     # second-meeting answers within 3 days of the date (PS section 8)


def pair_key(a, b):
    return '|'.join(sorted((a, b)))


def agreement(fa, fb):
    """+1 both observed and equal, -1 both observed and different, 0 if either is unknown."""
    out = []
    for k in SOFT:
        x, y = fa.get(k), fb.get(k)
        out.append(0 if x is None or y is None else (1 if x == y else -1))
    return out


def hard_state(m):
    """'complete', 'askable' (missing but not declined) or 'blocked' (a declined hard field)."""
    if any(m['field_status'].get(k) == 'declined' for k in HARD):
        return 'blocked'
    if any(m['fields'].get(k) is None for k in HARD):
        return 'askable'
    return 'complete'


class View:
    """Everything the decision layers need, derived from the request state only."""

    def __init__(self, state):
        self.state = state
        self.day = state['day']
        self.members = state['members']
        self.by_id = {m['member_id']: m for m in self.members}
        self.available = [m for m in self.members if m['available']]
        self.past = {pair_key(i['user_a'], i['user_b']) for i in state['introductions']}
        self.days_left = max(0, HORIZON - self.day)
        self._edges = None

    def feasible_edges(self):
        """All currently valid introductions: available, new, reciprocally feasible."""
        if self._edges is None:
            edges = []
            for a, b in itertools.combinations(self.available, 2):
                key = pair_key(a['member_id'], b['member_id'])
                if key in self.past:
                    continue
                if eligibility(a, b)['status'] == 'feasible':
                    edges.append((a['member_id'], b['member_id']))
            self._edges = edges
        return self._edges


def labels(state):
    """Matured labels from observable feedback; unresolved outcomes are censored (omitted).

    Returns:
      accept: list of (intro, actor, target, yes:int, observed_day) for recorded Yes/No responses.
      respond: dict member -> [responded, matured] counts over introduction responses.
      success: list of (intro, label:int, observed_day) for mutually accepted pairs whose
               outcome window (date <= 30 days, two Yes answers <= 3 days after it) has closed.
    """
    day = state['day']
    by_intro = {}
    for e in state['feedback']:
        by_intro.setdefault(e['introduction_id'], []).append(e)
    accept, respond, success = [], {}, []
    for intro in state['introductions']:
        iid = intro['introduction_id']
        es = by_intro.get(iid, [])
        pair = (intro['user_a'], intro['user_b'])
        rs = {e['member_id']: e for e in es if e['event'] == 'introduction_response'}
        for actor in pair:
            e = rs.get(actor)
            if e is None:
                continue  # pending: censored
            c = respond.setdefault(actor, [0, 0])
            c[1] += 1
            if e['value'] is None:
                continue  # no response by the deadline: not a preference label
            c[0] += 1
            target = pair[1] if actor == pair[0] else pair[0]
            accept.append((iid, actor, target, int(e['value'] == 'yes'), e['observed_day']))
        if not (len(rs) == 2 and all(e['value'] == 'yes' for e in rs.values())):
            continue
        dates = [e for e in es if e['event'] == 'date_happened']
        if not dates:
            if day - intro['assigned_day'] > DATE_WINDOW:
                success.append((iid, 0, day))  # no date inside the window
            continue
        d = dates[0]
        if d['value'] is not True or d['occurred_day'] - intro['assigned_day'] > DATE_WINDOW:
            success.append((iid, 0, d['observed_day']))
            continue
        ss = [e for e in es if e['event'] == 'second_meeting_intention']
        if len(ss) < 2:
            # After the 3-day window every timely answer is already observable, so a missing one
            # rules the outcome out; before that the outcome is still censored.
            if day - d['occurred_day'] > SECOND_WINDOW:
                success.append((iid, 0, day))
            continue
        ok = all(e['value'] == 'yes' and e['occurred_day'] is not None
                 and e['occurred_day'] - d['occurred_day'] <= SECOND_WINDOW for e in ss)
        success.append((iid, int(ok), max(e['observed_day'] for e in ss)))
    return accept, respond, success
