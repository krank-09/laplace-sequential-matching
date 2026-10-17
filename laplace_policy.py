"""Local entry point for `evaluate.py --policy laplace_policy.py`: runs the Laplace policy.

evaluate.py always appends `--baseline <name>` in local mode; it is accepted and ignored here.
Ablations: set LAPLACE_CFG to a JSON object of overrides (keys in laplace/core.py DEFAULT).
"""
import json
import os
import sys
from laplace.core import decide

if __name__ == '__main__':
    cfg = json.loads(os.environ.get('LAPLACE_CFG') or '{}')
    request = json.load(sys.stdin)
    print(json.dumps(decide(request, cfg), allow_nan=False))
