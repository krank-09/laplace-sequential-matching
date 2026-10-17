# Round 2 progress log (Team Laplace)

Deadline: 18 Oct 2026, 23:59 IST, via the Final submission issue form with a pinned 40-character SHA. The repo must be public by then.

## Done (17 Oct)

**Policy**
- Lives in `laplace/` (observe / model / asks / matching / core). `policy.py` defaults to Laplace and keeps the baselines; `laplace_policy.py` is the local entry point.
- Dockerfile and `.dockerignore` now include `laplace/`. The image builds (`sequential-policy:laplace`).
- Tests: the kit's 22 plus 5 in `test_laplace.py`, all passing.

**Experiment tooling**
- `scripts/fast_eval.py` produces the same action digests as `evaluate.py` (checked on seed 101, development and sparse).
- Priors were fitted on training seeds 1000–1059 (`laplace/priors.json`), 360 episodes.

**First design (v1)**
- Soft asks fired only when the partner already showed the field, so about 500 budget units went unused.
- Validation: MSMI 0.417, mutual 5.19 (kept in `results/validation_v1/`).

**Revised soft asks ('broad')**
- No gain: full scores MSMI 0.383, mutual 5.27.

**Key diagnostic**
- The feasible graph is tiny: about 109 feasible pairs per development episode, mean degree 2.6. Sparse has about 12.
- Every clarifying policy makes about 74 assignments per episode, so the problem is supply-limited.
- Cardinality-first weighting makes no difference.

**Validation (seeds 2000–2019), paired differences against full**
- Better than full, CI excluding zero:
  - thompson_off: +0.079
  - soft_asks_off: +0.062
  - no_discount: +0.033
- No other differences.
- No mutual-acceptance differences.
- no_asks is far worse.

## In progress / next

1. Validation runs for scarcity_off, wait_on, cardinality_first, greedy_matcher, exact_matcher, then the combined candidates `final` and `final_fixed`.
2. Choose the submitted config from validation only.
3. Reporting seeds 101–120:
   - `run_ablations.py --set reporting --keep`, covering baselines, full, final, key ablations and the full-info references.
   - Confirm digests through `evaluate.py`.
4. `scripts/matching_gap.py`, then `scripts/calibration.py` on the reporting `--keep` run.
5. Container check (`evaluate.py --image`).
6. Write `report/REPORT.md` and the PDF, then commit (`git add -f results/...`) and push.
7. User actions: make the repo public, file the Final submission issue, reply to confirm participation.
