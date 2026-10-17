# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Team **Laplace** (Kumar Ankit, sole member) in the IIT Madras x Romeo & Juliet / Vouchsafe **Sequential Matching Problem** hackathon. This is a clone of the official participant kit (release 1.0.0). The repo docs are the authoritative rules — quote `PROBLEM_STATEMENT.md` / `docs/*.md` rather than inventing rules. Technical questions go to the upstream repo's Issues tab (Question template), never with personal data.

Deadlines (IST): Round 1 research PDF via Google Form **9 Oct 2026 23:59**. The final build deadline was moved to **17 Oct 2026 23:59**, and submission is now through a Google Form (fields recorded in `submission/round2_form_answers.md`). Nothing can be fixed after the deadline, and one invalid private episode makes the entry ineligible.

`workshop-material/` is the user's untracked organiser workshop material (PDF/PPTX/MP4) — reference only, don't commit.

## Commands

The kit is stdlib-only Python 3.10+ (use `python3` on this Mac). No install step.

```bash
python3 -m unittest -v                         # all tests (test_kit, test_release, test_runner)
python3 -m unittest test_kit.ContractTests.test_temporal_integrity   # single test
python3 verify_data.py                         # checksum + leakage checks on data/
python3 evaluate.py --seeds 101 --output results/first_run.json      # one episode, current policy.py (greedy)
python3 evaluate.py --baseline no_asks --seeds 101,102,103 --variants all --output results/no_asks.json
python3 evaluate.py --policy my_policy.py --seeds 101 --variants sparse --output results/x.json
docker build -t sequential-policy:1.0 . && python3 evaluate.py --image sequential-policy:1.0 --seeds 101 --output results/container.json
```

Own helpers: `python3 scripts/summarise_results.py results/*.json` (per-variant funnel table) and `python3 scripts/baseline_stats.py` (seed-bootstrap CIs). Notes live in `notes/`, the Round 1 proposal/form in `research/`.

`evaluate.py` launches `policy.py` as a fresh subprocess twice per day (ask + match) for 60 days, so one episode = 120 process spawns; full `--variants all` × 3 seeds runs take minutes. `--baseline` is passed to the policy as a CLI flag, so any replacement `policy.py` must still accept `--baseline`. `results/` is git-ignored — final results/report must be committed explicitly (e.g. `git add -f`). CI (`.github/workflows/verify.yml`) runs tests, verify_data, a smoke eval and a container eval on Python 3.10 and 3.12.

## Architecture

- **`kit.py`** — everything about the world. `generate(seed, n, pool_id, variant)` builds a synthetic population (hidden `truth`, `bias`, `response_rate`, `second_bias`, `exit_day` per member). `eligibility(a, b)` is the reciprocal hard-constraint check → `feasible` / `infeasible` / `needs_clarification` (any missing HARD field blocks; a known failure wins over missing). `Simulator` holds state: `observe()` (only arrived members, no latent fields), `resolve_asks()` (budget 12/day; `constraints` bundle = 3, a SOFT field = 1; declined stays null), `advance(pairs)` (atomic batch validation, then samples responses/date/second-meeting events with delays), `metrics()`. Also `baseline_asks` / `baseline_match` used by `policy.py`.
- **`evaluate.py`** — the harness. `episode()` runs 60 decision days + 40 follow-up days through the JSON process boundary, then computes MSMI, coverage, waits, etc. `summarise()` averages per variant then across the six variants (primary score). `--image` switches to locked-down `docker run` (no network, 2 CPU, 1 GiB, read-only).
- **`policy.py`** — the only file the policy contract cares about: reads one JSON request on stdin, writes exactly `{"asks", "memory"}` or `{"pairs", "memory"}` on stdout. Memory is the only state carried between calls (≤1 MiB, null at episode start); no persistent filesystem. 10 s wall limit per call including startup. Diagnostics to stderr only.
- **`Dockerfile` / `.dockerignore`** — the image copies only `kit.py`, `policy.py` and `laplace/` (the `.dockerignore` whitelist covers `*.py` and `priors.json`). Any new module or model asset must be added to **both** files or the container eval fails.
- **`organise.py`** — organiser-side: generates 120 private worlds (6 variants × 20 random seeds, pool_id `private_pool`) with the same `generate()` and grades images. Public eval uses pool_id `evaluation`; never hard-code pool IDs.
- **`data/public_XX/`** — static day-30 snapshots (10 pools × 200). Split in `data_manifest.json`: train 01–06, validation 07–08, dev-test 09–10; keep disjoint. Snapshots are not rollouts of earlier decisions, so final-profile fields must not be attached to earlier outcomes (leakage).

## Team Laplace policy (laplace/)

- **Modules:**
  - `observe.py` builds the feasible graph and the censored labels.
  - `model.py` handles online MAP logistic fitting around the weak priors in `priors.json`.
  - `asks.py` handles clarification.
  - `matching.py` runs greedy plus local improvement.
  - `core.py` holds the defaults dict, the validator and the empty-action fallback.
- **Entry points:** `policy.py` with no arguments runs Laplace; `--baseline` runs a baseline. `laplace_policy.py` is the local entry for `evaluate.py --policy`. `LAPLACE_CFG` holds ablation overrides.
- **Fast harness:** `scripts/fast_eval.py` gives action digests identical to `evaluate.py`.
- **Experiments:**
  - `scripts/run_ablations.py --set validation|reporting [--keep] [--outdir DIR]`
  - `scripts/analyse.py`, `scripts/calibration.py`, `scripts/matching_gap.py` (needs SciPy)
- **Declared seeds:** training 1000–1059, validation 2000–2019, reporting 101–120.
- **Key finding:** the feasible graph is tiny (about 2.6 partners per feasible member), so every clarifying policy ties on MSMI. See `report/REPORT.md`.
- **OneDrive:** the repo lives in OneDrive, and syncing big result files slows runs. Write large outputs with `--outdir` outside the repo, then copy them in.

## Simulator mechanics worth knowing (from kit.py)

- Variants: `development`, `sparse` (12 single-zone geographies), `cold_start` (fewer complete profiles), `delayed` (+5–12 days before dates), `shift` (different soft-field outcome weights, lifestyle negative), `drift` (acceptance penalty from day 35).
- An introduction makes both people unavailable for 8 days; after mutual Yes they stay busy until date + 6; a mutual second-meeting Yes pauses (retires) them. Repeated pairs are forbidden.
- MSMI counts a pair only if a date happens ≤30 days after assignment and both say Yes within 3 days of the date. Denominator = members arrived by day 59 (fixed, regardless of how many are served).
- Members with any `declined` hard field can never become feasible — asking them for `constraints` wastes budget.

## Hard rules for policy code

- Use only the request JSON (`state`, `memory`) plus declared, committed training assets. Never read simulator internals, reconstruct hidden truths/seeds from IDs or generator code, or use future outcomes — this disqualifies regardless of score. Learning simulator behaviour from your own rollouts on declared training seeds is fine; document those seeds.
- Every returned pair must pass `kit.eligibility(...) == 'feasible'`, both members `available`, not previously introduced, no member reused in the batch. Asks: only available members, no duplicates, total cost ≤ `ask_budget_remaining`. Handle empty populations/graphs.
- Keep it deterministic (fixed inference seed) and offline; prefer stdlib/numpy/scipy, pin versions in `requirements.txt` and the Dockerfile if added.
- Missing feedback ≠ rejection; unresolved introductions are right-censored, not failures.
- Only report results that were actually run; cite real papers only.
