# Understanding: The Sequential Matching Problem (release 1.0.0)

Sources: `PROBLEM_STATEMENT.md`, `docs/POLICY_INTERFACE.md`, `docs/DATA_CONTRACT.md`, `docs/SUBMISSION.md`, `docs/FAQ.md`, `examples/REPORT_GUIDE.md`, and the public `kit.py` / `evaluate.py` source. Section numbers (§) refer to `PROBLEM_STATEMENT.md`.

## 1. The task

Each simulated day, a policy process is called twice through a JSON stdin/stdout protocol:

1. **Ask phase**: return `asks`, a list of `{member_id, field}`. `constraints` reveals all 11 hard fields for 3 units; one named soft field costs 1 unit. The budget is 12 units/day (§6). Answers arrive immediately, and **declined answers stay null forever**.
2. **Match phase**: on the refreshed state, return `pairs`, a set of non-overlapping, reciprocally feasible, never-before-introduced pairs of available members. An empty list means "wait".

There are 60 decision days, then 40 follow-up days with no actions. The policy is restarted on every call. Only `memory` (≤1 MiB JSON) carries state between calls. Each call has 10 s including startup. Assessment runs in an offline container with 2 CPU and 1 GiB (§11).

## 2. Scoring (§10)

- **Primary score** = the mean over the 6 scenario families (standard, sparse, cold_start, delayed, shift, drift) of the mean per-episode `MSMI per 100 arrived members = 100·M/N`. Each family has 20 private seeds, so 120 episodes in total.
- **M** = introductions where (a) both people replied Yes to the introduction, (b) a date happened **≤30 days after assignment**, and (c) **both** said Yes to a second meeting **within 3 days of the date**. One pair counts once.
- **N** = members who arrived by decision day 59, including those who left or were never served. N is fixed, so serving fewer people does not raise the score.
- Tie-breaks: coverage (distinct members introduced / N), then mutual acceptances per 100, then lower ask cost, then lower inference time.
- **Eligibility is all-or-nothing:** any invalid ask or pair, malformed JSON, crash or timeout in any of the 120 episodes makes the submission unranked.

## 3. Hard constraints (§7, `kit.eligibility`)

Reciprocal age range, `who_to_meet` gender, same relationship_structure, partner_smoking vs smoking (occasional counts as smoking), partner_children vs has_children, and no explicit yes/no conflict on wants_children. Each person's zone must be in the other's acceptable_zones, and their schedules must overlap. **Any missing hard field on either side gives `needs_clarification`, which blocks the pair.** A known failure takes precedence over missing data.

Operational rules: no repeated pair, one concurrent introduction per person, and each person used at most once per batch.

## 4. Simulator mechanics relevant to design (read from public `kit.py`)

These are structural facts about the public development simulator. The private worlds are produced by the same `generate()` with fresh seeds (`organise.py`). The policy may not reconstruct seeds or hidden values (§11). The plan is to *learn* response/acceptance behaviour from our own rollouts on declared training seeds rather than hard-coding the generator's coefficients.

- **Population:** 200 members. About 70% arrive on day 0 and the rest by day 20. About 12% exit between days 22 and 60. Gender mix is roughly 46/46/8 (woman/man/non_binary), and about 70% want exactly one gender.
- **Missingness:** about 35% of profiles are complete (20% in cold_start). Otherwise each field is observed with p≈0.10. About 7% of unobserved fields are `declined`.
  - In public seed 101 (development), 81/200 have complete hard fields and **53/200 have at least one declined hard field, which makes them permanently unmatchable**.
  - cold_start seed 101: 54 complete, 64 blocked.
- **Busy windows:** an introduction blocks both people for 8 days. After mutual acceptance they stay busy until date+6. A mutual second-meeting Yes "pauses" (retires) the pair. Over 60 days a person can receive about 7 introductions at most.
- **Funnel stochasticity:**
  - Each person replies with a member-specific response rate (U(0.55, 0.98)).
  - Acceptance is a logistic function of a member bias, soft-field agreement (relationship_goal ≫ pace > lifestyle > conversations) and pair noise.
  - A date happens with p=0.78, 1–14 days after the later reply.
  - The second-meeting reply arrives 1–5 days after the date, so only about 60% per person are on time (≤3 days).
  - Second-meeting Yes depends on a member bias and on relationship_goal agreement.
- **Variants:**
  - `sparse`: 12 single-zone geographies, so the feasible graph is very thin.
  - `cold_start`: fewer complete profiles.
  - `delayed`: dates pushed 5–12 days later, so many land after the 30-day window.
  - `shift`: soft-field weights change, and lifestyle agreement becomes *negative*.
  - `drift`: an acceptance penalty from day 35.

## 5. Implications

1. **Clarification sets who becomes matchable, and how early.** About 25–30% of members are blocked by a declined field, so asking them is wasted. Another ~30% need a constraints bundle, and at most 4 bundles fit in a day. Ask *order* sets how early people become matchable. The total budget is not binding after the first weeks (see §6).
2. The outcome is downstream and rare (low single digits per episode), so per-episode comparisons are noisy. Use many seeds, paired comparisons and the full funnel.
3. Each person has only ~7 introduction slots. A low-quality introduction blocks both people for 8 days and permanently uses up that pair, so waiting has a real option value. The time limit pushes the other way: slots unused by day 59 are worthless.
4. Soft-field agreement, especially relationship_goal, is likely the strongest observable signal. Asking relationship_goal costs only 1 unit. The `shift` variant shows the weights can change, so a fixed hand-tuned score could hurt there. Weights should be learned online from feedback or kept robust.
5. Learnable member-level signals: response rate, from no-response history, and acceptance propensity, from Yes/No history. Both are observable in `feedback` and must be treated as right-censored until the deadline passes.
6. Late introductions still count if the date falls within 30 days. The 40-day follow-up gives enough room, but in `delayed` late or slow pairs are more likely to miss the window.

## 6. Baseline results (actually run, 9 Oct 2026)

Command: `python3 evaluate.py --baseline {greedy,no_asks,random} --seeds 101,102,103 --variants all --output results/<b>.json`, in trusted local mode. All 54 episodes were valid. The full table is in `results/baseline_summary.md` and was produced by `scripts/summarise_results.py`.

| Baseline | Primary score (MSMI/100) | Mean coverage | Mean ask cost |
|---|---:|---:|---:|
| greedy | 0.389 | 0.372 | 207 |
| random-feasible | 0.278 | 0.377 | 207 |
| no_asks | 0.139 | 0.143 | 0 |

These are 3 seeds per variant, with 0–2 MSMI events per episode (`results/baseline_stats.txt`, `scripts/baseline_stats.py`).

- **greedy vs random:** greedy's point estimate is 0.11 MSMI/100 higher, about 40% relative (14 vs 10 events in 18 episodes). The paired seed-bootstrap 95% CI is [−0.03, 0.25] and includes zero, and the per-policy CIs overlap ([0.31, 0.50] vs [0.14, 0.44]). The effect may be real but these runs cannot resolve it. A bootstrap over 3 seeds is itself crude.
- **no_asks:** its CI [0.06, 0.22] does not overlap greedy's, and its coverage is about 3× lower.

### Where each baseline fails

- **Funnel (greedy, development, mean of 3):** 100 assignments → 13 mutual acceptances → 10 dates → 1 MSMI.
  - Roughly 1% of introductions turn into the outcome.
  - About 60 of the ~200 directional responses are missing.
  - The biggest losses happen at mutual acceptance (~13%) and after the date (~10%, partly the 3-day on-time rule).
- **Coverage stays at about 43%:** ~113 of ~200 arrived members are never introduced, even with clarification. A large share are blocked permanently by a declined hard field (53 in development seed 101). The rest have no feasible partner.
- **Clarification budget is not binding after the first weeks.** greedy/random spend about 200 of the 720 available units. `baseline_asks` runs out of askable members, so roughly 500 units per episode go unused that could buy soft fields such as relationship_goal. The real constraint is *latency*: 4 bundles/day means new members wait days to become matchable, and asks are made in list order, not by expected value.
- **greedy** scores pairs by counting equal observed soft fields. This ignores unknown fields, which count as 0, and ignores member-level response and acceptance history. It never waits, so it spends 8-day slots on weak pairs. It also allocates greedily rather than with a global matching.
- **random-feasible** has the same coverage as greedy, so pair quality makes only a small difference at n=3 (random has slightly fewer MSMI).
- **no_asks** can only match profiles that were complete at arrival, giving 18% coverage. It scores 0 in sparse and cold_start.
- **sparse:** all baselines reach only ~14% coverage and ~18 assignments, because the feasible graph is tiny. Each feasible edge is precious, so ask order and edge choice matter most here.
- **cold_start:** ask cost rises to 244, coverage is 36%, and waits are longer (median first wait ~5 days compared with ~3).
- **delayed / shift / drift** use the same populations as development for a given seed, so coverage is identical. Differences appear only in the funnel, and 3 seeds are too few to resolve them.

### Implication for evaluation

The MSMI signal is about 1 event per episode, which is Poisson-noisy. Method development should therefore:

- use paired seeds;
- track the higher-count proxies (mutual acceptances, dates) alongside MSMI;
- run ≥20 seeds per variant before claiming any MSMI difference.

A full 6×20 run takes about 1 h per policy in subprocess mode (~30 s/episode). An in-process fast evaluator that calls `decide()` directly is needed for iteration, with final numbers always re-run through `evaluate.py`.

## 7. Workshop material (workshop-material/, read 9 Oct 2026)

- **W1, Exploration/Exploitation (J. Liu).**
  - Selective feedback means greedy can lock in a wrong belief. Use Thompson sampling.
  - VOI = 0 unless some answer changes the action (Howard 1966).
  - Discount old evidence under drift (Garivier & Moulines 2011).
  - Spend some introductions on newcomers.
  - Check calibration.
- **W1 peer question bank.** Participant-compiled; it says it was not checked against the recording. Its inverse-propensity suggestion is *not* usable on the supplied files: propensities are null (PS §9).
- **W2, Robust ML from messy data** (car-park forecasting, slides only; the 1.4 GB video was not transcribed).
  - Compare baseline, best model and a perfect-information ceiling.
  - Use grouped hold-out validation.
  - Use a failure taxonomy.
  - Evaluate with a business metric.
- **W3, Hard rules, soft goals.**
  - Keep spec, prediction and decision separate, with a post-solve validator.
  - Write the hard/soft classification down.
  - Known-forbidden is not the same as not-observed.
  - The multiplier λ is the shadow price of a binding budget.
  - A product of marginal acceptances needs conditional independence; learning only from accepted matches biases.
  - The handout explicitly ties these to "learned pair scores s_ij optimised subject to binary assignments, eligibility and capacities".
