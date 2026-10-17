# Round 2 final submission: Google Form answers (Team Laplace, Kumar Ankit)

Paste each block into the matching field. Build commit: `79ef7f3efc35da24a6173cc3359fcd2ecc9875c5`.

---

## GitHub Repository Link*

https://github.com/krank-09/laplace-sequential-matching

## Demo Link (Optional)

(leave blank)

## Presentation or Report Link*

https://github.com/krank-09/laplace-sequential-matching/blob/main/report/REPORT.pdf

## Brief Solution Summary*

Team Laplace built an offline, standard-library Python policy for the Sequential Matching Problem. Each day it decides who to clarify, which reciprocally feasible pairs to introduce, and when to wait, using only the observable JSON state. It passes every hard rule and has run 0 invalid episodes across about 3,600 simulated episodes (training, validation and reporting) plus subprocess and container checks.

The policy has four layers:

1. A feasible-graph builder that uses the kit's reciprocal eligibility rules and censors pending outcomes.
2. A pair-value model, refit every call from the episode's own feedback around weak priors learned on declared training seeds.
3. Ranked clarification and a general-graph maximum-weight matching.
4. A final validator with a crash-safe empty-action fallback.

The main research finding is that this environment is supply-limited. Each member who has any feasible partner has only about 2–3 over the whole episode, so every policy that clarifies ends up introducing most feasible pairs.

On fresh reporting seeds (101–120, all six variants, 120 episodes):

- **MSMI per 100 arrived members:** 0.404, against greedy 0.433, random 0.367 and no-clarification 0.183. Differences among the clarifying policies are not statistically significant.
- **Mutual acceptances per 100:** 5.81 vs 5.19 for greedy, a paired gain of +0.62 with 95% CI [+0.35, +0.89]. This gain was not reproduced on validation seeds.

The report answers the three Round 1 reviewer questions with ablations, an exact-solver comparison and calibration checks.

## Methodology and Algorithms*

- **Hard vs soft.** Reciprocal eligibility (kit.eligibility), availability, one introduction per person per batch, no repeat pairs and the ask budget are hard rules. Predicted value, coverage, waiting and ask cost only rank feasible actions. Members with a declined hard field are never asked and never inferred.
- **Clarification.** Constraint bundles go to available members with missing (not declined) hard fields, ranked by unlock value: how many partners they could become feasible with.
- **Pair value.**
  - P(a says Yes | a responds) = logistic(w·g + u_a), where g is three-state soft-field agreement (agree / disagree / unknown) and u_a is a member effect.
  - Response propensities use Beta-binomial shrinkage.
  - A post-acceptance factor estimates P(MSMI | mutual Yes).
  - Value = (rho_a·rho_b)² · p_ab · p_ba · q_ab.
- **Learning.**
  - Online MAP (Laplace) logistic updates every call, from matured feedback only. Pending outcomes are censored, and a missing response is never treated as a No.
  - Features are the soft-field agreement as observed at assignment time, stored in policy memory, so later-clarified answers never leak into earlier outcomes.
  - Weak prior means (sd 0.6) were fitted on 360 of our own rollouts on training seeds 1000–1059.
- **Allocation.**
  - Daily maximum-weight general-graph matching: greedy, then local improvement (augmenting moves and 2-edge swaps).
  - A scarcity weight favours members with few options.
  - This was compared offline against an exact 0/1 integer program (SciPy HiGHS), which is never shipped.
- **Design selection.** Thompson sampling, broad soft-field asks, evidence discounting, a waiting threshold and cardinality-first matching were all implemented and tested. All were dropped or left off on validation evidence (seeds 2000–2019), following a parsimony rule.

## Baselines and Experimental Results*

**Baselines.** The supplied greedy, no-clarification and random-feasible baselines, run on identical seeds and all six scenario families. There were 20 fresh reporting seeds per family (101–120): 120 episodes per policy, all valid.

**Metrics.**

- MSMI per 100 arrived members (primary) with seed-bootstrap 95% CIs
- coverage
- the full funnel: assignments, mutual acceptances, dates, MSMI
- first-introduction waits and unserved members
- ask cost
- missing feedback
- runtime
- paired per-seed differences

**Results (reporting seeds).**

| Policy | MSMI/100 [95% CI] | Mutual acceptances/100 | Coverage | Ask cost |
|---|---|---|---|---|
| Submitted | 0.404 [0.325, 0.492] | 5.81 | 0.344 | 197 |
| Greedy | 0.433 [0.354, 0.517] | 5.19 | 0.345 | — |
| Random | 0.367 [0.300, 0.433] | 5.52 | 0.345 | — |
| No clarification | 0.183 [0.138, 0.233] | 1.73 | 0.123 | — |

- Paired against greedy: MSMI −0.03 [−0.13, +0.07]; mutual acceptances +0.62 [+0.35, +0.89]; dates +0.45 [+0.16, +0.73].

**Ablations.**

- **Online learning vs fixed prior:** MSMI +0.04 [−0.02, +0.10]. The weights do adapt correctly: in the shift variant the lifestyle weight flips sign. Outcomes are unchanged.
- **Exact matcher inside the policy:** −0.025 [−0.06, +0.01], not significant. On the 360 real daily graphs (mean 4.3 edges) the shipped matcher matched the exact optimum on every graph. On dense random graphs it loses 1–2% of weight, while greedy loses 2.5–3%.
- **Scarcity adjustment off:** −0.029 [−0.000, +0.062], borderline.
- **Calibration:**
  - Directional estimates are calibrated on average (ECE 0.05) but overconfident at the top, and their Brier score is no better than a constant rate, so they barely discriminate.
  - The two directions are dependent (residual correlation +0.16), so the product of directional probabilities mis-states joint acceptance.
  - P(MSMI) is under-predicted (0.8 vs 1.1 per 100 introductions) and only weakly discriminative.
- **Checks:** the official evaluate.py reproduces our fast-harness action digests 6/6. The Docker evaluation is valid, at 1.05 s per call against the 10 s limit.

Tables and per-variant results are in report/REPORT.pdf and results/.

## Limitations and Improvements from Round 1*

**What changed since Round 1.**

1. The Round 1 plan was implemented as a staged policy, with one ablation per stage and all design choices fixed on validation seeds before reporting.
2. **The first soft-ask rule failed, and was later dropped.** It asked a field only when a partner already showed it, so about 500 budget units per episode went unused. A broader rule spent the budget, but neither version improved outcomes, so soft asks are off in the submission.
3. **Thompson sampling was removed.** With almost no real choice among edges, exploration added noise and showed slight harm.
4. **A combination check prevented a false claim.** Three single-component "wins" did not reproduce when combined, so I treated them as noise instead of claiming them.
5. **New analyses were added:**
   - an exact-solver gap study across graph densities;
   - decision-time calibration logging;
   - an analysis-only full-information reference;
   - a failure taxonomy of who is left waiting and why.

**Limitations.**

- MSMI events are rare (about 0.8 per episode), so effects below about 15% of the score cannot be resolved with 120 episodes per policy.
- The joint acceptance model ignores the within-pair dependence it detected.
- The mutual-acceptance gain over greedy appears on reporting but not on validation seeds.
- Roughly half of unserved members are blocked by a declined hard field, and most of the rest have no feasible partner. No policy can serve either group.
- Results describe only this synthetic simulator.

## Setup Instructions, Dependencies and Dataset Details*

**Requirements.** Python 3.10+, standard library only, with no packages for inference. Docker image: python:3.12.7-slim, offline, CPU only. SciPy is used only by an offline analysis script (exact matcher) and is never in the image.

**Commands.**

```
git clone https://github.com/krank-09/laplace-sequential-matching
cd laplace-sequential-matching
python3 -m unittest -v
python3 verify_data.py
python3 evaluate.py --policy laplace_policy.py --seeds 101 --variants all --output results/check.json
docker build -t sequential-policy:laplace .
python3 evaluate.py --image sequential-policy:laplace --seeds 101 --output results/container_check.json
```

The container entrypoint (`python policy.py`, no arguments) runs the Laplace policy. To reproduce the experiments:

```
python3 scripts/fit_priors.py
python3 scripts/run_ablations.py --set validation
python3 scripts/run_ablations.py --set reporting --keep
python3 scripts/analyse.py --set reporting
python3 scripts/calibration.py --run results/reporting/submitted.json
```

**Data.** Only the official synthetic kit data and simulator; no external datasets.

**Declared seeds.**

- Training (prior means only): 1000–1059
- Validation (design choices): 2000–2019
- Reporting: 101–120
- Inference RNG: random.Random(1000003·day+17), deterministic

The static day-30 pools keep the manifest's train/validation/test split. The policy reads only the observable request JSON, never hidden simulator state.

**Asset.** laplace/priors.json, a few kB.

**Licences.** The kit is MIT and the data is under DATA_LICENSE; Team Laplace's code is MIT (LICENSE-LAPLACE).

## Did you utilize any AI-based tools?*

Select: **Yes, for both coding and report writing**

## If you used AI tools, please briefly specify which tools were used and how they contributed*

Anthropic's Claude, via the Claude Code CLI. It was used to:

- read the official documentation;
- write and refactor the policy, the test suite and the experiment scripts;
- run the baseline, validation and reporting experiments;
- check cited references;
- draft the README, the technical report and these answers.

I set the research direction, reviewed the design decisions, and chose the submitted configuration from validation evidence. Every reported number was produced by the committed scripts and can be reproduced from the repository. No AI tool is used during the policy's inference.

## Confirmation*

Select **Yes, I confirm**. The repository is public and the report link resolves (checked: HTTP 200).
