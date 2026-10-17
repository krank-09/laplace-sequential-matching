## Calibration and diagnostics: submitted on reporting seeds (120 episodes)

**Directional acceptance P(Yes | responded)**: n=13752, mean predicted 0.484, observed 0.471, Brier 0.2497 (constant-rate 0.2492), log loss 0.694, ECE 0.048

| bin mean predicted | observed rate | n |
|---|---|---|
| 0.273 | 0.361 | 1375 |
| 0.370 | 0.395 | 1375 |
| 0.414 | 0.433 | 1375 |
| 0.447 | 0.479 | 1375 |
| 0.475 | 0.466 | 1376 |
| 0.498 | 0.465 | 1375 |
| 0.524 | 0.535 | 1375 |
| 0.551 | 0.506 | 1375 |
| 0.591 | 0.523 | 1375 |
| 0.697 | 0.550 | 1376 |

**Joint acceptance P(both Yes | both responded) predicted by p_ab*p_ba**: n=5295, mean predicted 0.248, observed 0.263, Brier 0.1943 (constant-rate 0.1940), log loss 0.579, ECE 0.052

| bin mean predicted | observed rate | n |
|---|---|---|
| 0.081 | 0.151 | 529 |
| 0.141 | 0.196 | 530 |
| 0.175 | 0.219 | 529 |
| 0.201 | 0.225 | 530 |
| 0.226 | 0.280 | 529 |
| 0.248 | 0.285 | 530 |
| 0.272 | 0.325 | 529 |
| 0.301 | 0.300 | 530 |
| 0.348 | 0.285 | 529 |
| 0.487 | 0.366 | 530 |

Correlation of the two directional residuals within a pair: +0.159 (n=5295). A positive value means a shared pair component; the product then under-predicts the joint.

**Reciprocal outcome P(MSMI) predicted by pair value v**: n=8935, mean predicted 0.008, observed 0.011, Brier 0.0107 (constant-rate 0.0107), log loss 0.060, ECE 0.005

| bin mean predicted | observed rate | n |
|---|---|---|
| 0.002 | 0.004 | 893 |
| 0.003 | 0.004 | 894 |
| 0.004 | 0.009 | 893 |
| 0.006 | 0.008 | 894 |
| 0.007 | 0.006 | 893 |
| 0.008 | 0.016 | 894 |
| 0.009 | 0.019 | 893 |
| 0.010 | 0.013 | 894 |
| 0.012 | 0.018 | 893 |
| 0.018 | 0.011 | 894 |

**Per variant**: predicted vs observed totals

| variant | directional pred / obs | joint pred / obs | MSMI pred / obs per episode |
|---|---|---|---|
| development | 0.493 / 0.486 | 0.259 / 0.283 | 0.73 / 1.05 |
| sparse | 0.476 / 0.488 | 0.239 / 0.273 | 0.16 / 0.15 |
| cold_start | 0.483 / 0.470 | 0.242 / 0.247 | 0.53 / 0.90 |
| delayed | 0.491 / 0.484 | 0.254 / 0.270 | 0.70 / 1.00 |
| shift | 0.462 / 0.440 | 0.229 / 0.242 | 0.65 / 0.70 |
| drift | 0.493 / 0.473 | 0.258 / 0.269 | 0.73 / 1.05 |

**Online weight movement**: end-of-decision-horizon acceptance weights (mean over seeds) vs prior mean

| feature | prior | development | sparse | cold_start | delayed | shift | drift |
|---|---|---|---|---|---|---|---|
| intercept | -0.004 | -0.076 | +0.007 | -0.075 | -0.092 | -0.216 | -0.120 |
| relationship_goal | +0.383 | +0.536 | +0.453 | +0.378 | +0.523 | +0.305 | +0.532 |
| relationship_pace | +0.289 | +0.257 | +0.388 | +0.292 | +0.252 | +0.419 | +0.268 |
| lifestyle | +0.094 | +0.072 | +0.247 | +0.093 | +0.046 | -0.076 | +0.071 |
| conversations | +0.131 | +0.036 | +0.146 | +0.007 | +0.033 | +0.127 | +0.017 |
| emotional_availability | +0.021 | -0.016 | +0.017 | +0.050 | -0.074 | +0.045 | -0.032 |
| space_for_relationship | -0.033 | -0.116 | -0.160 | -0.028 | -0.108 | -0.075 | -0.112 |
| relocate | +0.015 | -0.069 | -0.090 | +0.036 | -0.154 | -0.114 | -0.076 |

**Who is left waiting** (per episode)

| group | members | coverage | median first wait (days) | 90th pct wait |
|---|---|---|---|---|
| arrived day 0 | 140.3 | 0.347 | 5 | 16 |
| arrived days 1-20 | 59.7 | 0.336 | 0 | 11 |

**Unserved members by cause** (per episode): blocked declined 67.9, hard fields never clarified 1.1, complete but never introduced 62.3

**Introduction exits** (per episode): no response 30.33, declined intro 32.51, no date or date failed 2.41, date after 30 days 0.07, second answer missing late or no 8.33, msmi 0.81
