# Candidates and the bench

Every possession pipeline we built sits behind one interface (`predict`, plus `predict_proba`
for the ensemble) and is scored by one shared bench, `bench.py`. Shipped: `ensemble_vote` and
the folders it needs (`temporal_tcn`, `handover_mstcn`, `ball_radar`, `learned_dfl_evidence`).
Built and dropped: `archive/` (see its README).

## Benches

| flag | data | labels | use |
|---|---|---|---|
| (none) | Metrica Sample Games 1 + 2, degraded with `tune_on_sample_data.degrade` x 2 seeds | practice labels from Metrica's event data | secondary check |
| `--holdout` | Metrica Sample Game 3, same degradation | same | run once per final candidate |
| `--dfl` | 7 open DFL / IDSSE Bundesliga games, `tune_on_sample_data.degrade` x 1 seed | DFL tracking flags (BallStatus / BallPossession), same meaning as the competition's | selection until 2026-11-07 |
| `--dfl2` | the same 7 games, `degrade_v2` x 3 seeds | same | **primary selection from 2026-11-08** |

On both DFL benches every model a candidate loads is swapped for its leave-one-game-out fold
model, so a game is only ever scored by models that never saw it (LightGBM boosters and
`torch.load` are swapped automatically, other files go through `dfl_folds.resolve`). Large work
files (DFL data, fold models, caches) live in `$MTAH_WORK`, default `<repo>/external/work`.

## Why bench v2: the simulator did not look like the real tracking

`match_stats.py` computes label-free statistics of a tracking feed. On the competition match
(BAR-ATH, positions only) vs the v1-degraded DFL games:

| | BAR-ATH | DFL, v1 degradation | DFL, `degrade_v2` |
|---|---|---|---|
| ball missing (graded frames) | 0.407 | 0.410 | 0.389 |
| ball gaps per 90 min | 127 | 1779 | 119 |
| gap length p10 / p50 / p90 (s) | 0.5 / 10.7 / 33.8 | 0.2 / 0.9 / 2.8 | 0.5 / 11.2 / 40.8 |
| share of missing frames in gaps > 5 s | 0.997 | 0.090 | 0.972 |
| ball jitter (m) / median acceleration (m/s²) | 0.009 / 5.7 | 0.214 / 363 | 0.007 / 5.6 |
| ball speed p50 / p90 (m/s), not fitted | 5.1 / 14.3 | 9.7 / 18.9 | 4.8 / 14.9 |
| visible players per team | 10.2 / 10.3 | 10.9 / 10.7 | 10.2 / 10.0 |
| tracklets per 90 min, median length (s) | 1036, 35.5 | 7178, 11.4 | 1237, 61.8 |
| player median acceleration (m/s²) | 0.85 | 6.25 | 0.92 |

The v1 simulator hid the ball in many short random bursts during play and added 0.2 m of
noise. The real ball is smooth and missing almost only in long gaps. During those gaps BAR-ATH's
players move like DFL players at dead ball: median team speed 1.0 m/s, against 1.02 dead and
2.27 alive on DFL. A mixture fit of the speed histograms puts about 95% of long-gap time at dead
ball. `degrade_v2.py` therefore hides the DFL ball around the DFL dead-ball runs (plus a few
short open-play gaps), smooths the ball and players, and fragments player tracks into longer
tracklets. Its knobs were fitted to the BAR-ATH statistics with `fit_degrade_v2.py`. Only
positions of the match are used, never a label (RULES.md allows inspecting the data). On the
v2-degraded DFL games the ball is hidden in about 80% of dead-ball frames and about 6% of alive
ones.

## Ranking on bench v2 (2026-11-08)

DFL leave-one-game-out, 7 games x 3 degradation seeds. SE is the standard error over games.
Possessions are per seed, summed over the 7 games (truth: 3061).

| candidate | spellF1 | SE | macroF1 | possessions |
|---|---|---|---|---|
| ensemble_vote | **0.6564** | 0.017 | 0.920 | 2245 |
| handover_mstcn | 0.6515 | 0.016 | 0.917 | 2343 |
| temporal_tcn | 0.6451 | 0.017 | 0.915 | 2385 |
| ball_radar | 0.6311 | 0.017 | 0.915 | 2244 |
| archive/semimarkov_lattice | 0.6257 | 0.015 | 0.910 | 2355 |
| archive/pitch_control | 0.6236 | 0.017 | 0.910 | 2181 |
| archive/ballfree_dead | 0.6167 | 0.017 | 0.911 | 2190 |
| archive/v2_stacked_fixes | 0.6161 | 0.016 | 0.908 | 2190 |
| learned_dfl_evidence | 0.6156 | 0.016 | 0.909 | 2158 |
| archive/skillcorner_domain | 0.6118 | 0.017 | 0.907 | 2137 |
| archive/pathcrf_dynamic | 0.6113 | 0.017 | 0.908 | 2159 |
| archive/hier_touch_decoder | 0.5949 | 0.017 | 0.899 | 2173 |
| archive/v1_original | 0.3697 | 0.011 | 0.786 | 2137 |

The order is close to the v1 bench's. `ball_radar` loses the most (0.647 on v1 to 0.631), as
expected once the ball is missing mainly at dead ball. Paired differences (same game and seed)
are far tighter than the per-candidate SE. The ensemble beats `temporal_tcn` by +0.011 (paired
SE 0.003, 17/21 wins), `ball_radar` by +0.025 (21/21) and `learned_dfl_evidence` by +0.041
(20/21). Against `handover_mstcn` it is +0.005 ± 0.004 (15/21), which is not resolved.
