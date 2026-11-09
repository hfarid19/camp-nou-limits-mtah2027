# ensemble_ctrl_v6: the vote with every member retrained on PFF World Cup 2022 + DFL

`v6_core.py` is ensemble_ctrl_v5c's vote machinery with the member list and the rule as
parameters: a hard majority vote of the members' own decoded codes (tie-break = member order),
then optional `merge_short_v2`, then `control.fold`, then `prekick_dead`. `members.py` and
`ensemble_core.py` are copies of v5c's.

| file | candidate |
|---|---|
| `pipeline_v5cpff.py` | **v5c+PFF**: v5c's three members with their exact recipes, retrained on 7 DFL + 48 PFF games |
| `pipeline.py` | the final pick (see the end of this file) |

Training data, folds, enforcement and the member recipes are in `../nn_pff/README.md`. The
reference for every comparison is the submission, `ensemble_ctrl_v5c`, with v4 also reported.
The selection protocol was fixed before any model was scored (`$MTAH_WORK/arch/nn_pff/prereg.txt`).

## v5c+PFF

Members, in tie-break order (DFL leave-one-game-out fine4, best first):

| member | recipe (= v5c member) | weights |
|---|---|---|
| `nn_pff_tcn/pipeline_v5cpff.py` | temporal_tcn_v3: TCN C 64, 12 blocks, LDE decoder; 4000 iterations | `tcn_v5cpff.npz` (0.9 MB) |
| `nn_pff/pipeline_v5cpff.py` | handover_mstcn_v3: ASRF MS-TCN, ASRF decoder; 40 x 1200 crops | `mstcn_v5cpff.npz` (1.3 MB) |
| `ball_radar_pff/pipeline.py` | ball_radar_v3: radar (3200 iterations) + stacked LightGBM d / ha / c | `radar.npz` 0.46 MB, `model_br*.txt` 0.9 / 1.8 / 1.7 MB |

**Rule, chosen nested.** The grid was smoothing {off, 1 s} x control fold {0, 0.5, 0.75, 1, 1.25, 1.5 s}.
The pre-registered criterion is PFF out-of-fold fine3 + DFL fine4. For each PFF fold the rule is
chosen on the other 3 folds + all DFL games; for each DFL game, on the other 6 games + all PFF
games. The picks were smoothing off with fold 0.75 or 1 s, or smoothing 1 s with fold 1 s, all
within 0.0002 of each other. The shipped rule is the argmax on all selection data: **smoothing
off, fold 0.75 s** (`vote_offline.py`; log `$MTAH_WORK/arch/nn_pff/logs/vote_v5cpff.txt`).

| (live runs; paired vs ensemble_ctrl_v5c, SE over games) | DFL fine4 (7 games x 3 seeds) | PFF out-of-fold fine3 (48 games) | PFF touch spellF1 | Metrica dev |
|---|---|---|---|---|
| ensemble_ctrl_v5c (submission) | 0.7171 | 0.6817 | 0.6635 | 0.6974 |
| ensemble_ctrl_v4 | 0.7095 (-0.0076 ± 0.0038, 4/21) | 0.6761 (-0.0056 ± 0.0025, 16/48) | 0.6580 | 0.7083 |
| **v5c+PFF** | **0.7327 (+0.0156 ± 0.0060, 15/21)** | **0.7887 (+0.1070 ± 0.0036, 48/48)** | **0.7718** | **0.7298** |
| v5c+PFF, nested rule estimate | 0.7315 (+0.0144 ± 0.0058) | 0.7885 (+0.1068 ± 0.0035) | 0.7711 | |

Members alone (same benches):

| member | DFL fine4 | vs its v5c member | PFF fine3 | Metrica dev |
|---|---|---|---|---|
| temporal_tcn_v3 -> t5c | 0.7028 -> 0.7300 | +0.0272 ± 0.0035 (21/21) | 0.6549 -> 0.7809 | 0.6653 -> 0.7142 |
| handover_mstcn_v3 -> m5c | 0.7080 -> 0.7251 | +0.0172 ± 0.0065 (16/21) | 0.6637 -> 0.7868 | 0.6544 -> 0.6979 |
| ball_radar_v3 -> radar_pff | 0.6781 -> 0.6988 | +0.0207 ± 0.0051 (19/21) | 0.6538 -> 0.7215 | 0.7038 -> 0.7345 |

**Read the PFF column with care.** PFF's ball is placed on the carrier by PFF's own events and
interpolated between them, and the touch labels come from the same events. So a model trained on
PFF partly learns PFF's ball rule, and PFF-trained models gain far more on PFF (+0.10) than on DFL
(+0.016 for the ensemble, +0.02 to +0.03 per member). DFL (real tracked ball, never in the scoring
model's training) and Metrica are the transfer tests, and both improve.

Other readings on PFF (vs v5c): T1.5 +0.074, nore +0.096, seq (PFF's own possession, coarser) +0.050.
On DFL, sel2 (the coarser 2026-11-08 metric) is -0.0048 ± 0.0056 (8/21). The model is finer-grained
than v5c: 7199 vs 6712 DFL possessions (truth 9183 raw), 14077 vs 12590 PFF (truth 16980).

**Checks.** Every DFL game used fold models (the bench's `fold files resolved` lists fold_tcn_0-6
for both networks and fold_radar_0-6; LightGBM folds via the bench's Booster swap). Every PFF game
was resolved to the out-of-fold model of all six weight files (`probs/nn_pff/ensemble_ctrl_v6_v5cpff/resolve.log`:
24 fold files x 12 games each). The live codes equal the offline vote of the saved member codes
on 21/21 DFL game-seeds and 48/48 PFF games (`../nn_pff/live_vs_offline.py`).

**Match** (positions only, never labels): two `bench.py ... --match` runs are `cmp`-identical,
sha256 `4d680a7aac55…`. `evaluation/validate_submission.py` passes. 368 possessions (v5c 329), dead
share 0.466, no X inside the halves. 4.1% of graded frames differ from v5c, 1.7% of them H<->A.

## Reproduce

```
W=$MTAH_WORK; A=$W/arch/ensemble_ctrl_v6
# members: see ../nn_pff/README.md (t5c, m5c) and ../ball_radar_pff/README.md
python candidates/nn_pff/vote_offline.py --select both --grid-all \
    --set "v5cPFF=nn_pff_dfl/t5c|t5c,nn_pff_dfl/m5c|m5c,nn_pff_dfl/radar_pff|radar_pff"
$A/live_checks.sh pipeline_v5cpff.py ensemble_ctrl_v6_v5cpff      # bench --dfl2, run_pff, Metrica dev
python candidates/nn_pff/live_vs_offline.py ensemble_ctrl_v6_v5cpff 0 0.75 \
    "nn_pff_dfl/t5c|t5c" "nn_pff_dfl/m5c|m5c" "nn_pff_dfl/radar_pff|radar_pff"
python candidates/bench.py candidates/ensemble_ctrl_v6/pipeline_v5cpff.py --match
python candidates/nn_pff/report.py                                # every table of this file
```
`$A/folds.json` = `arch/ensemble_ctrl_v5c/folds.json` + the six v5c+PFF weight files -> their DFL
fold folders. `$A/pff_folds.json` maps the same files to their PFF fold folders.

## Final pick (2026-11-09): v5c+PFF members with v5c's rule (`pipeline.py` = `pipeline_v5crule.py`)

An independent audit of `pipeline_v5cpff.py` reproduced every number above, found no leakage, and
confirmed it runs without torch. It also ran a check nobody had: the SkillCorner A-League bench
(`../skc/`, 20 games, never used for selection, labels with the fingerprint closest to the truth's).
There the 0.75 s rule was worse than v5c: fine3 −0.0073 ± 0.0040, 7/20 games. It also loses on
the coarse readings (T3 −0.025). The gain comes from the retrained members, not from the rule.

A decision rule was written down before the next results were seen
(`$MTAH_WORK/arch/ensemble_ctrl_v6/rule_choice_prereg.txt`). The candidate is the same members with
v5c's rule (merge_short 1 s, control.fold 1.25 s, prekick_dead). It replaces v5c if it wins DFL by at
least 2 SE and is not worse than v5c on SkillCorner beyond 1 SE. Both held. The 16-game PFF holdout
was then scored once, for this candidate only.

| live, vs v5c | v6 final (v5c rule) | v5c |
|---|---|---|
| DFL LOGO fine4 (7 games × 3 seeds) | **0.7286**, +0.0115 ± 0.0040 (16/21), sel2 +0.008 | 0.7171 |
| SkillCorner A-League fine3 (20 games) | 0.5964, +0.0028 ± 0.0044 (12/20) | 0.5936 |
| Metrica dev (Sample Games 1+2) | **0.7285** | 0.6974 |
| Metrica holdout (Game 3, once) | **0.7258** | 0.7188 |
| PFF holdout, 16 games (once) fine3 | 0.7869, +0.107 ± 0.005 (16/16); seq +0.048 | 0.6801 |

The PFF numbers are inflated: PFF's ball is placed on the carrier at event times, and the
members were trained on PFF. The holdout equals the out-of-fold estimate, so nothing leaked.
The DFL gain is the honest estimate of transfer.
Logs: `$MTAH_WORK/arch/ensemble_ctrl_v6/rulecheck/`.
