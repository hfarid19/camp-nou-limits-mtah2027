# ensemble_ctrl_v2: ensemble_ctrl with the sequence members retrained on the realistic degradation

`pipeline.py` combines three members:

1. a hard majority vote of `temporal_tcn_v2`, `handover_mstcn_v2` and `ball_radar`, in that
   tie-break order;
2. `learned_dfl_evidence`'s `merge_short_v2` at 1.0 s (ensemble_vote's "m10");
3. `control.fold(codes, 2.0 s)` over the graded blocks, as in `ensemble_ctrl`.

`ball_radar` is unchanged. The two TCN members are the retrained v2 versions (see their READMEs):
trained on `ds_v2` (randomised `degrade_v2`, where the ball is missing mostly at dead ball) with
controlled-possession targets. For each member, the version used is the one with the best
ctrl_mean on DFL bench v2.

`members.py` and `ensemble_core.py` are byte-identical copies from `ensemble_vote` (the
isolated member loader and the vote rules). `predict_proba` returns the vote shares, as
before.

**Rule as code, not config.** `ensemble_ctrl` reads `ensemble_vote/config_hard.json` through
`dfl_folds.resolve`, so under the bench every game uses a rule selected on the other 6 games.
On J03WOH that rule smooths at 0.5 s; on J03WR9 it adds pitch_control. Here the rule is fixed
in `pipeline.py`, and the full-data choice is used on every game, so only the members' models
need fold files. That change on its own is worth nothing: the same fixed rule with the old
members, replayed offline (`vote_offline.py`), gives 0.7287 against ensemble_ctrl's 0.7283.

## Results

### DFL bench v2 (7 games leave-one-game-out x 3 degrade_v2 seeds, live run, every member on its fold models)

| | ctrl_mean | ctrl T=1 / 2 / 3 | raw spellF1 | macroF1 | possessions |
|---|---|---|---|---|---|
| ensemble_ctrl | 0.7283 | 0.6912 / 0.7608 / 0.7328 | 0.6220 | | 5773 |
| **ensemble_ctrl_v2** | **0.7443** | 0.6947 / 0.7768 / 0.7613 | 0.6223 | 0.9227 | 5630 |

Paired against ensemble_ctrl (same game and seed, 21 pairs, SE over the 7 games with seeds averaged):

- **ctrl_mean: +0.0160 (SE 0.0027), 21/21 wins.** Per game: +0.011 +0.025 +0.016 +0.004 +0.016
  +0.024 +0.015.
- raw spellF1: +0.0003 (SE 0.0029), 10/21 wins.
- Against the fixed-rule replay with the old members: +0.0155 (SE 0.0028), 21/21. So the gain
  comes from the members.

The live codes equal the offline replay (`vote_offline.py` on the members' saved codes)
on all 21 game-seeds.

### Which member versions (offline replay, ctrl_mean)

| temporal_tcn | handover_mstcn | ctrl_mean | raw spellF1 |
|---|---|---|---|
| v1 | v1 | 0.7287 | 0.6229 |
| v2 raw | v1 | 0.7392 | 0.6321 |
| v2 ctrl | v1 | 0.7416 | 0.6269 |
| v2 raw | v2 raw | 0.7426 | 0.6353 |
| v2 raw | v2 ctrl | 0.7454 | 0.6296 |
| v2 ctrl | v2 raw | 0.7455 | 0.6223 |
| **v2 ctrl** | **v2 ctrl** | **0.7443** | 0.6223 |
| v2 ctrl_mix | v2 ctrl | 0.7458 | 0.6203 |

Every v2 pairing lies within 0.003 of the others, which is inside the paired noise. The shipped
pairing follows the rule fixed before the vote (the best version of each member on its own),
not the best row of this table.

### Metrica dev (Games 1, 2 x 2 seeds; secondary)

| | spellF1 | per run |
|---|---|---|
| ensemble_ctrl (saved codes) | 0.6727 | 0.684 0.671 0.678 0.659 |
| ensemble_ctrl_v2 | 0.6643 | 0.668 0.679 0.655 0.655 |

-0.008. Dev uses the OLD degradation (`tune_on_sample_data.degrade`: short random ball gaps
in live play) and every-touch practice labels from event data. A drop there is expected for
members retrained away from that regime, and it is within the dev noise band (about 0.04 per
run). DFL v2 remains the selection bench.

### Match (no score)

There are 262 possessions, mean 12.6 s, dead share 0.48 and no X inside the halves.
ensemble_ctrl gives 289 possessions, and 8951 of 158857 graded frames differ.

On DFL, ensemble_ctrl_v2 predicts 268 possessions per game. The controlled truth has 272 per
90 min at T = 2 s.

**Determinism:** two `--match` runs give byte-identical `submission.csv` (sha256
`2de360e9e2d3…`), `match.npy`, match codes and dev codes (checked with `cmp`).

## Fold models

`$MTAH_WORK/arch/ensemble_ctrl_v2/folds.json` merges:

- every entry of `arch/ensemble_vote/folds.json` (ball_radar's radar and LightGBM models, and
  LDE's);
- `arch/temporal_tcn_v2/folds.json`;
- `arch/handover_mstcn_v2/folds.json`.

In the live run, `tcn_weights.npz`, `model_tcn.npz` and `radar.npz` were resolved to
`fold_*_<gi>` for all 7 games, and ball_radar's boosters were swapped by the bench.

## Verdict

ensemble_ctrl_v2 is a consistent +0.016 ctrl_mean improvement over ensemble_ctrl on the
selection bench. It wins on all 21 game-seeds, with the paired SE at a sixth of the gain. The
gain comes entirely from the retrained members. On the raw every-touch labels it is a tie. On
the old-degradation Metrica dev bench it is slightly worse (-0.008, inside the noise). On the
match it predicts about 9% fewer possessions than ensemble_ctrl.

The caveat is that both DFL benches use simulated degradation: degrade_v2 is fitted to the
match's label-free statistics, but the training data uses the same simulator family, so part of
the gain may be simulator matching.

## Rebuild / rerun

```sh
W=$MTAH_WORK; PY=$W/venv/bin/python
# members: see temporal_tcn_v2/README.md and handover_mstcn_v2/README.md
$PY candidates/bench.py candidates/ensemble_ctrl_v2/pipeline.py --dfl2 \
    --folds $W/arch/ensemble_ctrl_v2/folds.json --save-probs $W/probs/ensemble_ctrl_v2
$PY candidates/temporal_tcn_v2/paired_codes.py $W/probs/ensemble_ctrl_v2 $W/probs/ensemble_ctrl
$PY candidates/bench.py candidates/ensemble_ctrl_v2/pipeline.py --match    # dev bench + submission.csv
# offline replay of member combinations (needs each member's --dfl2 --save-probs codes; ball_radar's are in
# $W/probs/ensemble_ctrl_v2/members/ball_radar)
$PY candidates/ensemble_ctrl_v2/vote_offline.py OUT_DIR <tcn codes> <handover codes> <ball_radar codes>
```
