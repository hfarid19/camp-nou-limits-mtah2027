# ensemble_ctrl_v3b: a learned spell-level control rule in place of the fixed 2 s fold

**Verdict: don't ship.** The nested gain over the shipped submission is not resolved
(sel2 +0.0022 ± 0.0017). It loses on ctrl_mean, and it is no better than simply choosing a
shorter fixed fold.

## What it is

`pipeline.py` runs `ensemble_ctrl_v2`'s vote unchanged: `temporal_tcn_v2`, `handover_mstcn_v2`,
`ball_radar`, then LDE merge_short at 1 s. After that, each contested spell is either kept or
handed back to the preceding team. A contested spell is a team spell that directly follows the
other team's spell, not one that follows D or a block start.

- **The rule.** A LightGBM model scores P(keep). A spell is handed back when P(keep) < thr. The
  passes follow `control.fold`: each pass re-segments the codes, scores the spells, then hands
  spells back left to right. Mode `iter` repeats this until nothing changes; `single` runs one
  pass.
- **Dependencies.** numpy, scipy and lightgbm only.
- **Files.**
  - `spellctl.py`: features and `apply()`, shared by the live and offline paths.
  - `train.py`: the offline fit and nested selection.
  - `data.py`, `fastscore.py`: data loading, and a spellF1 equal to `pm.evaluate`.
  - `analysis/diag.py`, `analysis/report.py`, `analysis/compare_live.py`: diagnostics, the regret
    table, and the live-vs-offline check.
  - `model.txt`, `config.json`: the full model and its rule, from a LOGO choice on all 7 games:
    variant `delta`, mode `iter`, thr 0.55.
  - In `$MTAH_WORK/arch/ensemble_ctrl_v3b/`: the fold models `fold_spell_<gi>.txt`, the fold
    configs `fold_cfg_<gi>.json` and `folds.json` (v2's folds plus these two).

**Features.** There are 62 per spell, computed only from the degraded inputs and the vote and
member outputs:

- duration and neighbour durations;
- each member's agreement with the spell team, and each member's mean P for the spell team and
  for the previous team;
- ball visibility;
- touch events: savgol acceleration > 5 m/s² or direction change > 10° with a player within
  1.5 m, one event per burst. These give counts by each team, touches in the first 1 s, completed
  passes (consecutive spell-team touches by different players more than 3 m apart), and the time
  since the previous team's last touch;
- opponents within 0.5, 1, 2, 3 and 5 m at the start;
- duel and contest shares;
- longest run inside the possession zone;
- ball speed entering and leaving the spell;
- long flight (> 15 m with no player within 1.5 m);
- distance to goal and to the touchline;
- regain by the previous team.

**Targets.** A spell's target is 1 when the label reading gives more of its frames to the spell
team than to the previous team. Variants: `evt3`, `evt2`, `evtavg` (declared primary in advance)
and `sel5`. `delta` was added after the first run: its target is the sign of Δsel2 when this one
spell alone is handed back, weighted by |Δ|.

**Nesting.** Each outer game's model is trained on the other 6 games. Its threshold, mode and
variant (for `joint`) are chosen on those 6 games, and each of them is scored by an inner model
trained on the remaining 5. Note that `joint` includes the post-hoc `delta` variant.

## Results (bench v2, 7 games x 3 seeds; paired vs shipped, SE over game means, wins of 21)

| codes | sel2 | Δ sel2 | Δ ctrl_mean | poss/90 | spells < 2 s |
|---|---|---|---|---|---|
| shipped fold 2 | 0.7485 | | | 253 | 3.6% |
| **v3b joint, nested (= live)** | 0.7506 | **+0.0022 ± 0.0017 (13/21)** | **−0.0028 ± 0.0016 (8/21)** | 261 | 6.3% |
| v3b evtavg iter, nested (primary) | 0.7483 | −0.0002 ± 0.0019 (11/21) | −0.0049 ± 0.0021 (5/21) | 262 | 6.9% |
| v3b delta iter, nested | 0.7512 | +0.0027 ± 0.0018 (14/21) | −0.0022 ± 0.0016 (9/21) | 261 | 6.3% |
| duration-only fold, T chosen nested (1.75 s on 6/7 games) | 0.7498 | +0.0014 ± 0.0018 (15/21) | −0.0018 ± 0.0018 (8/21) | 260 | 5.1% |

- **Against the duration-only fold, joint gains only +0.0008 ± 0.0021 (12/21).** The
  classifier does little more than choose a shorter fold.
- **In-sample**, with LOGO models and the threshold picked on all 7 games, the best result is
  0.7523 (delta). The evt variants reach only 0.7474–0.7501.
- **Event labels in the target don't help.** The evt-trained classifier separates the event
  labels slightly better than duration does (AUC 0.893 vs 0.869). But it is no better on the
  spells where a hand-back actually raises sel2 (AUC 0.786 vs 0.784).
- **Large headroom.** An oracle that handed back exactly the helpful spells would be worth about
  +0.05 per game-seed (sum of the single-spell deltas). The features don't capture it.
- **Short spells.** The truth has 4–30% of spells under 2 s, depending on the reading (Metrica:
  17–22%). v3b has 6.3%: the vote's merge_short has already removed most short spells, and the
  classifier can only remove spells, never restore them.

**Feature importance** (gain, full model): member probabilities and codes 40%,
proximity / speed / flight / pitch 34%, duration and context 19%, touch and pass 8%.
The top features are pm_c 25%, dur_s 10%, pm_p 6% and disp 5%.

**Regret table.** Score / regret against the best vote fold (0–3 s) at each reading:

| | raw | T1 | T1.5 | T2 | T2.5 | T3 | evt3 | evt2 | max regret |
|---|---|---|---|---|---|---|---|---|---|
| shipped | .622/.027 | .695/.023 | .746/.010 | .777/0 | .773/.017 | .761/.040 | .768/.011 | .757/0 | **0.040** |
| v3b joint | .639/.010 | .712/.006 | .753/.004 | .769/.008 | .759/.032 | .744/.057 | .763/.016 | .757/0 | 0.057 |
| v3b − shipped | +.017 | +.017 | +.006 | −.008 | −.014 | −.017 | −.005 | +.000 | |

v3b behaves like a fold of about 1.7 s. It wins when "controlled" is read loosely and loses when
it is read strictly, and its worst case is worse than shipped's. The small sel2 gain comes from
sel2 leaning towards loose readings.

## Checks

- **Live = offline.** `bench.py candidates/ensemble_ctrl_v3b/pipeline.py --dfl2 --folds
  $MTAH_WORK/arch/ensemble_ctrl_v3b/folds.json --save-probs $MTAH_WORK/probs/ensemble_ctrl_v3b_live`
  gives codes identical to the offline nested `joint` on **21/21** game-seeds (ctrl_mean 0.7414).
- **Match** (label-free): 258 possessions (shipped 262), dead share 0.48, no X in the halves, and
  1188 frames differ from the shipped CSV. `validate_submission.py` reports the file as valid.
- **Metrica dev:** spellF1 0.6585 (shipped 0.6643).

## Recommendation

Keep `ensemble_ctrl_v2`. v3b's +0.002 on sel2 is unresolved, and it equals what a 1.75 s fold
gets. It also costs ctrl_mean (−0.003) and the worst-case regret (0.057 vs 0.040). Content
features carry little control evidence beyond duration and the members' own probabilities on
this degraded data. A real gain probably needs the rule to restore the short changes that the
vote and merge_short drop, not only to remove spells.
