# adapt: label-free domain adaptation (round 5, track R5-2)

Goal: transfer to out-of-domain tracking. The proxy is the SkillCorner A-League bench (real broadcast tracking),
since BAR-ATH separates from DFL with AUC 0.99. Decision rule: `../../docs/prereg/round5_prereg.txt` (+ addendum:
the SkillCorner gain must hold at matched possession count). Variants fixed before any result:
`../../docs/prereg/prereg_adapt.txt`. All result logs: `$MTAH_WORK/arch/r5_adapt/results_*.txt`.

## Ideas and code

| idea | files | what it does |
|---|---|---|
| 1. per-game re-normalisation | `renorm.py`, `run_renorm.py`, `renorm_members.py`, `member_{tcn,mstcn}_a{50,100}.py`, `pipeline_lsw_renorm_a{50,100}.py` | The NN members' training mean / sd are moved towards the game's own feature moments over its graded blocks: `mu' = mu + a (mu_g - mu)`, `sd' = sd (sd_g / sd)^a` (ratio clipped to [1/3, 3]). The moments are mirror-pooled (features + their team-swapped copy), so the members' mirror pass sees the same normalisation. Variants: all features at a = 0.25 / 0.5 / 1, mean only, a fixed list of match-shifted features, or an auto-flag per game. |
| 2. test-time averaging | `tta.py`, `run_tta.py`, `renorm_members.tcn_tta`, `member_tcn_tta4[_a50].py`, `pipeline_mbr_tta4[_a50].py` | 4 jittered copies of the input plus the original. The jitter is smooth AR(1) offsets: players 0.5 m, ball 0.3 m, tau 1 s, seeds 1000-1003. Class (and hand-over) probabilities are averaged, then the member's own decoder runs. |
| 3. noisy student / distillation | `build_pl.py`, `train_student.py`, `run_student.py`, `skc_folds.json`, `adapt_match.py` | Teacher = lsw (vote under the v6 rule). Inductive: the recipes retrained on labelled data + pseudo-labelled SkillCorner training games (3 SKC folds, so no game is scored by a model that saw its pseudo-labels). Transductive (match-style): the shipped model fine-tuned on labelled data + the scored game's own pseudo-labels (x8), one fine-tune per bench unit. |
| stacking with R5-1 | `stack_mbr.py`, `nested_mbr.py` | mbr_decode's decoder on adapted member outputs. Fixed CFG; DFL also nested with R5-1's per-game picks. |
| scoring | `score_adapt.py` | DFL fine4 (LOGO, SE over 7 games), SkillCorner fine3 (20 games), Metrica dev; paired vs v6 / lsw (/ MBR); possessions per 90; the addendum's matched-count SKC check (v6 re-folded per game to the candidate's count). |
| match stats | `match_run.py` | BAR-ATH, label-free (Mac only): possessions/90, dead share, frames changed, determinism re-run. |

Every live pipeline here equals its offline evaluation (0 frames differ on SKC 1874553 / 2016236).
Inference is numpy + LightGBM only. The game statistics and the jitter are deterministic numpy functions of
the input positions.

## Results (paired vs v6; DFL fine4 LOGO, SE over 7 games; SKC fine3, 20 games; "matched" = SKC vs v6 re-folded to the candidate's possession count)

| candidate | DFL | SKC | SKC matched | Metrica | poss/90 DFL / SKC / Met |
|---|---|---|---|---|---|
| lsw (reference) | +0.0099 ± 0.0021 | +0.0054 ± 0.0026 | +0.0008 ± 0.0016 | +0.0035 | 305 / 284 / 281 |
| lsw + renorm all_a50 (a-priori primary) | +0.0075 ± 0.0029 | +0.0085 ± 0.0025 | +0.0035 ± 0.0022 | +0.0067 | 304 / 282 / 281 |
| lsw + renorm all_a100 | +0.0057 ± 0.0026 | +0.0115 ± 0.0032 | +0.0070 ± 0.0029 | +0.0011 | 305 / 280 / 278 |
| lsw, TTA on both NN members | +0.0023 ± 0.0027 | +0.0146 ± 0.0036 | +0.0064 ± 0.0032 | +0.0046 | 298 / 272 / 275 |
| lsw, TTA on the TCN only | +0.0057 ± 0.0016 | +0.0144 ± 0.0026 | +0.0079 ± 0.0022 | +0.0050 | 301 / 276 / 277 |
| MBR-lsw (R5-1; DFL nested) | +0.0137 ± 0.0044 | +0.0123 ± 0.0033 | +0.0100 ± 0.0024 | +0.0091 | 312 / 288 / 288 |
| MBR + renorm a50 (nested) | +0.0149 ± 0.0036 | +0.0128 ± 0.0034 | +0.0115 ± 0.0031 | +0.0076 | 312 / 287 / 288 |
| MBR + renorm a100 (nested) | +0.0142 ± 0.0034 | +0.0132 ± 0.0038 | +0.0091 ± 0.0029 | -0.0032 | 311 / 285 / 285 |
| **MBR + TCN-TTA (nested)** | +0.0134 ± 0.0041 | +0.0213 ± 0.0032 (20/20) | +0.0178 ± 0.0026 | +0.0043 | 307 / 283 / 284 |
| **MBR + TCN-TTA + a50 (nested)** | +0.0149 ± 0.0038 | +0.0206 ± 0.0035 (20/20) | +0.0169 ± 0.0030 | +0.0087 | 308 / 282 / 284 |
| lsw with TCN -> inductive PL student tS | +0.0068 ± 0.0021 | +0.0091 ± 0.0029 | +0.0028 ± 0.0022 | +0.0020 | 302 / 279 / 280 |
| lsw with TCN -> transductive student (match-style) | +0.0094 ± 0.0028 | +0.0060 ± 0.0026 | +0.0002 ± 0.0018 | +0.0064 | 304 / 282 / 280 |

The TCN-only TTA and its combination with a50 were chosen after the both-member TTA was seen, so they are post hoc.
Full logs, including every other variant: results_renorm / results_tta / results_mbr_stack / results_mbr_nested /
results_student / results_trans.
