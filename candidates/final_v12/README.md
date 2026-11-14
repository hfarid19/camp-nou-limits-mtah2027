# final_v12: submission v12 (v11 with 58 more input columns)

`pipeline.py` is v11 (`../final_v11`) with one change, and the stacker refitted to go with it:

1. **New members.** The TCN and the MS-TCN are replaced by retrained versions that read 281 inputs instead
   of v11's 223: the 194 repo features, v11's 29 "ACC + SHAPE" columns and 58 more ("REL", "KIN2" and
   "INT-hand"; `../newfeat_v12`). The training recipes and fold structure are v11's; only the input columns
   differ (experiment "newfeat", then "v12").
2. **The stacker** has v11's recipe (S2: LightGBM, 200 trees per class, depth 4, lr 0.05, on the four members'
   probabilities at 7 time offsets, 84 inputs, trained only on Metrica's three open sample games), refitted
   on the outputs of *these* four members.

Radar, pointer, decoder setting (`src hmm, alpha 0.1, rho 0.2, margin 0.0`, fixed sampling seed) and dead
ball before each half's kick-off are v9's, as in v11. Pre-registration: `../../docs/prereg/round5_prereg.txt`,
addenda of 13 Nov 2026 11:58 (REL), 12:04 and 18:24 (KIN2), 18:24 (INT), 20:52 (the inclusion rule) and of
14 Nov 2026 01:19 CST ("v12" bench).

**Status: the submission since 14 Nov 2026 (the user's choice; v11 was published that day and then
replaced).** `make_submission.py` runs it by default (`DEFAULT_CANDIDATE = "final_v12"`). The final checks
of 14 Nov are listed under "Final checks"; all of them wrote the same file.

```sh
python make_submission.py                                           # writes evaluation/results/camp_nou_limits.csv
python make_submission.py --candidate final_v12 --out <path>.csv   # the same pipeline, to another path
```

| slot | member file | what it is | weights |
|---|---|---|---|
| TCN | `newfeat_v12/member_tcn_v12_tta4_a50.py` | the `o_ls05` recipe on 281 inputs; test-time averaging over 4 jittered copies and all_a50 renormalisation as v9 and v11 | `newfeat_v12/tcn_v12.npz` (0.9 MB, sha256 `e518c074385a…`) |
| MS-TCN | `newfeat_v12/member_mstcn_v12_skcft_metft.py` | the `m5c_C192ls` → SkillCorner fine-tune → Metrica fine-tune chain on 281 inputs | `newfeat_v12/mstcn_v12_skcft_metft.npz` (10.7 MB, sha256 `be3cfbc1bf97…`) |
| radar | `ball_radar_pff/pipeline.py` | `radar_pff`, v9's | unchanged |
| pointer | `nn_pointer/pipeline_ballc_p20m.py` | `r5_ptr_ballc_p20M`, v9's; torch worker | unchanged |
| stacker | `stacker_v12.txt` (this folder), applied by `stacker_met/stacker.py` `predict` | S2 recipe; 84 inputs | 1.1 MB, sha256 `f518f1ac397a64f4261256fd6dbbdc36545eaee57aece71375cb424c172f4362` |

**The 58 new columns** (`../newfeat_v12/nf_rel.py`, `nf_kin2.py`, `nf_int.py`; numpy + scipy). Every column
is 0 where the ball is missing, and none depends on the attack direction.
- REL, 19 columns: the closing speed of the ball and each team's nearest player; a touch detector (a peak of
  the ball's turn or acceleration with a player within 1.5 m) and each team's touches in the last and next
  1 and 3 s; how often the nearest player and the nearest player's team changed in the last 1 and 3 s; the
  distance and speed difference of the two teams' nearest players; the time since the ball last turned sharply.
- KIN2, 26 columns: the ball's jerk; its turn angle and change of speed over 0.2, 0.4 and 1 s and the
  curvature of its path; the time since and until the nearest touch, with the toucher's team and distance,
  and touches per team in the last 2 s; the ball's motion relative to each team's nearest player.
- INT-hand, 13 columns: hand-picked products of ball acceleration, turn angle and speed with nearest-player
  proximity, closing speed, the distance difference of the two teams, ball-seen, last control and players
  within 2 m.

`../newfeat_v12/fe_v12.py` appends the blocks in the order ACC, SHAPE, REL, KIN2, INT to the repo features.
The module files are the experiment's code, with the path to `nf_features.py` changed to the sibling file.

**Order of operations** is v11's: member probabilities rounded to float32, stacker at every frame inside
the graded halves on one thread, its probabilities rounded to float32, `decoder.decode(four members +
stacker)`, then `prekick_dead`. One change for memory only: after the MS-TCN member, `pipeline.py` releases
the feature-block cache and the two members' per-unit caches before the radar and the pointer run. The
output is the same with and without it (see "The match").

## How the columns were chosen

Each group was screened on top of v11's 223 columns, with three seeds per variant, on PFF fold 0 (12 games),
SkillCorner (20) and two Metrica games, with the standard error over seeds (work log and tables:
`$MTAH_WORK/arch/r5_arch/newfeat/`, `tables_rel.md`, `tables_kin2.md`, `tables_int.md`, `v12/tables_dev.md`).

| added on top of the 223 columns | PFF fold 0 | SkillCorner | Metrica (2) |
|---|---|---|---|
| TCN + REL | +0.0065 ± 0.0018 | −0.0105 ± 0.0016 | +0.0160 ± 0.0049 |
| TCN + KIN2 | +0.0130 ± 0.0016 | −0.0088 ± 0.0007 | +0.0171 ± 0.0015 |
| TCN + INT-hand | +0.0094 ± 0.0021 | −0.0063 ± 0.0011 | +0.0127 ± 0.0011 |
| TCN, INT-gated input layer (dropped) | +0.0026 ± 0.0015 | −0.0029 ± 0.0015 | +0.0013 ± 0.0019 |
| MS-TCN + REL | +0.0026 ± 0.0016 | −0.0007 ± 0.0051 | +0.0196 ± 0.0025 |
| MS-TCN + KIN2 | +0.0077 ± 0.0019 | −0.0024 ± 0.0045 | +0.0252 ± 0.0045 |
| MS-TCN + INT-hand | +0.0145 ± 0.0018 | −0.0053 ± 0.0079 | +0.0185 ± 0.0024 |
| TCN, all three (281) | +0.0168 ± 0.0023 | −0.0100 ± 0.0015 | +0.0181 ± 0.0016 |
| MS-TCN, all three (281) | +0.0157 ± 0.0016 | −0.0104 ± 0.0077 | +0.0276 ± 0.0024 |

**The inclusion rule was changed after results.** v11's columns were admitted under a rule that allowed no
loss beyond noise on SkillCorner or Metrica. REL and KIN2 lost on SkillCorner in the TCN, and were first kept
out. At 20:52 CST on 13 Nov, after those results were seen, a new rule was adopted and recorded in the rule
file: a group qualifies for a member if it gains beyond seed noise (diff ≥ +2 seed SE and ≥ 8 of 9 cross-seed
wins) on at least 2 of the 3 dev benches, preferably including Metrica; a loss on the third does not exclude
it. Under it every group qualified for both members, except that REL in the MS-TCN passed only on Metrica
(+1.7 SE on PFF fold 0) and was included on our reading of the rule. INT-gated qualified nowhere and was
dropped. So v12's columns were selected on the same benches that then scored it.

## Evidence

All four benches, fixed decoder setting on every unit, as for v11. Every model that scored a game had never
seen it: DFL by leave-one-game-out fold models, SkillCorner by cross-validation models, PFF by fold models,
Metrica leave one game out for both the Metrica-trained MS-TCN and the stacker. On DFL, SkillCorner and PFF
the stacker trained on all three Metrica games is applied. The v12 models were trained on the Mac GPU (MPS)
and every member output was computed on the Mac CPU with the shipped members' numpy code. Work log and code:
`$MTAH_WORK/arch/r5_arch/newfeat/v12/` (`out/report_v12.txt`, `code/v12_bench.py`) and `../NOTES.md` there.

| bench | v9 | v11 | v12 without the stacker | **v12** | v12 − v11 ± SE (wins) | v12 without the stacker − v11 ± SE (wins) | v12 − v12 without the stacker ± SE (wins) |
|---|---|---|---|---|---|---|---|
| Metrica strict, 3 games | 0.7976 | 0.8155 | 0.8139 | **0.8197** | +0.0042 ± 0.0024 (2/3) | −0.0017 ± 0.0040 (2/3) | +0.0059 ± 0.0023 (3/3) |
| DFL fine4, 21 game-seeds | 0.7590 | 0.7673 | 0.7717 | **0.7735** | +0.0062 ± 0.0014 (18/21) | +0.0045 ± 0.0014 (15/21) | +0.0018 ± 0.0008 (13/21) |
| SkillCorner fine3, 20 games | 0.6270 | 0.6335 | 0.6289 | **0.6333** | −0.0002 ± 0.0025 (10/20) | −0.0046 ± 0.0022 (7/20) | +0.0044 ± 0.0011 (15/20) |
| PFF fine3, 48 games | 0.8227 | 0.8319 | 0.8384 | **0.8413** | +0.0094 ± 0.0013 (42/48) | +0.0065 ± 0.0014 (35/48) | +0.0029 ± 0.0008 (34/48) |

SE is over games (Metrica 3, DFL 7). v12 − v9 was not computed as a paired difference; the differences of
the means above are +0.0221 / +0.0145 / +0.0063 / +0.0186. Metrica per game, v12: 0.8343, 0.8087, 0.8162
(v11: 0.8261, 0.8042, 0.8162). MacroF1 (Metrica / DFL / SkillCorner / PFF): v11 0.8529 / 0.9432 / 0.9031 /
0.9543; v12 0.8523 / 0.9443 / 0.9035 / 0.9565 (v12 − v11: −0.0006 ± 0.0006, +0.0012 ± 0.0003, +0.0004 ±
0.0007, +0.0022 ± 0.0003). Possessions per 90, v11 / v12: DFL 308.2 / 309.2 (truth 412.1); SkillCorner
283.1 / 283.5 (truth 278.5); PFF 253.5 / 256.9 (truth 313.5); Metrica v12 297, 271, 316 (truth 307, 283, 324).

**No decision rule.** The "v12" addendum, written before the stacker was fitted, applied no rule: the choice
between v11 and v12 was the user's. Before reading the result, the bench code reproduced v11's scores of
record exactly.

**Caveats (stated plainly):**
- **SkillCorner did not improve** (−0.0002 ± 0.0025, 10 of 20 games), and without the stacker the new members
  lose there (−0.0046 ± 0.0022). Every new group lost on SkillCorner in the TCN alone. Label-free, the match's
  ball statistics resemble the SkillCorner bench (touches per minute: match 15.7, SkillCorner bench 14.9,
  degraded PFF training 33.5, Metrica bench 47.2; `newfeat/dist_match.md`).
- **None of these benches is untouched.** The groups were selected on PFF fold 0, SkillCorner and two Metrica
  games, under a rule changed after results. Only DFL and PFF folds 1-3 took no part in that choice.
- **The Metrica gain is 1.8 SE on 3 games** (2 of 3 games better; the third is equal to four decimals).
- **The PFF comparison has v11's mismatch.** On PFF the TCN slot runs without test-time averaging, while the
  stacker trained on averaged TCN outputs.
- **The stacker leans on one member.** 87.5% of its split gain comes from the MS-TCN's probabilities (pointer
  5.6%, TCN 3.5%, radar 3.3%), and 80% from the current frame.
- **Decoder setting not re-picked.** v8 and v9 took it from a grid on DFL; v11 and v12 reuse it.

## The match (label-free)

Two runs of the pipeline on the Mac (14 Nov, one thread, machine shared with other jobs), through a driver
that loads `final_v12/pipeline.py` and logs memory (`$MTAH_WORK/arch/r5_arch/newfeat/v12/code/run_match_v12.py`;
not through `make_submission.py`):

| run | pipeline | time | peak resident memory | CSV sha256 |
|---|---|---|---|---|
| 1 | without the cache release | 327 s | 7.41 GB (radar 6.9 GB, pointer 7.4 GB) | `eb249aa50cc9ab989514e982b9bfefabc3143f52b996811bb209f9429f5d849f` |
| 2 | as committed, with the cache release | 279 s | 5.77 GB | the same |

319 possessions (271.1 per 90 graded minutes), mean 10.9 s, dead share 0.455, no X inside the halves. v11:
325 possessions (276.2 per 90), mean 10.8 s, dead share 0.450. **97.46% of graded frames carry v11's label**
(4,031 of 158,857 differ); no frame outside the halves differs.

## Final checks (14 Nov 2026)

Verification ran once, on the version submitted. Every run wrote the CSV above (sha256 `eb249aa5…5d849f`),
and it passes `evaluation/validate_submission.py`.

| check | platform | wall time | peak memory (main process) |
|---|---|---|---|
| `make_submission.py` twice in a row | macOS arm64, Accelerate, torch 2.15.0, 1 thread | 267 s, 254 s | 5.9, 6.0 GB |
| same steps at 1 / 2 / 8 threads (driver outside the repo without the thread pinning) | macOS arm64 | 255 / 246 / 246 s | 6.1 / 5.9 / 6.0 GB |
| fresh clone of the commit that added v12, new venv from `requirements-submission.txt` | macOS arm64 | 280 s | 5.9 GB |
| clean clone in `python:3.14.6-slim` + libgomp1 | Linux arm64, OpenBLAS, torch 2.15.0+cpu | 10 min 55 s | 3.4 GB |
| same, x86-64 under Rosetta | Linux x86-64, torch BLAS MKL, AVX2 | 19 min 32 s | 3.3 GB |
| numpy pointer backend (`MTAH_PTR_BACKEND=numpy`) | macOS arm64 | 1197 s | 6.7 GB |

The Mac ran up to two checks at a time, so the times are not clean benchmarks. The torch worker that runs the
pointer is a separate process and is not in the memory column.

Live = offline: on DFL game J03WN1 at 3 degradation seeds, `pipeline.py` through `bench.py --dfl2 --folds` gave
exactly the codes that the bench decode gives from the saved member outputs. Largest probability differences:
TCN 0, MS-TCN 3.0e-8, radar 0, pointer 2.8e-6 (its saved outputs came from the float32 GPU path), stacker
4.8e-3; no member code differs.

Not done: native x86-64 hardware; live = offline on Metrica and SkillCorner units (the bench uses fold models
there and the pipeline the full models; v9 had extra full-model scoring runs for this, v12 does not); v9's
per-member noise probes.

v11's own platform checks were never run, because v12 replaced it.

## Training data

As v11: the 7 DFL / IDSSE open games (CC BY 4.0), the 48 PFF FC World Cup 2022 games, the 20 SkillCorner
A-League open-data games (MIT licence, © 2020 SkillCorner) and, for the MS-TCN and the stacker, Metrica
Sports' three open sample games (https://github.com/metrica-sports/sample-data; no licence file; its
README asks users to acknowledge the source in any public use, which we do). The training game counts of
every v12 model equal v11's. Nothing trained on the match, on FOOTPASS or on the 16 PFF holdout games.

## Files

| file | role |
|---|---|
| `pipeline.py` | the v12 pipeline (bench interface: `predict`, `predict_proba`) |
| `stacker_v12.txt`, `stacker_v12_meta.json` | the stacker and its record (LightGBM version, sha256, rows, gain shares) |
| `train_stacker.py` | trains the stacker from the saved Metrica outputs; its output is byte-identical to the all-games stacker scored above |
| `../newfeat_v12/nf_features.py` | v11's 29 columns (a copy of `../acc_shape/nf_features.py`) |
| `../newfeat_v12/nf_rel.py`, `nf_kin2.py`, `nf_int.py` | the 58 new columns |
| `../newfeat_v12/fe_v12.py` | base features + the column blocks, as a drop-in feature module, with an in-memory cache of the last inputs' blocks |
| `../newfeat_v12/_common.py`, `member_*.py` | member wrappers: the base pipelines unchanged, with features and weights replaced |
| `../newfeat_v12/tcn_v12.npz`, `mstcn_v12_skcft_metft.npz` | the two new weight files (full models) |

Nothing that v8, v9, v10 or v11 runs was changed.
