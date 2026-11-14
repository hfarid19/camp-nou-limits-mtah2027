# final_v11: submission v11 (new-feature members + stacker)

`pipeline.py` is v9 (`../final_f4spm`) with two changes. The first passed its rule on its own. The second (variant E3 of
"stack_ext") was ahead of v9 on all four benches but did not pass the rule of its own addendum, which compared it with
v10's stacker; the combination then passed the rule of the "v11" addendum (see Evidence):

1. **New-feature members.** The TCN and the MS-TCN are replaced by retrained versions that read 223 inputs
   instead of 194: the repo features plus 29 "ACC + SHAPE" columns (`../acc_shape`). The training recipes
   are the shipped ones; only the input columns differ (experiment "acc_ens").
2. **A stacker as a fifth member** of the same minimum-Bayes-risk decode, as in v10 (`../final_f4sps`): the
   S2 recipe (LightGBM on the four members' probabilities at 7 time offsets, trained only on Metrica's
   three open sample games), retrained on the outputs of *these* four members.

Radar, pointer, decoder setting (`src hmm, alpha 0.1, rho 0.2, margin 0.0`, fixed sampling seed) and dead
ball before each half's kick-off are v9's. Pre-registration: `../../docs/prereg/round5_prereg.txt`,
addendum of 13 Nov 2026 18:32 CST ("v11"); the two parents are the addenda of 11:59 ("acc_ens") and 10:30
("stack_ext", variant E3).

**Status: the intended submission since 13 Nov 2026 (the user's choice).** `make_submission.py` runs it by default
(`DEFAULT_CANDIDATE = "final_v11"`). The checks listed under "Deferred checks" have not been run yet. <!-- TODO final check -->

```sh
python make_submission.py                                           # writes evaluation/results/camp_nou_limits.csv
python make_submission.py --candidate final_v11 --out <path>.csv   # the same pipeline, to another path
```

| slot | member file | what it is | weights |
|---|---|---|---|
| TCN | `acc_shape/member_tcn_as_tta4_a50.py` | the `o_ls05` recipe on 223 inputs; test-time averaging over 4 jittered copies and all_a50 renormalisation as v9 | `acc_shape/tcn_as.npz` (0.9 MB, sha256 `56f2965647d1…`) |
| MS-TCN | `acc_shape/member_mstcn_as_skcft_metft.py` | the `m5c_C192ls` → SkillCorner fine-tune → Metrica fine-tune chain on 223 inputs | `acc_shape/mstcn_as_skcft_metft.npz` (10.6 MB, sha256 `e2ab3232c483…`) |
| radar | `ball_radar_pff/pipeline.py` | `radar_pff`, v9's | unchanged |
| pointer | `nn_pointer/pipeline_ballc_p20m.py` | `r5_ptr_ballc_p20M`, v9's; torch worker | unchanged |
| stacker | `stacker_v11.txt` (this folder), applied by `stacker_met/stacker.py` `predict` | S2 recipe: 200 trees per class, depth 4, lr 0.05; 84 inputs | 1.1 MB, sha256 `266a35b1a0e500bbdab4268bd8ac93401b72b14a29eda9eff9ad17f407c53c70` |

**The 29 columns** (`../acc_shape/nf_features.py`, numpy + `scipy.ndimage.maximum_filter1d`; scipy is in
`requirements-submission.txt`). ACC, 14 columns: ball acceleration and turn angle with their maxima over
±0.4 / 1 / 2 s, and the acceleration of each team's nearest player and of its players within 3 m of the
ball. SHAPE, 15 columns: each team's convex-hull area and the hulls' overlap, the angle and triangle area
the two nearest players of each team make at the ball, and open passing lanes. None depends on the attack
direction. The file is the experiment's module with two changes: the path to the repo's base features, and
the SHAPE pass works on blocks of 512 frames instead of 2,048 (memory only; see "The match").

**Order of operations** is v10's: member probabilities rounded to float32, stacker at every frame inside
the graded halves on one thread, its probabilities rounded to float32, `decoder.decode(four members +
stacker)`, then `prekick_dead`.

## Evidence

All four benches, fixed decoder setting on every unit. Every model that scored a game had never seen it:
DFL by leave-one-game-out fold models, SkillCorner by cross-validation models, PFF by fold models, Metrica
leave one game out for both the Metrica-trained MS-TCN and the stacker. On DFL, SkillCorner and PFF the
stacker trained on all three Metrica games is applied. Work log and code:
`$MTAH_WORK/arch/r5_arch/v11/` (`out/report_v11.txt`, `NOTES.md`).

| bench | v9 | v9 + new members | E3 (v9 + stacker) | **v11** | v11 − (v9 + new members) ± SE (wins) | v11 − E3 ± SE (wins) | v11 − v9 ± SE (wins) |
|---|---|---|---|---|---|---|---|
| Metrica strict, 3 games | 0.7976 | 0.8069 | 0.8089 | **0.8155** | +0.0087 ± 0.0019 (3/3) | +0.0067 ± 0.0052 (2/3) | +0.0179 ± 0.0063 (3/3) |
| DFL fine4, 21 game-seeds | 0.7590 | 0.7661 | 0.7641 | **0.7673** | +0.0012 ± 0.0016 (10/21) | +0.0032 ± 0.0028 (12/21) | +0.0083 ± 0.0024 (17/21) |
| SkillCorner fine3, 20 games | 0.6270 | 0.6322 | 0.6294 | **0.6335** | +0.0013 ± 0.0013 (11/20) | +0.0040 ± 0.0023 (14/20) | +0.0065 ± 0.0020 (16/20) |
| PFF fine3, 48 games | 0.8227 | 0.8283 | 0.8282 | **0.8319** | +0.0036 ± 0.0009 (38/48) | +0.0037 ± 0.0011 (33/48) | +0.0093 ± 0.0016 (39/48) |

SE is over games (Metrica 3, DFL 7). Metrica per game, v11: 0.8261, 0.8042, 0.8162. MacroF1 (Metrica /
DFL / SkillCorner / PFF): v9 0.8473 / 0.9393 / 0.9003 / 0.9500; v9 + new members 0.8503 / 0.9420 / 0.9014 /
0.9524; E3 0.8493 / 0.9414 / 0.9019 / 0.9527; v11 0.8529 / 0.9432 / 0.9031 / 0.9543. Possessions per 90,
v9 / v9 + new members / E3 / v11: DFL 308.8 / 307.3 / 310.0 / 308.2 (truth 412.1); SkillCorner 285.0 /
282.6 / 286.0 / 283.1 (truth 278.5); PFF 250.7 / 253.5 / 251.9 / 253.5 (truth 313.5); Metrica v11 290,
266, 316 (truth 307, 283, 324).

**Rule, written before the stacker was fitted:** v11 is "better" if, against v9 + new members and against
E3 separately, it wins on DFL or PFF by at least 2 game-level SE and loses on no bench by more than 1 SE.
Result: **passed**, on PFF against both (4.2 SE and 3.3 SE); no bench is lost.

Before reading the result: the scoring code refitted E3 and reproduced the earlier experiment's saved
outputs bit for bit (95/95 units), and v9, v9 + new members and E3 reproduced their earlier scores exactly.

**Caveats (stated plainly):**
- **On DFL the stacker adds nothing measurable on top of the new members**: +0.0012 ± 0.0016, 10 of 21
  units better. Its addition rests on PFF and on Metrica, which is 3 games. Against v9 the whole candidate
  is ahead on all four benches.
- **None of these benches is untouched.** Both parents were chosen on them, and the new-feature variant
  was picked as the best of four on PFF fold 0, SkillCorner and two Metrica games. Only DFL and PFF folds
  1-3 took no part in that choice. This run confirms a combination; it is not a fresh test.
- **The PFF comparison has a mismatch.** On PFF the TCN slot runs without test-time averaging (as for v8,
  v9 and v10), while the stacker trained on averaged TCN outputs.
- **The stacker leans on one member.** 89% of its split gain comes from the MS-TCN's probabilities (pointer
  6%, radar 4%, TCN 2%), and 79% from the current frame.
- **The stacker's MS-TCN inputs differ in kind between training and the match.** It trained on the
  MS-TCN's leave-one-game-out Metrica outputs (the game's fold model never saw it). At the match it gets
  the shipped MS-TCN, which trained on all three Metrica games but has not seen the match, so this is the
  same situation as on DFL, SkillCorner and PFF in the table.
- **Metrica is not independent** of the stacker or the MS-TCN; the Metrica numbers are leave one game out.
- **Decoder setting not re-picked.** v8 and v9 took their setting from a grid on DFL; v11 reuses it.
- **The new members were trained on a rented GPU (CUDA)**, the shipped TCN on the Mac GPU. All benchmark
  outputs of the new members were computed there too, with the numpy path on Linux (see "Deferred").

## The match (label-free)

One run of `make_submission.py --candidate final_v11` on the Mac (13 Nov, one thread, machine shared with
other jobs at load 8-9): **519 s**, peak resident memory **8.4 GB** in the main process (v10: 271 s,
5.2 GB). That run used blocks of 2,048 frames in the SHAPE pass. With blocks of 512 frames, now the value
in `nf_features.py`, a second run of the same pipeline (through a driver that set the block size and logged
memory, not through `make_submission.py`) peaked at **5.9 GB** and wrote the **same CSV, byte for byte**.
On the match input the 29 columns are bit-identical for block sizes 256, 512, 1,024 and 2,048. Of the
5.9 GB, the TCN reaches 4.3 GB and the MS-TCN 4.8 GB; the peak falls in the radar member, which is v9's.
Time per member in that run: TCN 290 s (five feature passes, about 55 s each for the SHAPE columns),
MS-TCN 58 s, radar 65 s, pointer 221 s. CSV sha256 `9ba4f14d75d872284ecd76cd1c6c6308ad5bf7f9d6c9fedcf3d2b2ce46586938`; format valid
(`evaluation/validate_submission.py`). 325 possessions (276.2 per 90 graded minutes), mean 10.8 s, dead
share 0.450, no X inside the halves. v9 has 321 possessions (272.8 per 90), mean 10.8 s, dead share 0.456.
**96.65% of graded frames carry v9's label** (5,328 of 158,857 differ); no frame outside the halves differs.

**Are the new columns at the match like those the models trained on?** (`match_dist.py`,
`report_match.py` of the acc_ens experiment; table `$MTAH_WORK/arch/r5_arch/v11/tables_match.md`.) Per
column, the share of non-zero values and six quantiles were computed for the match and for each of the 95
bench units (degraded inputs). Of 203 column statistics, **170 lie inside the range of at least one
benchmark's units and 33 lie outside all four**. Ball seen: 59.3% of frames, inside the range.
- `ball_turn`, one of the two columns the gain rests on: all 7 statistics inside.
- `ball_acc`, the other: the middle of the distribution is inside; the tails are narrower at the match
  than on any bench unit. The 99th percentile is 27.5 m/s² (bench units: 33.5 to 59.9) and the 95th
  percentile of the ±0.4 s maximum is 27.1 (bench: 33.5 to 70). The simulated ball degradation produces
  larger acceleration spikes than the real tracking has.
- Player acceleration: the upper tail is at or just above the highest bench unit (95th percentile of the
  nearest player's acceleration: home 5.48 m/s² against at most 5.44, away 5.50 against at most 5.48).
- Team shape: in 1-2% of match frames a team has fewer than 3 tracked players, so its hull area is 0;
  that never happens on the benches. The two hulls overlap more (95th percentile 0.97; bench at most 0.94)
  and the home hull exceeds the away hull by more than on any bench unit.

So the match is mostly, not entirely, inside the benchmark range. The departures are in the tails of the
ball acceleration and in the hull columns. What they cost cannot be measured without labels.

## Checks done

**The repo's member wrappers reproduce the experiment's outputs** (13 Nov, Mac numpy; block size 2,048 at
the time). The wrappers in `../acc_shape` were run on one unit per bench with the fold model that never
saw the game, and compared with the saved Linux outputs every benchmark number above was computed from:

| bench unit | TCN form and model | TCN: codes differing, max abs ΔP | MS-TCN model | MS-TCN: codes differing, max abs ΔP |
|---|---|---|---|---|
| DFL J03WMX, seed 2027 | averaged + all_a50, `fold_tcn_0` | 0, 3.0e-8 | `fold_tcn_0` | 0, 2.4e-12 |
| SkillCorner 1874553 | averaged + all_a50, full model | 0, 3.0e-8 | `fold_skc_3` | 0, 2.3e-11 |
| Metrica Sample Game 1, seed 2027 | averaged + all_a50, full model | 0, 3.0e-8 | `fold_met_1` | 0, 4.4e-16 |
| PFF 10504 | **plain** (`member_tcn_as.py`), `fold_pff_0` | 0, 2.2e-16 | `fold_pff_0` | 0, 1.0e-9 |

No code differs on any of the 8 comparisons (145,006 to 151,824 frames each). The averaged TCN's outputs
were saved as float32; rounded to float32, the Mac probabilities equal them exactly (the 3.0e-8 is that
rounding). The other outputs were saved as float64 and differ from the Mac's by at most 1e-9, which is
Mac-versus-Linux floating-point rounding; after the pipeline's own float32 rounding, at most one float32
step (3.0e-8) remains. So the new columns and both members agree across the two platforms on these units.
PFF was scored with the plain TCN, as for v8, v9 and v10; v11 itself runs the averaged form.

## Deferred checks

By the user's decision (13 Nov), verification runs once, on the version finally submitted. **None of the
following has been done for v11:**
- Benchmark pipeline = this folder's `predict` on a unit per bench (live = offline) for the whole
  five-member pipeline: the numbers above come from the experiments' scoring scripts on saved member
  outputs, not from `pipeline.py`. The radar and pointer are v9's; the stacker step and the decode are
  v10's code with a different model file. The one-float32-step differences above reach the stacker,
  whose trees are sensitive to such steps (see v10's README); whether any output frame moves is not
  known.
- `make_submission.py --candidate final_v11` run twice for byte identity (the two runs above went through
  different drivers and gave the same CSV); thread counts; Linux containers; a fresh clone; the noise,
  rounding and threshold-margin probes of v10.
- Re-running `make_submission.py` after its edits (first `final_v11` added to `CANDIDATES`; then, on 13 Nov, the
  default changed to `final_v11` and the docstring rewritten), for the default and for `--candidate final_f4spm`.

Ready for those checks: fold mapping `$MTAH_WORK/arch/r5_arch/v11/folds_v11.json` (v9's entries plus the
two new weight files and the stacker), DFL / SkillCorner / PFF / Metrica fold models under
`$MTAH_WORK/pod_rescue/root/acc_ens/models/`, saved member outputs under
`$MTAH_WORK/pod_rescue/root/acc_ens/mw/probs/accens/`, stacker outputs under `$MTAH_WORK/probs/v11/S2n/`.
`acc_shape/member_tcn_as.py` is the TCN without test-time averaging, the form the PFF numbers used.

## Training data

As v9: the 7 DFL / IDSSE open games (CC BY 4.0), the 48 PFF FC World Cup 2022 games, the 20 SkillCorner
A-League open-data games (MIT licence, © 2020 SkillCorner) and, for the MS-TCN and the stacker, Metrica
Sports' three open sample games (https://github.com/metrica-sports/sample-data; no licence file; its
README asks users to acknowledge the source in any public use, which we do). Nothing trained on the match,
on FOOTPASS or on the 16 PFF holdout games.

## Files

| file | role |
|---|---|
| `pipeline.py` | the v11 pipeline (bench interface: `predict`, `predict_proba`) |
| `stacker_v11.txt`, `stacker_v11_meta.json` | the stacker and its record (LightGBM version, sha256, rows, gain shares) |
| `train_stacker.py` | trains the stacker from the saved Metrica outputs; its output is byte-identical to the all-games stacker scored above |
| `../acc_shape/nf_features.py` | the 29 columns |
| `../acc_shape/fe_as.py` | base features + the 29 columns, as a drop-in feature module; keeps the last 6 inputs' extra columns in memory |
| `../acc_shape/_common.py`, `member_*.py` | member wrappers: the base pipelines unchanged, with features and weights replaced |
| `../acc_shape/tcn_as.npz`, `mstcn_as_skcft_metft.npz` | the two new weight files (full models) |

Nothing that v8, v9 or v10 runs was changed.
