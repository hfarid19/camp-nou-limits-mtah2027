# final_f4sps: submission candidate v10 ("F4sP + S2"), not the submission

`pipeline.py` is `../final_f4sp` (F4sP, submission v8) with a **fifth member** in the same minimum-Bayes-risk
decode: the stacker **S2** (`../stacker_met`). S2 is a small LightGBM model, trained only on Metrica's three
open sample games, that takes the four members' probabilities at 7 time offsets as its inputs. v10 keeps
everything else from v8: the four members and their weights, the decoder setting (`src hmm, alpha 0.1,
rho 0.2, margin 0.0`, fixed sampling seed) and dead ball before each half's kick-off. It is the
candidate "S2a" of the round-5 stack_aug experiment
(`../../docs/prereg/round5_prereg.txt`, addendum of 13 Nov 2026 09:02 CST and the post-hoc addition of
09:27).

**Status: a candidate, not the submission.** `make_submission.py` still runs v9 (`candidates/final_f4spm`)
by default. v10 runs only on request:

```sh
python make_submission.py --candidate final_f4sps --out <path>.csv   # --out is required for a non-default candidate
python candidates/final_f4sps/match_run.py --tag <tag> [--probes]    # writes to $MTAH_WORK/arch/r5_arch/final_f4sps/csv/; refuses evaluation/
```

| slot | member file | what it is | weights |
|---|---|---|---|
| TCN | `adapt/member_tcn_tta4_a50.py` | `o_ls05`, v8's (test-time averaging over 4 jittered copies, all_a50 renormalisation) | `nn_pff_tcn/tcn_ls.npz` |
| MS-TCN | `nn_pff/pipeline_c192ls_skcft.py` | `m5c_C192ls_skcft`, **v8's** (not v9's Metrica fine-tuned MS-TCN) | `nn_pff/mstcn_c192ls_skcft.npz` |
| radar | `ball_radar_pff/pipeline.py` | `radar_pff`, v8's | `radar.npz`, `model_br*.txt` |
| pointer | `nn_pointer/pipeline_ballc_p20m.py` | `r5_ptr_ballc_p20M` (19.3M parameters), v8's; torch worker | `nn_pointer/ptr_ballc_p20m.npz` |
| stacker (new) | `stacker_met/stacker.py` | S2: LightGBM, 200 trees per class, depth 4, lr 0.05; 84 inputs = 4 members x (H, A, D) x offsets −2, −1, −0.4, 0, +0.4, +1, +2 s (clamped inside each half) | `stacker_met/stacker_s2.txt` (1.1 MB, sha256 `877902c48d56683a989858ababb7f647f0c61b5beabf61862b293ce6634aca1b`; trained on Metrica Sample Games 1-3) |

**Order of operations.** The four members' probabilities are rounded to float32. The stacker then
evaluates **every frame** inside the graded halves (25 per second; it was trained on every 5th frame) on a
single thread. Its codes are the argmax of its float64 probabilities, and the probabilities are then
rounded to float32, as the experiment saved them. Next comes `decoder.decode(four members + stacker, CFG)`
and then `prekick_dead`. This is how the experiment scored S2a (`stack_aug/code/stack.py` `predict`,
`score_S.py` `bench_job`); it was not redesigned here.

## How it was chosen

Full account: `$MTAH_WORK/arch/r5_arch/stack_aug/report.txt` and `NOTES.md`; tables in `out/S_report.txt`
and `out/S_pff_report.txt`. Under the 09:02 addendum, four stacking candidates were scored, all on v8's
four frozen members: S1 (logistic regression) and S2 (LightGBM), each used either (a) as a fifth MBR member
or (b) alone with the base rule. On Metrica, every stacker was trained on two games and scored on the
third (leave one game out). On DFL and SkillCorner, the stacker trained on all 3 games was applied. All
used the fixed decoder setting above.

| bench | v8 (F4sP) | v9 (F4sP-MF2) | v10 (S2a) | v10 − v9 ± SE (wins) | v10 − v8 ± SE (wins) |
|---|---|---|---|---|---|
| DFL fine4, 21 game-seeds | 0.7615 | 0.7590 | 0.7642 | +0.0052 ± 0.0013 (17/21) | +0.0026 ± 0.0018 (12/21) |
| SkillCorner-CV fine3, 20 games | 0.6285 | 0.6270 | 0.6281 | +0.0011 ± 0.0019 (11/20) | −0.0004 ± 0.0023 (11/20) |
| Metrica, leave one game out, strict, Sample Game 1 | 0.7912 | 0.8120 | 0.8041 | −0.0080 | +0.0128 |
| Metrica, leave one game out, strict, Sample Game 2 | 0.7883 | 0.7948 | 0.7927 | −0.0020 | +0.0044 |
| Metrica, leave one game out, strict, Sample Game 3 | 0.7808 | 0.7859 | 0.7920 | +0.0061 | +0.0112 |
| Metrica mean, strict | 0.7868 | 0.7976 | 0.7963 | **−0.0013 (1/3)** | +0.0095 (3/3) |
| PFF World Cup, 48 games out of fold, fine3 (**post hoc**) | 0.8252 | 0.8227 | 0.8288 | +0.0062 ± 0.0011 (38/48) | +0.0036 ± 0.0008 (36/48) |

Possessions per 90: DFL 310.4 / 308.8 / 310.8 (truth 412.1); SkillCorner 285.1 / 285.0 / 285.4 (truth
278.9); Metrica 299, 277, 316 / 294, 271, 312 / 301, 274, 316 (strict truth 307, 283, 324); PFF 13,618 /
13,561 / 13,627 possessions in all (truth 16,980). Metrica practice labels: v10 − v9 −0.0012 (1/3); v10 − v8
+0.0092 (3/3). MacroF1 DFL / SkillCorner / Metrica strict: v8 0.9409 / 0.9005 / 0.8450, v9 0.9393 / 0.9003 /
0.8473, v10 0.9413 / 0.9007 / 0.8475. The other three stacking candidates: S1a Metrica −0.0040 (1/3),
DFL +0.0048, SkillCorner +0.0017 vs v9. S1b and S2b were worse than v9 and v8 on every bench.

In the v8 and v9 rows, Metrica, SkillCorner and PFF were decoded with the same setting as v10. DFL uses
their official nested scores, which equal the fixed-setting decode on every DFL unit. On PFF, the "v8" and
"v9" references are the PFF-fold versions of their members ("v8 (PFF folds)", "v9 (PFF folds)").

**Caveats (stated plainly):**
- **v10 failed the pre-registered rule against v9.** The 09:02 rule needs held-out Metrica to be better on
  at least 2 of 3 games, with a mean gain of at least +0.005. v10 scored **−0.0013, better on 1 of 3
  games**. Its DFL and SkillCorner parts pass. Under the rule written before the results, v9 stands.
- **The comparison that favours it was read after the results.** The DFL gain (+0.0052 ± 0.0013) is part
  of the rule only as a "not worse by more than 0.005" check. PFF was added **post hoc**, at 09:27, after
  the S results were seen. It has no rule attached, and its TCN inputs do not match: on PFF the TCN
  outputs carry no test-time averaging, while the stacker trained on TTA outputs.
- **Selection.** v10 is the best of four stacking candidates (S1a, S1b, S2a, S2b) scored on three Metrica
  games, on top of the benches already reused for v8's and v9's design and selection. There is no clean
  confirmation: the FOOTPASS holdout was spent on v7.
- **Metrica is not independent.** The stacker trains on Metrica: the shipped model on all 3 games. The
  Metrica numbers above are leave one game out (3 games), and from now on no Metrica number for v10 is
  untouched evidence.
- **v10 uses v8's MS-TCN, not v9's Metrica fine-tuned one.** Metrica enters v10 only through the
  stacker, so v10 is not "v9 plus a stacker". Against v8, the base it was built on, v10 is better on
  Metrica (3/3), DFL and PFF, and level on SkillCorner.
- **Per frame vs per decode.** Per frame, S2 beats every member on held-out Metrica (log-loss 0.400 vs
  0.599 for the members' mean). Its argmax flickers, though: 1,688 label changes per unit, against 748 for
  the members' mean. Alone it loses (S2b). Its value comes only through the MBR decode.
- **Numerical margin** (match, see Checks). Some stacker inputs sit right at tree thresholds. A 1e-6
  relative perturbation of the member probabilities changes the stacker's probabilities by up to 0.09 and
  its codes at 1-3 frames. It changed **no output frame** in any probe, and none when every
  threshold-adjacent input was moved one float32 step across its threshold.

## Checks

All checks ran on the Mac (Apple M5 Max) or in local containers, on positions and tags of the match only,
with no labels. Work log: `$MTAH_WORK/arch/r5_arch/final_f4sps/NOTES.md`. Unless noted, all runs used the
torch backend for the pointer.

**Done:**
- **Stacker file = the experiment's stacker** (`../stacker_met/train_stacker.py check`). On 21 DFL
  game-seeds, 20 SkillCorner games and 6 in-sample Metrica units, the shipped `stacker_s2.txt` applied to
  the saved member outputs equals the experiment's all-games S2 outputs: max |ΔP| = 0, no frame's code
  differs. Training it twice gave byte-identical files.
- **Live = offline** (`live_vs_offline.py --members-dir`). The live codes of `pipeline.py`, run through
  the benches by `dump_members.py` (which wraps it unchanged), were compared with the experiment's
  offline S2a decode of saved outputs. They are identical on all **7 units** tried:
  - DFL J03WN1, all 3 degradation seeds. The bench resolved the member weights to their
    leave-one-game-out `fold_*_1` models, as for v8. The stacker passed through the bench's Booster
    swap; it never saw DFL, so its "fold model" is the same file.
  - Metrica Sample Game 1 and Sample Game 2, both seeds each. The offline reference here uses the
    **all-games** stacker (in-sample `S2_full/insample_met_*`), because that is the stacker `pipeline.py`
    ships. It is not the leave-one-game-out stacker the scored S2a used on Metrica. The member inputs are
    the ones the experiment trained and scored on.

  Slot by slot: the TCN, MS-TCN and radar probabilities equal the saved ones (max |ΔP| = 0). The pointer
  differs from its saved outputs, which came from the float32 GPU path, by 1.7e-6 to 2.8e-6, as in v8.
  Through tree thresholds, that difference moves the live stacker's probabilities by up to 2.8e-3 to
  1.6e-2 from the saved stacker's. No stacker code and no final code differs on any of the 7 units. The
  shipped stacker applied to the saved members reproduces the saved stacker exactly.
- **Match** (`match_run.py`: positions and tags only, no labels):
  - 326 possessions (277.0 per 90 graded minutes), mean 10.6 s, dead share 0.454, no X inside the
    halves;
  - possession lengths per 90 graded minutes, v10 / v9 / v8: under 1 s 1.7 / 3.4 / 4.2; under 2 s
    36.5 / 33.1 / 37.4; under 3 s 71.4 / 74.8 / 74.8. Quick changes per 90: A-B-A with the middle
    possession under 2 s, 17.0 / 12.7 / 14.4; team changes less than 2 s apart, 17.8 / 14.4 / 17.0. v9 has
    321 possessions (272.8 per 90), mean 10.8 s, dead 0.456; v8 has 326 (277.0), 10.6 s, 0.457;
  - 2.46% of graded frames differ from v9 (3,914 frames), 1.99% from v8 (3,165), 2.93% from v7 and
    3.88% from v6; no frame outside the graded halves differs;
  - the four members' match outputs equal v8's saved ones, and decoding them without the stacker gives
    v8's CSV, so the stacker is the only difference;
  - CSV sha256 `28ff7e1f3728dd50cbbde933f1dd7fbfea588131c8273b3c3d29c813c1fc7e14`;
    `validate_submission.py`: valid.
- **Byte-identical on macOS arm64** (Accelerate, torch backend): `make_submission.py --candidate
  final_f4sps` (one thread) and `match_run.py` with 1, 2 and 8 threads all gave the sha256 above. Every
  saved array (member and stacker probabilities and codes) is identical across 1, 2 and 8 threads.
- **Perturbation and threshold margin** (`match_run.py --probes`, `$MTAH_WORK/.../final_f4sps/scripts/ulp_probe.py`):
  - Noise probe (member probabilities × (1 + 1e-6 u), 3 draws): the stacker's codes change at 1 frame
    per draw and its probabilities by up to 0.043. **0 output frames change.**
  - Rounding to 1e-6: the stacker changes at 3 frames (max |ΔP| 0.090), 0 output frames. Noise then
    rounding: 0 output frames.
  - Margin: some stacker inputs sit right at tree thresholds. The smallest relative distance is 1e-16.
    14,467 of the 158,857 × 84 inputs lie within 1e-6 relative of a threshold, and 5,209 within 1e-7.
    4,677 inputs (3,431 frames) would cross some split if moved by one float32 step; 4,014 of these are
    the value float32(0.9998), the radar's clamp ceiling. Moving all 4,677 one step across at once
    changes the stacker's probabilities at 1,789 frames (max 0.032) but **no stacker code and no output
    frame**.
  - So at this match the output is insensitive to 1e-6-level changes, but the stacker itself is not.
    Cross-platform identity therefore depends on the members' float32 probabilities being bit-identical
    across platforms, which the Linux runs would have tested.
- **v9 stays the default.** After the change, `python make_submission.py` with no arguments rewrote
  `evaluation/results/camp_nou_limits.csv` with sha256 `9926b7c56557a7aedb39597b7c2f52b37c82028d01b394021a360bb3aabcb3e2`
  (v9: 321 possessions). `--candidate final_f4sps` without `--out` is refused.
- **Run time** (Mac shared with other agents' jobs, load 6-10): `make_submission.py --candidate
  final_f4sps` took 271 s on one thread, with a peak resident memory of 5.2 GB in the main process plus
  the torch worker. `match_run.py`: 282-367 s for 1, 2 and 8 threads, of which the pointer took
  198-262 s and the stacker plus decode about 3 s.

**Not done (stopped at the user's decision, 13 Nov ~10:40):**
- Linux arm64 container (python:3.14.6-slim + libgomp1, `requirements-submission.txt`);
- a fresh clone plus this change's files in such a container;
- Linux x86-64 under emulation;
- live = offline on SkillCorner (game 1874553): started and stopped before it finished;
- the numpy backend (`MTAH_PTR_BACKEND=numpy`): not run, as for v9;
- re-running `--candidate final_f4sp` (v8) after the change. The edit to `make_submission.py` only added
  `final_f4sps` to `CANDIDATES` and a docstring sentence.

v10 is therefore **byte-identical on macOS only**. Unlike v8 and v9, it has no cross-platform reproduction.

## Training data

The four members are v8's: the 7 DFL / IDSSE open games (CC BY 4.0) and the 48 PFF FC World Cup 2022
games, plus, for the MS-TCN, the 20 SkillCorner A-League open-data games (MIT licence, © 2020
SkillCorner). The stacker trained only on Metrica Sports' three open sample games,
https://github.com/metrica-sports/sample-data. That repository has no licence file. Its README asks users
to "be responsible with the use of this data" and, "if you use it for anything public, please acknowledge
the source". We acknowledge it: Metrica Sports sample data, Sample Games 1-3. Nothing trained on FOOTPASS
or on the 16 PFF holdout games.

## Files

| file | role |
|---|---|
| `pipeline.py` | the v10 pipeline (bench interface: `predict`, `predict_proba`; proba = vote shares of the five members' codes) |
| `match_run.py` | label-free BAR-ATH run: timing, statistics vs v6 / v7 / v8 / v9, the v8 check, sha256, the noise, rounding and threshold-margin probes; never writes under `evaluation/` |
| `live_vs_offline.py` | live codes vs the experiment's offline S2a decode of saved outputs (and, with `--members-dir`, slot by slot) |
| `dump_members.py` | bench wrapper for that check: runs `pipeline.py` unchanged and saves the five slots' live outputs |
| `../stacker_met/` | the stacker: model, meta, `stacker.py`, `train_stacker.py`, README |

Nothing that v8 or v9 runs was changed. `make_submission.py` gained `final_f4sps` in `CANDIDATES`. The
default is still `final_f4spm` (v9).
