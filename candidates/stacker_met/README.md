# stacker_met: the Metrica-trained stacker "S2" (fifth member of candidate v10, `../final_f4sps`)

A small LightGBM model that reads the per-frame probabilities of v8's four members and outputs
P(home, away, dead) per frame. In candidate v10 its probabilities and codes enter the same
minimum-Bayes-risk decode as a fifth member. It was designed and scored in the round-5
experiment "stack_aug" (pre-registration addendum of 13 Nov 2026, 09:02 CST, `../../docs/prereg/round5_prereg.txt`; candidate "S2a").

| file | what it is |
|---|---|
| `stacker_s2.txt` | the LightGBM booster (text model, LightGBM 4.7.0), trained on all three Metrica games. sha256 `877902c48d56683a989858ababb7f647f0c61b5beabf61862b293ce6634aca1b`, 1.1 MB |
| `stacker_s2_meta.json` | LightGBM version, model sha256, training parameters, row and class counts, input layout. The pipeline checks the version and sha256 when it loads the model |
| `stacker.py` | input construction (`feats`) and prediction (`load`, `predict`), used by `../final_f4sps/pipeline.py` |
| `train_stacker.py` | trains `stacker_s2.txt` from the members' saved outputs (`train`), and compares a model file with the experiment's saved stacker outputs (`check`) |

## Inputs

There are 84 predictors: 4 members x 7 time offsets x 3 classes. The members are, in slot order, the TCN
(o_ls05 with test-time averaging), the MS-TCN `m5c_C192ls_skcft`, the radar `radar_pff` and the pointer
`r5_ptr_ballc_p20M`. These are v8's members, and none of them trained on Metrica. The 7 offsets are
−2, −1, −0.4, 0, +0.4, +1 and +2 s (frames −50, −25, −10, 0, +10, +25, +50 at 25 fps). Each source frame is
clamped inside the graded block (half) of the frame being predicted. The member probabilities are rounded
to float32 first.

**How it is evaluated per frame.** Training used every 5th frame. At prediction time the model is
evaluated at every frame inside the graded blocks, 25 per second. This is what the experiment's scoring
did (`stack.py`, `predict`); it was not redesigned here. Outside the blocks the probabilities are 1/3 each
and the code is X. The codes are the argmax of the float64 probabilities, and the probabilities are then
rounded to float32, as the experiment saved them. Prediction runs on one thread (`num_threads=1`).

## Training

- **Rows:** frames t with t % 5 == 0 inside the blocks that carry a strict Metrica-rule label H, A or D
  (`../control_events/metrica_rules.py`). They come from Metrica Sample Games 1, 2 and 3, each degraded
  with bench seeds 2027 and 2028 (`../bench.py`). Every row also enters a second time as a
  home/away-swapped copy: the H and A columns of every member and offset are exchanged, and so are the
  labels. This gives 339,912 rows (H 121,978, A 121,978, D 95,956).
- **Inputs:** the members' saved bench outputs on those games, listed in `train_stacker.py`. The
  pointer's outputs there come from its float32 GPU forward pass.
- **Model:** `LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, objective="multiclass",
  random_state=0, deterministic=True, force_row_wise=True, n_jobs=2)`. Everything else is left at the
  defaults, as fixed in the pre-registration.
- **Gain share:** by member, TCN 0.034, MS-TCN 0.026, radar 0.347, pointer 0.593. By offset (−2 s to
  +2 s): 0.027, 0.009, 0.010, 0.608, 0.218, 0.072, 0.058.

```sh
MTAH_WORK=... python -I candidates/stacker_met/train_stacker.py train [--out DIR]   # rebuilds stacker_s2.txt
MTAH_WORK=... python -I candidates/stacker_met/train_stacker.py check               # vs the experiment's outputs
```

**Reproducibility.** Training twice gave byte-identical files with the sha256 above. `check` applies the
shipped file through `stacker.predict` (one thread) to the members' saved outputs. On all 47 units it
equals the experiment's own all-games stacker outputs (`$MTAH_WORK/probs/stack_aug/S2_full`): 21 DFL
game-seeds, 20 SkillCorner games and the 6 in-sample Metrica units. The largest |ΔP| is 0, and no frame's
code differs.

## Training data

The stacker trained only on Metrica Sports' three open sample games:
https://github.com/metrica-sports/sample-data. That repository has no licence file. Its README asks users
to "be responsible with the use of this data" and says: "If you use it for anything public, please
acknowledge the source." We acknowledge it here: the Metrica Sports sample data (Sample Games 1-3),
github.com/metrica-sports/sample-data. The four members whose outputs it reads trained on the 7 DFL /
IDSSE open games (CC BY 4.0) and the 48 PFF FC World Cup 2022 games. The MS-TCN also trained on the 20
SkillCorner A-League open-data games (MIT licence, © 2020 SkillCorner). None of these trained on Metrica,
FOOTPASS or the 16 PFF holdout games.

Because the stacker trains on all three Metrica games, Metrica is not an independent benchmark for
anything that uses it.
