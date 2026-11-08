# pathcrf_dynamic: a CRF whose transition costs depend on the frame

**What it is.** A linear-chain CRF over the three frame states (H, A, D). The transition
potentials are not fixed numbers: each one is computed from features of the frame itself. The
CRF is trained on the open DFL data and decoded with Viterbi. `predict_proba` returns the CRF's
forward-backward posterior marginals.

**Result.** It matches LDE on DFL but does not beat it. DFL LOGO spellF1 is **0.6248**
(macroF1 0.9147); LDE scores 0.6243 on the same bench. Metrica dev is 0.6649 against LDE's 0.6725,
and the Metrica holdout (Game 3, run once at the end) is 0.6812 against LDE's 0.6845. All of these
differences are inside the bench's noise band (about 0.04 spellF1). Learned dynamic transitions
also do **not** fix the under-splitting: the CRF still predicts 304-348 possessions per DFL game
where the truth has 370-511.

## Method

Reference: H. Kim, K. Lee, S. Seo, S.-K. Ko, J. Yoon, C. Park, *PathCRF: Ball-Free Soccer Event
Detection via Possession Path Inference from Player Trajectories*, arXiv:2602.12080 (2026),
code at github.com/hyunsungkim-ds/pathcrf. PathCRF's design has four parts:
- At each step it chooses one possession edge among 26x26 player and "outside" nodes.
- An MLP produces the emission score from the edge embedding.
- The transition score is dynamic, `psi_t(e', e) = MLP_trans([z_e',t-1 ; z_e,t])`, with impossible
  transitions masked out.
- It is trained with CRF negative log-likelihood (forward algorithm) plus auxiliary losses, and
  decoded with Viterbi.

The challenge asks only for the possessing team and has no player identities, so the state space
here is reduced to the team level, {H, A, D}. The masking becomes symmetry constraints between
states:

```
S(y) = sum_t f_t(y_t) + sum_{t>=1} psi_t(y_{t-1}, y_t)
f_t      = tau * LDE emission: (log pH + log(1-pD), log pA + log(1-pD), log pD), tau = 0.15
psi_t(A->H) = h_s([ssym_t,  santi_t])   psi_t(H->A) = h_s([ssym_t, -santi_t])   (team-symmetric)
psi_t(H->D) = psi_t(A->D) = h_x(die_t)
psi_t(D->H) = h_r([rsym_t,  ranti_t])   psi_t(D->A) = h_r([rsym_t, -ranti_t])
psi_t(i->i) = 0
h(x) = w.x + v.tanh(W x + b)            (8 hidden units: the analogue of PathCRF's MLP_trans)
```

**Emissions.** The emissions are LDE's evidence models: model_d, model_ha and model_c, copied
byte-identically. Its local-contrast team logit and long-gap dead term are used as well.

**Transition features** (`crf_features.py`; each looks both ways around frame t):
- **Switch:**
  - the log-odds of LDE's hand-over detector (model_c), and how far it is below its local ±0.5 s
    maximum;
  - ball seen or not, ball speed, change in ball speed, ball-gap length, and the nearest player's
    distance to the ball;
  - how much the team logit changes, plus an oriented part: the team-logit change over ±0.24 s and
    ±1 s, which team is nearer the ball, a change in the last confident controller, and the team
    logit itself.
- **Die:** the dead log-odds and how it changes over ±0.24 s and ±1 s, ball-out onset, ball
  seen, gap length, players' mean speed, the share of stationary players, and the long-gap
  indicator.
- **Revive:** the reverse of the die features, plus the restarting team's logit.

**Training** (`train_crf.py`, numpy and scipy only, no torch):
- **Data (`prep.py`).** Fresh domain-randomised degradations of the 7 DFL games, using LDE's
  `make_dataset.degrade_dfl` recipe copied into `dfl_degrade.py`. The seeds are 61 and 62; LDE
  used 11-13 and the bench uses 2027. On each game gi the emissions are computed with LDE's
  **leave-one-game-out fold boosters** for gi, cut to the full models' number of trees exactly as
  `bench.py` does. So every training emission is out-of-fold, the same condition as the bench and
  the match. The bench caches are never read.
- **Windows.** The graded halves are cut into 30 s windows at full 25 fps (2662 windows).
- **Loss.** The final model uses a **structured hinge (margin-rescaled structural SVM)**,
  `max_y' [S(y') + Delta(y, y')] - S(y)`, where Delta is built to match spellF1:
  - a per-frame Hamming term weighted `0.05 * (1 + 75 / length of the gold run)`, so every gold
    spell, short or long, costs at least about 3.75 when it is missed;
  - 0.5 for each predicted change that has no gold change of the same type within ±12 frames
    (a one-frame offset is free in spellF1).

  It is optimised with full-batch subgradient Adam (lr 0.02, 150 iterations, Polyak averaging over
  the second half). The pairwise biases start at LDE's tuned constant costs (switch 0.75, dead 5 at
  tau 0.15), and an L2 penalty of 1e-5 is applied. All exact gradients come from
  forward-backward or Viterbi edge counts; a numerical gradient check was run for each loss.
- **Leave-one-game-out.** `fold_crf_<gi>.npz` is trained on the other six games (both seeds).
  `crf.npz` is trained on all seven. The bench loads `crf.npz` through `dfl_folds.resolve`, which
  swaps in the fold file.
- **Leakage note.** The emissions on game j come from LDE fold model j, which was trained on games
  that include gi. The CRF for fold gi therefore sees game gi only second-hand, through LDE's
  behaviour on other games, and never sees its labels. This is the usual stacking approximation; a
  fully nested set of 42 LDE models was not affordable.

**Decoding.** Viterbi per graded half, with no post-processing. On the match only, LDE's Smart
Tagging priors are folded into the evidence (team ±0.1, goal-gap dead +1) and the pre-kick-off D
is kept; both are no-ops on the bench. `predict_proba` returns the forward-backward marginals of
the same potentials. Scale 1 is close to NLL-optimal on held-out degradations (NLL 0.30 / 0.26
against 0.42 / 0.35 at scale 2 and 0.58 / 0.64 at 0.5), so no recalibration is applied.

## Variants tried (bench --dfl, LOGO fold models, seed 2027; kept only if DFL spellF1 improved)

| variant | DFL spellF1 / macroF1 | possessions (pred, 7 games) |
|---|---|---|
| reference: LDE round 3 (fixed costs + full post chain) | 0.6243 / 0.9144 | |
| `const`: this CRF code with LDE's constant costs (0.75/5/5), no post chain | 0.6212 / 0.914 | 309-373 |
| v1: learned unary + linear dynamic transitions, CRF NLL | not benched. Offline on games 0-1: Viterbi 0.275 (145/131 poss), MPM 0.509. The unary collapsed to a local contrast of the team logit and the team was carried by the transitions: degenerate for MAP | |
| v2: fixed LDE unary + linear dynamic transitions, CRF NLL (L-BFGS, converged) | 0.6020 / 0.9126 | 257-302 |
| v2 + switch bias +1 | 0.6060 / 0.9131 | |
| v2 + switch bias +1 + LDE post chain | 0.6050 / 0.913 | |
| v2, MPM decode (argmax of marginals + merge_short_v2) | 0.6134 / 0.9141 | 272-322 |
| v4: softmax-margin (Gimpel & Smith 2010) with the same Delta, L-BFGS | 0.5989 / 0.9122 | |
| v3: structured hinge, biases initialised at -4/-6, lr 0.05, 200 it, cfp 1 | 0.6191 / 0.9139 | 283-338 |
| v5: structured hinge, biases initialised at LDE's costs, lr 0.02, 150 it, cfp 0.5 | 0.6240 / 0.9146 | 304-349 |
| v5 + LDE post chain | 0.6188 / 0.9144 | |
| v5 + switch bias +0.3 (more splits) | 0.6203 / 0.9146 | 310-370 |
| **v6 = v5 with 8-unit tanh MLP potentials (final)** | **0.6248 / 0.9147** | 304-348 |

**Why maximum likelihood fails here.** The NLL-trained CRF learns calibrated transition rates. On
correlated 25 fps evidence each individual hand-over stays uncertain, so the MAP path skips it, and
the model makes switching more expensive than LDE's tuned constants (learned bias -2.2 against
-0.75). The structured hinge trains the MAP decoder directly; started from LDE's costs it gives
+0.0036 over `const`. That gain is real but small: the dynamic features move the change points,
not their number. Every attempt to force more splits made the score worse (switch bias,
post chain).

## Final scores (bench runs, all with `--save-probs`)

- DFL LOGO: spellF1 **0.6248**, macroF1 0.9147. Per game, spellF1 and possessions predicted/true:
  | game | spellF1 | possessions |
  |---|---|---|
  | WMX | 0.5612 | 323/511 |
  | WN1 | 0.5758 | 317/500 |
  | WOH | 0.6686 | 304/402 |
  | WOY | 0.6195 | 324/456 |
  | WPY | 0.6345 | 309/421 |
  | WQQ | 0.6398 | 348/401 |
  | WR9 | 0.6742 | 307/370 |
- Metrica dev: 0.6649 / 0.8117
- Metrica holdout G3 (run once, at the end): 0.6812 / 0.8459
- Match: 334 possessions, mean 10.1 s, dead share 0.470, no X inside the halves. Two `--match`
  runs gave byte-identical files (`cmp`); sha256 fefa7f54...84b9.

## Data and licences

- DFL / IDSSE open tracking. Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of
  spatiotemporal and event data in elite soccer*, Scientific Data 12, figshare 28196177,
  doi:10.6084/m9.figshare.28196177, CC BY 4.0. It is fetched by
  `learned_dfl_evidence/fetch_idsse.py` and is never committed. The labels are the tracking
  feed's BallStatus and BallPossession flags.
- LDE's models and code are reused under the repository licence: `model_*.txt`,
  `model_meta.json`, `features.py`, `lde_base.py` (= LDE `pipeline.py`) and `dfl_degrade.py`
  (= LDE `make_dataset.py`).
- Nothing is trained on Metrica or on the competition match, and no hand labels are used.
  Decoder choices were made on DFL LOGO only.

## Rebuild

```
PY=<venv python>; S=<scratchpad>; A=$S/arch/pathcrf_dynamic
$PY -I candidates/pathcrf_dynamic/prep.py --out $A/data --seeds 61,62 --jobs 2      # ~2 min
sh candidates/pathcrf_dynamic/train_all.sh $PY $A/data $A/v6 --tau 0.15 --loss ssvm --iters 150 \
   --init-lde --lr 0.02 --cfp 0.5 --hidden 8                                     # ~12 min, 2 procs
cp $A/v6/crf.npz candidates/pathcrf_dynamic/crf.npz
mkdir -p $A/folds && cp $A/v6/folds/fold_crf_*.npz $A/folds/
$PY -I candidates/pathcrf_dynamic/folds_json.py candidates/pathcrf_dynamic/crf.npz $A/folds $A/folds.json
$PY candidates/bench.py candidates/pathcrf_dynamic/pipeline.py --dfl --folds $A/folds.json --save-probs $S/probs/pathcrf_dynamic
$PY candidates/bench.py candidates/pathcrf_dynamic/pipeline.py --save-probs $S/probs/pathcrf_dynamic
$PY candidates/bench.py candidates/pathcrf_dynamic/pipeline.py --match --save-probs $S/probs/pathcrf_dynamic
```

**Determinism.** Fixed seeds, numpy/BLAS with OMP_NUM_THREADS=1, and lightgbm with
num_threads=1 at inference. The training script is deterministic by construction, but a
byte-identical *retrain* of `crf.npz` has not been checked. The match output has been checked
byte-identical across two runs.

`PCD_SWEEP=1` with `PCD_<KEY>` / `PCD_CRF` overrides is for tuning only. The default run ignores
the environment.
