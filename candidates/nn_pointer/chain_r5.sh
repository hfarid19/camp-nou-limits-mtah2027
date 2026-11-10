#!/bin/zsh
# Round 5 copy of nn_pff/chain_r4.sh (adds KIND ptr = nn_pointer). Round 4 live checks of one config as its models arrive from the cloud (= chain.sh + Metrica dev + SkillCorner):
#   candidates/nn_pff/chain_r4.sh <config> tcn|mstcn [dfl-only]
# 1. (unless dfl-only) PFF out-of-fold: one run_pff.py per fold as soon as fold_pff_<k>.npz exists, the 4
#    folds in parallel into probs/nn_pff/<config> (disjoint games, so resolve.log's per-game --expect check holds).
#    dfl-only = models trained on all 64 PFF games (NN_PFF_ALLOW_HOLDOUT=1): no PFF out-of-fold score exists.
# 2. when full.npz and fold_tcn_0-6.npz exist: folds JSON = ensemble_ctrl_v5c/folds.json + sha(full) -> config dir,
#    bench.py --dfl2 (leave-one-DFL-game-out fold models) -> probs/nn_pff_dfl/<config>;
#    in parallel Metrica dev (full model) -> probs/nn_pff_metrica/<config> and SkillCorner 20 games -> probs/skc/<config>.
# Scoring processes on the Mac are capped (wait while >= MAXP bench / run_pff / bench_skc processes run).
NAME=$1; KIND=$2; MODE=$3
export MTAH_WORK=${MTAH_WORK:?set MTAH_WORK to the work directory} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
R=${0:A:h:h:h}  # repo root (this script is candidates/<family>/<name>.sh)
P=$MTAH_WORK/venv/bin/python
D=$MTAH_WORK/arch/nn_pff/$NAME
MAXP=${MAXP:-10}
case $KIND in mstcn) PIPE=$R/candidates/nn_pff/pipeline_cfg.py;; tcn) PIPE=$R/candidates/nn_pff_tcn/pipeline_cfg.py;; ptr) PIPE=$R/candidates/nn_pointer/pipeline_cfg.py;; seg) PIPE=$R/candidates/nn_seg/pipeline_cfg.py;; esac
slot() { while [ $(pgrep -f "candidates/bench.py|nn_pff/run_pff.py|skc/bench_skc.py|nn_asformer/run_pff.py" | wc -l) -ge $MAXP ]; do sleep 30; done; }
cd $R
if [ "$MODE" != dfl-only ]; then
  for k in 0 1 2 3; do
    while [ ! -f $D/fold_pff_$k.npz ]; do sleep 30; done
    [ -f $D/pff_folds.json ] || $P candidates/nn_pff/folds_json.py $D > /dev/null
    G=$($P -c "import json;print(' '.join(json.load(open('$MTAH_WORK/arch/nn_pff/selection_folds.json'))['folds'][$k]))")
    slot
    NN_PFF_MODEL=$D/fold_pff_0.npz $P candidates/nn_pff/run_pff.py $PIPE --name $NAME --folds $D/pff_folds.json --model $D/fold_pff_0.npz --games ${=G} > $D/oof_$k.log 2>&1 &
    sleep 5
  done
fi
while [ ! -f $D/full.npz ]; do sleep 30; done
for g in 0 1 2 3 4 5 6; do while [ ! -f $D/fold_tcn_$g.npz ]; do sleep 30; done; done
$P - <<PY
import hashlib, json
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tcn"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
PY
slot
NN_PFF_MODEL=$D/full.npz $P candidates/bench.py $PIPE --dfl2 --folds $D/folds_dfl.json \
    --save-probs $MTAH_WORK/probs/nn_pff_dfl/$NAME > $D/bench_dfl2.log 2>&1 &
slot
NN_PFF_MODEL=$D/full.npz $P candidates/bench.py $PIPE --save-probs $MTAH_WORK/probs/nn_pff_metrica/$NAME > $D/bench_dev.log 2>&1 &
slot
NN_PFF_MODEL=$D/full.npz $P candidates/skc/bench_skc.py run $PIPE --name $NAME > $D/skc.log 2>&1 &
wait
echo "CHAIN_DONE $NAME $(tail -1 $D/bench_dfl2.log | cut -c1-120)"
