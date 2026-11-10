#!/bin/zsh
# Round 4 live checks of one nn_asformer config (= ../nn_pff/chain_r4.sh for this family):
#   candidates/nn_asformer/chain_r4.sh <config> numpy|torch [pff-only|no-pff|extras]
# extras = only Metrica dev + SkillCorner (configs whose PFF / DFL codes another chain makes).
# numpy = pipeline_cfg.py (asf_np, the shipping path); torch = pipeline_torch.py (asf_torch, screening only:
# the full-attention arch costs ~50 min per match in numpy).
# 1. PFF out-of-fold (unless no-pff): one run_pff.py per fold as fold_pff_<k>.npz appears, in parallel -> probs/nn_asformer/<config>.
# 2. (unless pff-only) when full.npz + fold_tcn_0-6.npz exist: bench --dfl2 -> probs/nn_asformer_dfl/<config>,
#    Metrica dev -> probs/metrica_dev/<config>, SkillCorner 20 games -> probs/skc/<config> (full model), in parallel.
NAME=$1; BACK=$2; MODE=$3
export MTAH_WORK=${MTAH_WORK:?set MTAH_WORK to the work directory} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
R=${0:A:h:h:h}  # repo root (this script is candidates/<family>/<name>.sh)
P=$MTAH_WORK/venv/bin/python
D=$MTAH_WORK/arch/nn_asformer/$NAME
MAXP=${MAXP:-10}
[ $BACK = torch ] && PIPE=$R/candidates/nn_asformer/pipeline_torch.py || PIPE=$R/candidates/nn_asformer/pipeline_cfg.py
slot() { while [ $(pgrep -f "candidates/bench.py|nn_pff/run_pff.py|skc/bench_skc.py|nn_asformer/run_pff.py" | wc -l) -ge $MAXP ]; do sleep 30; done; }
cd $R
if [ "$MODE" != no-pff ] && [ "$MODE" != extras ]; then
  for k in 0 1 2 3; do
    while [ ! -f $D/fold_pff_$k.npz ]; do sleep 30; done
    [ -f $D/pff_folds.json ] || $P candidates/nn_pff/folds_json.py $D > /dev/null
    G=$($P -c "import json;print(' '.join(json.load(open('$MTAH_WORK/arch/nn_pff/selection_folds.json'))['folds'][$k]))")
    slot
    NN_ASF_MODEL=$D/fold_pff_0.npz $P candidates/nn_asformer/run_pff.py $PIPE --name $NAME --folds $D/pff_folds.json \
        --model $D/fold_pff_0.npz --games ${=G} > $D/oof_$k.log 2>&1 &
    sleep 5
  done
  wait
  echo "PFF_DONE $NAME"
fi
[ "$MODE" = pff-only ] && exit 0
while [ ! -f $D/full.npz ]; do sleep 30; done
for g in 0 1 2 3 4 5 6; do while [ ! -f $D/fold_tcn_$g.npz ]; do sleep 30; done; done
$P - <<PY
import hashlib, json
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tcn"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
PY
if [ "$MODE" != extras ] && [ ! -f $MTAH_WORK/probs/nn_asformer_dfl/$NAME/dfl2_DFL-MAT-J03WR9_2029_codes.npy ]; then
  slot
  NN_ASF_MODEL=$D/full.npz $P candidates/bench.py $PIPE --dfl2 --folds $D/folds_dfl.json \
      --save-probs $MTAH_WORK/probs/nn_asformer_dfl/$NAME > $D/bench_dfl2.log 2>&1 &
fi
if [ ! -f $MTAH_WORK/probs/metrica_dev/$NAME/dev_Sample_Game_2_2028_codes.npy ]; then
  slot
  NN_ASF_MODEL=$D/full.npz $P candidates/bench.py $PIPE --save-probs $MTAH_WORK/probs/metrica_dev/$NAME > $D/bench_dev.log 2>&1 &
fi
slot
NN_ASF_MODEL=$D/full.npz $P candidates/skc/bench_skc.py run $PIPE --name $NAME > $D/skc.log 2>&1 &
wait
echo "CHAIN_DONE $NAME"
