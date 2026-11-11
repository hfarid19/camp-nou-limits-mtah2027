#!/bin/zsh
# Round-5 "+SKC" scoring units on the Mac (OMP_NUM_THREADS=2; run through lanes.py, at most 2 at once):
#   run_mac.sh <variant dir> ptr|mstcn <name> skccv <k>      SkillCorner CV fold k: its 5 games with fold_skc_<k>.npz only
#   run_mac.sh <variant dir> ptr|mstcn <name> nowin <a> <b>  10-minutes-out model on games a..b-1 (of the sorted 20)
#   run_mac.sh <variant dir> ptr|mstcn <name> dfl <i,j,..>   DFL LOGO bench on the listed DFL games (indices)
#   run_mac.sh <variant dir> ptr|mstcn <name> met            Metrica dev (full model)
D=$1; KIND=$2; NAME=$3; WHAT=$4
: ${MTAH_WORK:?set MTAH_WORK}; export MTAH_WORK OMP_NUM_THREADS=2
R=${0:A:h:h:h}; P=$MTAH_WORK/venv/bin/python
case $KIND in mstcn) PIPE=$R/candidates/nn_pff/pipeline_cfg.py;; ptr) PIPE=$R/candidates/nn_pointer/pipeline_cfg.py;; esac
cd $R
case $WHAT in
skccv)
  K=$5
  G=$($P -I - <<PY
import sys; sys.path.insert(0, "candidates/nn_pff")
import numpy as np, data as D
games = D.skc_cv_folds()[$K]
tg = {str(g) for g in np.load("$D/fold_skc_$K.npz")["train_games"]}
assert not tg & set(games), ("fold model trained on a scored game", tg & set(games))
assert len([g for g in tg if g.startswith("SKC-")]) == 15
print(" ".join(g[4:] for g in games))
PY
)
  [ -z "$G" ] && { echo "fold $K check FAILED"; exit 1; }
  echo "fold $K games $G (check OK)"
  NN_PFF_MODEL=$D/fold_skc_$K.npz $P -I candidates/nn_pointer/skc_both.py $PIPE $NAME --games ${=G};;
nowin)
  G=$(ls $MTAH_WORK/skc_aleague/npz | sed 's/SKC-//; s/.npz//' | sed -n "$(($5+1)),$6p")
  NN_PFF_MODEL=$D/nowin.npz $P -I candidates/nn_pointer/skc_both.py $PIPE ${NAME}_nowin --games ${=G};;
dfl)
  $P -I - <<PY || exit 1
import hashlib, json, os, numpy as np
dfl = sorted(f[:-4] for f in os.listdir("$MTAH_WORK/idsse/npz") if f.endswith(".npz"))
for g, m in enumerate(dfl):
    tg = {str(x) for x in np.load("$D/fold_tcn_%d.npz" % g)["train_games"]}
    assert m not in tg and len([x for x in tg if x.startswith("SKC-")]) == 20, (g, m)
f = json.load(open("$MTAH_WORK/arch/ensemble_ctrl_v5c/folds.json"))
f[hashlib.sha256(open("$D/full.npz", "rb").read()).hexdigest()[:12]] = ["$D", "tcn"]
json.dump(f, open("$D/folds_dfl.json", "w"), indent=1)
print("DFL fold check OK")
PY
  M=$($P -c "import os;d=sorted(f[:-4] for f in os.listdir('$MTAH_WORK/idsse/npz') if f.endswith('.npz'));print(','.join(d[int(i)] for i in '$5'.split(',')))")
  BENCH_DFL_ONLY=$M NN_PFF_MODEL=$D/full.npz $P -I candidates/bench.py $PIPE --dfl2 --folds $D/folds_dfl.json \
      --save-probs $MTAH_WORK/probs/nn_pff_dfl/$NAME;;
met)
  NN_PFF_MODEL=$D/full.npz $P -I candidates/bench.py $PIPE --save-probs $MTAH_WORK/probs/nn_pff_metrica/$NAME;;
esac
