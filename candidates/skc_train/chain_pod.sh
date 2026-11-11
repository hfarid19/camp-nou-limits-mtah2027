#!/bin/bash
# Round-5 "+SKC": score one variant on the pod as its models land (score_pod.sh): SkillCorner CV when fold_skc_0-3
# exist, the 10-minutes-out run when nowin.npz exists, the DFL LOGO bench when full + fold_tcn_0-6 exist.
#   candidates/skc_train/chain_pod.sh <variant dir> ptr|mstcn <name> [skip-skccv]
D=$1; KIND=$2; NAME=$3; SKIP=${4:-}
S=$(dirname "$0")/score_pod.sh
if [ "$SKIP" != skip-skccv ]; then
  for k in 0 1 2 3; do until [ -f $D/fold_skc_$k.npz ]; do sleep 30; done; done
  bash $S $D $KIND $NAME skccv
fi
until [ -f $D/nowin.npz ]; do sleep 30; done
bash $S $D $KIND $NAME nowin &
until [ -f $D/full.npz ]; do sleep 30; done
for g in 0 1 2 3 4 5 6; do until [ -f $D/fold_tcn_$g.npz ]; do sleep 30; done; done
/root/skcscore/bin/python -I - <<PY || { echo "DFL fold check FAILED"; exit 1; }
import json, os, numpy as np
dfl = sorted(f[:-4] for f in os.listdir("/root/work/idsse/npz") if f.endswith(".npz"))
for g, m in enumerate(dfl):
    tg = {str(x) for x in np.load("$D/fold_tcn_%d.npz" % g)["train_games"]}
    assert m not in tg and len([x for x in tg if x.startswith("SKC-")]) == 20, (g, m)
print("DFL fold check OK: fold_tcn_g never saw DFL game g; each has the 20 SKC games")
PY
bash $S $D $KIND $NAME dfl
wait
echo "CHAIN_DONE $NAME"
