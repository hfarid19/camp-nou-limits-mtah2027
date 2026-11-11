#!/bin/zsh
# append the Mac scoring units of one variant to the lanes.py jobs file:  mkjobs.sh <jobs file> <variant> ptr|mstcn [skc|rest]
J=$1; N=$2; K=$3; PART=${4:-all}
W=/path/to/mtah-work; D=$W/arch/r5_arch/skc_train/$N
S=${0:A:h}/run_mac.sh
if [ $PART = all ] || [ $PART = skc ]; then
  for k in 0 1 2 3; do echo "REQ $D/fold_skc_$k.npz :: $S $D $K $N skccv $k" >> $J; done
fi
if [ $PART = all ] || [ $PART = rest ]; then
  echo "REQ $D/nowin.npz :: $S $D $K $N nowin 0 10" >> $J
  echo "REQ $D/nowin.npz :: $S $D $K $N nowin 10 20" >> $J
  F=$D/full.npz; for g in 0 1 2 3 4 5 6; do F=$F,$D/fold_tcn_$g.npz; done
  for part in 0,1 2,3 4,5 6; do echo "REQ $F :: $S $D $K $N dfl $part" >> $J; done
  echo "REQ $D/full.npz :: $S $D $K $N met" >> $J
fi
