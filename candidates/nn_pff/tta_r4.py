#!/usr/bin/env python3
"""Round 4: R5's test-time averaging (candidates/adapt/run_tta.py, unchanged) for round-4 member configs.
run_renorm.KIND only lists the lsw/v6 members, so the round-4 configs are added here (kind from their trainer),
then run_tta.main() runs with the same arguments, e.g.
    python -I candidates/nn_pff/tta_r4.py --out $MTAH_WORK/probs/adapt32 --p32 --cfgs o_ls05_s8 --k 4 --renorm all_a50 --shard 0 --nshard 4
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "adapt"))
import run_renorm as rr  # noqa: E402

for c in ("o_ls05_s8", "o_ls05_64", "o_ls05_64s8", "t5c_64"):
    rr.KIND[c] = "tcn"
for c in ("m5c_W_s8", "m5c_W_64", "m5c_W_64s8", "m5c_Wls", "m5c_Xls", "m5c_WXls", "m5c_X", "m5c_WX", "m5c_64"):
    rr.KIND[c] = "mstcn"
import run_tta  # noqa: E402

if __name__ == "__main__":
    run_tta.main()
