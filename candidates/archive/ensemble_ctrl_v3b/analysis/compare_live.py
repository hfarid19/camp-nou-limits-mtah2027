"""Live bench codes (bench.py --dfl2 --save-probs) vs the offline nested joint codes, frame by frame."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
live, off = data.W + "/ensemble_ctrl_v3b_live", data.W + "/ensemble_ctrl_v3b_joint"
same = 0
fs = sorted(glob.glob(live + "/dfl2_*_codes.npy"))
for f in fs:
    a, b = np.load(f), np.load(os.path.join(off, os.path.basename(f)))
    same += int((a == b).all())
    print(os.path.basename(f), "identical" if (a == b).all() else "DIFF {} frames".format(int((a != b).sum())))
print("identical on {}/{}".format(same, len(fs)))
