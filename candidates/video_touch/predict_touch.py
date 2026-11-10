"""Per-frame touch probabilities with a trained run.
usage: predict_touch.py RUN_DIR HALF [HALF...] [--cpu THREADS] [--range a:b] [--tag name]
HALF is <match>_<half> (SoccerTrack crops) or BARATH_full (match crops)."""
import sys, os, time, hashlib, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vt_model import TouchNet
from train_touch import predict_half, CROPS

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('halves', nargs='+')
ap.add_argument('--cpu', type=int, default=0); ap.add_argument('--range', default=None)
ap.add_argument('--tag', default='')
a = ap.parse_args()
dev = 'cpu' if a.cpu else 'mps'
if a.cpu:
    torch.set_num_threads(a.cpu)
torch.use_deterministic_algorithms(True)
m = TouchNet(T=64, pretrained=False)
m.load_state_dict(torch.load(f'{a.run}/model.pt', map_location='cpu'))
m = m.to(dev).eval()
for h in a.halves:
    arr = np.load(f'{CROPS}/crops_{h}.npy', mmap_mode='r')
    if a.range:
        s, e = map(int, a.range.split(':')); arr = arr[s:e]
    t = time.time()
    p = predict_half(m, arr, 64, 128, dev)
    dt = time.time() - t
    out = f'{a.run}/prob_{h}{a.tag}.npy'
    np.save(out, p)
    print(h, dev, a.cpu, f'{len(p)} frames {dt:.0f}s {len(p)/dt:.0f} fps sha {hashlib.sha256(p.tobytes()).hexdigest()[:16]}', flush=True)
