import os, sys, time, hashlib, torch, numpy as np
sys.path.insert(0, os.path.join(os.environ['MTAH_WORK'], 'video_touch/scripts'))
from vt_model import TouchNet
S = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0)
m = TouchNet(T=64, pretrained=True).eval()
x = torch.randn(1, 64, 3, S, S, generator=torch.Generator().manual_seed(1))
res = {}
for thr in [1, 4, 8]:
    torch.set_num_threads(thr)
    with torch.inference_mode():
        m(x)
        t = time.time(); outs = [m(x) for _ in range(3)]; dt = (time.time() - t) / 3
    hs = {hashlib.sha256(o.numpy().tobytes()).hexdigest()[:12] for o in outs}
    res[thr] = hs
    print(f'CPU threads={thr} S={S}: {64/dt:.0f} frames/s, repeat hashes {hs}')
print('cross-thread identical:', len(set().union(*res.values())) == 1)
if torch.backends.mps.is_available():
    mm = m.to('mps'); xm = x.to('mps')
    with torch.inference_mode():
        mm(xm); torch.mps.synchronize(); t = time.time()
        for _ in range(5): o = mm(xm)
        torch.mps.synchronize(); dt = (time.time() - t) / 5
    print(f'MPS: {64/dt:.0f} frames/s; max |mps-cpu| = {(o.cpu()-outs[0]).abs().max().item():.2e}')
