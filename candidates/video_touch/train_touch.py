"""Train TouchNet on SoccerTrack crops; game-grouped split. Per-frame touch target.

Target: frame t is positive if a touch event (TOUCH classes) has its 0-based frame within +-R of t
(R=1: soft label 1 at 0, 0.5 at +-1). Loss: BCE with pos_weight. Clips of T frames sampled with
50% centred near events, 50% uniform. Augment: random 128 crop of 160 (+-16 px), hflip, colour jitter.
Inference: sliding windows of T with stride T/2, keep centre halves; writes per-frame probs.
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_common import events, TOUCH
from vt_model import TouchNet

CROPS = os.path.join(os.environ['MTAH_WORK'], 'video_touch/crops')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1) * 255


def load_half(m, h):
    arr = np.load(f'{CROPS}/crops_{m}_{h}.npy', mmap_mode='r')
    E = [e for e in events(m, '1' if h == '1st' else '2') if e['label'] in TOUCH]
    fr = np.array([e['frame'] - 1 for e in E], int)
    team = np.array([0 if e['team'] == 'left' else 1 for e in E], int)
    y = np.zeros(len(arr), np.float32)
    for f in fr:
        if 0 <= f < len(y):
            y[f] = 1.0
            for d in (-1, 1):
                if 0 <= f + d < len(y):
                    y[f + d] = max(y[f + d], 0.5)
    return dict(m=m, h=h, arr=arr, y=y, ev=fr, team=team)


def batch(halves, B, T, crop, rng, train=True):
    X = np.zeros((B, T, crop, crop, 3), np.uint8); Y = np.zeros((B, T), np.float32)
    for b in range(B):
        d = halves[rng.integers(len(halves))]
        N = len(d['y'])
        if rng.random() < 0.5 and len(d['ev']):
            c = d['ev'][rng.integers(len(d['ev']))] + rng.integers(-T // 2 + 2, T // 2 - 2)
        else:
            c = rng.integers(T, N - T)
        s = int(np.clip(c - T // 2, 0, N - T))
        o = (160 - crop) // 2
        dy, dx = (rng.integers(0, 160 - crop + 1, 2) if train else (o, o))
        x = np.asarray(d['arr'][s:s + T, dy:dy + crop, dx:dx + crop])
        if train and rng.random() < 0.5:
            x = x[:, :, ::-1]
        X[b] = x; Y[b] = d['y'][s:s + T]
    X = torch.from_numpy(X).permute(0, 1, 4, 2, 3).float()
    if train:
        g = torch.empty(B, 1, 3, 1, 1).uniform_(0.8, 1.2); X = (X * g).clamp(0, 255)
    return (X - MEAN) / STD, torch.from_numpy(Y)


@torch.inference_mode()
def predict_half(model, arr, T, crop, dev, bs=8):
    N = len(arr); o = (160 - crop) // 2
    p = np.zeros(N, np.float32); stride = T // 2
    starts = list(range(0, max(N - T, 0) + 1, stride))
    if starts[-1] != N - T:
        starts.append(N - T)
    for i in range(0, len(starts), bs):
        ss = starts[i:i + bs]
        X = np.stack([arr[s:s + T, o:o + crop, o:o + crop] for s in ss])
        X = torch.from_numpy(X).permute(0, 1, 4, 2, 3).float()
        X = ((X - MEAN) / STD).to(dev)
        out = torch.sigmoid(model(X)[..., 0]).float().cpu().numpy()
        for s, q in zip(ss, out):
            a = T // 4 if s > 0 else 0; b = T - T // 4 if s + T < N else T
            p[s + a:s + b] = q[a:b]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--T', type=int, default=32); ap.add_argument('--B', type=int, default=12)
    ap.add_argument('--crop', type=int, default=128); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--pos_weight', type=float, default=5.0); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = 'mps'
    tr = [load_half(*g.split('_')) for g in a.train.split(',')]
    te = [load_half(*g.split('_')) for g in a.test.split(',')]
    model = TouchNet(T=a.T).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    pw = torch.tensor(a.pos_weight, device=dev)
    t0 = time.time(); run = 0
    import queue, threading
    q = queue.Queue(maxsize=6)
    def producer(seed):
        r = np.random.default_rng(seed)
        while True:
            q.put(batch(tr, a.B, a.T, a.crop, r))
    for k in range(2):
        threading.Thread(target=producer, args=(a.seed * 100 + k,), daemon=True).start()
    for step in range(a.steps):
        model.train()
        X, Y = q.get()
        with torch.autocast('mps', dtype=torch.bfloat16):
            out = model(X.to(dev))[..., 0].float()
        loss = F.binary_cross_entropy_with_logits(out, Y.to(dev), pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if (step + 1) % 200 == 0:
            print(f'step {step+1} loss {run:.4f} {time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{a.out}/model.pt')
    model.eval()
    for d in te:
        p = predict_half(model, d['arr'], 64, a.crop, dev)
        np.save(f"{a.out}/prob_{d['m']}_{d['h']}.npy", p)
        print('predicted', d['m'], d['h'], flush=True)
    json.dump(vars(a), open(f'{a.out}/args.json', 'w'))


if __name__ == '__main__':
    main()
