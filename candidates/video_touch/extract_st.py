"""Extract ball-centred crops for every frame of a SoccerTrack half.

Centre: ball/ track (pitch m; NaN rows interpolated), Gaussian-smoothed sigma=SIG frames so the
crop path does not reveal the event-anchored kinks of the interpolated ball. Extent EXT metres
horizontally, square in pixels using local horizontal px/m of the fitted pitch->image map.
Output: crops_<m>_<h>.npy uint8 (N, S, S, 3) RGB, meta_<m>_<h>.npz (u, v, side per frame).
"""
import sys, os, time, numpy as np, av, cv2
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_common import geom, ball, video

EXT, S, SIG = 14.0, 160, 25.0
OUT = os.path.join(os.environ['MTAH_WORK'], 'video_touch/crops')


def interp_nan(a):
    a = a.astype(float).copy(); i = np.arange(len(a)); ok = np.isfinite(a)
    a[~ok] = np.interp(i[~ok], i[ok], a[ok]); return a


def centre_path(m, h):
    proj, pxm = geom(f'{m}_{h}')
    bx, by, _ = ball(m, h)
    x = gaussian_filter1d(interp_nan(bx), SIG); y = gaussian_filter1d(interp_nan(by), SIG)
    u, v = proj(x, y); side = EXT * pxm(x, y)
    return u, v, np.clip(side, 64, 900)


def crop_yuv(fr, u, v, side, W, H):
    s = int(round(side)) // 2 * 2
    u0 = int(round(u - s / 2)) // 2 * 2; v0 = int(round(v - s / 2)) // 2 * 2
    Y = np.frombuffer(fr.planes[0], np.uint8).reshape(H, fr.planes[0].line_size)[:, :W]
    U = np.frombuffer(fr.planes[1], np.uint8).reshape(H // 2, fr.planes[1].line_size)[:, :W // 2]
    V = np.frombuffer(fr.planes[2], np.uint8).reshape(H // 2, fr.planes[2].line_size)[:, :W // 2]
    yb = np.zeros((s, s), np.uint8); ub = np.full((s // 2, s // 2), 128, np.uint8); vb = ub.copy()
    a0, a1 = max(0, v0), min(H, v0 + s); b0, b1 = max(0, u0), min(W, u0 + s)
    if a1 > a0 and b1 > b0:
        yb[a0 - v0:a1 - v0, b0 - u0:b1 - u0] = Y[a0:a1, b0:b1]
        ub[(a0 - v0) // 2:(a1 - v0) // 2, (b0 - u0) // 2:(b1 - u0) // 2] = U[a0 // 2:a1 // 2, b0 // 2:b1 // 2]
        vb[(a0 - v0) // 2:(a1 - v0) // 2, (b0 - u0) // 2:(b1 - u0) // 2] = V[a0 // 2:a1 // 2, b0 // 2:b1 // 2]
    i420 = np.concatenate([yb.ravel(), ub.ravel(), vb.ravel()]).reshape(s * 3 // 2, s)
    rgb = cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)
    return cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA if s > S else cv2.INTER_LINEAR)


def main(m, h):
    os.makedirs(OUT, exist_ok=True)
    fn = f'{OUT}/crops_{m}_{h}.npy'
    if os.path.exists(fn.replace('.npy', '.done')):
        return
    u, v, side = centre_path(m, h)
    N = len(u)
    arr = np.lib.format.open_memmap(fn, 'w+', np.uint8, (N, S, S, 3))
    c = av.open(video(m, h)); st = c.streams.video[0]; st.thread_type = 'AUTO'; st.thread_count = 4
    t0 = time.time(); n = 0
    for fr in c.decode(st):
        if n >= N:
            break
        assert fr.format.name == 'yuv420p'
        arr[n] = crop_yuv(fr, u[n], v[n], side[n], fr.width, fr.height); n += 1
        if n % 10000 == 0:
            print(m, h, n, f'{n / (time.time() - t0):.0f} fps', flush=True)
    arr.flush()
    np.savez(f'{OUT}/meta_{m}_{h}.npz', u=u, v=v, side=side, n_decoded=n, N=N)
    open(fn.replace('.npy', '.done'), 'w').write(f'{n} {N}\n')
    print('DONE', m, h, n, N, flush=True)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        m, h = arg.split('_'); main(m, h)
