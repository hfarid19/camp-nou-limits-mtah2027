"""Ball-centred crops for every frame of BAR-ATH tactical video (inference only; no labels used).
Same recipe as extract_st.py: tracking ball (NaN interpolated) smoothed sigma 25 frames, projected by
the per-frame homography (normalised pitch -> px), 14 m extent using local horizontal px/m.
"""
import sys, os, time, numpy as np, av
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_st import crop_yuv, interp_nan, EXT, S, SIG

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(os.environ['MTAH_WORK'], 'video_touch/crops')
L, W = 105.3, 68.0


def centre_path():
    b = np.load(os.path.join(os.environ['MTAH_WORK'], 'match_cache.npz'))['b']
    x = gaussian_filter1d(interp_nan(b[:, 0] / L), SIG); y = gaussian_filter1d(interp_nan(b[:, 1] / W), SIG)
    d = np.genfromtxt(f'{REPO}/data/tactical/BAR-ATH_tactical_homography.csv', delimiter=',', skip_header=1)
    H = d[:, 1:10].reshape(-1, 3, 3)
    def proj(xx, yy):
        v = np.einsum('nij,nj->ni', H, np.stack([xx, yy, np.ones_like(xx)], 1)); return v[:, 0] / v[:, 2], v[:, 1] / v[:, 2]
    u, v = proj(x, y); u1, _ = proj(x + 1.0 / L, y)
    side = np.clip(EXT * np.abs(u1 - u), 64, 900)
    return u, v, side, np.isnan(b[:, 0])


def main():
    fn = f'{OUT}/crops_BARATH_full.npy'
    u, v, side, miss = centre_path(); N = len(u)
    arr = np.lib.format.open_memmap(fn, 'w+', np.uint8, (N, S, S, 3))
    c = av.open(f'{REPO}/data/tactical/BAR-ATH_tactical.mp4'); st = c.streams.video[0]
    st.thread_type = 'AUTO'; st.thread_count = 4
    t0 = time.time(); n = 0
    for fr in c.decode(st):
        if n >= N:
            break
        arr[n] = crop_yuv(fr, u[n], v[n], side[n], fr.width, fr.height); n += 1
        if n % 20000 == 0:
            print(n, f'{n / (time.time() - t0):.0f} fps', flush=True)
    arr.flush()
    np.savez(f'{OUT}/meta_BARATH_full.npz', u=u, v=v, side=side, ball_missing=miss, n_decoded=n)
    print('DONE', n, N, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
