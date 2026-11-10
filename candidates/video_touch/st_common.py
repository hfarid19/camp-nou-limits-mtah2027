import os, numpy as np, json
ST = os.path.join(os.environ['MTAH_WORK'], 'video_touch/stmeta')
VID = os.path.join(os.environ['MTAH_WORK'], 'soccertrack/videos')
GEOM = None
TOUCH = {'PASS', 'DRIVE', 'HIGH PASS', 'CROSS', 'THROW IN', 'BALL PLAYER BLOCK', 'PLAYER SUCCESSFUL TACKLE', 'SHOT', 'FREE KICK', 'HEADER'}
def feats(x, y, deg=4):
    x = np.asarray(x, float) / 52.5; y = np.asarray(y, float) / 34.0
    return np.stack([x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)
def geom(key):
    global GEOM
    if GEOM is None:
        GEOM = json.load(open(ST + '/geom/geom.json'))
    g = GEOM[key]
    cu, cv = np.array(g['cu']), np.array(g['cv'])
    def proj(x, y):
        A = feats(x, y); return A @ cu, A @ cv
    def pxm(x, y):
        u0, _ = proj(x, y); u1, _ = proj(np.asarray(x) + 1.0, y); return np.abs(u1 - u0)
    return proj, pxm
def events(m, half):
    A = json.load(open(f'{ST}/bas/{m}/{m}_12_class_events.json'))['actions']
    return [a for a in A if a['gameTime'][0] == half]
def ball(m, h):
    z = np.load(f'{ST}/ball/{m}_{h}_ball.npz'); return z['x'], z['y'], z['status']
def video(m, h):
    return f'{VID}/{m}/{m}_panorama_{h}_half.mp4'
