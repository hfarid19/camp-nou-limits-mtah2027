"""Range-fetch slices of each GSR half JSON (2.7 GB each) to get (image feet px, pitch m, box h) pairs."""
import http.client, sys, re, json, os, urllib.request, numpy as np
from huggingface_hub import get_token, hf_hub_url
OUT = sys.argv[1]
tok = get_token()
SIZES = json.load(open(os.path.join(os.path.dirname(OUT.rstrip('/')), '..', 'meta', 'tree.json')))
SIZES = {e['path']: e.get('size') for e in SIZES}
rec = re.compile(r'"image_id":\s*"(\d+)".*?"attributes":\s*\{[^}]*"role":\s*"(\w+)"[^}]*"team":\s*"?(\w+)"?[^}]*\}.*?"bbox_image":\s*\{([^}]*)\}.*?"bbox_pitch":\s*\{([^}]*)\}', re.S)
num = lambda s, k: float(re.search(r'"%s":\s*(-?[\d.eE+-]+)' % k, s).group(1))
for m in ['117092','117093','118575','118576','118577','118578','128057','128058','132831','132877']:
    for h in ['1st', '2nd']:
        p = f'gsr/{m}/{m}_{h}.json'; fn = os.path.join(OUT, f'{m}_{h}.npy')
        if os.path.exists(fn): continue
        url = hf_hub_url('atomscott/soccertrack-v2', p, repo_type='dataset')
        size = SIZES[p]; rows = []
        for k in range(12):
            off = int(size * (0.08 + 0.07 * k))
            req = urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}', 'Range': f'bytes={off}-{off + 3_000_000}'})
            data = b''
            for _try in range(4):
                try:
                    data = urllib.request.urlopen(req, timeout=120).read(); break
                except http.client.IncompleteRead as e:
                    if len(e.partial) > 200_000:
                        data = e.partial; break
                except Exception as e:
                    print('retry', e, flush=True)
            txt = data.decode('utf-8', 'ignore')
            for mo in rec.finditer(txt):
                bi, bp = mo.group(4), mo.group(5)
                try:
                    rows.append([int(mo.group(1)[-6:]), num(bi, 'x_center'), num(bi, 'y') + num(bi, 'h'), num(bi, 'h'),
                                 num(bp, 'x_bottom_middle'), num(bp, 'y_bottom_middle'), mo.group(2) == 'player', {'left': 0, 'right': 1}.get(mo.group(3), -1)])
                except Exception:
                    pass
        a = np.array(rows, float); np.save(fn, a); print(m, h, a.shape, flush=True)
