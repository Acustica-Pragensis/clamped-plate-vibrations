from pathlib import Path
import csv
import re
import numpy as np

IN_DIR  = Path('data/em/raw')
OUT_DIR = Path('data/em/chosen_modes')
MODES   = Path('data/comsol_modes.csv')
WANT    = [11, 13, 31, 33]          # -> columns W11, W13, W31, W33

RE_WITH = re.compile(r'\((\d+)[;,:]')   # with(3;W) (1/m) @ ...  -> 3
RE_PLAIN = re.compile(r'^W(\d+)')       # fallback: plain header "W3"


def raw_index(name):
    """Eigenmode index encoded in a COMSOL column header, None for coordinates."""
    m = RE_WITH.search(name) or RE_PLAIN.match(name)
    return int(m.group(1)) if m else None


def read_raw(path):
    """Header line = last '%' line above the data with the same field count."""
    lines = path.read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if re.match(r'^\s*[-+.0-9]', l))
    n = len(lines[i].split(','))
    hdr = None
    for j in range(i - 1, -1, -1):
        s = lines[j].strip()
        if s.startswith('%') and len(s.lstrip('%').split(',')) == n:
            hdr = [t.strip() for t in s.lstrip('%').split(',')]
            break
    if hdr is None:
        raise ValueError(f'{path.name}: no header line with {n} fields')
    arr = np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=i))
    return hdr, arr


def load_modes(path):
    """{1.03: {'f11': 1, 'f12': 2, 'f13': 5, ...}}  label -> raw mode index."""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    keys = [k.strip() for k in rows[0][1:]]
    out = {}
    for r in rows[1:]:
        d = {}
        for k, lab in zip(keys, r[1:]):
            lab = lab.strip()
            if lab and lab.lower() != 'nan':
                d[lab] = int(k)
        out[round(float(r[0]), 6)] = d
    return out


def write(path, hdr, arr):
    with open(path, 'w') as fh:
        fh.write(','.join(hdr) + '\n')
        np.savetxt(fh, arr, delimiter=',', fmt='%.15g')


modes = load_modes(MODES)
OUT_DIR.mkdir(exist_ok=True)

for f in sorted(IN_DIR.glob('em_ba*_a*_h*.csv')):
    m = re.match(r'em_ba([\d.]+)_a([\d.eE+-]+)_h([\d.eE+-]+)_([XYW])\.csv$', f.name)
    if not m:
        print('skip', f.name)
        continue

    lmap = modes.get(round(float(m.group(1)), 6), {})
    names, arr = read_raw(f)

    idx = [raw_index(n) for n in names]
    kmap = {k: j for j, k in enumerate(idx) if k is not None}   # mode index -> column
    icoord = [j for j, k in enumerate(idx) if k is None]

    cols, hdr = [arr[:, icoord]], [names[j] for j in icoord]
    for n in WANT:
        k = lmap.get(f'f{n}')
        if k in kmap:
            cols.append(arr[:, kmap[k]])
        else:
            cols.append(np.full(arr.shape[0], np.nan))
            print(f'  warning: f{n} -> raw index {k} missing in {f.name}')
        hdr.append(f'W{n}')

    write(OUT_DIR / f.name, hdr, np.column_stack(cols))
    print(f'{f.name}: {",".join(hdr)}')