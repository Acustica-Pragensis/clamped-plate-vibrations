import re
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

from xl import xl_assign, xl_eigenpairs, xl_field

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# ======================================================================
# 1. Configuration
# ======================================================================
DATA_DIR  = Path('data/em/chosen_modes')
FILE_RE   = re.compile(r'em_ba([\d.]+)_a([\d.eE+-]+)_h([\d.eE+-]+)_W\.csv$')
COLS      = ('cpl2x', 'cpl2y')
MODES     = ['11', '13', '31', '33']

# ---- requested geometry sets -----------------------------------------
# Training geometries (all lengths in metres).
A_TRAIN  = [300e-6, 600e-6, 900e-6, 1200e-6, 1500e-6]
BA_TRAIN = [1.03, 1.10, 1.18, 1.40, 1.50, 1.70, 1.80]
H_TRAIN  = [4e-6, 8e-6, 12e-6, 16e-6]

# Disjoint test geometries.
A_TEST   = [400e-6, 700e-6, 1000e-6, 1300e-6]
BA_TEST  = [1.30, 1.60, 1.90, 2.00]
H_TEST   = [5e-6, 9e-6, 11e-6]

# Only these b/a values are evaluated and shown.
BA_EVAL  = [1.10, 1.30, 1.50, 1.70, 1.90]

ALIGN_OWN   = True
SELFTEST    = True

# Colors: myorange / mypurple, as used in the existing TikZ files.
PALETTE = {'own': '#fa7528', 'xl': '#9F0954'}
LABELS  = {'own': 'present approximation', 'xl': 'Xing & Liu'}

# TikZ output: both variants are written directly to these files.
# Whiskers are at the 8th/92nd percentiles.
BOX_STYLES = {
    'own': 'color=myorange, fill=myorange!20',
    'xl':  'color=mypurple, fill=mypurple!20',
}
TIKZ_MIN_YMAX = 11.0
TIKZ_FIT      = Path('figs/em/em_boxplot_fit.tex')     # present approximation only
TIKZ_FIT_XL   = Path('figs/em/em_boxplot_fit_xl.tex')  # present approximation + Xing & Liu


# ======================================================================
# 2. Closed-form approximations, X = x/(2a), Y = y/(2b) in [-0.5, 0.5]
# ======================================================================
def mode11(X, Y):
    return (1 - 4*X**2)**2 * (1 - 4*Y**2)**2

def mode13(X, Y):
    px = (1 - 4*X**2)**1.65
    py = (1 - (6.83*Y)**2) * (np.cos(np.pi*Y)**2) / 1.1
    return px * py

def mode31(X, Y):
    px = (1 - (6.83*X)**2) * (np.cos(np.pi*X)**2) / 1.1
    py = (1 - 4*Y**2)**1.65
    return px * py

def mode33(X, Y):
    px = (1 - (6.6*X)**2) * (np.cos(np.pi*X)**2)
    py = (1 - (6.6*Y)**2) * (np.cos(np.pi*Y)**2)
    return px * py

OWN = {'11': mode11, '13': mode13, '31': mode31, '33': mode33}

# ======================================================================
# 3. Xing & Liu (2009)
# ======================================================================
# The solver and mode assignment live in xl.py.

def overlap(a, b):
    na, nb = np.dot(a, a), np.dot(b, b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return abs(np.dot(a, b)) / np.sqrt(na * nb)

# ======================================================================
# 5. Geometry selection
# ======================================================================
def geom_key(a, ba, h):
    return (round(float(a), 12), round(float(ba), 6), round(float(h), 12))

def ratio_key(ba):
    return round(float(ba), 6)

def _matches(value, targets):
    return any(np.isclose(value, t, rtol=1e-9, atol=1e-15) for t in targets)

def _eval_values(values, selected):
    return [v for v in values if _matches(v, selected)]

# Restrict the requested sets to the b/a values that are actually shown.
TRAIN_GEOMS = {
    geom_key(a, ba, h)
    for a in A_TRAIN
    for ba in _eval_values(BA_TRAIN, BA_EVAL)
    for h in H_TRAIN
}
TEST_GEOMS = {
    geom_key(a, ba, h)
    for a in A_TEST
    for ba in _eval_values(BA_TEST, BA_EVAL)
    for h in H_TEST
}
if TRAIN_GEOMS & TEST_GEOMS:
    raise RuntimeError('train and test geometry sets overlap')

def split_of(a, ba, h):
    key = geom_key(a, ba, h)
    if key in TRAIN_GEOMS:
        return 'train'
    if key in TEST_GEOMS:
        return 'test'
    return None

def fmt_geom(key):
    a, ba, h = key
    return f'a={a*1e6:.0f} um, b/a={ba:.2f}, h={h*1e6:.0f} um'

# ======================================================================
# 6. Helpers
# ======================================================================
def reader(path):
    """Reduced file: plain header line, then numeric rows."""
    with open(path) as fh:
        hdr = [h.strip() for h in fh.readline().split(',')]
    return hdr, np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1))

def err_w(w_fem, w_pred):
    return 100 * np.sum(np.abs(w_fem - w_pred)) / np.sum(np.abs(w_fem))

def stat_or_nan(values, func):
    arr = np.asarray(values, dtype=float)
    return func(arr) if arr.size else np.nan

if SELFTEST:
    print("Self-test of the Xing & Liu solver vs their Table 5 (a=1, b=1.2):")
    ref_ka = [5.49, 7.42, 8.30, 9.56, 9.76]
    ref_b1 = [4.33, 3.90, 7.72, 7.45, 3.68]
    for i, (b1, a1) in enumerate(xl_eigenpairs(1.0, 1.2)[:5]):
        print(f"  mode {i+1}:  ka = {np.hypot(b1, a1):5.2f} (ref {ref_ka[i]:5.2f})"
              f"    b1 = {b1:5.2f} (ref {ref_b1[i]:5.2f})")
    print()

# ======================================================================
# 7. Main loop
# ======================================================================
store = {'own': {m: {} for m in MODES},
         'xl':  {m: {} for m in MODES}}

files = sorted(DATA_DIR.glob('em_ba*_a*_h*_W.csv'))
print(f"{len(files)} plane files in {DATA_DIR}")
print(f"evaluated b/a values: {', '.join(f'{r:.2f}' for r in BA_EVAL)}")
print(f"train geometries at these b/a: {len(TRAIN_GEOMS)}; "
      f"test geometries: {len(TEST_GEOMS)}\n")

reported = set()
seen = set()
skipped = 0

for f in files:
    m = FILE_RE.search(f.name)
    if not m:
        print(f'skip {f.name}')
        continue

    b_ratio = ratio_key(float(m.group(1)))
    a_val   = float(m.group(2))
    h_val   = float(m.group(3))

    split = split_of(a_val, b_ratio, h_val)
    if split is None:
        skipped += 1
        continue
    seen.add(geom_key(a_val, b_ratio, h_val))

    try:
        hdr, arr = reader(f)
        x_fem = arr[:, hdr.index(COLS[0])]
        y_fem = arr[:, hdr.index(COLS[1])]
    except Exception as e:
        print(f'Error reading {f.name}: {e}')
        continue

    Lx, Ly = 2.0*a_val, 2.0*b_ratio*a_val
    Xn, Yn = x_fem / Lx, y_fem / Ly

    try:
        assign = xl_assign(b_ratio)
    except Exception as e:
        print(f'  warning: {f.name}: {e}')
        continue

    # one diagnostic table per aspect ratio, first file only
    if b_ratio not in reported:
        reported.add(b_ratio)
        print(f'ba={b_ratio:.2f} [{split}]: assignment ({len(assign)} labels)')
        for lab, (pair, k) in sorted(assign.items(), key=lambda kv: kv[1][1]):
            col = 'W' + lab[1:]
            if col in hdr:
                fem = arr[:, hdr.index(col)]
                fem = fem / np.max(np.abs(fem)) if np.max(np.abs(fem)) else fem
                ov = overlap(xl_field(pair, b_ratio, Xn, Yn), fem)
                flag = '   <-- CHECK' if ov < 0.95 else ''
                print(f'    {lab:>4}  k={k:7.3f}  overlap={ov:.4f}{flag}')
            else:
                print(f'    {lab:>4}  k={k:7.3f}  overlap=   n/a')
        print()

    for mn in MODES:
        col = f'W{mn}'
        lab = f'f{mn}'
        if col not in hdr:
            print(f'  warning: {col} missing in {f.name}')
            continue
        if lab not in assign:
            print(f'  warning: {lab} not in mode table for ba={b_ratio:.2f}')
            continue

        fem = arr[:, hdr.index(col)]
        if not np.isfinite(fem).all():
            print(f'  warning: non-finite {col} in {f.name}')
            continue
        mx = np.max(np.abs(fem))
        if mx == 0.0:
            print(f'  warning: {col} is zero in {f.name}')
            continue
        fem = fem / mx

        own = OWN[mn](Xn, Yn)
        if ALIGN_OWN and np.dot(own, fem) < 0:
            own = -own

        pair, k = assign[lab]
        w_xl = xl_field(pair, b_ratio, Xn, Yn)
        if np.dot(w_xl, fem) < 0:
            w_xl = -w_xl

        store['own'][mn].setdefault(b_ratio, []).append(err_w(fem, own))
        store['xl'][mn].setdefault(b_ratio, []).append(err_w(fem, w_xl))

    if store['own']['11'].get(b_ratio):
        print(f'{f.name}: ba={b_ratio:.2f} a={a_val:.1e} h={h_val:.1e} '
              f'({len(store["own"]["11"][b_ratio])} files)')

print(f'\n{skipped} file(s) outside the requested geometry sets were ignored.')

# ---- coverage --------------------------------------------------------
print('\nGeometry coverage:')
for name, geoms in (('train', TRAIN_GEOMS), ('test', TEST_GEOMS)):
    missing = geoms - seen
    print(f'  {name}: {len(geoms) - len(missing)}/{len(geoms)} found, '
          f'{len(missing)} missing')
    for key in sorted(missing)[:8]:
        print(f'    missing {fmt_geom(key)}')
    if len(missing) > 8:
        print(f'    ... and {len(missing) - 8} more')
print()

# ======================================================================
# 8. Box plots (Matplotlib)
# ======================================================================
def shade(colour, factor):
    return tuple(min(1.0, max(0.0, c * factor)) for c in to_rgb(colour))

def style(colour):
    edge = shade(colour, 0.75)
    return dict(
        boxprops     = dict(facecolor=colour, edgecolor=edge,
                            alpha=0.55, linewidth=1.3),
        whiskerprops = dict(color=edge, linewidth=1.5),
        capprops     = dict(color=edge, linewidth=1.5),
        medianprops  = dict(color=shade(colour, 0.55), linewidth=2.4),
        flierprops   = dict(marker='o', markersize=3.5,
                            markerfacecolor=colour, markeredgecolor=edge,
                            alpha=0.75, linestyle='none'),
    )

def plot_comparison(store, stem='modes_l1_own_vs_xl'):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    for ax, mn in zip(axes, MODES):
        pos = np.arange(1, len(BA_EVAL) + 1)
        for off, key in ((-0.18, 'own'), (+0.18, 'xl')):
            data, where = [], []
            for p, r in zip(pos, BA_EVAL):
                d = store[key][mn].get(ratio_key(r), [])
                if len(d) == 0:
                    continue
                data.append(d)
                where.append(p + off)
            if data:
                # Matplotlib default whiskers; the 8/92 percentiles are
                # requested only for the TikZ export.
                ax.boxplot(data, positions=where, widths=0.32,
                           patch_artist=True, showfliers=True,
                           **style(PALETTE[key]))
        ax.set_xticks(pos)
        ax.set_xticklabels([f'{r:.2f}' for r in BA_EVAL], rotation=45)
        ax.set_title(f'Mode {mn}')
        ax.set_xlabel('$b/a$ ratio')
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[0].set_ylabel('L1 error (%)')
    axes[0].legend(handles=[Patch(facecolor=PALETTE[k],
                                  edgecolor=shade(PALETTE[k], 0.75),
                                  alpha=0.55, label=LABELS[k])
                            for k in ('own', 'xl')],
                   loc='upper left', fontsize=9)
    fig.suptitle('Mode-shape error against FEM, selected $b/a$')
    plt.tight_layout()
    Path('figs').mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        plt.savefig(f'figs/em/{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.show()

plot_comparison(store)

# ---- numeric summary -------------------------------------------------
print("\nMedian / max L1 error [%]")
print(f"  {'mode':>4} {'b/a':>6} {'N':>4} | {'own med':>8} {'own max':>8}"
      f" | {'xl med':>8} {'xl max':>8}")
for mn in MODES:
    for r in BA_EVAL:
        key = ratio_key(r)
        own = store['own'][mn].get(key, [])
        xl = store['xl'][mn].get(key, [])
        n_ = max(len(own), len(xl))
        print(f"  {mn:>4} {r:6.2f} {n_:4d} | "
              f"{stat_or_nan(own, np.median):8.3f} "
              f"{stat_or_nan(own, np.max):8.3f} | "
              f"{stat_or_nan(xl, np.median):8.3f} "
              f"{stat_or_nan(xl, np.max):8.3f}")

# ======================================================================
# 9. TikZ boxplots (8th/92nd-percentile whiskers)
# ======================================================================
def box_stats(values):
    d = np.asarray(values, dtype=float)
    if d.size == 0:
        return None
    return dict(
        med=float(np.median(d)),
        q1=float(np.percentile(d, 25)),
        q3=float(np.percentile(d, 75)),
        lo=float(np.percentile(d, 8)),
        hi=float(np.percentile(d, 92)),
    )

def tikz_addplot(key, ratio, position, st, active=True):
    line = (
        f"\\addplot [{BOX_STYLES[key]}, "
        f"boxplot/draw position={position:g}, "
        f"boxplot prepared={{median={st['med']:.3f}, "
        f"upper quartile={st['q3']:.3f}, lower quartile={st['q1']:.3f}, "
        f"upper whisker={st['hi']:.3f}, lower whisker={st['lo']:.3f}}}] "
        f"coordinates {{}}; % b/a={ratio:.2f} {key}"
    )
    return line if active else '% ' + line

def tikz_ymax(store):
    values = []
    for mn in MODES:
        for r in BA_EVAL:
            key = ratio_key(r)
            for series in ('own', 'xl'):
                values.extend(store[series][mn].get(key, []))
    if not values:
        return TIKZ_MIN_YMAX
    return max(TIKZ_MIN_YMAX, float(np.ceil(np.max(values))))

def tikz_document(store, include_xl, title):
    """Full standalone TikZ document as a list of lines."""
    lines = [f"% ==================== TikZ: {title} ===================="]
    out = lines.append
    out(r"\documentclass{standalone}")
    out(r"\usepackage{pgfplots}")
    out(r"\usepgfplotslibrary{statistics, groupplots}")
    out(r"\pgfplotsset{compat=1.18}")
    out("")
    out(r"\definecolor{mypurple}{HTML}{9F0954}")
    out(r"\definecolor{myorange}{HTML}{fa7528}")
    out("")
    out(f"% Whiskers: lower = 8th percentile, upper = 92nd percentile.")
    out(f"% Shown b/a: {', '.join(f'{r:.2f}' for r in BA_EVAL)}")
    out("")
    out(r"\begin{document}")
    out(r"\begin{tikzpicture}")
    out(r"\begin{groupplot}[")
    out(r"    group style={")
    out(r"        group size=4 by 1,")
    out(r"        horizontal sep=0.8cm,")
    out(r"        vertical sep=0.4cm,")
    out(r"        y descriptions at=edge left,")
    out(r"        x descriptions at=edge bottom,")
    out(r"    },")
    out(r"    width=5.5cm, height=7cm,")
    out(r"    boxplot={")
    out(r"        draw direction=y,")
    out(r"        box extend=0.45,")
    out(r"    },")
    out(f"    xtick={{{','.join(str(i) for i in range(1, len(BA_EVAL) + 1))}}},")
    out(f"    xticklabels={{{','.join(f'{r:.2f}' for r in BA_EVAL)}}},")
    out(r"    xticklabel style={rotate=0, font=\small, inner sep=2pt},")
    out(f"    xmin=0.5, xmax={len(BA_EVAL) + 0.5:.1f},")
    out(r"    xlabel={$b/a$},")
    out(r"    ylabel={$\mathrm{Err}_W$/ \%},")
    out(f"    ymin=0, ymax={tikz_ymax(store):g},")
    out(r"    grid=both,")
    out(r"    grid style={dotted, gray!60},")
    out(r"]")
    out("")

    for mn in MODES:
        out(f"\\nextgroupplot[title={{Mode $({mn[0]},{mn[1]})$}}]")
        for i, r in enumerate(BA_EVAL, start=1):
            key = ratio_key(r)
            own = box_stats(store['own'][mn].get(key, []))
            xl = box_stats(store['xl'][mn].get(key, []))
            if own is None:
                out(f"% b/a={r:.2f} own: no data")
            else:
                out(tikz_addplot('own', r, i, own, active=True))
            if xl is None:
                out(f"% b/a={r:.2f} xl: no data")
            else:
                out(tikz_addplot('xl', r, i, xl, active=include_xl))
        out("")

    out(r"\end{groupplot}")
    out(r"\end{tikzpicture}")
    out(r"\end{document}")
    return lines


def write_tikz(path, store, include_xl, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tikz_document(store, include_xl, title)) + "\n")
    print(f"wrote {path}")

# Variant 1: present approximation only (Xing & Liu commented out).
write_tikz(TIKZ_FIT, store, include_xl=False,
           title='em_boxplot_fit style (present approximation only)')

# Variant 2: both series at the same draw position, as in the _xl file.
write_tikz(TIKZ_FIT_XL, store, include_xl=True,
           title='em_boxplot_fit_xl style (both series)')
