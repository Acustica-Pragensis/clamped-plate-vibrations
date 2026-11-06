"""Present eigenfrequency approximation vs Xing & Liu (2009).

Both models are evaluated against the COMSOL FEM reference on the train and
test splits, and the errors are shown as overlaid boxplots per aspect ratio
and per mode, mirroring the Mitchell & Hazell comparison.

Outputs
    figs/ef/ef_xl_by_ba.png   / .tex
    figs/ef/ef_xl_by_mode.png / .tex
"""
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from xl import xl_mode_frequencies

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# ======================================================================
# 1. Configuration
# ======================================================================
SETS = [
    ("train", "data/ef/ef_selected_train.csv"),
    ("test",  "data/ef/ef_selected_test.csv"),
]
FIG_DIR    = "figs/ef"
MODEL_NAME = "present approximation"
BA_ROUND   = 2
SAVE_FIGS  = True

FIG_BA   = "ef_xl_by_ba"
FIG_MODE = "ef_xl_by_mode"

# Material of the plate (same values as ef_6_test.py).
E     = 170e9       # Pa
nu    = 0.28        # -
rho_p = 2329.0      # kg/m^3

# (mode m,n) -> column index in ef_selected_{train,test}.csv
FEM_COLS = {(1, 1): 3,  (1, 2): 4,  (1, 3): 5,
            (2, 1): 6,  (2, 2): 7,  (2, 3): 8,
            (3, 1): 9,  (3, 2): 10, (3, 3): 11}
MODES = sorted(FEM_COLS)

# Boxplot / TikZ style.
WHISKER_PCT = (8, 92)
BOX_HALF    = 0.20
BOX_WIDTH   = 0.30
COLORS      = {'own': 'tab:orange', 'xl': '#9F0954'}
TIKZ_STYLES = {'own': 'color=myorange, fill=myorange!20',
               'xl':  'color=mypurple, fill=mypurple!20'}
TIKZ_LABELS = {'own': 'present approximation',
               'xl':  r'Xing \& Liu'}
SERIES_ORDER = ('own', 'xl')


# ======================================================================
# 2. Models
# ======================================================================
def f_new(a, b, h, m, n, E, nu, rho):
    """Present approximation, as used in ef_6_test.py."""
    return (np.pi / 8 * np.sqrt(E * h**2 / (12 * (1 - nu**2) * rho))
            * (((m + ((m - 0.224) / (a * n / b + 1.923 * m - 0.717))) / a)**2
               + ((n + ((n - 0.224) / (1.923 * n - 0.717 + b * m / a))) / b)**2))


def pct_err(pred, ref):
    return np.abs((pred - ref) / ref) * 100.0


# ======================================================================
# 3. Data and predictions
# ======================================================================
def load_set(name, path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    ba = data[:, 0]
    a  = data[:, 1]
    h  = data[:, 2]
    b  = ba * a

    ok = (np.isfinite(ba) & np.isfinite(a) & np.isfinite(h)
          & (ba > 0) & (a > 0) & (h > 0))
    fem = {md: data[:, FEM_COLS[md]] for md in MODES}
    return dict(name=name, path=path, ba=ba, a=a, b=b, h=h, fem=fem, ok=ok)


def pred_xl(s, md):
    """Xing & Liu frequency of one mode for every row of the split."""
    label = f"f{md[0]}{md[1]}"
    out = np.full(s['a'].shape, np.nan)
    cache = {}
    for i in np.flatnonzero(s['ok']):
        key = (round(float(s['ba'][i]), 6),
               round(float(s['a'][i]), 12),
               round(float(s['h'][i]), 12))
        if key not in cache:
            cache[key] = xl_mode_frequencies(s['a'][i], s['ba'][i], s['h'][i],
                                             E, nu, rho_p)
        out[i] = cache[key][label]
    return out


def predictions(s):
    return {
        'own': {md: f_new(s['a'], s['b'], s['h'],
                          float(md[0]), float(md[1]), E, nu, rho_p)
                for md in MODES},
        'xl':  {md: pred_xl(s, md) for md in MODES},
    }


def buckets_by_ba(s, preds, series):
    key = np.round(s['ba'], BA_ROUND)
    labels, buckets = [], []
    for bv in np.unique(key):
        mask = (key == bv) & s['ok']
        e = []
        for md in MODES:
            f_fem = s['fem'][md][mask]
            keep = np.isfinite(f_fem)
            e.extend(pct_err(preds[series][md][mask][keep], f_fem[keep]))
        labels.append(f"{bv:.{BA_ROUND}f}")
        buckets.append(np.array(e, dtype=float))
    return labels, buckets


def buckets_by_mode(s, preds, series):
    labels, buckets = [], []
    for md in MODES:
        mask = s['ok']
        f_fem = s['fem'][md][mask]
        keep = np.isfinite(f_fem)
        labels.append(f"({md[0]},{md[1]})")
        buckets.append(pct_err(preds[series][md][mask][keep], f_fem[keep]))
    return labels, buckets


# ======================================================================
# 4. Matplotlib boxplots
# ======================================================================
def overlay_panel(ax, labels, buckets_own, buckets_xl, xlabel, title):
    pos = np.arange(len(labels), dtype=float)
    common = dict(widths=BOX_WIDTH, patch_artist=True, showfliers=False,
                  manage_ticks=False, whis=WHISKER_PCT)

    ax.boxplot(buckets_xl, positions=pos + BOX_HALF, **common,
               boxprops    =dict(facecolor=COLORS['xl'], alpha=0.35,
                                 edgecolor=COLORS['xl']),
               medianprops =dict(color=COLORS['xl'], linewidth=2.0),
               whiskerprops=dict(color=COLORS['xl']),
               capprops    =dict(color=COLORS['xl']))

    ax.boxplot(buckets_own, positions=pos - BOX_HALF, **common,
               boxprops    =dict(facecolor=COLORS['own'], alpha=0.35,
                                 edgecolor=COLORS['own']),
               medianprops =dict(color=COLORS['own'], linewidth=2.0),
               whiskerprops=dict(color=COLORS['own']),
               capprops    =dict(color=COLORS['own']))

    ax.set_xticks(pos, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')


def make_png(panels, fname):
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4), sharey=True,
                             squeeze=False)
    axes = axes[0]

    for ax, panel in zip(axes, panels):
        overlay_panel(ax, panel['labels'], panel['own'], panel['xl'],
                      panel['xlabel'], panel['name'])
    axes[0].set_ylabel(r"$\mathrm{Err}_f\,/\,\%$")

    axes[0].legend(handles=[
        Patch(facecolor=COLORS['own'], alpha=0.35, edgecolor=COLORS['own'],
              label=TIKZ_LABELS['own'].replace(r'\&', '&')),
        Patch(facecolor=COLORS['xl'], alpha=0.35, edgecolor=COLORS['xl'],
              label=TIKZ_LABELS['xl'].replace(r'\&', '&'))],
        fontsize=8, loc='upper left')

    fig.suptitle(f"{MODEL_NAME} vs Xing & Liu vs FEM")
    fig.tight_layout()
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ======================================================================
# 5. TikZ output
# ======================================================================
def box_stats(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    lo, hi = np.percentile(d, WHISKER_PCT)
    return dict(med=float(med), q1=float(q1), q3=float(q3),
                lo=float(lo), hi=float(hi))


def axis_ticks(all_buckets):
    """Shared y range / ticks covering every box in the figure."""
    his = [box_stats(b)['hi'] for b in all_buckets if box_stats(b) is not None]
    top = max(his) if his else 1.0
    top = max(top, 0.05)
    raw = top / 6.0
    mag = 10.0 ** np.floor(np.log10(raw))
    step = 10.0 * mag
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * mag
        if top / step <= 7.0:
            break
    ymax = float(np.ceil(top / step) * step)
    ticks = np.arange(0.0, ymax + 0.5 * step, step)
    return ymax, ticks


def _fmt_tick(v):
    return f"{v:g}"


def _box_plots(panel):
    lines = []
    for i, lab in enumerate(panel['labels'], start=1):
        for off, key in zip((-BOX_HALF, +BOX_HALF), SERIES_ORDER):
            st = box_stats(panel[key][i - 1])
            if st is None:
                continue
            lines.append(
                f"\\addplot [{TIKZ_STYLES[key]}, "
                f"boxplot/draw position={i + off:g}, boxplot prepared={{"
                f"median={st['med']:.3f}, upper quartile={st['q3']:.3f}, "
                f"lower quartile={st['q1']:.3f}, "
                f"upper whisker={st['hi']:.3f}, "
                f"lower whisker={st['lo']:.3f}}}] "
                f"coordinates {{}}; % {lab} {key}")
    return lines


def _legend_lines():
    return [
        r"\addlegendimage{area legend, color=myorange, fill=myorange!20}",
        r"\addlegendentry{present approximation}",
        r"\addlegendimage{area legend, color=mypurple, fill=mypurple!20}",
        r"\addlegendentry{Xing \& Liu}",
    ]


def _write_tex(path, body):
    with open(path, 'w') as fh:
        fh.write("\n".join(body) + "\n")
    print(f"wrote {path}")


def _header(comment):
    return [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{statistics, groupplots}",
        r"\pgfplotsset{compat=1.18}",
        "",
        r"\definecolor{myorange}{HTML}{fa7528}",
        r"\definecolor{mypurple}{HTML}{9F0954}",
        "",
        r"\begin{document}",
        "",
        f"% ---------------- {comment} ----------------",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={",
        r"        group size=2 by 1,",
        r"        horizontal sep=.5cm,",
        r"        y descriptions at=edge left,",
        r"        x descriptions at=edge bottom,",
        r"    },",
    ]


def emit_tex_by_ba(path, panels):
    ymax, ticks = axis_ticks([b for p in panels for key in SERIES_ORDER
                              for b in p[key]])
    body = _header("left: training, right: test")
    body += [
        r"    height=6.0cm,",
        r"    boxplot={draw direction=y, box extend=0.30},",
        r"    xlabel style={yshift=-7pt},",
        f"    ymin=-0.05, ymax={ymax:g},",
        f"    ytick={{{','.join(_fmt_tick(t) for t in ticks)}}},",
        r"    xlabel={$b/a$},",
        r"    ylabel={$\mathrm{Err}_f$/ \%},",
        r"    xticklabel style={rotate=0, inner sep=1.5pt, yshift=-3pt},",
        r"    grid=both,",
        r"    grid style={dotted, gray!60},",
        r"]",
    ]

    for ipanel, panel in enumerate(panels):
        n = len(panel['labels'])
        head = []
        if ipanel == 0:
            head = [
                r"\nextgroupplot[legend columns=2,",
                r"               legend style={draw=none, at={(0.02,0.98)}, anchor=north west},",
                r"               legend cell align=left,",
            ]
        else:
            head = [r"\nextgroupplot["]
        xticks = ",".join(str(i) for i in range(1, n + 1))
        xlabels = ",".join(panel['labels'])
        width = max(5.0, 1.1 * n + 1.5)
        head += [
            f"    xtick={{{xticks}}},",
            f"    xticklabels={{{xlabels}}},",
            f"    xmin=0.5, xmax={n + 0.5:.1f},",
            f"    width={width:.1f}cm,",
            r"    ]",
        ]
        body += head
        if ipanel == 0:
            body += _legend_lines()
        body += _box_plots(panel)

    body += [r"\end{groupplot}", r"\end{tikzpicture}", "", r"\end{document}"]
    _write_tex(path, body)


def emit_tex_by_mode(path, panels):
    ymax, ticks = axis_ticks([b for p in panels for key in SERIES_ORDER
                              for b in p[key]])
    n = len(panels[0]['labels'])
    body = _header("left: training, right: test")
    body += [
        r"    width=10.0cm, height=6.0cm,",
        r"    boxplot={draw direction=y, box extend=0.30},",
        f"    xtick={{{','.join(str(i) for i in range(1, n + 1))}}},",
        f"    xticklabels={{{','.join(panels[0]['labels'])}}},",
        r"    xticklabel style={rotate=0, inner sep=1.5pt},",
        f"    ytick={{{','.join(_fmt_tick(t) for t in ticks)}}},",
        f"    xmin=0.5, xmax={n + 0.5:.1f},",
        f"    ymin=-0.05, ymax={ymax:g},",
        r"    xlabel={Mode $(m,n)$},",
        r"    ylabel={$\mathrm{Err}_f$/ \%},",
        r"    grid=both,",
        r"    grid style={dotted, gray!60},",
        r"]",
    ]

    body += [
        r"\nextgroupplot[legend columns=2,",
        r"               legend style={draw=none, at={(0.02,0.98)}, anchor=north west},",
        r"               legend cell align=left,",
        r"    ]",
    ]
    body += _legend_lines()
    body += _box_plots(panels[0])

    body += [r"\nextgroupplot[ylabel=, yticklabels=\empty]"]
    body += _box_plots(panels[1])

    body += [r"\end{groupplot}", r"\end{tikzpicture}", "", r"\end{document}"]
    _write_tex(path, body)


# ======================================================================
# 6. Main
# ======================================================================
def main():
    sets = []
    for name, path in SETS:
        s = load_set(name, path)
        s['preds'] = predictions(s)
        sets.append(s)

    panels_ba, panels_mode = [], []
    print(f"{'split':>6} {'model':>10} {'mean':>8} {'median':>8} {'max':>8}")
    print("-" * 46)

    for s in sets:
        lbl_ba, own_ba = buckets_by_ba(s, s['preds'], 'own')
        _, xl_ba = buckets_by_ba(s, s['preds'], 'xl')
        lbl_md, own_md = buckets_by_mode(s, s['preds'], 'own')
        _, xl_md = buckets_by_mode(s, s['preds'], 'xl')

        panels_ba.append(dict(name=s['name'], xlabel=r"$b/a$",
                              labels=lbl_ba, own=own_ba, xl=xl_ba))
        panels_mode.append(dict(name=s['name'], xlabel="mode (m,n)",
                                labels=lbl_md, own=own_md, xl=xl_md))

        for key in SERIES_ORDER:
            flat = np.concatenate(own_ba if key == 'own' else xl_ba)
            label = MODEL_NAME if key == 'own' else 'Xing & Liu'
            print(f"{s['name']:>6} {label:>10} "
                  f"{flat.mean():8.3f} {np.median(flat):8.3f} "
                  f"{flat.max():8.3f}")

    make_png(panels_ba, f"{FIG_DIR}/{FIG_BA}.png")
    make_png(panels_mode, f"{FIG_DIR}/{FIG_MODE}.png")

    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        emit_tex_by_ba(f"{FIG_DIR}/{FIG_BA}.tex", panels_ba)
        emit_tex_by_mode(f"{FIG_DIR}/{FIG_MODE}.tex", panels_mode)


if __name__ == "__main__":
    main()
