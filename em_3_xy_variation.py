import re
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# ======================================================================
# 1. Configuration
# ======================================================================
MODES       = ['11', '13', '31', '33']
DATA_DIR    = Path('data/em/chosen_modes/')
UNIT_W_FILE = Path('data/em/chosen_modes/unit_sq_W.csv')   # unit plate 2D fields (volume reference)
UNIT_X_FILE = Path('data/em/chosen_modes/unit_sq_X.csv')   # unit plate 1D profiles along x
UNIT_Y_FILE = Path('data/em/chosen_modes/unit_sq_Y.csv')   # unit plate 1D profiles along y
FILE_GLOB   = 'em_ba*_a*_h*_{suffix}.csv'
FILE_RE     = re.compile(r'em_ba([\d.]+)_a([\d.eE+-]+)_h([\d.eE+-]+)_[WXY]\.csv$')
KEEP_RATIOS = [1.1, 1.3, 1.5, 1.7, 1.9, 2.0]   # only these are compared and plotted
EXPORT_DIR  = Path('figs/em')
OUT_X_PATTERN = 'mode{mn}_X_components.csv'
OUT_Y_PATTERN = 'mode{mn}_Y_components.csv'
FIG_PATTERN   = 'mode{mn}_components_with_variation.png'


# ======================================================================
# 2. Data loading
# ======================================================================
def reader(path):
    """Reduced file: plain header line, then numeric rows."""
    with open(path) as fh:
        hdr = [h.strip() for h in fh.readline().split(',')]
    return hdr, np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1))


def unit_reader(path):
    """Raw COMSOL export: '%' comment lines, header = last '%' line above
    the data with the same field count (as in em_prepare_data.py)."""
    lines = Path(path).read_text().splitlines()
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


def load_profiles(suffix, modes, unit_ref):
    """Self-normalised 1D W{mn} profiles from the _X/_Y files, each file
    read once, all modes at once, with the matching b/a values. Signs are
    aligned to the unit-plate reference profile (same convention as
    ALIGN_OWN in em_test_modes_all.py)."""
    store = {mn: ([], []) for mn in modes}  # mn -> (profiles, b_values)
    for f in sorted(DATA_DIR.glob(FILE_GLOB.format(suffix=suffix))):
        m = FILE_RE.search(f.name)
        if not m:
            print(f'skip {f.name}')
            continue

        b_val = round(float(m.group(1)), 6)
        if not any(np.isclose(b_val, r, rtol=0.0, atol=1e-6) for r in KEEP_RATIOS):
            continue

        try:
            hdr, arr = reader(f)
        except Exception as e:
            print(f"Error in {f.name}: {e}")
            continue

        for mn in modes:
            prof = arr[:, hdr.index(f'W{mn}')]
            prof = prof / np.max(np.abs(prof))
            ref = unit_ref[mn][0] if suffix == 'X' else unit_ref[mn][1]
            if np.dot(prof, ref) < 0:
                prof = -prof
            store[mn][0].append(prof)
            store[mn][1].append(b_val)
    return {mn: (np.array(p), np.array(b)) for mn, (p, b) in store.items()}


def load_shapes(modes):
    """Mean |w| shape descriptor of the max-normalised W{mn} field, from the _W files
    (same files as em_test_modes_all.py), each file read once, all modes
    at once."""
    store = {mn: ([], []) for mn in modes}  # mn -> (volumes, b_values)
    for f in sorted(DATA_DIR.glob(FILE_GLOB.format(suffix='W'))):
        m = FILE_RE.search(f.name)
        if not m:
            print(f'skip {f.name}')
            continue

        b_val = round(float(m.group(1)), 6)
        if not any(np.isclose(b_val, r, rtol=0.0, atol=1e-6) for r in KEEP_RATIOS):
            continue

        try:
            hdr, arr = reader(f)
        except Exception as e:
            print(f"Error in {f.name}: {e}")
            continue

        for mn in modes:
            w = arr[:, hdr.index(f'W{mn}')]
            store[mn][0].append(np.mean(np.abs(w)) / np.max(np.abs(w)))
            store[mn][1].append(b_val)
    return {mn: (np.array(v), np.array(b)) for mn, (v, b) in store.items()}


# ======================================================================
# 3. Unit plate reference (b/a = 1.0)
# ======================================================================
hdr_uw, arr_uw = unit_reader(UNIT_W_FILE)
hdr_ux, arr_ux = unit_reader(UNIT_X_FILE)
hdr_uy, arr_uy = unit_reader(UNIT_Y_FILE)

N = arr_ux.shape[0]  # number of profile points

unit_ref = {}
for mn in MODES:
    w = arr_uw[:, hdr_uw.index(f'W{mn}')]
    x_ref = arr_ux[:, hdr_ux.index(f'W{mn}')]
    y_ref = arr_uy[:, hdr_uy.index(f'W{mn}')]
    unit_ref[mn] = (x_ref / np.max(np.abs(x_ref)),
                    y_ref / np.max(np.abs(y_ref)),
                    np.mean(np.abs(w)) / np.max(np.abs(w)))  # vol_ref

# Mode 31 uses the opposite component sign convention.  Flip both the X
# and Y references so the sign alignment below and the exported `unit`
# column use the same convention.  The volume reference is |w| based.
x31, y31, vol31 = unit_ref['31']
unit_ref['31'] = (-x31, -y31, vol31)


# ======================================================================
# 4. Load all data
# ======================================================================
profiles_X = load_profiles('X', MODES, unit_ref)
profiles_Y = load_profiles('Y', MODES, unit_ref)
volumes    = load_shapes(MODES)


# ======================================================================
# 5. Per-mode volume analysis and plots
# ======================================================================
def plot_panel(profiles, b_array, ref_profile, color_dict, x_axis,
               title, xlabel, ylabel=None):
    # Track labels we have already added
    added_labels = set()

    for i in range(len(profiles)):
        current_b = b_array[i]
        color = color_dict[current_b]

        # Check if this b_ratio has been labeled yet
        label_str = f"b/a={current_b:.2f}"
        if label_str not in added_labels:
            lbl = label_str
            added_labels.add(label_str)
        else:
            lbl = None  # Don't add to legend if already present

        plt.plot(x_axis, profiles[i], color=color, alpha=0.5, linewidth=1, label=lbl)

    # Plotting the unit plate reference data (b/a=1.0) on top for comparison
    plt.plot(x_axis, ref_profile, color='black', linestyle='--', linewidth=2, label='b/a=1.00')

    # Formatting
    plt.grid(True, linestyle=':', alpha=0.6)

    # Sort the legend entries so b/a = 1.10 comes before 1.90
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        # Sort by the numerical value in the label string
        labels_handles = sorted(zip(labels, handles), key=lambda t: float(t[0].split('=')[1]))
        sorted_labels, sorted_handles = zip(*labels_handles)
        plt.legend(sorted_handles, sorted_labels, loc='upper right', fontsize='x-small', ncol=2)

    plt.title(title)
    plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)


def export_profiles(axis_array, b_array, filename, unit_profile=None,
                    unit_name='unit'):
    """Mean profile per b/a as CSV for TikZ (numpy only).

    If unit_profile is given, it is exported as an extra column (the
    b/a = 1.00 unit-plate reference)."""
    unique_b = np.unique(b_array)
    # Start with the coordinate column
    cols = [np.linspace(-0.5, 0.5, N)]
    col_names = ['coord']

    if unit_profile is not None:
        col_names.append(unit_name)
        cols.append(unit_profile)

    for b in unique_b:
        # Group by b_ratio and take mean
        mask = (b_array == b)
        mean_profile = np.mean(axis_array[mask], axis=0)

        # Create a LaTeX-safe column name: e.g., 1.03 -> ba103
        col_name = f"ba{int(round(b * 100))}"
        col_names.append(col_name)
        cols.append(mean_profile)

    np.savetxt(EXPORT_DIR / filename, np.column_stack(cols),
               delimiter=',', header=','.join(col_names), comments='',
               fmt='%.15g')
    print(f"Exported {filename} with columns: {col_names}")


x_axis = np.linspace(-0.5, 0.5, N)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

for mn in MODES:
    x_ref, y_ref, vol_ref = unit_ref[mn]

    X_array, b_array   = profiles_X[mn]
    Y_array, b_array_Y = profiles_Y[mn]
    vol_array, b_array_W = volumes[mn]
    print(f"\nMode {mn}: {len(X_array)} profiles loaded, "
          f"{len(np.unique(b_array))} unique b/a values")

    unique_b = np.unique(np.concatenate((b_array, b_array_Y)))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_b)))
    color_dict = dict(zip(unique_b, colors))

    # --- Volume Displacement Analysis ---
    # 1. Reference volume comes from the unit plate w (b/a = 1.0)
    unique_b_w = np.unique(b_array_W)

    avg_vols = []
    rel_diffs = []  # List to store percentage differences relative to the unit plate

    for b in unique_b_w:
        mask = (b_array_W == b)
        mean_vol_b = np.mean(vol_array[mask])
        avg_vols.append(mean_vol_b)

        # Calculate relative difference: ((Current - Reference) / Reference) * 100
        rel_diff = (mean_vol_b - vol_ref) / vol_ref * 100
        rel_diffs.append(rel_diff)

    rel_diffs = np.array(rel_diffs)
    max_abs_variation = np.max(np.abs(rel_diffs))

    print("\n" + "="*50)
    print(f"VOLUME DISPLACEMENT ANALYSIS (MODE {mn})")
    print(f"Reference Volume (unit plate, b/a=1.0): {vol_ref:.6f}")
    print(f"Max absolute variation from ref:    {max_abs_variation:.2f}%")
    print(f"Relative difference range:          [{np.min(rel_diffs):+.2f}%, {np.max(rel_diffs):+.2f}%]")

    # Identify where the most and least deviation occurs
    idx_min_abs = np.argmin(np.abs(rel_diffs))
    idx_max_abs = np.argmax(np.abs(rel_diffs))
    print(f"Closest match to ref at b/a = {unique_b_w[idx_min_abs]:.2f} ({rel_diffs[idx_min_abs]:+.2f}%)")
    print(f"Largest deviation at b/a =   {unique_b_w[idx_max_abs]:.2f} ({rel_diffs[idx_max_abs]:+.2f}%)")
    print("="*50 + "\n")

    # 2. Find the specific index for b/a = 2.00
    idx_200 = np.where(np.isclose(unique_b_w, 2.00))[0]
    if len(idx_200) > 0:
        val_200 = rel_diffs[idx_200[0]]
        print(f"Variation at b/a = 2.00:            {val_200:+.2f}%")
    else:
        print("Variation at b/a = 2.00:            Data not available")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_panel(X_array, b_array, x_ref, color_dict, x_axis,
               '$X_{mn}(\\bar{x})$', '$\\bar{x}$', 'Scaled Deflection')
    plt.subplot(1, 2, 2)
    plot_panel(Y_array, b_array_Y, y_ref, color_dict, x_axis,
               '$Y_{mn}(\\bar{y})$', '$\\bar{y}$')

    plt.tight_layout()
    # plt.savefig(FIG_PATTERN.format(mn=mn), dpi=300)
    plt.show()
    plt.close()

    # exporting data for TikZ
    export_profiles(X_array, b_array, OUT_X_PATTERN.format(mn=mn),
                    unit_profile=x_ref, unit_name='unit')
    export_profiles(Y_array, b_array_Y, OUT_Y_PATTERN.format(mn=mn),
                    unit_profile=y_ref, unit_name='unit')
