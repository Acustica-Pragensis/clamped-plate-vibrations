# Free vibrations of fully clamped rectangular plates

## Analytical approximation of eigenmodes and eigenfrequencies

Based on data-driven discovery, we provide analytical approximations obtained using symbolic regression,
which help understand the direct influence of geometry parameters, and speed up the MEMS transducers/microphones design.

![Graphical Abstract](figs/paper/graphical_abstract/ga.png)

### Project Structure

```text
ef_nonperf/
├── data/
│   ├── comsol/                 # raw COMSOL output, never edited by hand
│   │   ├── freqs_full20261016_112507_v1.csv
│   │   ├── freqs_full20261016_133325_v2.csv
│   │   └── mesh_study_v{1,2,3}.csv
│   ├── reference/              # hand-made lookup tables
│   │   └── comsol_modes.csv
│   └── processed/              # everything Python produces downstream
│       ├── ef_selected_modes_v1.csv
│       ├── ef_selected_modes_v2.csv
│       └── ef_pysr_v1.csv
├── figs/
├── outputs/                    # PySR runs
├── 0_run_sweep.m
├── 0_eigenfunctions_rectangle_full.m
├── 0_run_mesh_study.m
├── 1_prepare_data_stage1.py
├── 2_prepare_data_stage2.py
├── 3_ef_sr_rectangle.py
├── 3_mesh_study.py
├── 4_ef_hof_to_py.py
├── 4_ef_hof_to_latex.py
├── 5_ef_revision1_all_models.py
├── hof_equations_v1.py         # generated, imported by step 5
└── paths.py                    # every data path as one constant
```

```text
PLATE_EIGENFUNCTIONS/
│
├── src/                             # utilities
│   ├── __init__.py
│   ├── utils.py                     # e.g., phi_series_term
│   └── visualize.py                 # plotting templates
│
├── ef_non_perf/                     # Eigenfrequencies of non-perforated plates
│   ├── data/                        # data for data-driven discovery and validation
│   ├── figs/                        # exported figures
│   ├── search_equations.py          # Symbolic regression loop
│   ├── validate_equations.py        # Equation validation against data
│   └── generate_figs.py             # Script for figures & final CSV export
│
├── ef_perf/                         # Eigenfrequencies of perforated plates
│   ├── data/
│   ├── figs/
│   ├── 
│   ├── 
│   └── 
│
├── em_non_perf/                     # Eigenmodes of non-perforated plates
│   ├── data/
│   ├── figs/
│   ├── 
│   ├── 
│   └── 
│
└── em_perf/                         # Eigenmodes of perforated plates
    ├── data/
    ├── figs/
    ├── em_perf_micromachines.py
    └── em_perf_variation.py

```
