# Figure 6 — memento inside CZI CELLxGENE Discover

Regenerates the analysis panels of Figure 6 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

| Panel | Shows | Original code |
| --- | --- | --- |
| A | UMAP of the SLE dataset in CELLxGENE | screenshot — not code |
| B | Enumeration of possible comparisons | schematic — not code |
| C | Precomputed vs full mode, differential mean | `cxg_comparison/cellxgene_comparison.ipynb` |
| D | Precomputed vs full mode, differential variability | `cxg_comparison/cellxgene_comparison.ipynb` |
| E | Query runtime vs number of comparisons | `cxg_comparison/cellxgene_comparison.ipynb` |
| F | Schematic of the multi-dataset pDC/cDC analysis | schematic — not code |
| G | QQ plot, datasets pooled vs each alone | `rare_celltype_comparison.py`, `cellxgene_crossdata.ipynb` |

Panels A, B and F are figure art with no generating code, so four panels are reproduced.

## Requirements

Python with scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib, and the TileDB
stack:

```bash
pip install tiledb tiledbsoma cellxgene-census
```

Figure 6 uses **this repository's** memento package. Building the comparison cube also
needs a [memento-cxg](https://github.com/mincheoly/memento-cxg) checkout — see
[memento-cxg patches](#memento-cxg-patches).

## Data

Unlike the other figures, almost nothing here is a file on the volume. Panels C, D and G
stream cells and metadata from the public CELLxGENE census at run time. Two precomputed
estimator cubes are involved, and they are not interchangeable:

**Panel G** reads the full-census cube, which ships as a tar and must be unpacked once:

```bash
mkdir -p /memento_data/precomputation/extracted
tar -xf /memento_data/precomputation/stimators_cube.2023-10-23-homo_sapiens-full.tar \
    -C /memento_data/precomputation/extracted     # ~17 GB, several minutes

python data_manifest.py check                     # confirms the cube is readable
```

**Panels C and D** build their own cube, because the full-census one was produced with
the variance estimators switched off and those panels are precisely a test of the
variance path:

```bash
python build_cube.py         # ~2 min, writes intermediate/estimators_cube (1.2 MB)
```

It is scoped to the one donor and two cell types the comparison uses, and takes its
capture rate from `config.CAPTURE_RATE` so it cannot drift from the full memento run.

Override locations with `MEMENTO_CUBE_PATH`, `MEMENTO_COMPARISON_CUBE_PATH` and
`MEMENTO_CXG_PATH`, and the census release with `CENSUS_VERSION`.

## Run

```bash
python build_cube.py         # once, if you have not already
python make_figure6.py       # all four panels + figure6.png
```

Panel C/D caches its census query as `intermediate/panel_cd_donor_cells.h5ad` and its
memento run as a csv, so reruns are fast. Panel G is the slow one — it fits every gene
across 23 datasets, pooled and then individually.

## What the panels should show

- **C** — the two routes agree on the mean: log fold changes fall on the diagonal over
  1,498 genes (**r = 0.999**, slope 1.00, intercept 0.02), and the p-values follow at
  **r = 0.92** over 1,359, the precomputed route running slightly conservative.
- **D** — the two routes agree on variability over 1,487 genes: **slope 1.005**,
  intercept 0.05, r = 0.92, with the p-values at r = 0.95.
- **E** — the precomputed mode is 211× to 380× faster at query time (median 283×),
  against 9.3 minutes of one-off precomputation.
- **G** — the pooled fit yields 10,624 genes with 7,001 at p < 0.05, and departs from the
  null further than any single dataset.

## Notes

- **The census release the notebooks used is gone.** CZI retired 2023-10-30; `config`
  points at 2023-12-15, the nearest surviving release. The SLE dataset carries an
  identical 1,263,676 cells in every available release and all 25 dataset ids the
  cross-dataset panel needs are still present, so the substitution is safe.
- **Panels C and D show effect sizes alongside p-values.** The effect-size panels are the
  direct test of whether the precomputed mode is sound; the p-value panels are what the
  published figure shows.
- **The cube's means must be renormalized before they are compared.** The cube estimates
  each (cell type, dataset, donor) group independently, so two cell types' stored means
  do not sum to the same total — 41.42 for monocytes against 14.48 for CD4 T cells here.
  Taking a ratio without rescaling shifts every gene's log fold change by
  log(14.48/41.42) = −1.05. Correlation cannot see this (it is a pure intercept) but the
  p-values move by orders of magnitude: uncorrected, panel C's p-value agreement is
  r = 0.18, and corrected it is r = 0.92. `normalize_to_relative_abundance` divides each
  group by its own total, matching the default route, which normalizes both cell types
  together.
- **Why panels C and D build their own cube.** The full-census cube on the volume has its
  variance fields all but unpopulated — for this donor, `var`/`sev`/`selv` are non-zero
  for 119 of 15,106 monocyte genes and 50 of 13,893 CD4 T cell genes, and exactly 0.0
  everywhere else. `publication/cellxgene/make_cube.py` shows why: `ESTIMATOR_NAMES` is
  truncated to seven entries and the `compute_variance` / `compute_sev` calls are
  commented out. Run against that cube, panel D compares 44 genes instead of ~1,500, and
  the mean-variance trend behind `res_var` gets fit on those few points and overfits.
  `build_cube.py` sidesteps this by building a small cube with the full estimator set.
  The notebook that produced the published panels read a locally built cube from the
  memento-cxg working directory (`cube_path = '/home/ubuntu/Github/memento-cxg/'`), not
  the census cube, so this is a gap in what was archived rather than anything about the
  original analysis.
- **The two routes must assume the same capture rate `q`, and getting this wrong is
  silent.** `q` enters the second-moment correction in `compute_variance`, and its effect
  is mean-dependent, so it does not cancel when two cell types are ratioed. Building the
  cube at q = 0.1 while the full run uses q = 0.07 shrinks panel D's slope from 1.005 to
  0.699 — the correlation stays at 0.96 either way, so it looks fine unless you fit a
  line. `build_cube.py` takes q from `config.CAPTURE_RATE` for this reason. This is a
  hazard in reproducing the comparison today, not a defect in the original: the cube the
  notebook used is not among the archived files, and its q cannot be recovered.
- **The cube's dimension names depend on which build you have.** The notebook rotates
  `feature_id`/`cell_type`/`dataset_id` on read, which is correct for a cube from the
  `make_cube.py` lineage (`CUBE_DIMS_VAR + CUBE_TILEDB_DIMS_OBS`). Both the census cube
  and anything built by current memento-cxg order them the other way
  (`CUBE_TILEDB_DIMS_OBS + CUBE_DIMS_VAR`) and are labelled correctly as stored, so the
  rename is *not* applied here — applying it would scramble the query.
- **Panel G substitutes the full census cube.** The original script read a purpose-built
  `estimators_cube_dcs_many`, which is not on the volume. The full cube is used instead.
  It carries dendritic-cell estimators for 12 of the 23 listed datasets, and only 4 of
  those have both pDCs and cDCs from the same donors — the structure a per-dataset fit
  needs. So the grey per-dataset curves are fewer than in the published panel. The
  pooled fit is unaffected: 10,624 genes, 7,001 at p < 0.05, and it sits above the
  individual datasets, which is the point the panel makes.
- **`get_groups` now returns label columns numerically encoded.** The notebook's
  `groups[['cell_type']] == ct2` therefore compares floats to a string and yields an
  all-zero treatment, which current memento drops as constant, leaving an empty design.
  The treatment is derived from the group labels instead.
- **`approx` changed from a boolean to a named method** (`boot`, `norm`, `gdp`). `norm`
  is used; `gdp` saturates around −log10(P) = 6 on this two-group comparison.
- **Panel E replots recorded measurements.** Re-timing the default mode means rerunning
  memento once per comparison, which is the cost the panel exists to show.

## memento-cxg patches

`build_cube.py` runs [memento-cxg](https://github.com/mincheoly/memento-cxg)'s
`cell_census_summary_cube.py` unmodified, but that code needs two changes to run here.
`build_cube.py` checks for both and exits with an explanation rather than failing
obscurely.

1. **numpy 2 compatibility.** `estimators.py::compute_variance` ends with
   `float(variance)` where `variance` is a shape-`(1,)` array. NumPy 1.25 deprecated the
   implicit size-1-array-to-scalar conversion and 2.0 removed it, so every gene raises
   `TypeError: only 0-dimensional arrays can be converted to Python scalars` in every
   worker and pass 2 produces nothing. Unwrap with `.item()`.

2. **A configurable capture rate.** `Q` is a module constant. Pass 2 runs in *spawned*
   processes, which re-import the module, so assigning `builder.Q` from a caller has no
   effect on the workers — the build silently uses the default. Reading it from the
   environment (`Q = float(os.environ.get("MEMENTO_CUBE_Q", 0.1))`) is what lets
   `build_cube.py` hold q equal to `config.CAPTURE_RATE`.
