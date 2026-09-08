# Figure 6 — memento inside CZI CELLxGENE Discover

Regenerates the analysis panels of Figure 6 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

| Panel | Shows | Original code |
| --- | --- | --- |
| A | UMAP of the SLE dataset in CELLxGENE | screenshot — not code |
| B | Enumeration of possible comparisons | schematic — not code |
| C | Precomputed vs full mode, differential mean | `cxg_comparison/cellxgene_comparison.ipynb` |
| D | Precomputed vs full mode, differential variability | `cxg_comparison/cellxgene_comparison.ipynb` — **incomplete, see below** |
| E | Query runtime vs number of comparisons | `cxg_comparison/cellxgene_comparison.ipynb` |
| F | Schematic of the multi-dataset pDC/cDC analysis | schematic — not code |
| G | QQ plot, datasets pooled vs each alone | `rare_celltype_comparison.py`, `cellxgene_crossdata.ipynb` |

Panels A, B and F are figure art with no generating code, so four panels are reproduced.

> **Status:** C, E and G reproduce. **D does not yet run on the right input** — the
> estimators cube recovered for this repository has its variance fields unpopulated for
> all but a few dozen genes per cell type. Its output is a placeholder; see the notes.

## Requirements

Python with scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib, and the TileDB
stack:

```bash
pip install tiledb tiledbsoma cellxgene-census
```

Figure 6 uses **this repository's** memento package.

## Data

Unlike the other figures, almost nothing here is a file on the volume. Panels C, D and G
stream cells and metadata from the public CELLxGENE census at run time; the one large
local input is the precomputed estimators cube, which ships as a tar and must be
unpacked once:

```bash
mkdir -p /memento_data/precomputation/extracted
tar -xf /memento_data/precomputation/stimators_cube.2023-10-23-homo_sapiens-full.tar \
    -C /memento_data/precomputation/extracted     # ~17 GB, several minutes

python data_manifest.py check                     # confirms the cube is readable
```

Override the location with `MEMENTO_CUBE_PATH`, and the census release with
`CENSUS_VERSION`.

## Run

```bash
python make_figure6.py       # all four panels + figure6.png
```

Panel C/D caches its census query as `intermediate/panel_cd_donor_cells.h5ad` and its
memento run as a csv, so reruns are fast. Panel G is the slow one — it fits every gene
across 23 datasets, pooled and then individually.

## What the panels should show

- **C** — the two routes agree on the mean: log fold changes fall on the diagonal over
  1,498 genes (**r = 0.999**, slope 1.00, intercept 0.02), and the p-values follow at
  **r = 0.92** over 1,359, the precomputed route running slightly conservative.
- **D** — **incomplete: the required input is not on the volume.** The cube here has its
  variance fields populated for only a few dozen genes per cell type, so the panel runs
  on 44 genes instead of the full set and the output should not be read as a comparison
  of the two routes. See the note below.
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
- **The cube on the volume has its variance fields almost entirely unpopulated.** For the
  donor panel D uses, `var`/`sev`/`selv` are non-zero for 119 of 15,106 monocyte genes
  and 50 of 13,893 CD4 T cell genes; every other entry is exactly 0.0. Only 44 genes
  reach the comparison, and the mean-variance trend that `res_var` divides out is then
  fit on those few points, where it overfits.

  This is a statement about `estimators_cube_v2` as it sits on the volume, and nothing
  more. It says the variability comparison cannot be run from this input — not that the
  published panel is wrong. The panel was generated from a cube built at the time, which
  is not the artifact recovered here; a cube with the variance estimators filled in
  would let the comparison run as intended. **Treat panel D's current output as a
  placeholder pending that input, and do not cite the 44-gene numbers.** The script and
  the panel-C machinery around it are believed correct and will work unchanged once the
  right cube is available.
- **The cube's dimension names are correct as stored.** The notebook rotated
  `feature_id`/`cell_type`/`dataset_id` on read to fix an earlier build; applying that
  rename to `estimators_cube_v2` would scramble the query, so it is not applied.
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
