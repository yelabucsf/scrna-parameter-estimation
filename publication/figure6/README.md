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

- **C** — mean log fold changes agree almost exactly between the two routes,
  **Pearson r = 0.999** over 1,498 genes.
- **D** — variability log fold changes agree at **r = 0.79**.
- **E** — the precomputed mode is 211× to 380× faster at query time (median 283×),
  against 9.3 minutes of one-off precomputation.
- **G** — the pooled fit yields 10,624 genes with 7,001 at p < 0.05, and departs from the
  null further than any single dataset.

## Notes

- **The census release the notebooks used is gone.** CZI retired 2023-10-30; `config`
  points at 2023-12-15, the nearest surviving release. The SLE dataset carries an
  identical 1,263,676 cells in every available release and all 25 dataset ids the
  cross-dataset panel needs are still present, so the substitution is safe.
- **Panels C and D show effect sizes alongside p-values.** The two routes agree on the
  estimates but not on the p-value scale: the precomputed route tests analytically from
  the stored standard errors while the default route bootstraps, so the precomputed
  p-values are systematically smaller. Effect-size agreement (r = 0.999) is the claim
  that the precomputed mode is sound; the p-value scatter carries a visible offset.
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
