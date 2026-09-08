# Figure 5 — eQTL, vQTL and cQTL mapping in SLE

Regenerates every panel of Figure 5 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

| Panel | Shows | Original code |
| --- | --- | --- |
| A | QQ plots for eQTLs, vQTLs and cQTLs | `run_memento/qqplots.ipynb` |
| B | ROC for recovering OneK1K eQTLs | `auc_curve/roc_curve.ipynb` |
| C | Power vs number of individuals | `power_analysis/sample_power.ipynb` |
| D | eQTL enrichment across ATAC lineages | `atac_enrichment/atac_plots.ipynb` |
| E | Matched-lineage enrichment, both methods | `atac_enrichment/atac_plots.ipynb` |
| F–G | A vQTL: HLA-C at chr6:31326612 | `run_memento/analyze_variability.ipynb` |
| H–I | A cQTL: JUNB-LYZ at chr12:69688073 | `run_memento/analyze_coexpression.ipynb` |

Analyses cover two ancestry groups (`asian`, `eur`) and six cell types (T4, T8, B, NK,
cM, ncM).

## Requirements

Python with scanpy, pandas, scipy, scikit-learn, seaborn and matplotlib.

Figure 5 uses **this repository's** memento package. No separate clone needed.

## Data

Panels A–E are published as one archive, 21 GB unpacked — the largest of the five, since
panels B and C rest on 480 resampled-cohort result files.

All five figures share one Zenodo record; download just this figure's archive.
DOI: _pending_

```bash
mkdir -p ~/memento_bundles && cd ~/memento_bundles
curl -L -O https://zenodo.org/records/<RECORD>/files/figure5_data.tar.gz
curl -L -O https://zenodo.org/records/<RECORD>/files/figure5_data.tar.gz.sha256
sha256sum -c figure5_data.tar.gz.sha256      # macOS: shasum -a 256 -c
tar -xzf figure5_data.tar.gz
export MEMENTO_DATA_PATH=~/memento_bundles
```

The archive unpacks to `figure5_data/`, organized by panel:

| Directory | Feeds |
| --- | --- |
| `panelA_qq/` | A — QTL summary statistics, and per-variant allele frequencies |
| `panelBC_replication/` | B, C |
| `panelDE_atac/` | D, E |
| `panelFI_examples/` | F–I — single-cell data (genotypes not included, see below) |

### Panels F–I need controlled-access data

These four panels group individuals by their genotype at a single variant. Those
genotypes are individual-level data for the CLUES cohort, released by
[Perez et al. 2022](https://www.science.org/doi/10.1126/science.abf1970) only under
[dbGaP phs002812.v1.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs002812.v1.p1)
with a signed Data Use Certification, so they are not in the archive and cannot be.

**Panels A–E are unaffected**, and `make_figure5.py` assembles them without F–I rather
than failing. Panel A's minor-allele-frequency filter needs only an aggregate frequency
per variant, which is aggregate, identifies nobody, and ships in the archive.

With an approved dbGaP request, place the matrices at
`figure5_data/genotypes/{asian,eur}_genos.tsv` — a `CHROM:POS` index and one column per
individual — and F–I run.

## Run

```bash
python make_figure5.py     # all panels + figure5.png
```

Panel A is the slow one — it reads ~7.5 GB of full-cohort QTL results and computes minor
allele frequencies over 3.3M variants in each population. Individual panels can be
plotted alone (`python panel_c_power.py`). Figures land in `figures/`, intermediates in
`intermediate/`.

## What the panels should show

Each script prints its key numbers, so a drift is visible without opening the figure:

- **B** — memento AUC ≈ 0.79 against pseudobulk ≈ 0.77.
- **C** — memento roughly doubles pseudobulk's power at every cohort size; in B cells at
  50 individuals, 0.45 against 0.18.
- **D** — the matched-lineage diagonal is far stronger for memento (mean −log10 P ≈ 5.8)
  than for pseudobulk (≈ 0.9).
- **F** — HLA-C variability by genotype at chr6:31326612: 0.68 / 0.22 / 0.20.
- **H** — JUNB–LYZ correlation by genotype at chr12:69688073: 0.21 / 0.11 / 0.06.

## Notes

- **Panel F plots the residual variance**, element 1 of `get_1d_moments`. The notebook
  took element 0 — the mean — while labelling the axis "Variability"; that is a slip,
  since a vQTL panel is about variability.
- **Minor allele frequencies are computed vectorised.** The notebook used a row-wise
  `value_counts` lambda over 3.3M variants, which takes hours; the chunked NumPy version
  here gives the same quantity in about 30 seconds.
- **Panel A skips the per-gene mean merge** the notebook performed on the vQTL table. That
  column is not used by the plot, so the twelve single-cell h5ads it required are not read.
- **Panels B and C use the `asian` cohort**, as the published panels do. The `eur`
  equivalents are on the volume and the scripts take a population constant if you want them.
- The stored ATAC filenames spell the pseudobulk method `matqetl` in one directory and
  `mateqtl` in another; both spellings are handled.
