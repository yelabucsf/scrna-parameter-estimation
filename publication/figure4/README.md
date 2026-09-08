# Figure 4 — T cell regulatory networks from Perturb-seq

Runnable reproduction of every panel of Figure 4 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9),
following the layout established in `publication/figure2/` and `figure3/`.

The original code is in `publication/perturbseq/`. Panel-to-notebook mapping was
recovered by grepping every `savefig` call there.

## Panels

| Panel | What it shows | Source notebook | Its output name |
| --- | --- | --- | --- |
| A | Regulator selection: expression and binding | — (see below) | — |
| B | sgRNA-by-gene differential mean matrix | `cd4_wt_coex.ipynb` | — |
| C | DMG effect sizes, and their WT coexpression | `cd4_wt_coex.ipynb` | `effect_size_heatmap.png`, `wt_coexpression.png` |
| D | Regulator-target correlation in WT | `interaction.ipynb` | `tf_deg_wt_corrs.pdf` |
| E | Bipartite network from DM alone | `cd4_tf_coex_analysis.ipynb` | `de_graph.png` |
| F | Network including regulator interactions | `cd4_tf_coex_analysis.ipynb` | `interact_graph.png` |
| G | Shared binding sites vs TSS window | `cd4_tf_coex_analysis.ipynb` | `chipseq.pdf` |
| H | The LGALS3BP locus with IRF1/PRDM1 peaks | — (see below) | — |

## Setup

```bash
python data_manifest.py check     # 99 files, 5.3 GB
python data_manifest.py link      # build the panel-organized tree
```

Needs scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib and networkx.

Like Figure 3, Figure 4 uses **this repository's** memento package, not the
object-oriented rewrite Figure 2 needs. It also needs
[github.com/mincheoly/misc-seq](https://github.com/mincheoly/misc-seq) for
`GRCh38Genes.bed`; point `MISCSEQ_PATH` at its `miscseq/` directory.

## Running

```bash
python run_wt_coexpression.py   # ~1 min, 64,261 gene pairs in WT cells
python make_figure4.py          # all panels, then figures/figure4.png
```

## Validation

The selection logic — which sgRNAs are tested, and which genes count as differentially
expressed — is the foundation every panel rests on, and it reproduces **exactly**.
`cd4_wt_coex.ipynb` cell 49 left its 359-gene DMG list in the stored output, and
`perturbseq_data.dmg_set()` returns the same 359 genes: Jaccard 1.0, nothing missing,
nothing extra. The pipeline selects 84 sgRNAs over 63 regulators, matching the 63
regulators in `encode_result.csv`.

The panel E/F interaction edges also match the published Cytoscape files exactly: 564
`interacts` edges in `cytoscape_SIF.csv` and 1,128 in the `_explicit` variant, both
reproduced on the nose.

## Reconstructed panels

**Panel A** has no source in the repository — it documents experimental design that
predates the analysis code. It is rebuilt from the three inputs that design used, all of
which are on hand: the ENCODE TF ChIP-seq query in
`experiment_report_2022_6_15_19h_31m.tsv` (1,059 candidate regulators), expression of
those candidates in the Perturb-seq counts, and binding breadth from `encode_result.csv`.
Candidates are plotted against the 63 regulators actually perturbed, so the implied
thresholds are visible rather than asserted.

**Panel H** likewise has no source cell. The notebooks' `encode.Encode` helper streamed
ENCODE peak files at run time and deleted them afterwards, leaving nothing on the volume.
Its own selection rule — highest-ranked IDR thresholded peaks on GRCh38, no audit errors
— picks ENCFF557FUM (IRF1, K562) and ENCFF719BHI (PRDM1, A549); both are downloaded and
staged at `tfko140/encode_peaks/`. The TSS is defined as in `Encode.get_tss_window`.
Both factors bind within a few hundred bases of the LGALS3BP TSS, which is the point the
panel makes.

## Deviations

* **The `regulates` edges in the Cytoscape output do not match the stored file.** The
  interaction edges match exactly (564 / 1,128), but the stored `cytoscape_SIF.csv` has
  14,095 `regulates` edges where notebook cell 64's thresholds (`de_fdr < 0.001` and
  `|de_coef| > 0.1`) yield 703. No threshold on the stored `de_fdr` column reproduces
  14,095 — the closest, `de_fdr < 0.2`, gives 13,901 — and the stored file uses a
  `regulates` relation where cell 64 writes `upRegulates`/`downRegulates`, so it predates
  the current code. This script follows cell 64 and reports both counts on every run.
* **Panels E and F are rendered with NetworkX**, following the notebook's own layout code.
  The published panels were laid out in Cytoscape; the edge lists are the reproducible
  artifact and are written to `intermediate/cytoscape_SIF*.csv`.
* **Panel B shows all 84 selected sgRNAs against the 4,892 genes** present in the effect
  matrix, rather than every gene tested, since the matrix is built from the guides that
  pass selection.

## Missing inputs that turned out not to matter

`tfko140/1d_tests/` and `tfko140/cd4_cropseq_data/` are referenced by the notebooks, are
not on the volume, and never appeared in the S3 sync log. Neither is needed: they are
per-guide intermediates, and `1d/filtered_1d_result.csv` holds the aggregated results
(709,340 guide-gene tests) the panels actually read.

## A memento bug this figure surfaced

`memento/main.py` and `memento/estimator.py` called `np.in1d`, removed in numpy 2.0, so
`compute_1d_moments` raised `AttributeError` on any current environment. Fixed to
`np.isin` in the same commit that adds this figure's scaffolding.

That was masked by a second bug: `REPO_ROOT` in the figure3 and figure4 configs resolved
one directory short, so both were importing a pip-installed memento from site-packages
rather than this repository's package. Figure 3's results are unaffected — `estimator.py`,
`bootstrap.py` and `util.py` are byte-identical between the two, the differing `main.py`
functions differ only in hypothesis-test paths the moment computation never enters, and
recomputing Figure 3's control gene-by-gene matrix under the repo package reproduces the
stored one bit for bit.
