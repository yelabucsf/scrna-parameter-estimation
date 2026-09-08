# Figure 4 — T cell regulatory networks from Perturb-seq

Regenerates every panel of Figure 4 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

| Panel | Shows | Original code |
| --- | --- | --- |
| A | Regulator selection: expression and binding | reconstructed (see Notes) |
| B | sgRNA-by-gene differential mean matrix | `cd4_wt_coex.ipynb` |
| C | DMG effect sizes, and their WT coexpression | `cd4_wt_coex.ipynb` |
| D | Regulator-target correlation in WT cells | `interaction.ipynb` |
| E | Bipartite network from DM alone | `cd4_tf_coex_analysis.ipynb` |
| F | Network including regulator interactions | `cd4_tf_coex_analysis.ipynb` |
| G | Shared binding sites vs window around the TSS | `cd4_tf_coex_analysis.ipynb` |
| H | The LGALS3BP locus with IRF1/PRDM1 peaks | reconstructed (see Notes) |

## Requirements

Python with scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib and networkx.

Figure 4 uses **this repository's** memento package, like Figure 3. It also needs
`GRCh38Genes.bed` from a separate repository:

```bash
git clone https://github.com/mincheoly/misc-seq ~/Github/misc-seq   # set MISCSEQ_PATH
```

## Data

```bash
python data_manifest.py check     # 99 files, 5.3 GB
python data_manifest.py link      # build the panel-organized tree the scripts read
```

`check --root DIR` validates a copy instead of the source volume; `bundle --root DIR`
writes a standalone directory that `FIGURE4_DATA` can point at.

Panel H needs two ENCODE peak files, staged on the volume at `tfko140/encode_peaks/`.
If they are missing:

```bash
mkdir -p /memento_data/tfko140/encode_peaks && cd $_
for acc in ENCFF557FUM ENCFF719BHI; do
  curl -sSL -o $acc.bed.gz "https://www.encodeproject.org/files/$acc/@@download/$acc.bed.gz"
done
```

## Run

```bash
python run_wt_coexpression.py   # ~1 min, 64,261 gene pairs in WT cells
python make_figure4.py          # all panels + figure4.png
```

Individual panels can be plotted alone (`python panel_g_chipseq.py`). Figures land in
`figures/` as pdf and png; intermediates in `intermediate/`, including the Cytoscape edge
lists for panels E and F.

## Checks built in

`perturbseq_data.py` prints the selection it derives — **84 sgRNAs over 63 regulators,
359 DMGs**. That DMG set matches the list recorded in `cd4_wt_coex.ipynb` exactly, so a
change there means something upstream has shifted.

`panel_ef_networks.py` compares its edge lists to the published Cytoscape files and
prints the result; the interaction counts should read 564 and 1,128.

## Notes

- **Panels E and F are laid out with NetworkX here**, but the published figures were laid
  out in Cytoscape. The reproducible artifact is the edge list —
  `intermediate/cytoscape_SIF.csv` and `cytoscape_SIF_explicit.csv` — which can be loaded
  into Cytoscape directly.
- **The `regulates` edge count does not match the stored SIF.** Interaction edges match
  exactly, but the stored file has 14,095 regulatory edges where the notebook's stated
  thresholds give 703; the stored file predates the current code. Both counts are printed
  on every run.
- **Panel A is reconstructed.** No notebook draws it — it documents experimental design
  predating the analysis code — so it is rebuilt from the inputs that design used: the
  ENCODE candidate query in `experiment_report_2022_6_15_19h_31m.tsv`, expression in the
  Perturb-seq counts, and binding breadth from `encode_result.csv`.
- **Panel H is reconstructed.** The notebooks streamed ENCODE peak files at run time and
  deleted them. The same selection rule picks ENCFF557FUM (IRF1) and ENCFF719BHI (PRDM1),
  which the command above downloads.
- `tfko140/1d_tests/` and `cd4_cropseq_data/` are referenced by the notebooks but are not
  on the volume and are not needed — `1d/filtered_1d_result.csv` holds the aggregated
  results the panels read.
