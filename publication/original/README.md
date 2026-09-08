# Original analysis notebooks

The code as it was actually run for [Kim et al., *Cell*
2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9), kept unmodified for
provenance.

| Directory | Figure |
| --- | --- |
| `validation/` | 2 — method validation and comparisons |
| `hbec_interferon/` | 3 — interferon stimulation in airway epithelium |
| `perturbseq/` | 4 — Perturb-seq of transcription factor knockouts |
| `genetics/` | 5 — eQTL, vQTL and cQTL analysis |
| `cellxgene/` | 6 — memento in CZI CELLxGENE Discover |
| `other/` | exploratory work not in the paper |

**These are a record, not a runnable pipeline.** They carry hardcoded paths to machines
and volumes that no longer exist (`/home/ubuntu/Data/`, `/data_volume/memento/`), import
package versions whose APIs have since changed, and in places read intermediate files that
were never archived.

To reproduce a figure, use [`../figure2/`](../figure2/) through
[`../figure6/`](../figure6/), which are maintained, run against a published data bundle,
and document every place they depart from the code here.

Three of these files are still read at run time by the reproductions, so the directory is
a dependency rather than an archive:

| File | Read by | For |
| --- | --- | --- |
| `hbec_interferon/classify_isg/select_isgs.ipynb` | `figure3/isg_gene_lists.py` | gene lists preserved only in stored cell outputs |
| `perturbseq/encode_tf/metadata.tsv` | `figure4/perturbseq_data.py` | the ENCODE experiment index |
| `cellxgene/rare_celltype_comparison.py` | `figure6/panel_g_crossdataset.py` | the 23 dataset ids, read from the source so the two cannot drift |
