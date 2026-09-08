# Reproducing the figures

Code and data for the figures of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

Each figure has a directory that regenerates its panels from a published input bundle.
Download one archive, point one environment variable at it, run one script:

```bash
mkdir -p ~/memento_bundles && cd ~/memento_bundles
curl -L -O https://zenodo.org/records/<RECORD>/files/figure3_data.tar.gz
sha256sum -c figure3_data.tar.gz.sha256
tar -xzf figure3_data.tar.gz
export MEMENTO_DATA_PATH=~/memento_bundles

cd <repo>/publication/figure3 && python make_figure3.py
```

Each figure's README has its own DOI, its exact size, and anything specific to it.

| Figure | Directory | Subject | Data | DOI |
| --- | --- | --- | --- | --- |
| 1 | — | conceptual; no code | — | — |
| 2 | [`figure2/`](figure2/) | method validation and comparisons | 4.1 GB | _pending_ |
| 3 | [`figure3/`](figure3/) | interferon stimulation in airway epithelium | 9.6 GB | _pending_ |
| 4 | [`figure4/`](figure4/) | Perturb-seq of transcription factor knockouts | 5.3 GB | _pending_ |
| 5 | [`figure5/`](figure5/) | eQTL, vQTL and cQTL analysis | 21 GB | _pending_ |
| 6 | [`figure6/`](figure6/) | memento in CZI CELLxGENE Discover | 42 MB | _pending_ |

You need only the bundle for the figure you are reproducing. Nothing requires the full
346 GB working volume, and nothing requires AWS credentials.

Two figures need something extra, for reasons outside our control:

- **Figure 5, panels F–I** need individual-level genotypes, which are controlled-access
  under [dbGaP phs002812.v1.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs002812.v1.p1)
  and cannot be redistributed. Panels A–E are unaffected and the figure assembles without
  F–I.
- **Figure 6** streams cells from the public CELLxGENE census at run time, so panels C and
  D need network access.

## The original notebooks

`validation/`, `hbec_interferon/`, `perturbseq/`, `genetics/`, `cellxgene/` and `other/`
are the **original, unmodified** analysis notebooks, kept for provenance. They carry
hardcoded paths that no longer exist and depend on package versions that have since
changed; they are a record of what was run, not something to run.

`figure2/` through `figure6/` are the maintained reproductions. Where one departs from the
original, its README says so and why.

## Maintainers

[`MAINTAINING.md`](MAINTAINING.md) covers regenerating and republishing the bundles.
