# Reproducing the figures

Code and data for the figures of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

Each figure has a directory that regenerates its panels from a published input bundle.
Download one archive, point one environment variable at it, run one script:

```bash
mkdir -p ~/memento_bundles && cd ~/memento_bundles
curl -L -O https://zenodo.org/records/22667586/files/figure3_data.tar.gz
sha256sum -c figure3_data.tar.gz.sha256
tar -xzf figure3_data.tar.gz
export MEMENTO_DATA_PATH=~/memento_bundles

cd <repo>/publication/figure3 && python make_figure3.py
```

All the archives live in one Zenodo record — DOI [10.5281/zenodo.22667586](https://doi.org/10.5281/zenodo.22667586) — and are independent of each
other, so download only the one you need.

| Figure | Directory | Subject | Download | Unpacked |
| --- | --- | --- | --- | --- |
| 1 | — | conceptual; no code | — | — |
| 2 | [`figure2/`](figure2/) | method validation and comparisons | 1.3 GB | 4.1 GB |
| 3 | [`figure3/`](figure3/) | interferon stimulation in airway epithelium | 2.7 GB | 9.6 GB |
| 4 | [`figure4/`](figure4/) | Perturb-seq of transcription factor knockouts | 1.6 GB | 4.9 GB |
| 5 | [`figure5/`](figure5/) | eQTL, vQTL and cQTL analysis | 5.8 GB | 21 GB |
| 6 | [`figure6/`](figure6/) | memento in CZI CELLxGENE Discover | 44 MB | 44 MB |

Nothing here requires the 346 GB working volume the analysis was run against, and nothing
requires AWS credentials.

Two figures need something extra, for reasons outside our control:

- **Figure 5, panels F–I** need individual-level genotypes, which are controlled-access
  under [dbGaP phs002812.v1.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs002812.v1.p1)
  and cannot be redistributed. Panels A–E are unaffected and the figure assembles without
  F–I.
- **Figure 6** streams cells from the public CELLxGENE census at run time, so panels C and
  D need network access.

## Layout

```
publication/
  figure2/ .. figure6/   the maintained reproductions — start here
  original/              the analysis as it was actually run, unmodified
  MAINTAINING.md         regenerating and republishing the data bundles
```

[`original/`](original/) holds the notebooks behind the paper, kept for provenance. They
carry hardcoded paths to machines that no longer exist and depend on package versions
whose APIs have changed — a record of what was run, not something to run. Three files in
there are still read at run time by the reproductions; `original/README.md` says which.

Where a reproduction departs from the original, its README says so and why.

## Maintainers

[`MAINTAINING.md`](MAINTAINING.md) covers regenerating and republishing the bundles.
