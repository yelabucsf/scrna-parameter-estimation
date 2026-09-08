# Maintaining the figure reproductions

For whoever regenerates or republishes the figure data. Readers reproducing a figure need
none of this — they download one archive and run one script, as each figure's README
describes.

## The two layers of data

**The source volume**, `/memento_data`, is a flat sync of `s3://memento-paper/revision/`
organized by *dataset*, which says nothing about which figure needs what. It is ~346 GB
and private.

**The figure bundles** are what gets published: one per figure, organized by *panel and
role*, containing only the files that figure reads. Together they are a few percent of the
volume.

`figureN/data_manifest.py` is the mapping between them, and the single source of truth for
what a figure needs.

## data_manifest.py

```bash
cd publication/figureN
python data_manifest.py check                # is everything present on the volume?
python data_manifest.py check --root DIR     # ... or in a downloaded bundle
python data_manifest.py link                 # panel-organized symlink tree, for local work
python data_manifest.py bundle --root DIR    # real copies into a staging directory
python data_manifest.py archive --root DIR   # ... and a .tar.gz + .sha256 to publish
```

Each entry is `(panel, tier, destination in the tree, source path)`. The tiers:

| Tier | Meaning | In the bundle? |
| --- | --- | --- |
| `required` | a panel script reads it directly | yes |
| `provenance` | not read at plot time, but needed to regenerate a `required` file | yes |
| `restricted` | may not be redistributed | **never** |

`bundle` and `archive` require `--root`. They write real copies, and earlier they defaulted
to `config.FIGUREN_DATA` — which meant a bare `python data_manifest.py bundle` buried the
volume's symlink tree under gigabytes of duplicates. `link` still defaults there, because
that is what it is for.

`archive` refuses to build if any `required` source is missing. A bundle silently missing
a panel's input is worse than no bundle.

## The layout contract

An archive's single top-level directory is `figureN_data/`. That is not cosmetic:
`config.FIGUREN_DATA` defaults to `MEMENTO_DATA_PATH + 'figureN_data/'`, so a reader
extracts anywhere, points `MEMENTO_DATA_PATH` at the parent, and every path resolves. One
environment variable, no per-figure configuration.

Renaming that directory breaks the instruction in all five READMEs.

Paths are built by string concatenation, so each `config.py` runs its directory variables
through `_dir()`, which appends a missing trailing slash and expands `~`. Without it
`MEMENTO_DATA_PATH=~/bundles` silently yields `~/bundlesfigure2_data/`.

## Restricted data

`figure5` withholds `genotypes/{eur,asian}_genos.tsv`: individual-level genotypes for 91
named CLUES participants, released by Perez et al. 2022 (*Science*) only under
**dbGaP phs002812.v1.p1** with a signed Data Use Certification. They are tagged
`restricted`, so `link` still builds them into a local tree — a maintainer with dbGaP
access runs everything — while `bundle` and `archive` refuse to copy them.

What ships instead: panel A only ever needed a minor-allele frequency per variant, which
is aggregate and identifies nobody. Regenerate it after any genotype change with

```python
python -c "import panel_a_qqplots; panel_a_qqplots.write_minor_allele_frequencies()"
```

writing to `/memento_data/lupus/mateqtl_input/maf/`. The csv round-trip is accurate to
~1e-16 and no variant sits within 1e-12 of the 10% filter threshold, so the substitution
cannot change which variants pass.

Panels F–I need per-individual calls at two variants and cannot be served this way. They
are skipped when the genotypes are absent, and `make_figure5.py` assembles A–E instead of
failing.

**Before adding any new data source to a bundle, check its redistribution terms.** The
third-party files currently kept as fetch-it-yourself rather than redistributed are
figure 3's `mostafavi2016_mmc2.xls` (Elsevier supplementary) and figure 4's ENCODE peaks.

## Regeneration paths kept out of the user READMEs

- **Figure 2, BASiCS** — `panel_a_run_simulations.py variance --dump-basics-inputs` then
  `Rscript run_basics_simulation.R` (conda env `r-env`, R 4.5 + BASiCS 2.22, spec in
  `figure2/r-env.yml`). Both write to `figure2/intermediate/basics_simulation/`; override
  with `FIGURE2_BASICS_DIR`. The bundle ships the resulting parameter csvs, so readers
  never run this.
- **Figure 4, guide selection** — `data_manifest.TESTED_GUIDES` is a frozen copy of
  `perturbseq_data.selected_guides()`, because the manifest must enumerate itself before
  the tree it describes exists. After changing the selection filters, run
  `python -c "import data_manifest; data_manifest.verify_guides()"`, which re-derives the
  list and fails on drift.
- **Figure 6, cubes** — `build_dc_subset.py` slices the 17 GB census cube to the five
  dendritic-cell types panel G reads (45 MB); that slice is what ships. `build_cube.py`
  builds panels C/D's comparison cube from the census in ~2 min, and needs
  [memento-cxg](https://github.com/mincheoly/memento-cxg) at PR #5 or later. The full
  census cube stays on the volume.

## Guard against reaching past the bundle

Every panel script must read from `config.FIGUREN_DATA`, never from the raw volume, or it
will work for you and fail for everyone else. This should print nothing:

```bash
grep -rn 'config\.\(DATA_PATH\|HBEC_PATH\|TFKO_PATH\|LUPUS_PATH\|MISCSEQ_PATH\)' \
  publication/figure{2,3,4,5,6}/*.py | grep -v data_manifest.py | grep -v config.py
```

## Publishing to Zenodo

One record per figure, so each has its own DOI to cite in its README and can be
re-versioned independently. `zenodo_metadata.py N` emits the record metadata.

**Rehearse on `sandbox.zenodo.org` first — published Zenodo files are immutable.** A
mistake means a new version, not an edit.

Reserve the DOI before writing it into the README, so the docs and the upload land in one
commit:

```bash
export ZENODO_TOKEN=...                 # scopes: deposit:write, deposit:actions
export ZENODO=https://zenodo.org/api    # sandbox.zenodo.org/api to rehearse
N=3

# 1. Draft the record and reserve its DOI.
curl -sS -X POST "$ZENODO/deposit/depositions?access_token=$ZENODO_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"metadata":{"prereserve_doi":true}}' > deposit.json
DEP=$(jq -r .id deposit.json)
BUCKET=$(jq -r .links.bucket deposit.json)
jq -r .metadata.prereserve_doi.doi deposit.json    # -> paste into figureN/README.md

# 2. Upload. Use the bucket API: the deposit/files API does not scale past a few GB,
#    and figure 5 is ~20 GB.
curl -sS --progress-bar -X PUT \
     "$BUCKET/figure${N}_data.tar.gz?access_token=$ZENODO_TOKEN" \
     --upload-file figure${N}_data.tar.gz
curl -sS -X PUT \
     "$BUCKET/figure${N}_data.tar.gz.sha256?access_token=$ZENODO_TOKEN" \
     --upload-file figure${N}_data.tar.gz.sha256

# 3. Attach metadata and publish.
python zenodo_metadata.py $N > metadata_figure${N}.json
curl -sS -X PUT "$ZENODO/deposit/depositions/$DEP?access_token=$ZENODO_TOKEN" \
     -H 'Content-Type: application/json' -d @metadata_figure${N}.json
curl -sS -X POST \
     "$ZENODO/deposit/depositions/$DEP/actions/publish?access_token=$ZENODO_TOKEN"
```

Confirm the paper DOI in `zenodo_metadata.py` before the first upload — it is marked
TODO there, and a wrong identifier cannot be corrected in place.

After publishing, download one record from a clean directory and run the figure from it.
That is the only check that covers the whole path.

## Republishing a bundle

1. Fix whatever changed, and re-run the guard above.
2. `python data_manifest.py archive --root /staging/figureN_data --out /dist/figureN_data.tar.gz`
3. **Validate offline**, which is the step that actually catches a missed path: extract to
   a clean directory, `export MEMENTO_DATA_PATH=<that directory>`, and run
   `make_figureN.py` with the source volume unreachable. Compare each panel's printed
   summary statistics against the figure's README.
4. Upload as a **new version** of the figure's existing Zenodo record, so the DOI keeps
   resolving. Published Zenodo files are immutable — rehearse on `sandbox.zenodo.org`
   first.
5. Update the version and size in the figure's README if they changed.
