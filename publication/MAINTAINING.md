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

**Two records, deliberately.** The data lives in its own dataset record; the code lives in
the existing record at concept DOI **10.5281/zenodo.13637731**, which is minted
automatically from GitHub releases on `mincheoly/scrna-parameter-estimation` (a fork of
the canonical repo — an artefact of who had release rights at publication time).

They are kept apart because a concept DOI resolves to the *latest* version. If the data
were added as a version of the code record, the next GitHub release would mint a
code-only version, and the DOI people cite would quietly stop pointing at the data. The
two records cross-reference instead: the data record declares `isSupplementedBy` the code
concept DOI, and the code record can carry the reverse link.

To refresh the code, fast-forward the fork to the canonical repo's master and cut a
release; Zenodo mints a new version of the code record on its own, leaving the data record
untouched.

One dataset record holds all five archives. Zenodo serves files individually, so a reader
still downloads only the figure they want. `zenodo_metadata.py` emits its metadata.

Reserve the DOI first, on the website or via the API, so the READMEs and the upload land
in one commit. Then:

```bash
# Put the API token somewhere the shell can read it but history cannot. Run this in a
# real terminal -- `read -rs` needs a TTY and silently writes nothing without one.
read -rs ZENODO_TOKEN && printf '%s' "$ZENODO_TOKEN" > ~/.zenodo_token \
  && chmod 600 ~/.zenodo_token && unset ZENODO_TOKEN
wc -c ~/.zenodo_token          # ~61 bytes; 0 means the read got nothing

DEPOSITION=<draft record id> BUNDLE_DIST=~/bundle_dist ./upload_to_zenodo.sh
```

`upload_to_zenodo.sh` uploads the ten files and stops. It does not publish — that is
irreversible, so review the draft on the website and press Publish there.

**Zenodo returns 504s, and not only under load.** During one upload run every endpoint
started timing out, including read-only ones, for tens of minutes. The script therefore
retries *every* API call with backoff, including the bucket lookup that everything else
depends on — losing that one call means the run never starts. It also resumes: before
uploading it reads the draft's file list with Zenodo's own md5 per file and skips anything
already there with a matching checksum, so an interrupted 11 GB run picks up where it
stopped rather than starting over.

Every accepted file's md5 is compared against the local one, so a truncated upload is
caught rather than sitting in the record looking healthy.

Afterwards, add the reverse link on the code record (edit its metadata, add
`isSupplementTo` pointing at the data DOI), then download one archive from the published
record into a clean directory and run its figure. That is the only check covering the
whole path.

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
