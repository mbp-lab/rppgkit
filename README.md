<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/svg/rppgkit-logo-dark.svg">
    <img src="assets/brand/svg/rppgkit-logo.svg" alt="rppgkit" width="360">
  </picture>
</p>

<!-- --8<-- [start:body] -->
# rppgkit

rppgkit is a framework for configuration-driven, reproducible remote
photoplethysmography (rPPG) experiments. Every setting is defined in a YAML file under
`configs/`, so an experimental variant is specified by a change to the configuration
rather than to the code.

The current version supports the complete pipeline on UBFC-rPPG with POS: face
cropping, preprocessing, subject-level splits and evaluation. Support for trainable
methods (PhysNet) is in development.

## Installation

rppgkit requires [uv](https://docs.astral.sh/uv/). Clone the repository and install
the locked dependencies:

```bash
git clone <repo-url> rppgkit && cd rppgkit
uv sync --locked
```

Before the first run, set two paths:

- `root` in `configs/dataset/ubfc_rppg.yaml`: the UBFC-rPPG dataset folder.
- `cache_root` in `configs/paths.yaml`: the folder in which caches are written.

## Quick start

The following commands evaluate POS on UBFC-rPPG:

```bash
uv run python scripts/1_extract_roi.py -m                    # build the ROI cache (SLURM, GPU)
uv run python scripts/3_make_splits.py split=test_only       # create the test-only split
uv run python scripts/evaluate.py method=pos split=test_only # write metrics.json
```

## Documentation

- Tutorial: [Your first POS evaluation](https://github.com/mbp-lab/rppgkit/blob/main/docs/tutorials/first-pos-evaluation.md), the quick start in detail.
- How-to guides: [How to extract the ROI](https://github.com/mbp-lab/rppgkit/blob/main/docs/how-to/extract-the-roi.md), [How to evaluate a method](https://github.com/mbp-lab/rppgkit/blob/main/docs/how-to/evaluate-a-method.md).
- Explanation: [How rppgkit works](https://github.com/mbp-lab/rppgkit/blob/main/docs/how-it-works.md), [Design principles](https://github.com/mbp-lab/rppgkit/blob/main/docs/design-principles.md).
- Reference: [References](https://github.com/mbp-lab/rppgkit/blob/main/docs/reference/references.md), full citations for the datasets, methods and tools used in rppgkit.

### Building the documentation locally

The documentation is not yet hosted online and must be built from a local clone. The
pages are in `docs/`, the site configuration is in `mkdocs.yml`, and the API reference
is generated from the docstrings in `src/`. The build requires only the `docs`
dependency group:

```bash
uv run --only-group docs mkdocs serve   # live preview at http://127.0.0.1:8000
```

The preview reloads whenever a page or a docstring changes. Before pushing changes to
the documentation, build the site as the CI does. With `--strict`, broken links are
treated as errors:

```bash
uv run --only-group docs mkdocs build --strict --site-dir site
```

Serve the built site with `python -m http.server -d site`. Opening the files directly
does not work, because the page links point to folders rather than to `index.html`
files.

## Licence

rppgkit is distributed under the GNU General Public License v3.0 only
([LICENSE](https://github.com/mbp-lab/rppgkit/blob/main/LICENSE)). It includes
third-party code under GPL-compatible licences: the YOLO5Face face detector (GPL-3.0),
the PhysNet network (MIT) and an OpenCV Haar cascade (Intel License Agreement). See
[THIRD_PARTY_NOTICES.md](https://github.com/mbp-lab/rppgkit/blob/main/THIRD_PARTY_NOTICES.md).
<!-- --8<-- [end:body] -->
