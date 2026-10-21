# Design principles

We designed rppgkit to run many controlled variants of rPPG experiments, which differ
for example in the face crop, detector, preprocessing, method or evaluation setting,
and to compare their results reliably. Three principles follow from this aim. The
mechanisms that implement them are described in [How rppgkit works](how-it-works.md).

## Reproducibility through configuration

Every setting is defined in a YAML file under `configs/`, together with its value and a
comment that explains it. No setting is read from environment variables;
machine-specific paths are set once in `configs/paths.yaml`. The settings of a run can
therefore be read in one place, and code and configuration cannot diverge. A variant is
a change to the configuration, not to the code, so two runs differ exactly where their
configurations differ.

We record everything that produced a result. Each cache is named by a hash of its
configuration and of the `version` of every class that the configuration names, and
the key of each stage includes the key of the preceding stage. `metrics.json` records
the cache keys, a hash of the split file, the method and evaluation settings, and an
`experiment_key` that is shared by all folds of one experiment. Provenance does not
depend on git, because experiments are often run from a copy of the code without its
history. Variants coexist on disk and never overwrite one another.

This design has two limitations. A code change that alters the output of a class
requires a manual increment of its `version`; if the increment is omitted, an outdated
cache is reused. In addition, scripts that read a cache must be given the same upstream
settings as the script that built it; otherwise they stop with "cache not found".

## Fair evaluation

We split by subject rather than by recording. All recordings of a subject fall into the
same part of a fold, and a split cannot be constructed if a subject occurs in two
parts. Split files are created once, committed to the repository and never
overwritten, so every ablation is tested on the same subjects, regardless of its ROI or
preprocessing settings.

Methods never receive test labels. Test recordings are loaded without any PPG array, so
a method cannot use the ground truth, not even inadvertently.

Prediction and ground truth are processed identically. Both undergo the same
post-processing (detrending, band-pass filtering, spectral peak detection) with the
same settings. The ground truth is always the raw PPG from the ROI cache, never a
preprocessed label. Methods therefore perform no filtering of their own, so no signal
is filtered twice. Our POS implementation follows the original description
([Wang et al., 2017](reference/references.md#wang2017)) and contains no filter.

The enforcement of these rules is described in
[Splits and test labels](how-it-works.md#splits-and-test-labels) and
[Evaluation](how-it-works.md#evaluation).

## Readability and comparability

Each component has a single responsibility. Each dataset has one reader, each face
detector one folder, and each method one file that contains everything it requires.
Preprocessing steps and methods share small interfaces, and the scripts compose the
stages; no base class combines parsing, splitting, preprocessing and caching. Adding a
dataset requires one reader and one configuration file. The library in `src/rppgkit/`
never reads configuration files, so it can also be used from a notebook.

We introduce an abstraction only at its second use. Code is moved into a shared module
only when a second method, dataset or stage requires it. Until then, it remains where
it is used and can be read from top to bottom.

Our defaults follow published tools. The evaluation defaults reproduce the
post-processing of rPPG-Toolbox and were verified against it numerically. The YOLO5Face
network and its pre- and post-processing are verbatim copies of the code in
rPPG-Toolbox, and our detector returns the same boxes as the rPPG-Toolbox detector.
[Comparison with rPPG-Toolbox](reference/rppg-toolbox.md) lists every correspondence
and every deliberate difference, and each change to copied code is marked with a
comment beginning `rppgkit:`. Our results can therefore be compared with published
results, and every difference can be traced.

This approach has two limitations. Copied code retains its original style and is
excluded from linting, so that it remains line-for-line comparable with its source.
Moreover, code can be copied only if its licence permits distribution under GPL-3.0;
the code of rPPG-Toolbox itself (RAIL licence) is therefore reimplemented rather than
copied. `THIRD_PARTY_NOTICES.md` lists every copied component and its licence.
