---
title: 'BCI-IM Suite: An end-to-end toolkit for EEG motor-imagery acquisition, validation, and analysis'
tags:
  - Python
  - Neuroscience
  - Electroencephalography
  - Brain-computer interface
  - Motor imagery
  - Signal processing
authors:
  - name: "João Alfredo S de Meireles"
    orcid: 0000-0003-4284-874X  
    corresponding: true
    affiliation: 1
affiliations:
  - name: "Instituto Tecnológico de Aeronáutica (ITA), São José dos Campos, Brasil"
    index: 1
bibliography: paper.bib
---

# Summary

Brain–computer interfaces (BCIs) translate brain activity into control signals,
combined motor imagery (MI), the mental rehearsal of a movement without execution, 
are one of the most widely studied paradigms for non-invasive,
electroencephalography (EEG) based BCIs. Building an MI experiment from scratch,
however, requires stitching together several distinct stages: presenting timed
stimuli and recording sample-aligned event markers, verifying that the recorded
signals are usable before any analysis, and finally extracting features and
decoding the mental states. BCI-IM Suite is an end-to-end, GUI-driven toolkit
that covers this entire pipeline for the low-cost OpenBCI Cyton board, requiring
no command-line interaction from the experimenter. It consists of three
self-contained tools that share a common CSV data format and event-marker
scheme: a data-collection application that runs cued MI and executed-movement
protocols, a dependency-free data-integrity validator, and an offline analysis
and classification suite. The toolkit targets an 8-channel sensorimotor montage
(C3, Cz, C4, P3, P4, F3, F4, Pz) sampled at 250 Hz, and its analysis stage
combines conventional spatial-filtering decoders with recurrence-based and
nonlinear-dynamics methods that are rarely available together in a single
open-source package.

# Statement of need

Researchers and students setting up an EEG motor-imagery experiment with
consumer-grade hardware such as the OpenBCI Cyton typically assemble their
pipeline from a patchwork of scripts: one for stimulus presentation and
acquisition, ad-hoc notebooks for quality control, and a separate analysis
codebase. This fragmentation makes experiments hard to reproduce, error-prone at
the acquisition stage (mislabeled trials, dropped samples, railed channels that
go unnoticed until analysis), and raises the barrier to entry for students who
are not yet proficient with low-level signal-processing libraries.

BCI-IM Suite addresses this gap by providing the three stages as integrated,
graphical tools built on a single, documented data contract. Event markers are
inserted directly into the BrainFlow [@brainflow] data stream so that cues stay
sample-aligned with the EEG, and every recording carries a commented metadata
header (subject, session, protocol, board, sampling rate, marker map). A
standalone HTML/JavaScript validator lets an experimenter confirm timestamp
continuity, channel presence, marker ordering, per-trial durations, and the
effective sampling rate before trusting a recording. The software is intended for BCI
researchers prototyping MI experiments, for instructors teaching neural signal
processing, and for students who need a reproducible, inspectable pipeline
rather than a black box.

# State of the field

Several mature open-source tools exist in the EEG/BCI space. MNE-Python
[@mne] is the de-facto general-purpose library for EEG/MEG analysis, and
BrainFlow [@brainflow] provides a uniform acquisition API across many boards,
including the Cyton. Dedicated BCI frameworks such as OpenViBE and BCI2000
offer real-time stimulus presentation and online decoding, while pyRiemann
[@congedo] supplies Riemannian-geometry classifiers that are now standard in MI
decoding competitions.

BCI-IM Suite is not intended to replace these libraries; rather, it composes
several of them (BrainFlow for acquisition, scikit-learn [@scikit-learn] for
classification, and optionally pyRiemann and XGBoost [@xgboost]) into a single,
opinionated, end-to-end workflow specialised for a fixed sensorimotor montage
and a documented data format. Two design choices distinguish it from the general
toolkits above. First, it ships an explicit, GUI-based *validation* stage as a
first-class citizen of the workflow, encoding the quality-control checks that
practitioners usually perform informally. Second, alongside conventional Common
Spatial Patterns (CSP) [@ramoser] and Filter-Bank CSP [@ang] decoders, the
analysis suite integrates recurrence-based functional connectivity
(space-time-recurrence / cross-recurrence quantification, after
[@rodrigues2019]) and a battery of nonlinear-dynamics descriptors (Lempel-Ziv
complexity, Katz fractal dimension, sample entropy, detrended fluctuation
analysis, and Hjorth parameters) within the same point-and-click interface.
This combination of standard and nonlinear methods is uncommon in existing
open, GUI-driven BCI tools.

# Software design

The suite follows a three-stage, loosely-coupled architecture in which the
stages communicate only through the shared CSV-with-metadata format, so each
tool can be used, replaced, or extended independently.

interface.py is a Tkinter acquisition application wrapping BrainFlow. It
implements two stimulus protocols (a two-class Graz-B left/right hand MI
protocol and a three-class executed-movement protocol), cued trial sequencing
with audible beeps and on-screen fixation/cue display, a synthetic-board mode
for hardware-free testing, a live signal preview with flat/railed channel
detection, and CSV export with sample-aligned event markers.

validador_coleta_bci.html is a single, dependency-free HTML/JavaScript page:
the user drags a CSV onto it and receives a per-check pass/warning/fail report
covering timestamp continuity, channel presence, signal sanity, marker
well-formedness and ordering, per-trial duration consistency, effective
sampling rate, and protocol auto-detection cross-checked against the header.

bci_analysis_suite.py is a Tkinter split-pane analysis application. It
exposes feature extractors (CSP, FBCSP, band power, Hjorth, nonlinear/complexity
features, connectivity, and Riemannian tangent-space features when pyRiemann is
available), classifiers (shrinkage-LDA, RBF-SVM, Random Forest, and optionally
XGBoost) evaluated with GroupKFold cross-validation reporting accuracy,
Cohen's kappa, and confusion matrices, an ERD/ERS analysis following
[@pfurtscheller], a Recurrence Quantification Analysis engine computing eleven
standard metrics plus cross-recurrence connectivity, and high-DPI figure export
(power spectral densities, topographic maps, ERD/ERS time courses, recurrence
plots, and connectivity maps) alongside CSV result tables.

# Research impact statement

BCI-IM Suite was developed to support the author's research on EEG-based
motor-imagery decoding and is the acquisition-and-analysis backbone of that
work. Its scholarly contribution lies less in any single algorithm — most of
which are established in the literature — than in lowering the barrier to
reproducible MI experiments: it couples a documented data contract, an
explicit validation stage, and a unified analysis interface that places
conventional spatial-filtering decoders side-by-side with recurrence-based and
nonlinear-dynamics methods that practitioners would otherwise have to implement
themselves. By packaging these as inspectable, GUI-driven tools that run on
inexpensive OpenBCI hardware, the suite is positioned for use both in BCI
research prototyping and in teaching settings where students benefit from an
open, end-to-end pipeline they can read and modify.

<!--
  REVIEWER-FACING IMPACT EVIDENCE — ADICIOANR PRE-PRINT DO PAPER NOSSO 
-->

# AI usage disclosure

Generative AI (Anthropic Claude Opus 4.6) was used to assist with the
implementation of the graphical user-interface (front-end) layer of the
acquisition and analysis applications. The signal-processing, feature-extraction,
and classification logic was designed and written by the author. The author reviewed, edited, and validated all AI-assisted outputs,
tested every function to verify correctness and integrity, and retained all core
design decisions and scientific claims.

# Acknowledgements

<<< Optional: acknowledge advisors, funding agencies (e.g. CAPES, CNPq, FAPESP),
and collaborators here, or delete this section. >>>

# References
