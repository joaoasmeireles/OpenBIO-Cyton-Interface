# BCI-IM Suite — EEG Motor Imagery Acquisition & Analysis

An end-to-end toolkit for **EEG-based Brain–Computer Interface (BCI)** experiments built around the **OpenBCI Cyton** board. It covers the full research pipeline: recording labeled motor-imagery / movement sessions, checking the integrity of the recordings, and running a rich offline analysis and classification suite — all from desktop GUIs, no command line required.

The suite was designed for **motor imagery (MI)** and **executed movement** paradigms using an 8-channel sensorimotor montage (`C3, Cz, C4, P3, P4, F3, F4, Pz`) at **250 Hz**.

---

## Components

The repository contains three self-contained tools that share a common data format and event-marker scheme.

### 1. `interface_v03.py` — Data Collection Suite
A Tkinter GUI for acquiring labeled EEG sessions from an OpenBCI Cyton (via [BrainFlow](https://brainflow.org/)).

- **Two stimulus protocols:**
  - **Graz B** — 2-class motor imagery (left vs. right hand), with configurable fixation, beep, imagery, rest, and pause durations.
  - **Movement Protocol** — 3-class executed movement (Hand Grip, Elbow Flexion, Shoulder Extension).
- **Cued trial sequencing** with audible beeps (`winsound` on Windows) and on-screen fixation/cue display.
- **Synthetic board mode** for testing the full pipeline with no hardware attached.
- **Live signal preview** with per-channel coloring and basic flat/railed channel detection (impedance/quality hints).
- **Precise event markers** inserted into the BrainFlow stream so cues stay sample-aligned with the EEG.
- **CSV export** including timestamps, 8 EEG channels (scaled to µV), accelerometer axes, and the marker channel, with a commented metadata header (subject, session, protocol, board, sampling rate, marker map).

### 2. `validador_coleta_bci.html` — Data Integrity Validator
A standalone, **dependency-free HTML/JavaScript** page (just open it in a browser, drag a CSV onto it) that validates recordings before you trust them in analysis. Checks include:

- Timestamp presence, monotonicity, and continuity (flags temporal gaps).
- All 8 expected EEG channels present.
- Flat / railed channel detection.
- Event markers present and well-formed (session start/end, block start/end, rest start/end).
- Trial counts per cue, marker ordering, and per-trial duration consistency (with outlier flagging).
- Effective sampling rate vs. the expected 250 Hz.
- Protocol auto-detection and cross-check against the file header.

Each check is reported as **pass / warning / fail** with a human-readable detail line and a marker-count table.

### 3. `bci_analysis_suite_v2.py` — Analysis Suite
A Tkinter split-pane GUI for offline analysis and decoding of the collected sessions.

- **Feature extraction:** CSP, Filter-Bank CSP (FBCSP), Band Power, Hjorth parameters, nonlinear/complexity features, connectivity, and **Riemannian tangent-space** features (when `pyriemann` is installed).
- **Classifiers:** LDA (shrinkage), RBF-SVM, Random Forest, and **XGBoost** (optional), evaluated with `GroupKFold` cross-validation reporting accuracy, Cohen's κ, and confusion matrices.
- **ERD/ERS** analysis (event-related desynchronization/synchronization) following Pfurtscheller & Lopes da Silva (1999), with rest-baseline normalization.
- **Recurrence Quantification Analysis (RQA)** engine computing 11 metrics (RR, DET, L, Lmax, ENTR, LAM, TT, Vmax, LWVL, W, RTE) and **cross-recurrence STR connectivity** (Rodrigues et al., 2019).
- **Nonlinear dynamics:** Lempel-Ziv complexity, Katz fractal dimension, sample entropy, DFA, and Hjorth descriptors.
- **Visualizations:** PSD, topographic maps, ERD/ERS time courses, recurrence plots, connectivity maps, and summary figures, exported as high-DPI PNGs alongside CSV result tables.

---

## Data Format

All recordings are CSV files with a commented metadata header followed by a standard table:

```
# BCI-IM Collection Suite v2.0
# Subject: ...
# Session: ...
# Protocol: graz_b | movement
# Date: YYYY-MM-DD HH:MM:SS
# Board: Cyton | Synthetic
# Sampling Rate: 250 Hz
# Channels: C3, Cz, C4, P3, P4, F3, F4, Pz
# Markers: { ... }
#
Timestamp,C3,Cz,C4,P3,P4,F3,F4,Pz,Accel_X,Accel_Y,Accel_Z,Marker
```

### Event Markers
| Marker | Value | Marker | Value |
|---|---|---|---|
| session_start | 100 | rest_start | 50 |
| session_end | 200 | rest_end | 51 |
| block_start | 90 | pause_start | 99 |
| block_end | 91 | pause_end | 98 |
| fixation | 1 | mi_left | 10 |
| beep | 2 | mi_right | 11 |
| mov_hand | 20 | mov_elbow | 21 |
| mov_shoulder | 22 | | |

---

## Installation

Requires **Python 3.9+**.

```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Data acquisition (Cyton board)
pip install brainflow

# Optional — enables extra classifiers / features in the analysis suite
pip install xgboost pyriemann
```

> **Note:** Audible cues use `winsound` and are Windows-only; the tools run on other platforms without sound.

---

## Usage

**Collect data:**
```bash
python interface_v03.py
```
Select board/port (or enable Synthetic mode), enter subject & session info, choose a protocol, then run. A timestamped CSV is written to your chosen folder.

**Validate a recording:**
Open `validador_coleta_bci.html` in any browser and drag your CSV onto the drop zone.

**Analyze & classify:**
```bash
python bci_analysis_suite_v2.py
```
Load one or more validated CSVs, choose features/classifiers and analysis techniques, and export figures and result tables.

A typical workflow is: **collect → validate → analyze**.
