# quantum-kernel-eeg-emotion

## Overview

We evaluate quantum kernel methods for cross-subject EEG emotion recognition on the DEAP dataset. Three quantum feature maps (domain-informed, generic, random) are compared against classical SVM baselines under leave-one-subject-out (LOSO) cross-validation with within-fold feature selection.

Beyond classification accuracy, we provide eigenspectrum analysis, kernel-target alignment, feature space geometry visualization, and an RFF dequantization test.

## Setup

```bash
conda create -n qk_eeg python=3.11
conda activate qk_eeg
pip install pennylane numpy scikit-learn scipy matplotlib
```

## Data

Download the DEAP dataset (preprocessed Python format, `s01.dat` to `s32.dat`)) and place the files in `data/deap/`.

## Usage

Run experiments:

```bash
# Filtered (primary results)
DEAP_DIR=./data/deap OUTPUT_DIR=./results_val_12q_filtered MIN_MINORITY_RATIO=0.15 \
  python run_experiment.py --task valence --n-qubits 12 --n-features 20

# No filter
DEAP_DIR=./data/deap OUTPUT_DIR=./results_val_12q_nofilter MIN_MINORITY_RATIO=0.0 \
  python run_experiment.py --task valence --n-qubits 12 --n-features 20
```

Generate the circuit diagram:

```bash
python draw_circuit.py
```
