# Geometric Attractor Steering in Non-Equilibrium Tumor Microenvironments

Repository: https://github.com/sukuyoshi/Geometric_Cancer_Attractors

Dissipative systems framework and operator-theoretic control for cancer differentiation therapy, pre-metastatic early warning, and immune checkpoint bifurcation.

## Overview

This repository provides an open-source (MIT Licensed) implementation of a multi-scale dissipative systems framework for oncology:
1. **Multidimensional Tumor Microenvironment (TME) Dynamics**: Modeling the coupled non-linear interactions of cancer stem cells (CSCs), differentiated tumor cells, healthy stroma, cytotoxic lymphocytes, immunosuppressive populations, metabolic stress (acidosis/hypoxia), and extracellular matrix stiffness.
2. **Differentiation Therapy via Least Action Steering**: Formulating optimal epigenetic pulses to escape malignant atavistic attractors into differentiated benign basins without cytotoxic mutational selection.
3. **Sturm-Liouville Gap Softening for Metastatic Transition**: Tracking the spectral gap Omega_1 of the local Koopman generator to forecast drug resistance and epithelial-mesenchymal transition (EMT) prior to clinical dissemination.
4. **Raychaudhuri Convergence Diagnostics for Immunotherapy**: Monitoring phase-space volume expansion rate theta to achieve early discrimination of immune checkpoint blockade responders versus non-responders.

## Architecture

- `src/tumor_microenvironment_dynamics.py`: Coupled non-linear differential equations governing the 7-dimensional TME state vector.
- `simulate_differentiation_therapy.py`: Phase-portrait simulation of differentiation pulse steering versus cytotoxic chemotherapy relapse.
- `simulate_metastasis_early_warning.py`: Koopman operator identification and spectral turbulence gap softening during metastatic cascade onset.
- `simulate_checkpoint_bifurcation.py`: Raychaudhuri convergence diagnostics in immune checkpoint blockade responders versus non-responders.
- `results/`: Trajectory CSVs and high-resolution phase portraits.

## License

MIT License. Copyright 2026 Reiji Yoshikai.
