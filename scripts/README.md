<!--
---
title: "Scripts"
description: "Phase-organized execution scripts for the RBH-1 validation pipeline"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: [phase-0, phase-1, phase-2, phase-3, phase-4, phase-5, phase-6, phase-7, phase-8, ard]
tags:
  - domain: extraction
  - type: source-code
related_documents:
  - "[Main README](../README.md)"
  - "[Source Library](../src/README.md)"
---
-->

# Scripts

## 1. Overview

This folder contains phase-organized execution scripts for the RBH-1 validation pipeline. Scripts are entry points that orchestrate the pipeline — they import reusable functionality from `src/` and produce outputs consumed by subsequent phases. Each phase subfolder corresponds to a discrete validation step with clear inputs and outputs.

---

## 2. Contents

*None — this folder contains only subdirectories.*

---

## 3. Subdirectories

| Directory | Purpose | Status | Key Scripts |
|-----------|---------|--------|-------------|
| `00-ideation-and-setup/` | Repository scaffolding | ✅ Complete | `create-repo.ps1` |
| `01-data-acquisition/` | MAST data retrieval and validation | 🔄 In Progress | `acquire_data.py`, `validate_data.py` |
| `02-standard-extraction/` | Baseline 1D spectra extraction | ⏳ Pending | — |
| `03-cube-differencing/` | Systematic artifact characterization | ⏳ Pending | — |
| `04-noise-model/` | Empirical covariance estimation | ⏳ Pending | — |
| `05-kinematic-fitting/` | Tied velocity field posteriors | ⏳ Pending | — |
| `06-mappings-inference/` | MAPPINGS V shock parameter inference | ⏳ Pending | — |
| `07-robustness-tests/` | Jackknife and multi-line coherence | ⏳ Pending | — |
| `08-galaxy-falsification/` | Edge-on hypothesis forward modeling | ⏳ Pending | — |
| `09-ard-materialization/` | ARD packaging for Zenodo | ⏳ Pending | — |

---

## 4. Usage Pattern

```bash
# Activate environment on cluster
source /mnt/ai-ml/venvs/venv-ml-py312/bin/activate

# Run phase script
python scripts/01-data-acquisition/validate_data.py

# Scripts produce outputs in work-logs/ and validation/
```

---
