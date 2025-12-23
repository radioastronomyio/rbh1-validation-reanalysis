<!--
---
title: "Data"
description: "Data manifest and staging areas for the RBH-1 validation pipeline"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: [phase-1, phase-2, phase-3, phase-4, phase-5, phase-6, phase-7, phase-8, ard]
tags:
  - domain: data
  - type: data-manifest
related_documents:
  - "[Main README](../README.md)"
  - "[Phase 01 Worklog](../work-logs/01-data-acquisition/README.md)"
---
-->

# Data

## 1. Overview

This folder contains data manifests and staging area structure for the RBH-1 validation pipeline. Actual observational data lives on cluster storage (`/mnt/ai-ml/data/`) — this directory tracks metadata, provenance, and provides a logical organization for pipeline outputs at each stage.

**Note:** Large data files are not committed to the repository. This structure defines the data flow; actual files reside on cluster storage.

---

## 2. Contents

*None — this folder contains only subdirectories.*

---

## 3. Subdirectories

| Directory | Purpose | Phase | README |
|-----------|---------|-------|--------|
| `01_raw/` | Pointers to raw HST/JWST observations | Phase 1 | [README](01_raw/README.md) |
| `02_reduced/` | Extracted 1D spectra and reduced products | Phase 2-4 | [README](02_reduced/README.md) |
| `03_inference/` | MCMC chains and posterior samples | Phase 5-7 | [README](03_inference/README.md) |
| `04_ard/` | ARD package staging before Zenodo upload | ARD | [README](04_ard/README.md) |

---

## 4. Cluster Storage Layout

| Path | Contents | Size |
|------|----------|------|
| `/mnt/ai-ml/data/hst/` | HST GO-17301 products | ~44 GB |
| `/mnt/ai-ml/data/jwst/` | JWST GO-3149 products | ~5 GB |
| `/mnt/ai-ml/data/data_manifest.csv` | Acquisition manifest | — |

---
