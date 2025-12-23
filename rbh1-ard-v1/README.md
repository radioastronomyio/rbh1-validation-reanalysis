<!--
---
title: "RBH-1 ARD v1"
description: "Analysis-Ready Dataset release package for Zenodo"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: ard
tags:
  - domain: data
  - type: data-manifest
related_documents:
  - "[Main README](../README.md)"
  - "[ARD Specification](../docs/ard-spec.md)"
---
-->

# RBH-1 ARD v1

## 1. Overview

This folder contains the official Analysis-Ready Dataset release package for the RBH-1 validation project. The complete package will be uploaded to Zenodo with a DOI upon paper submission. The ARD enables community reanalysis without requiring access to HPC resources or weeks of compute time.

**Status:** ⏳ Pending — Materialization begins with Phase 9.

---

## 2. Contents

*Planned contents for v1.0 release:*

| Component | Description | Status |
|-----------|-------------|--------|
| `DATASET_CARD.md` | Academic documentation following Gebru et al. standards | ⏳ Pending |
| `DATA_DICTIONARY.md` | Complete field specifications for all layers | ⏳ Pending |
| `likelihood/` | Observed cube, weights, masks, instrument kernels | ⏳ Pending |
| `inference/` | Full MCMC chains (~10M samples) | ⏳ Pending |
| `representative/` | ~30 synthetic cubes + intrinsic velocity fields | ⏳ Pending |
| `validation/` | Jackknife distributions, rejection surfaces | ⏳ Pending |

---

## 3. Subdirectories

*None — subdirectories will be created during ARD materialization.*

---

## 4. ARD Layers

| Layer | Purpose | Enables |
|-------|---------|---------|
| **Likelihood Interface** | Contract for forward model evaluation | Testing alternative physics |
| **Inference Layer** | Full posterior samples | Prior reweighting, credible intervals |
| **Representative Sample** | Visualizable synthetic observations | Residual analysis, sanity checks |
| **Validation Layer** | Robustness artifacts | Jackknife verification, falsification |

---
