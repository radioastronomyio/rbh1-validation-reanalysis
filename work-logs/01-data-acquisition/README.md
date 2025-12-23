<!--
---
title: "Phase 1: Data Acquisition"
description: "HST and JWST data download, integrity validation, and QA visualization"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: phase-1
tags:
  - domain: data
  - type: documentation
related_documents:
  - "[Work Logs README](../README.md)"
  - "[Phase 00 Worklog](../00-ideation-and-setup/README.md)"
  - "[Data Acquisition Script](../../scripts/01-data-acquisition/acquire_data.py)"
---
-->

# Phase 1: Data Acquisition

## 1. Overview

This worklog documents the acquisition of HST WFC3/UVIS (GO-17301) and JWST NIRSpec IFU (GO-3149) observations for RBH-1 validation. The phase covers MAST data retrieval, FITS integrity validation, and generation of QA visualizations to confirm data quality before proceeding to extraction.

**Status:** 🔄 In Progress — Data acquired, validation underway.

---

## 2. Contents

| File | Purpose | Phase |
|------|---------|-------|
| `aquired-files-jwst-and-hst-console-output.txt` | Console log from overnight acquisition run | Phase 1 |
| `aquired-files-jwst-and-hst-directory-list.txt` | Directory listing of all downloaded products | Phase 1 |
| `data_manifest.csv` | Structured manifest with file paths, sizes, checksums | Phase 1 |

---

## 3. Subdirectories

*None — this folder has no subdirectories.*

---

## 4. Acquisition Summary

### HST GO-17301 (WFC3/UVIS)

| Metric | Value |
|--------|-------|
| Products | 142 files |
| Size | ~44 GB |
| Filters | F200LP, F350LP |
| Product Types | FLC (CTE-corrected), DRC (drizzled), HAP skycell mosaics |

### JWST GO-3149 (NIRSpec IFU)

| Metric | Value |
|--------|-------|
| Products | 52 files |
| Size | ~5 GB |
| Configuration | G140M/F100LP |
| Product Types | CAL (Level 2b), S3D (cubes), X1D (1D extracted) |

### Data Location

Actual data staged on cluster storage:

- HST: `/mnt/ai-ml/data/hst/`
- JWST: `/mnt/ai-ml/data/jwst/`

---

## 5. Next Steps

- [ ] Run comprehensive FITS integrity validation
- [ ] Generate HST DRC thumbnails with RBH-1 FOV marked
- [ ] Generate JWST S3D cube slices at key wavelengths
- [ ] Extract and plot 1D spectra with emission lines annotated
- [ ] Create inventory summary figures
- [ ] Complete phase worklog with validation results

---
