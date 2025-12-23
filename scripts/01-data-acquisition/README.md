<!--
---
title: "Data Acquisition Scripts"
description: "MAST data retrieval and validation scripts for HST and JWST observations"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: phase-1
tags:
  - domain: data
  - type: source-code
related_documents:
  - "[Scripts README](../README.md)"
  - "[Phase 01 Worklog](../../work-logs/01-data-acquisition/README.md)"
---
-->

# Data Acquisition Scripts

## 1. Overview

This folder contains scripts for acquiring HST and JWST observations from MAST and validating their integrity. These scripts implement idempotent, resumable data retrieval with full provenance tracking.

---

## 2. Contents

| File | Purpose | Phase |
|------|---------|-------|
| `acquire_data.py` | Downloads HST GO-17301 and JWST GO-3149 products from MAST | Phase 1 |
| `validate_data.py` | FITS integrity checks, header validation, WCS consistency | Phase 1 |

---

## 3. Subdirectories

*None — this folder has no subdirectories.*

---

## 4. Usage

```bash
# Activate environment
source /mnt/ai-ml/venvs/venv-ml-py312/bin/activate

# Run acquisition (idempotent - skips existing files)
python scripts/01-data-acquisition/acquire_data.py

# Run validation
python scripts/01-data-acquisition/validate_data.py
```

---
