<!--
---
title: "Source Library"
description: "Reusable library modules for the RBH-1 validation pipeline"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: [phase-2, phase-3, phase-4, phase-5, phase-6, phase-7, phase-8, ard]
tags:
  - domain: extraction
  - type: source-code
related_documents:
  - "[Main README](../README.md)"
  - "[Scripts](../scripts/README.md)"
---
-->

# Source Library

## 1. Overview

This folder contains reusable library modules imported by phase scripts. Unlike scripts (which are execution entry points), `src/` modules encapsulate domain logic that may be used across multiple phases. This separation ensures scripts remain thin orchestration layers while core functionality lives in testable, importable modules.

---

## 2. Contents

*None — this folder contains only subdirectories.*

---

## 3. Subdirectories

| Directory | Purpose | Status | README |
|-----------|---------|--------|--------|
| `extraction/` | Spectral extraction from IFU cubes | ⏳ Pending | [README](extraction/README.md) |
| `inference/` | MCMC sampling and likelihood evaluation | ⏳ Pending | [README](inference/README.md) |
| `falsification/` | Edge-on galaxy forward modeling | ⏳ Pending | [README](falsification/README.md) |
| `visualization/` | Figure generation for paper and QA | ⏳ Pending | [README](visualization/README.md) |

---

## 4. Import Pattern

```python
# Scripts import from src modules
from src.extraction import extract_spectra
from src.inference import run_mcmc
from src.visualization import plot_posterior
```

---
