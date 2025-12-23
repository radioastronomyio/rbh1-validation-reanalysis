<!--
---
title: "Validation"
description: "Data integrity checks and QA outputs for the validation pipeline"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5 (Anthropic)"
date: "2024-12-23"
version: "1.0"
phase: [phase-1, phase-2, phase-3, phase-4, phase-5, phase-6, phase-7, phase-8]
tags:
  - domain: validation
  - type: documentation
related_documents:
  - "[Main README](../README.md)"
  - "[Phase 01 Worklog](../work-logs/01-data-acquisition/README.md)"
---
-->

# Validation

## 1. Overview

This folder contains data validation outputs, QA figures, and integrity check results from each phase of the validation pipeline. Each phase that produces data also produces validation artifacts demonstrating fitness for downstream use.

---

## 2. Contents

*None — this folder will contain phase-specific validation subdirectories.*

---

## 3. Subdirectories

| Directory | Purpose | Status |
|-----------|---------|--------|
| `phase-01-data-integrity/` | FITS readability, header completeness, WCS consistency | 🔄 In Progress |

*Additional phase validation directories will be created as pipeline progresses.*

---

## 4. Validation Philosophy

Every phase produces artifacts with:

1. **Integrity checks** — Technical validation (file readability, schema compliance)
2. **Sanity checks** — Scientific validation (reasonable values, expected distributions)
3. **Visual QA** — Figures enabling human inspection of key properties

This ensures data quality issues are caught early rather than propagating through the pipeline.

---
