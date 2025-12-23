<!--
---
title: "Tagging Strategy"
description: "Controlled vocabulary for document classification and RAG retrieval in rbh1-validation-reanalysis"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4 (Anthropic)"
date: "2024-12-21"
version: "1.0"
phase: [phase-1, phase-2, phase-3, phase-4, phase-5, phase-6, phase-7, phase-8, ard]
tags:
  - domain: documentation
  - type: specification
related_documents:
  - "[Interior README Template](interior-readme-template.md)"
  - "[General KB Template](kb-general-template.md)"
---
-->

# Tagging Strategy

## 1. Purpose

This document defines the controlled tag vocabulary for all documentation in rbh1-validation-reanalysis, enabling consistent classification for human navigation and RAG system retrieval.

---

## 2. Scope

Covers all tag categories, valid values, and usage guidance. Does not cover front-matter field structure—see individual templates for field requirements.

---

## 3. Tag Categories

### Phase Tags

Pipeline execution phases. Documents may belong to multiple phases.

| Tag | Description |
|-----|-------------|
| `phase-1` | Data acquisition + provenance |
| `phase-2` | Standard extraction baseline |
| `phase-3` | Cube-space differencing |
| `phase-4` | Empirical noise model |
| `phase-5` | Tied kinematic fitting |
| `phase-6` | MAPPINGS V inference |
| `phase-7` | Robustness tests |
| `phase-8` | Edge-on galaxy falsification |
| `ard` | Analysis-Ready Dataset materialization |

**Usage**: Tag with all phases a document supports. A methodology doc explaining background subtraction used in phases 2-4 would carry `phase-2`, `phase-3`, `phase-4`.

---

### Domain Tags

Primary functional area. Usually one per document.

| Tag | Description |
|-----|-------------|
| `extraction` | Spectral extraction from IFU cubes |
| `inference` | MCMC sampling, posterior generation |
| `falsification` | Galaxy hypothesis forward modeling |
| `validation` | Robustness and QA checks |
| `documentation` | Methodology, specifications, standards |
| `data` | Data manifests, acquisition, provenance |

**Usage**: Choose the primary domain. A document about validating inference results is `validation`, not `inference`.

---

### Type Tags

Document purpose and structure.

| Tag | Description |
|-----|-------------|
| `methodology` | How we do something |
| `reference` | Lookup information |
| `guide` | Step-by-step procedures |
| `decision-record` | Why we chose X over Y |
| `specification` | Formal requirements |
| `source-code` | Code files and scripts |
| `configuration` | Config files, parameters |
| `data-manifest` | Data inventory and provenance |

**Usage**: One type per document. If a document explains both *how* and *why*, choose the dominant purpose.

---

### Tech Tags

Data sources and external dependencies.

| Tag | Description |
|-----|-------------|
| `hst` | Hubble Space Telescope data (GO-17301) |
| `jwst` | JWST NIRSpec IFU data (GO-3149) |
| `mappings-v` | MAPPINGS V shock model grids |

**Usage**: Tag when the document is specific to that data source. A general extraction methodology doc doesn't need `jwst`; a doc about NIRSpec cube structure does.

---

## 4. References

| Reference | Link |
|-----------|------|
| Main README | [../../README.md](../../README.md) |
| Interior README Template | [interior-readme-template.md](interior-readme-template.md) |
| General KB Template | [kb-general-template.md](kb-general-template.md) |

---
