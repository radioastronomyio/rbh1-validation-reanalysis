# Interior README Template

> Template Version: 1.0  
> Applies To: All subdirectory README.md files in rbh1-validation-reanalysis  
> Last Updated: 2024-12-21

---

## Template

```markdown
<!--
---
title: "[Folder Name]"
description: "One-line description of this folder's role in the validation pipeline"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "[Full AI Model Name/Version]"
date: "YYYY-MM-DD"
version: "X.Y"
phase: [phase-1, phase-2, ...]
tags:
  - domain: [extraction/inference/falsification/validation/documentation/data]
  - type: [source-code/configuration/documentation/data-manifest]
related_documents:
  - "[Parent Directory](../README.md)"
  - "[Related Doc](path/to/doc.md)"
---
-->

# [Folder Name]

## 1. Overview

[2-4 sentences explaining what this folder level represents in the RBH-1 validation pipeline. Answer: "What does this folder do, and why does it exist as a distinct unit?"]

---

## 2. Contents

| File | Purpose | Phase |
|------|---------|-------|
| `filename.py` | [One-line description] | Phase X |
| `another-file.md` | [One-line description] | Phase X |

*None — this folder contains only subdirectories.*

---

## 3. Subdirectories

| Directory | Purpose | README |
|-----------|---------|--------|
| `child-folder/` | [One-line description] | [README](child-folder/README.md) |

*None — this folder has no subdirectories.*

---
```

---

## Style Guide

### Fixed Semantic Numbering

Sections 1-3 maintain their position regardless of content. If a section doesn't apply, include it with a "None" placeholder. This enables predictable RAG retrieval—Section 2 is always Contents, Section 3 is always Subdirectories.

### Front-Matter

The front-matter block is wrapped in HTML comments (`<!-- ... -->`) to hide from rendering while remaining parseable for RAG import.

Required fields:
- `title` — Folder name
- `description` — One-line role description
- `author` — Primary author with GitHub link
- `ai_contributor` — AI model/version if applicable
- `date` — Creation date (YYYY-MM-DD)
- `version` — Semantic version (X.Y)
- `phase` — Pipeline phases this folder serves
- `tags` — Domain and type classification
- `related_documents` — Links to parent and related docs

Phase Tags — Use pipeline phases from the main README:

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

A folder may serve multiple phases. List all that apply.

Domain Tags — Primary functional area:

- `extraction` — Spectral extraction from IFU cubes
- `inference` — MCMC sampling, posterior generation
- `falsification` — Galaxy hypothesis forward modeling
- `validation` — Robustness and QA checks
- `documentation` — Methodology, specifications
- `data` — Data manifests, acquisition scripts

### Overview Section

The Overview answers one question: *What role does this folder play in validating RBH-1?*

Good example (`src/inference/`):
> This folder contains the Bayesian inference pipeline for shock parameter estimation. Scripts here consume extracted spectra from `src/extraction/` and produce posterior samples over shock velocity, metallicity, and magnetic parameters. These posteriors form the core statistical claims of the validation paper and materialize directly to the ARD inference layer.

Weak example:
> This folder has inference code.

The good example establishes: inputs, outputs, downstream purpose, and ARD connection.

### Contents Table

Three columns: `File | Purpose | Phase`

- File: Exact filename as link or code-formatted
- Purpose: One line, verb-led ("Extracts...", "Computes...", "Defines...")
- Phase: Which pipeline phase this file primarily serves

Example (`src/extraction/`):

| File | Purpose | Phase |
|------|---------|-------|
| `extract_spectra.py` | Extracts 1D spectra from NIRSpec IFU cubes at defined apertures | Phase 2 |
| `background_model.py` | Fits and subtracts sky/CGM background from cube slices | Phase 3 |
| `noise_empirical.py` | Builds empirical noise model from off-source regions | Phase 4 |

### Subdirectories Table

Only list directories that have their own README. Three columns: `Directory | Purpose | README`

Example (`src/`):

| Directory | Purpose | README |
|-----------|---------|--------|
| `extraction/` | Spectral extraction pipeline | [README](extraction/README.md) |
| `inference/` | MCMC sampling and shock fitting | [README](inference/README.md) |
| `falsification/` | Edge-on galaxy forward modeling | [README](falsification/README.md) |
| `visualization/` | Figure generation for paper | [README](visualization/README.md) |

### Empty Sections

When a section has no items, preserve the section with a placeholder:

```markdown
## 3. Subdirectories

*None — this folder has no subdirectories.*
```

This maintains fixed semantic numbering for RAG systems expecting Section 3 to contain subdirectory information.

### Length Guidelines

Interior READMEs should be 50-100 lines including front-matter. They are navigation aids, not methodology documents. If you're explaining *how* something works rather than *what* it is, that content belongs in a dedicated methodology article.

---

## Relationship to Other Templates

| Template | Purpose | When to Use |
|----------|---------|-------------|
| Interior README (this) | Folder-level navigation and context | Every subdirectory with multiple files |
| General KB Article | Standalone knowledge documentation | Methodology, references, guides, decisions |
| Worklog KB | Session-based development documentation | After implementation sessions |

---

## Validation Checklist

Before committing an interior README:

- [ ] Front-matter wrapped in HTML comments
- [ ] All required front-matter fields populated
- [ ] Overview explains folder's role in validation pipeline
- [ ] All files in folder are listed in Contents table
- [ ] All child directories with READMEs are linked
- [ ] Empty sections use placeholder text, not omission
- [ ] Total length under 100 lines
