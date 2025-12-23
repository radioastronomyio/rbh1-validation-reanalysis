<!--
---
title: "Phase 0: Ideation and Setup"
description: "Project conception, repository scaffolding, and infrastructure decisions for RBH-1 independent validation"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "Claude Opus 4.5"
date: "2024-12-23"
version: "1.0"
phase: phase-0
tags:
  - domain: project-setup
  - type: methodology
  - tech: hst, jwst, infrastructure
related_documents:
  - "[Main README](../../README.md)"
  - "[Validation Spec](../../docs/validation-spec.md)"
  - "[ARD Spec](../../docs/ard-spec.md)"
---
-->

# Phase 0: Ideation and Setup

> Compiled from: Multiple sessions | December 2024  
> Status: Complete  
> Key Outcome: Repository scaffolding, methodology documentation, and infrastructure allocation for independent RBH-1 validation

---

## 1. Objective

Establish the foundation for an independent scientific validation of the RBH-1 hypervelocity SMBH candidate (van Dokkum et al. 2025). This phase defined the project's neutral validation stance, designed the twin-deliverable architecture (validation paper + Analysis-Ready Dataset), and scaffolded the repository structure to support a 9-phase pipeline from data acquisition through ARD materialization.

---

## 2. Scientific Context

### The Object

RBH-1 is a 62 kiloparsec linear feature extending from a compact source at z ≈ 0.96. If the interpretation holds, this represents the first direct observation of gravitational wave recoil — a supermassive black hole ejected from its host galaxy, now traveling at ~1000 km/s through the circumgalactic medium and creating a supersonic bow shock.

### The Competing Hypotheses

| Hypothesis | Proponents | Key Predictions |
|------------|------------|-----------------|
| Supersonic bow shock from ejected SMBH | van Dokkum et al. (2025) | Velocity discontinuity at tip, shock-consistent line ratios, CGM metallicity |
| Edge-on bulgeless dwarf galaxy | Sánchez Almeida et al. (2023) | Stellar continuum, disk kinematics, Toomre-stable configuration |

### Our Stance

Neutral validation. Contributing to the scientific discussion around extraordinary objects is an honor, not an adversarial exercise. The project either strengthens the bow shock interpretation or identifies where additional work is needed — both outcomes advance the field.

---

## 3. Key Architectural Decisions

### Decision 1: Twin Co-Equal Deliverables

Decision: Produce a validation paper and an Analysis-Ready Dataset (ARD) as inseparable, co-equal outputs.

Rationale: Astronomical reanalysis has a hidden cost — weeks of CPU time between downloading calibrated data and having usable posteriors. Every researcher faces this compute wall, often regenerating artifacts that others already produced and discarded. The effort to package computed artifacts (MCMC chains, synthetic cubes, likelihood interfaces) is negligible compared to regeneration cost. This isn't methodological innovation; it's just not throwing away what we already computed.

Implications: ARD specification must be designed upfront, not retrofitted. Pipeline outputs are structured for preservation from the start.

### Decision 2: Cube-Native Likelihood Evaluation

Decision: Models meet data in IFU space rather than reduced constraints.

Rationale: Traditional approaches reduce 3D spectral cubes to 1D spectra or 2D maps before fitting. This discards spatial-spectral covariance information. Cube-native evaluation preserves the full likelihood surface, enabling proper marginalization over nuisance parameters (PSF variations, background structure).

Implications: Requires GPU-accelerated convolution. Forward model generates synthetic cubes that are compared directly to observed cubes after instrument effects are applied.

### Decision 3: Producer-Consumer Pipeline Architecture

Decision: Separate galactic dynamics computation (CPU) from likelihood evaluation (GPU) via task brokering.

Rationale: Heterogeneous cluster with 3 CPU-heavy nodes and 1 GPU node. Dynamics calculations (galpy, Toomre Q stability) are embarrassingly parallel but not GPU-friendly. Convolution and likelihood evaluation benefit from GPU acceleration. Decoupling ensures neither resource sits idle.

Implications: DragonFlyDB for ephemeral task brokering, PostgreSQL for durable chain storage. CPU nodes produce velocity fields, GPU node consumes and evaluates.

### Decision 4: 9-Phase Pipeline Structure

Decision: Organize work into discrete phases with clear handoffs and validation criteria.

| Phase | Name | Primary Output |
|-------|------|----------------|
| 00 | Ideation and Setup | Repository, methodology docs |
| 01 | Data Acquisition | Validated HST/JWST products |
| 02 | Standard Extraction | Baseline 1D spectra, comparison to published |
| 03 | Cube Differencing | Systematic artifact characterization |
| 04 | Noise Model | Empirical covariance structure |
| 05 | Kinematic Fitting | Tied velocity field posteriors |
| 06 | MAPPINGS Inference | Shock parameter posteriors |
| 07 | Robustness Tests | Jackknife, multi-line coherence |
| 08 | Galaxy Falsification | Edge-on hypothesis rejection surface |
| 09 | ARD Materialization | Zenodo-ready dataset package |

Rationale: Phase boundaries create natural checkpoints for validation and documentation. Each phase produces artifacts that become inputs to subsequent phases, enabling modular development and clear provenance chains.

---

## 4. Repository Structure

```
rbh1-validation-reanalysis/
├── .kilocode/              # AI agent memory bank and rules
│   └── rules/memory-bank/  # Persistent context across sessions
├── assets/                 # Hero images, diagrams
├── data/                   # Data manifest (actual data on cluster storage)
├── docs/                   # Methodology specifications
│   └── documentation-standards/  # Templates and conventions
├── notebooks/              # Exploratory analysis
├── rbh1-ard-v1/           # ARD package staging area
├── scripts/               # Phase-organized execution scripts
│   ├── 00-ideation-and-setup/
│   ├── 01-data-acquisition/
│   └── ...
├── src/                   # Reusable library code
│   ├── extraction/
│   ├── inference/
│   ├── falsification/
│   └── visualization/
├── validation/            # Data validation outputs and figures
├── work-logs/             # Phase worklogs (this directory)
└── README.md              # Project overview
```

### Key Conventions

- Scripts vs src: Scripts are phase-specific execution entry points. src/ contains reusable library code imported by scripts.
- Work-logs: Phase worklogs are synthesis documents compiled from multiple sessions, focused on outcomes rather than session-by-session granularity.
- Interior READMEs: Each significant directory self-documents via README with contents description and links to child READMEs.

---

## 5. Infrastructure Allocation

### Compute Cluster

| Node | CPU | RAM | GPU | Role |
|------|-----|-----|-----|------|
| proj-gpu01 | AMD 5950X (16c) | 48 GB | NVIDIA A4000 16GB | GPU inference, task broker, primary compute |
| proj-cpu01 | Intel 12900K (12c) | 48 GB | — | MCMC walker pool |
| proj-cpu02 | Intel 12900K (12c) | 48 GB | — | MCMC walker pool |
| proj-cpu03 | Intel 12900K (12c) | 48 GB | — | MCMC walker pool |

### Storage Layout

| Path | Purpose | Capacity |
|------|---------|----------|
| `/mnt/ai-ml/data/hst/` | HST GO-17301 products | ~50 GB |
| `/mnt/ai-ml/data/jwst/` | JWST GO-3149 products | ~5 GB |
| `/mnt/ai-ml/rbh1/` | Working directory, scripts | — |

### Runtime Environment

- Python: 3.12 via `venv-ml-py312`
- Key packages: astropy, astroquery, numpy, scipy, matplotlib, seaborn, JAX, galpy
- Task broker: DragonFlyDB (ephemeral)
- Chain storage: PostgreSQL (durable)

---

## 6. Data Sources Identified

### HST WFC3/UVIS — Program GO-17301

| Filter | Exposure | Products |
|--------|----------|----------|
| F200LP | ~15 ks | FLC (CTE-corrected), DRC (drizzled) |
| F350LP | ~15 ks | FLC (CTE-corrected), DRC (drizzled) |

6 visits across the RBH-1 field. HAP skycell mosaics provide wide-field context.

### JWST NIRSpec IFU — Program GO-3149

| Configuration | Exposure | Products |
|---------------|----------|----------|
| G140M/F100LP | ~7 ks | CAL (Level 2b), S3D (cubes), X1D (1D extracted) |

Two target positions along the linear feature (t001, t002).

### MAPPINGS V Shock Models

Pre-computed grids from 3MdB database for radiative shock parameter inference.

---

## 7. Validation Criteria Established

### Confirmation Conditions

The bow shock interpretation is validated if:

- 95% CI of shock velocity from line ratios overlaps spatial gradient measurement
- Difference-of-pointings artifacts account for <20% of wake flux
- Inferred metallicity consistent with CGM (Z < 0.5 Z☉)
- Velocity discontinuity persists across all jackknife subsets

### Tension Conditions

Tension with the interpretation is declared if:

- Marginalizing over magnetic parameter broadens v_s posterior to include virial velocities
- Standard Level 3 extraction significantly diminishes wake signal
- Jackknife analysis shows exposure/region dependence

---

## 8. ARD Specification

### Layers Defined

| Layer | Contents | Purpose |
|-------|----------|---------|
| Likelihood Interface | Observed cube, weights, masks, instrument kernels | Drop-in forward model testing |
| Inference Layer | Full MCMC chains (~10M samples) | Statistical reanalysis, prior reweighting |
| Representative Sample | ~30 synthetic cubes + intrinsic velocity fields | Visual inspection, residual analysis |
| Validation Layer | Jackknife distributions, rejection surfaces | Robustness verification |

### What This Enables

A researcher receiving the ARD can:

- Skip weeks of compute — posterior samples already exist
- Test alternative priors — reweight existing chains instead of rerunning
- Run their own forward model — likelihood interface provides the contract
- Validate against new data — future observations plug directly in
- Reproduce exactly — full provenance chain from MAST to final measurement

---

## 9. Artifacts Produced

| Artifact | Purpose | Location |
|----------|---------|----------|
| `README.md` | Project overview and scientific context | Repository root |
| `ROADMAP.md` | Implementation timeline | Repository root |
| `acquire_data.py` | MAST data retrieval script | `scripts/` |
| `validate_data.py` | FITS validation framework | `scripts/` |
| Phase worklog template | Documentation standard | `docs/documentation-standards/` |
| Interior README template | Navigation standard | `docs/documentation-standards/` |
| Memory bank templates | AI context persistence | `.kilocode/rules/memory-bank/` |

---

## 10. Lessons Learned

| Challenge | Resolution |
|-----------|------------|
| Scope definition for "validation" | Established neutral stance — validation strengthens or identifies gaps, not adversarial debunking |
| ARD design timing | Specified ARD layers upfront rather than retrofitting after analysis complete |
| Documentation overhead | Adopted phase worklog pattern — synthesis documents rather than session-by-session logs |

Key Insight: The ARD-as-deliverable approach inverts traditional thinking. Instead of treating computed artifacts as disposable byproducts archived after publication, treat them as first-class deliverables designed from project inception. The marginal cost of packaging is trivial compared to the regeneration cost others would face.

---

## 11. Next Phase

Enables: Phase 01 (Data Acquisition) now has clear targets — HST GO-17301 and JWST GO-3149 products with defined validation criteria.

Dependencies resolved: Repository structure, documentation standards, infrastructure allocation, scientific methodology all established.

Open items: None blocking Phase 01.

---

## 12. Provenance

| Item | Value |
|------|-------|
| Repository created | December 2024 |
| Python target | 3.12 |
| Primary runtime | venv-ml-py312 on proj-gpu01 |
| Methodology references | van Dokkum et al. (2025), Sánchez Almeida et al. (2023) |

---

Next: [Phase 01: Data Acquisition](../01-data-acquisition/README.md)
