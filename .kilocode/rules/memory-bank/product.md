# RBH-1 Validation & Reanalysis Product Overview

## Problems Solved

This project addresses:

- Compute wall for reanalysis: Researchers wanting to revisit RBH-1 results face weeks of CPU time regenerating posteriors. The ARD eliminates this barrier by preserving computed artifacts.
- Methodological opacity: Published results often lack sufficient detail for exact reproduction. Full provenance chains from MAST download to final measurement enable verification.
- Binary validation framing: Claims are often framed as "confirmed" or "debunked." Neutral validation identifies where interpretations are robust and where additional work is needed.
- Edge-on galaxy alternative: The competing hypothesis (serendipitously aligned thin galaxy) requires quantitative falsification, not dismissal.

## How It Works

The validation operates through a 9-phase pipeline from data acquisition through ARD materialization. Each phase produces artifacts that become inputs to subsequent phases, creating clear provenance chains.

Key components:

- Cube-native likelihood evaluation: Models meet data in IFU space rather than reduced constraints, preserving spatial-spectral covariance
- Producer-Consumer architecture: CPU nodes compute galactic dynamics (galpy), GPU node handles convolution and likelihood evaluation (JAX), DragonFlyDB brokers tasks
- MAPPINGS V inference: Pre-computed shock model grids enable Bayesian parameter estimation
- Galaxy falsification campaign: Forward modeling of edge-on hypothesis with Toomre Q stability filtering

The ARD materializes four layers: likelihood interface (for drop-in forward model testing), inference layer (full MCMC chains), representative sample (~30 synthetic cubes), and validation layer (jackknife distributions).

## Goals and Outcomes

### Primary Goals

1. Test bow shock interpretation: Determine if 95% CI of shock velocity from line ratios overlaps spatial gradient measurement
2. Characterize systematics: Quantify difference-of-pointings artifacts, background subtraction sensitivity
3. Falsify edge-on alternative: Produce rejection surface for galaxy hypothesis parameter space
4. Package ARD: Release complete Analysis-Ready Dataset on Zenodo with DOI

### Success Metrics

- Validation completeness: All four confirmation/tension conditions evaluated with quantitative results
- ARD utility: Dataset enables posterior reweighting without recomputation
- Reproducibility: Independent researcher can regenerate key results from ARD + published methodology
- Community value: ARD downloaded and cited by subsequent RBH-1 studies
