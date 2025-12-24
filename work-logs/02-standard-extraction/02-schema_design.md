# RBH-1 Schema Design Document

## Overview

This document describes the PostgreSQL 16 schema for the RBH-1 Analysis-Ready Dataset (ARD). The schema implements a four-zone scalar materialization architecture that transforms raw observational data into queryable physical inferences.

| Zone | Purpose | Tables |
|------|---------|--------|
| 0 | Provenance Anchor | `observations`, `pipeline_runs` |
| 1 | Geometric Materialization | `wcs_solutions`, `spectral_grids` |
| 2 | Spectral Materialization | `ifu_spaxels`, `regions_of_interest` |
| 3 | Physical Materialization | `emission_lines`, `physical_properties` |
| — | Artifact References | `mcmc_metadata`, `mcmc_parameter_summary`, `noise_artifacts` |

## Design Sources

This schema consolidates recommendations from three research passes:

1. GDR Schema Research (2024-12-24): Core four-zone architecture, PostgreSQL 16 optimizations, PostGIS spatial indexing, JWST GWA tilt handling
2. MCMC Storage Research: HDF5 format recommendation based on DESI/SDSS patterns, hybrid DB/filesystem architecture, convergence diagnostics
3. GPT5.2 Review: Provenance table recommendations, extensibility analysis, line covariance considerations

## Key Design Decisions

### Namespace

Schema uses `rbh1.*` namespace within the pgsql01 research data lake, alongside existing cosmic void schemas.

### GWA Tilt Handling

JWST NIRSpec wavelength solutions are exposure-specific due to Grating Wheel Assembly non-repeatability. The `spectral_grids` table materializes the actual wavelength array from each exposure's WCS-TABLE extension, eliminating ~20 km/s systematic errors from assuming static wavelength solutions.

### Spaxel-Centric Design

IFU cubes are decomposed into individual spaxel rows in `ifu_spaxels`. Each row contains:

- Pre-computed sky coordinates (enables PostGIS spatial queries)
- Detector pixel coordinates (enables artifact tracing)
- Full spectrum as TOAST-compressed float arrays
- Feature-centric coordinates (distance along/perpendicular to wake axis)

This design leverages NVMe random-read IOPS for scatter-gather queries.

### Hypothesis Testing

The `physical_properties.bayes_factor` column stores ln(evidence_shock) - ln(evidence_galaxy) as a generated column. Views enforce quality gates:

- `view_gold_sample`: S/N > 5, Bayes factor > 10
- `view_kinematic_gradient`: Valid [OIII] measurements along wake

### MCMC Chain Storage

Full posterior chains are stored as external HDF5 files (too large for inline storage). The schema provides:

- `mcmc_metadata`: File references with convergence diagnostics
- `mcmc_parameter_summary`: Posterior moments for fast queries without file access

File layout: `data/03_inference/mcmc/{spaxel_id}_{line_name}.hdf5`

### Covariance Matrix Storage

Spaxel-spaxel noise covariance from Phase-04 stored as external HDF5/ASDF files. The `noise_artifacts` table references these with summary statistics for quick filtering.

## Scientific Completeness

The schema directly supports the three validation anchors:

| Anchor | Schema Support |
|--------|----------------|
| 600 km/s velocity discontinuity | `emission_lines.velocity` + `view_kinematic_gradient` |
| Shock vs. galaxy model evidence | `physical_properties.bayes_factor` + `view_gold_sample` |
| CGM metallicity test | `physical_properties.metallicity_12_log_oh` + `view_metallicity_profile` |

## Usage

### Deploy Schema

```bash
psql -h pgsql01 -U rbh1_admin -d research -f 02-schema_ddl.sql
```

### Example Queries

Velocity discontinuity at bow shock tip:

```sql
SELECT dist_from_shock_tip_kpc, velocity_kms, velocity_err
FROM rbh1.view_kinematic_gradient
WHERE dist_from_shock_tip_kpc BETWEEN 0 AND 5
ORDER BY dist_from_shock_tip_kpc;
```

Spaxels favoring shock model:

```sql
SELECT COUNT(*), AVG(bayes_factor)
FROM rbh1.view_gold_sample
WHERE region = 'WAKE';
```

Metallicity in CGM vs host galaxy:

```sql
SELECT region, AVG(metallicity_12_log_oh), STDDEV(metallicity_12_log_oh)
FROM rbh1.view_metallicity_profile
GROUP BY region;
```

## Extensibility Notes

- New models: Current Bayes factor assumes two-model comparison. Adding a third model requires schema migration (new column or normalized `model_evidence` table).
- HST imaging: Zone 2 is JWST-centric. HST FLC/DRC products need parallel structure if queryable pixel-level data is required (currently only metadata in Zone 0).
- Multi-epoch: Schema supports multiple observations via `obs_id` foreign keys. Additional epochs require only new rows, not schema changes.

## Provenance

All derived tables include `*_by_run` foreign keys to `pipeline_runs`, enabling full audit trail from raw observation to published figure.
