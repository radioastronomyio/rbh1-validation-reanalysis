# Phase 01 Data Validation Report

**Generated:** 2025-12-23T20:45:54.531870+00:00
**Config:** `validation_config.yaml`
**Data Path:** `/mnt/ai-ml/data`

## Summary

✅ 12 passed | ⚠️ 0 warnings | ❌ 0 failed | ℹ️ 6 info

## Validation Results

| Check | Status | Expected | Actual | Detail |
|-------|--------|----------|--------|--------|
| hst_original_flc_count | ✅ PASS | 36 | 36 | Original pipeline (if3x*) |
| hst_hap_flc_count | ✅ PASS | 36 | 36 | HAP reprocessed (hst_17301*) |
| hst_original_drc_count | ℹ️ INFO | - | 12 | Original pipeline DRC |
| hst_hap_drc_count | ℹ️ INFO | - | 58 | HAP reprocessed DRC |
| hst_skycell_count | ℹ️ INFO | - | 4 | HAP skycell mosaics |
| hst_primary_source | ℹ️ INFO | - | HAP FLC | Used for integration time and downstream analysis |
| jwst_cal_count | ✅ PASS | 16 | 16 | - |
| jwst_s3d_count | ℹ️ INFO | - | 18 | - |
| jwst_x1d_count | ℹ️ INFO | - | 18 | - |
| fits_readable | ✅ PASS | All files readable | 198 files OK | - |
| hst_provenance | ✅ PASS | 17301 | All 146 files match | - |
| jwst_provenance | ✅ PASS | 3149 | All 52 files match | - |
| hst_filter_coverage | ✅ PASS | ['F350LP', 'F200LP'] | ['F350LP', 'F200LP', 'detection'] | - |
| jwst_grating | ✅ PASS | G140M | ['G140M'] | - |
| jwst_filter | ✅ PASS | F100LP | ['F100LP'] | - |
| hst_pointing | ✅ PASS | Within 1.0 arcmin | Max offset: 0.588 arcmin | - |
| jwst_pointing | ✅ PASS | Within 1.0 arcmin | Max offset: 0.624 arcmin | - |
| hst_total_integration | ✅ PASS | 30000s ±10% | 29898.0s from 36 HAP FLC files | - |

## Data Sources

### HST Products

MAST delivers two versions of HST FLC (flat-fielded, CTE-corrected) products:

| Type | Pattern | Description |
|------|---------|-------------|
| Original | `if3x*_flc.fits` | Original pipeline calibration |
| HAP | `hst_17301*_flc.fits` | Hubble Advanced Products reprocessing |

HAP products use current best calibration files and improved CTE correction models.
This validation uses **HAP FLC as primary** for integration time calculation,
with original products retained for comparison analysis.

## Validation Criteria

This validation compares acquired data against expected values from:

- van Dokkum et al. (2025) discovery paper
- MAST program metadata for GO-17301 (HST) and GO-3149 (JWST)

Checks verify file integrity, inventory completeness, provenance, instrument configuration,
pointing accuracy, and integration time totals as part of independent validation methodology.