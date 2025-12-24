#!/usr/bin/env python3
"""
===============================================================================
Script       : 02-validate_data.py
Phase        : Phase 01 - Data Acquisition
Project      : RBH-1 Independent Validation Study
Repository   : rbh1-validation-reanalysis
-------------------------------------------------------------------------------
Author       : CrainBramp / VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-23
===============================================================================

SCIENTIFIC CONTEXT
==================
This script performs independent validation of the acquired HST and JWST data
against the claims in van Dokkum et al. (2025). An independent validation study
must verify that the data being analyzed actually matches what the discovery
paper describes—this is distinct from simply checking file integrity.

VALIDATION PHILOSOPHY
=====================
We distinguish between two types of validation:

1. Generic validation: "Is this valid astronomical data?"
   - Files open correctly
   - Headers parse
   - No corruption

2. Independent validation: "Does this match what the paper claims?"
   - Correct observing programs (GO-17301, GO-3149)
   - Expected file counts (36 FLC exposures)
   - Target coordinates match (RA 40.43°, Dec -8.35°)
   - Integration time matches (~30,000s)
   - Instrument configuration matches (F200LP/F350LP, G140M/F100LP)

This script implements both layers, with emphasis on the second. Every check
that passes strengthens our confidence that we're analyzing the same data
that led to the discovery claim.

DATA PRODUCTS
=============
HST products come in two versions from MAST:

- Original pipeline (if3x*): Calibration files from observation epoch
- HAP reprocessed (hst_17301*): Hubble Advanced Products with current calfiles

We validate both but use HAP as primary for downstream analysis (better
CTE correction, improved astrometry). The integration time check uses HAP
FLC files only to avoid double-counting.

CONFIGURATION
=============
Expected values are loaded from validation_config.yaml rather than hardcoded.
This allows easy updates if the paper's claimed values are revised, and makes
the validation criteria explicit and auditable.

OUTPUT
======
- validation_results.json: Machine-parseable results for automation
- validation_report.md: Human-readable summary for documentation

Exit codes:
- 0: All checks passed
- 1: One or more checks failed
- 2: Warnings only (no failures)

Usage:
    python 02-validate_data.py /path/to/data --config validation_config.yaml

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================


def load_config(config_path: Path) -> dict:
    """
    Load validation configuration from YAML file.

    The config contains expected values derived from van Dokkum et al. (2025)
    and MAST program metadata. Externalizing these values makes the validation
    criteria transparent and easily auditable.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary with target, hst, and jwst sections.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# RESULT FORMATTING
# =============================================================================


def check_result(name: str, status: str, expected=None, actual=None, detail: str = None) -> dict:
    """
    Create a standardized check result dictionary.

    Standardized format enables both human review and automated processing
    of validation results.

    Parameters
    ----------
    name : str
        Unique identifier for this check (e.g., "hst_flc_count").
    status : str
        One of: "PASS", "WARN", "FAIL", "INFO"
        - PASS: Check succeeded, data matches expectations
        - WARN: Potential issue but not blocking
        - FAIL: Data does not match expectations
        - INFO: Informational only, not a pass/fail check
    expected : any, optional
        What we expected to find (from paper/config).
    actual : any, optional
        What we actually found in the data.
    detail : str, optional
        Additional context or explanation.

    Returns
    -------
    dict
        Structured result for inclusion in output.
    """
    result = {"name": name, "status": status}
    if expected is not None:
        result["expected"] = expected
    if actual is not None:
        result["actual"] = actual
    if detail:
        result["detail"] = detail
    return result


# =============================================================================
# FITS HEADER UTILITIES
# =============================================================================


def get_header_value(header, *keys):
    """
    Get first available header value from list of possible keys.

    FITS headers vary between instruments and pipeline versions. HST and JWST
    use different keywords for the same information (e.g., RA_TARG vs TARG_RA).
    This function provides a robust way to extract values regardless of which
    keyword convention is used.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header object.
    *keys : str
        Header keywords to try, in order of preference.

    Returns
    -------
    any or None
        First found value, or None if no keys present.

    Examples
    --------
    >>> ra = get_header_value(header, 'RA_TARG', 'TARG_RA', 'CRVAL1')
    """
    for key in keys:
        if key in header:
            return header[key]
    return None


# =============================================================================
# INTEGRITY VALIDATION
# -----------------------------------------------------------------------------
# These checks verify basic data integrity: can we read the files?
# =============================================================================


def validate_fits_readable(fits_files: list[Path]) -> list[dict]:
    """
    Check that each FITS file can be opened and its HDU list accessed.
    
    Returns a single check-result describing overall readability: `PASS` if all files opened successfully, or `FAIL` with details (up to five unreadable files and their error messages) if any files could not be read.
    
    Parameters:
        fits_files (list[Path]): Paths to FITS files to validate.
    
    Returns:
        list[dict]: A list containing one check-result dictionary summarizing the readability check.
    """
    results = []
    unreadable = []

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                # Force access to HDU list to catch lazy-load errors
                _ = len(hdul)
        except Exception as e:
            unreadable.append((fpath.name, str(e)))

    if unreadable:
        results.append(check_result(
            "fits_readable",
            "FAIL",
            expected="All files readable",
            actual=f"{len(unreadable)} unreadable",
            detail="; ".join([f"{n}: {e}" for n, e in unreadable[:5]])
        ))
    else:
        results.append(check_result(
            "fits_readable",
            "PASS",
            expected="All files readable",
            actual=f"{len(fits_files)} files OK"
        ))

    return results


# =============================================================================
# INVENTORY VALIDATION
# -----------------------------------------------------------------------------
# These checks verify we have the expected number and types of files.
# =============================================================================


def validate_hst_inventory(hst_path: Path, config: dict) -> tuple[list[dict], list[Path], list[Path], list[Path]]:
    """
    Validate HST file inventory against expected counts.

    MAST delivers two versions of HST products:

    1. Original pipeline (if3x*): Uses calibration files from observation epoch
    2. HAP reprocessed (hst_17301*): Uses current best calibration files

    We validate both independently because:
    - Original matches what van Dokkum initially analyzed
    - HAP provides best available calibration for our reanalysis

    Parameters
    ----------
    hst_path : Path
        Directory containing HST FITS files.
    config : dict
        Validation configuration with expected counts.

    Returns
    -------
    tuple
        (results, original_flc_files, hap_flc_files, all_hst_files)
        Returns file lists for use in downstream validation checks.
    """
    results = []

    # ---- Identify files by pipeline version ----
    # Original pipeline products have ipppssoot naming (e.g., if3x01abq)
    original_flc = list(hst_path.glob("if3x*_flc.fits"))
    original_drc = list(hst_path.glob("if3x*_drc.fits"))

    # HAP products have systematic naming (e.g., hst_17301_01_wfc3_...)
    hap_flc = list(hst_path.glob("hst_*_flc.fits"))
    hap_drc = list(hst_path.glob("hst_*_drc.fits"))

    # Skycell mosaics are HAP-only products covering the full field
    skycell = list(hst_path.glob("hst_skycell*"))

    # ---- Validate original FLC count ----
    # Paper states 36 FLC exposures
    expected_orig = config['hst']['expected_original_flc_count']
    actual_orig = len(original_flc)
    if actual_orig == expected_orig:
        results.append(check_result("hst_original_flc_count", "PASS", expected_orig, actual_orig,
                                    detail="Original pipeline (if3x*)"))
    elif actual_orig > 0:
        results.append(check_result("hst_original_flc_count", "WARN", expected_orig, actual_orig,
                                    detail="Original pipeline count mismatch"))
    else:
        results.append(check_result("hst_original_flc_count", "FAIL", expected_orig, actual_orig))

    # ---- Validate HAP FLC count ----
    # Should also be 36 (same exposures, different calibration)
    expected_hap = config['hst']['expected_hap_flc_count']
    actual_hap = len(hap_flc)
    if actual_hap == expected_hap:
        results.append(check_result("hst_hap_flc_count", "PASS", expected_hap, actual_hap,
                                    detail="HAP reprocessed (hst_17301*)"))
    elif actual_hap > 0:
        results.append(check_result("hst_hap_flc_count", "WARN", expected_hap, actual_hap,
                                    detail="HAP reprocessed count mismatch"))
    else:
        results.append(check_result("hst_hap_flc_count", "FAIL", expected_hap, actual_hap))

    # ---- Informational counts (no pass/fail criteria) ----
    results.append(check_result("hst_original_drc_count", "INFO", actual=len(original_drc),
                                detail="Original pipeline DRC"))
    results.append(check_result("hst_hap_drc_count", "INFO", actual=len(hap_drc),
                                detail="HAP reprocessed DRC"))
    results.append(check_result("hst_skycell_count", "INFO", actual=len(skycell),
                                detail="HAP skycell mosaics"))

    # ---- Document which source we'll use for integration time ----
    primary = config['hst'].get('primary_flc_source', 'hap')
    results.append(check_result("hst_primary_source", "INFO",
                                actual=f"{'HAP' if primary == 'hap' else 'Original'} FLC",
                                detail="Used for integration time and downstream analysis"))

    all_files = original_flc + original_drc + hap_flc + hap_drc + skycell
    return results, original_flc, hap_flc, all_files


def validate_jwst_inventory(jwst_path: Path, config: dict) -> tuple[list[dict], list[Path]]:
    """
    Validate JWST file inventory against expected counts.

    JWST NIRSpec IFU observations produce three product types:
    - CAL: Calibrated 2D spectra (one per dither position)
    - S3D: Reconstructed 3D spectral cubes
    - X1D: Extracted 1D spectra

    We expect 16 CAL files from the dither pattern (2 targets × 4 dithers × 2 nods).

    Parameters
    ----------
    jwst_path : Path
        Directory containing JWST FITS files.
    config : dict
        Validation configuration with expected counts.

    Returns
    -------
    tuple
        (results, all_jwst_files)
    """
    results = []

    cal_files = list(jwst_path.glob("*_cal.fits"))
    s3d_files = list(jwst_path.glob("*_s3d.fits"))
    x1d_files = list(jwst_path.glob("*_x1d.fits"))

    # ---- Validate CAL count ----
    expected_cal = config['jwst']['expected_cal_count']
    actual_cal = len(cal_files)

    if actual_cal == expected_cal:
        results.append(check_result("jwst_cal_count", "PASS", expected_cal, actual_cal))
    elif actual_cal > 0:
        results.append(check_result("jwst_cal_count", "WARN", expected_cal, actual_cal,
                                    detail="Count mismatch - verify dither pattern"))
    else:
        results.append(check_result("jwst_cal_count", "FAIL", expected_cal, actual_cal))

    # ---- Informational counts ----
    results.append(check_result("jwst_s3d_count", "INFO", actual=len(s3d_files)))
    results.append(check_result("jwst_x1d_count", "INFO", actual=len(x1d_files)))

    return results, cal_files + s3d_files + x1d_files


# =============================================================================
# PROVENANCE VALIDATION
# -----------------------------------------------------------------------------
# These checks verify the data comes from the correct observing programs.
# =============================================================================


def validate_provenance(fits_files: list[Path], expected_program: int, telescope: str) -> list[dict]:
    """
    Check that each FITS file lists the expected observing program identifier.
    
    Parameters:
        fits_files (list[Path]): FITS file paths to inspect.
        expected_program (int): Expected proposal/program identifier to match.
        telescope (str): Telescope label used in the result name (e.g., "HST" or "JWST").
    
    Returns:
        list[dict]: A single-element list containing a validation result dict. The result
            has status "PASS" if all files match, "WARN" if up to 10% of files mismatch,
            and "FAIL" if more than 10% mismatch. The `expected` field contains the
            expected program id, `actual` summarizes the number of mismatches, and
            `detail` includes up to three example mismatched file messages.
    """
    results = []
    mismatched = []

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                program_id = get_header_value(header, 'PROPOSID', 'PROGRAM', 'PROGID')

                if program_id is None:
                    mismatched.append((fpath.name, "No program ID found"))
                elif int(program_id) != expected_program:
                    mismatched.append((fpath.name, f"Got {program_id}"))
        except Exception as e:
            mismatched.append((fpath.name, f"Error: {e}"))

    if mismatched:
        # If >10% mismatch, it's a failure; otherwise just a warning
        status = "FAIL" if len(mismatched) > len(fits_files) * 0.1 else "WARN"
        results.append(check_result(
            f"{telescope.lower()}_provenance",
            status,
            expected=expected_program,
            actual=f"{len(mismatched)} mismatched",
            detail="; ".join([f"{n}: {r}" for n, r in mismatched[:3]])
        ))
    else:
        results.append(check_result(
            f"{telescope.lower()}_provenance",
            "PASS",
            expected=expected_program,
            actual=f"All {len(fits_files)} files match"
        ))

    return results


# =============================================================================
# INSTRUMENT CONFIGURATION VALIDATION
# -----------------------------------------------------------------------------
# These checks verify the correct filters/gratings were used.
# =============================================================================


def validate_hst_filters(fits_files: list[Path], config: dict) -> list[dict]:
    """
    Check that the provided HST FITS files include the filters declared in the config.
    
    Parameters:
        fits_files (list[Path]): HST FITS files to inspect.
        config (dict): Configuration containing expected filters under config['hst']['filters'].
    
    Returns:
        list[dict]: A single validation result dict named "hst_filter_coverage".
            - Status is `PASS` if all expected filters are present, `FAIL` if any are missing.
            - `expected` lists the filters from the configuration, `actual` lists detected filters.
            - `detail` (present on failure) lists which expected filters are missing.
    """
    results = []
    filters_found = set()

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                filt = get_header_value(header, 'FILTER', 'FILTER1', 'FILTER2')
                if filt:
                    filters_found.add(filt.strip())
        except Exception:
            pass

    expected_filters = set(config['hst']['filters'])

    # We need both filters present; additional filters (e.g., 'detection') are OK
    if expected_filters.issubset(filters_found):
        results.append(check_result(
            "hst_filter_coverage",
            "PASS",
            expected=list(expected_filters),
            actual=list(filters_found)
        ))
    else:
        missing = expected_filters - filters_found
        results.append(check_result(
            "hst_filter_coverage",
            "FAIL",
            expected=list(expected_filters),
            actual=list(filters_found),
            detail=f"Missing: {missing}"
        ))

    return results


def validate_jwst_config(fits_files: list[Path], config: dict) -> list[dict]:
    """
    Validate that JWST FITS headers contain the expected disperser (grating) and filter as specified in the config.
    
    Checks the GRATING or DISPERSER header values against config['jwst']['grating'] and the FILTER or FWA_POS header values against config['jwst']['filter'].
    
    Parameters:
        fits_files (list[Path]): Paths to JWST FITS files to inspect.
        config (dict): Validation configuration; must include keys 'jwst' -> 'grating' and 'jwst' -> 'filter'.
    
    Returns:
        list[dict]: Two standardized validation result dictionaries (one for grating, one for filter), produced by check_result, containing status, expected value, observed values, and optional detail.
    """
    results = []
    gratings_found = set()
    filters_found = set()

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                grating = get_header_value(header, 'GRATING', 'DISPERSER')
                filt = get_header_value(header, 'FILTER', 'FWA_POS')
                if grating:
                    gratings_found.add(grating.strip())
                if filt:
                    filters_found.add(filt.strip())
        except Exception:
            pass

    expected_grating = config['jwst']['grating']
    if expected_grating in gratings_found:
        results.append(check_result("jwst_grating", "PASS", expected_grating, list(gratings_found)))
    else:
        results.append(check_result("jwst_grating", "WARN", expected_grating, list(gratings_found),
                                    detail="Grating mismatch or not found in headers"))

    expected_filter = config['jwst']['filter']
    if expected_filter in filters_found:
        results.append(check_result("jwst_filter", "PASS", expected_filter, list(filters_found)))
    else:
        results.append(check_result("jwst_filter", "WARN", expected_filter, list(filters_found),
                                    detail="Filter mismatch or not found in headers"))

    return results


# =============================================================================
# POINTING VALIDATION
# -----------------------------------------------------------------------------
# These checks verify all observations target the correct sky position.
# =============================================================================


def validate_pointing(fits_files: list[Path], config: dict, telescope: str) -> list[dict]:
    """
    Check that observations' pointing falls within the configured target tolerance.
    
    Parameters:
        fits_files (list[Path]): FITS files whose headers will be inspected for RA/Dec.
        config (dict): Configuration containing target coordinates ('target.ra_deg', 'target.dec_deg')
            and pointing tolerance in arcminutes ('target.pointing_tolerance_arcmin').
        telescope (str): Telescope identifier used in the result name (e.g., "HST" or "JWST").
    
    Returns:
        list[dict]: A list with a single validation result dict indicating PASS if all files are
        within tolerance, FAIL if one or more files exceed the tolerance (includes count and examples),
        or WARN if no coordinates could be extracted.
    """
    results = []

    target_coord = SkyCoord(
        ra=config['target']['ra_deg'] * u.deg,
        dec=config['target']['dec_deg'] * u.deg
    )
    tolerance = config['target']['pointing_tolerance_arcmin'] * u.arcmin

    offsets = []
    failed_files = []

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header

                # Try multiple possible keywords for coordinates
                ra = get_header_value(header, 'RA_TARG', 'TARG_RA', 'CRVAL1')
                dec = get_header_value(header, 'DEC_TARG', 'TARG_DEC', 'CRVAL2')

                if ra is not None and dec is not None:
                    obs_coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
                    sep = target_coord.separation(obs_coord)
                    offsets.append(sep.arcmin)

                    if sep > tolerance:
                        failed_files.append((fpath.name, f"{sep.arcmin:.2f} arcmin"))
        except Exception:
            pass

    if not offsets:
        results.append(check_result(
            f"{telescope.lower()}_pointing",
            "WARN",
            detail="Could not extract coordinates from headers"
        ))
    elif failed_files:
        results.append(check_result(
            f"{telescope.lower()}_pointing",
            "FAIL",
            expected=f"Within {tolerance.value} arcmin",
            actual=f"{len(failed_files)} files outside tolerance",
            detail="; ".join([f"{n}: {o}" for n, o in failed_files[:3]])
        ))
    else:
        max_offset = max(offsets)
        results.append(check_result(
            f"{telescope.lower()}_pointing",
            "PASS",
            expected=f"Within {tolerance.value} arcmin",
            actual=f"Max offset: {max_offset:.3f} arcmin"
        ))

    return results


# =============================================================================
# INTEGRATION TIME VALIDATION
# -----------------------------------------------------------------------------
# Verifies total exposure time matches paper claims.
# =============================================================================


def validate_hst_integration(primary_flc_files: list[Path], config: dict, source_name: str) -> list[dict]:
    """
    Check that the summed HST FLC exposure time from the chosen primary source matches the expected total in the config.
    
    Sums EXPTIME/TEXPTIME from the provided primary FLC files and compares the total to config['hst']['expected_total_integration_sec'] using config['hst']['integration_tolerance_pct'] percent; produces a PASS if within tolerance or a WARN if outside it.
    
    Parameters:
        primary_flc_files (list[Path]): FLC files from the selected primary source (HAP or original).
        config (dict): Validation configuration containing `hst.expected_total_integration_sec` and `hst.integration_tolerance_pct`.
        source_name (str): Human-readable name of the primary source used in reporting (e.g., "HAP" or "original").
    
    Returns:
        list[dict]: A single-element list containing the validation result dictionary for the total HST integration check.
    """
    results = []

    total_exptime = 0.0
    flc_count = 0

    for fpath in primary_flc_files:
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                exptime = get_header_value(header, 'EXPTIME', 'TEXPTIME')
                if exptime:
                    total_exptime += float(exptime)
                    flc_count += 1
        except Exception:
            pass

    expected = config['hst']['expected_total_integration_sec']
    tolerance_pct = config['hst']['integration_tolerance_pct']
    tolerance = expected * tolerance_pct / 100

    diff = abs(total_exptime - expected)

    if diff <= tolerance:
        results.append(check_result(
            "hst_total_integration",
            "PASS",
            expected=f"{expected}s ±{tolerance_pct}%",
            actual=f"{total_exptime:.1f}s from {flc_count} {source_name} FLC files"
        ))
    else:
        results.append(check_result(
            "hst_total_integration",
            "WARN",
            expected=f"{expected}s ±{tolerance_pct}%",
            actual=f"{total_exptime:.1f}s from {flc_count} {source_name} FLC ({diff:.1f}s difference)"
        ))

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_markdown_report(results: list[dict], config_path: Path, data_path: Path) -> str:
    """
    Generate a human-readable Markdown report summarizing validation results.
    
    Builds a Markdown document containing generation timestamp, config and data path references, a summary of pass/warn/fail/info counts, a results table (name, status, expected, actual, detail), and explanatory sections on data sources and validation criteria.
    
    Parameters:
        results (list[dict]): Sequence of validation check dictionaries produced by the validation functions.
        config_path (Path): Path to the validation configuration file (used only for inclusion in the report).
        data_path (Path): Root path of the validated data (used only for inclusion in the report).
    
    Returns:
        str: Complete Markdown document as a single string.
    """
    pass_count = sum(1 for r in results if r['status'] == 'PASS')
    warn_count = sum(1 for r in results if r['status'] == 'WARN')
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')
    info_count = sum(1 for r in results if r['status'] == 'INFO')

    status_icons = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'INFO': 'ℹ️'}

    lines = [
        "# Phase 01 Data Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Config:** `{config_path.name}`",
        f"**Data Path:** `{data_path}`",
        "",
        "## Summary",
        "",
        f"✅ {pass_count} passed | ⚠️ {warn_count} warnings | ❌ {fail_count} failed | ℹ️ {info_count} info",
        "",
        "## Validation Results",
        "",
        "| Check | Status | Expected | Actual | Detail |",
        "|-------|--------|----------|--------|--------|",
    ]

    for r in results:
        icon = status_icons.get(r['status'], '?')
        expected = str(r.get('expected', '-'))
        actual = str(r.get('actual', '-'))
        detail = r.get('detail', '-')

        # Escape pipe characters for Markdown table
        expected = expected.replace('|', '\\|')
        actual = actual.replace('|', '\\|')
        detail = detail.replace('|', '\\|')

        lines.append(f"| {r['name']} | {icon} {r['status']} | {expected} | {actual} | {detail} |")

    lines.extend([
        "",
        "## Data Sources",
        "",
        "### HST Products",
        "",
        "MAST delivers two versions of HST FLC (flat-fielded, CTE-corrected) products:",
        "",
        "| Type | Pattern | Description |",
        "|------|---------|-------------|",
        "| Original | `if3x*_flc.fits` | Original pipeline calibration |",
        "| HAP | `hst_17301*_flc.fits` | Hubble Advanced Products reprocessing |",
        "",
        "HAP products use current best calibration files and improved CTE correction models.",
        "This validation uses **HAP FLC as primary** for integration time calculation,",
        "with original products retained for comparison analysis.",
        "",
        "## Validation Criteria",
        "",
        "This validation compares acquired data against expected values from:",
        "",
        "- van Dokkum et al. (2025) discovery paper",
        "- MAST program metadata for GO-17301 (HST) and GO-3149 (JWST)",
        "",
        "Checks verify file integrity, inventory completeness, provenance, instrument configuration,",
        "pointing accuracy, and integration time totals as part of independent validation methodology.",
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main entry point for data validation.

    Parses arguments, runs all validation checks, and outputs results in
    both JSON and Markdown formats.
    """
    parser = argparse.ArgumentParser(
        description="Validate RBH-1 HST/JWST data acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("data_path", type=Path,
                        help="Root path containing hst/ and jwst/ subdirectories")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "validation_config.yaml",
                        help="Path to validation config YAML (default: ./validation_config.yaml)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results (default: same as config)")

    args = parser.parse_args()

    # ---- Validate input paths ----
    if not args.data_path.exists():
        print(f"ERROR: Data path does not exist: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    hst_path = args.data_path / "hst"
    jwst_path = args.data_path / "jwst"

    if not hst_path.exists() or not jwst_path.exists():
        print(f"ERROR: Expected hst/ and jwst/ subdirectories in {args.data_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.config.parent

    # ---- Load configuration ----
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # ---- Run validation checks ----
    all_results = []

    print("Validating HST inventory...")
    hst_inv_results, original_flc, hap_flc, hst_files = validate_hst_inventory(hst_path, config)
    all_results.extend(hst_inv_results)

    print("Validating JWST inventory...")
    jwst_inv_results, jwst_files = validate_jwst_inventory(jwst_path, config)
    all_results.extend(jwst_inv_results)

    print("Checking FITS readability...")
    all_results.extend(validate_fits_readable(hst_files + jwst_files))

    print("Validating HST provenance...")
    all_results.extend(validate_provenance(hst_files, config['hst']['program_id'], "HST"))

    print("Validating JWST provenance...")
    all_results.extend(validate_provenance(jwst_files, config['jwst']['program_id'], "JWST"))

    print("Validating HST filters...")
    all_results.extend(validate_hst_filters(hst_files, config))

    print("Validating JWST configuration...")
    all_results.extend(validate_jwst_config(jwst_files, config))

    print("Validating HST pointing...")
    all_results.extend(validate_pointing(hst_files, config, "HST"))

    print("Validating JWST pointing...")
    all_results.extend(validate_pointing(jwst_files, config, "JWST"))

    # ---- Integration time uses primary FLC source only ----
    primary_source = config['hst'].get('primary_flc_source', 'hap')
    if primary_source == 'hap':
        primary_flc = hap_flc
        source_name = "HAP"
    else:
        primary_flc = original_flc
        source_name = "original"

    print(f"Validating HST integration time ({source_name} FLC)...")
    all_results.extend(validate_hst_integration(primary_flc, config, source_name))

    # ---- Compute summary statistics ----
    pass_count = sum(1 for r in all_results if r['status'] == 'PASS')
    warn_count = sum(1 for r in all_results if r['status'] == 'WARN')
    fail_count = sum(1 for r in all_results if r['status'] == 'FAIL')

    # ---- Write JSON results ----
    json_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_file": str(args.config),
        "data_path": str(args.data_path),
        "checks": all_results,
        "summary": {"pass": pass_count, "warn": warn_count, "fail": fail_count}
    }

    json_path = output_dir / "validation_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Wrote: {json_path}")

    # ---- Write Markdown report ----
    md_report = generate_markdown_report(all_results, args.config, args.data_path)
    md_path = output_dir / "validation_report.md"
    with open(md_path, 'w') as f:
        f.write(md_report)
    print(f"Wrote: {md_path}")

    # ---- Print summary ----
    print()
    print(f"Summary: ✅ {pass_count} passed | ⚠️ {warn_count} warnings | ❌ {fail_count} failed")

    # ---- Exit code reflects validation status ----
    if fail_count > 0:
        sys.exit(1)
    elif warn_count > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()