#!/usr/bin/env python3
"""
Script Name  : 01-fits_introspection.py
Description  : Deep introspection of FITS files to build data dictionary for schema design
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-23
Phase        : Phase 02 - Standard Extraction
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Extracts comprehensive metadata from all FITS file types in the RBH-1 dataset
(HST FLC/DRC, JWST CAL/S3D/X1D) including HDU inventory, header keywords by
category, data characteristics, and derived metadata like wavelength/spatial
coverage. Output drives PostgreSQL schema design for the Analysis-Ready Dataset.

Usage
-----
    python 01-fits_introspection.py <data_root> [--output-dir <dir>]

Examples
--------
    python 01-fits_introspection.py /mnt/ai-ml/data --output-dir ./
        Run on proj-gpu01 where data resides; outputs JSON and MD summaries.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


# =============================================================================
# CONFIGURATION
# =============================================================================

# File type patterns and their semantic categories
FILE_TYPES = {
    'hst_original_flc': {
        'pattern': 'if3x*_flc.fits',
        'subdir': 'hst',
        'description': 'HST WFC3/UVIS flat-fielded CTE-corrected (original pipeline)',
        'sample_count': 2,
    },
    'hst_hap_flc': {
        'pattern': 'hst_*_flc.fits',
        'subdir': 'hst',
        'description': 'HST WFC3/UVIS flat-fielded CTE-corrected (HAP reprocessed)',
        'sample_count': 2,
    },
    'hst_original_drc': {
        'pattern': 'if3x*_drc.fits',
        'subdir': 'hst',
        'description': 'HST WFC3/UVIS drizzle-combined (original pipeline)',
        'sample_count': 2,
    },
    'hst_hap_drc': {
        'pattern': 'hst_17301*_drc.fits',
        'subdir': 'hst',
        'description': 'HST WFC3/UVIS drizzle-combined (HAP reprocessed)',
        'sample_count': 2,
    },
    'hst_skycell': {
        'pattern': 'hst_skycell*_drc.fits',
        'subdir': 'hst',
        'description': 'HST HAP skycell mosaic',
        'sample_count': 2,
    },
    'jwst_cal': {
        'pattern': '*_cal.fits',
        'subdir': 'jwst',
        'description': 'JWST NIRSpec IFU calibrated 2D spectra',
        'sample_count': 2,
    },
    'jwst_s3d': {
        'pattern': '*_s3d.fits',
        'subdir': 'jwst',
        'description': 'JWST NIRSpec IFU reconstructed 3D spectral cube',
        'sample_count': 2,
    },
    'jwst_x1d': {
        'pattern': '*_x1d.fits',
        'subdir': 'jwst',
        'description': 'JWST NIRSpec IFU extracted 1D spectrum',
        'sample_count': 2,
    },
}

# Header keyword categories for organization
# Patterns ending with '*' are prefix matches (e.g., 'TUNIT*' matches TUNIT1, TUNIT2, etc.)
# This enables capturing indexed keywords without enumerating all possible indices.
KEYWORD_CATEGORIES = {
    'target': [
        'TARGNAME', 'TARGPROP', 'RA_TARG', 'DEC_TARG', 'TARG_RA', 'TARG_DEC',
        'PROPOSID', 'PROGRAM', 'PROGID', 'PI_NAME', 'VISITYPE',
    ],
    'observation': [
        'DATE-OBS', 'TIME-OBS', 'DATE-BEG', 'DATE-END', 'MJD-OBS', 'MJD-BEG',
        'EXPTIME', 'TEXPTIME', 'EFFEXPTM', 'EXPSTART', 'EXPEND',
        'OBSTYPE', 'OBSMODE', 'OPMODE',
    ],
    'instrument': [
        'TELESCOP', 'INSTRUME', 'DETECTOR', 'CHANNEL',
        'FILTER', 'FILTER1', 'FILTER2', 'FILTNAM1', 'FILTNAM2',
        'GRATING', 'DISPERSR', 'GWA_POS', 'GWA_TILT', 'GWA_XTIL', 'GWA_YTIL',
        'APERTURE', 'APERNAME', 'FXD_SLIT', 'SLITNAME',
        'READPATT', 'NGROUPS', 'NINTS', 'NFRAMES',
    ],
    'calibration': [
        'PHOTFLAM', 'PHOTPLAM', 'PHOTBW', 'PHOTZPT', 'PHOTMODE',
        'BUNIT', 'BUNITS', 'TUNIT*',
        'FLATFILE', 'DARKFILE', 'BIASFILE', 'PHTFLAM1', 'PHTFLAM2',
        'CAL_VER', 'CRDS_VER', 'CRDS_CTX',
        'S_FLAT', 'S_DARK', 'S_PHOTOM', 'S_WCS',
    ],
    'wcs_spatial': [
        'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
        'CDELT1', 'CDELT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
        'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
        'ORIENTAT', 'PA_V3', 'ROLL_REF',
        'RADESYS', 'EQUINOX', 'WCSAXES',
    ],
    'wcs_spectral': [
        'CRPIX3', 'CRVAL3', 'CDELT3', 'CD3_3',
        'CTYPE3', 'CUNIT3', 'SPECSYS', 'VELOSYS',
        'WAVSTART', 'WAVEND', 'SPORDER',
    ],
    'data_quality': [
        'EXPFLAG', 'QUALCOM*', 'DQ_INIT', 'SATURATE',
        'NGOODPIX', 'MEANDARK', 'MEANBLEV',
        'SDQFLAGS', 'NREJECTS',
    ],
    'detector': [
        'BINAXIS1', 'BINAXIS2', 'CCDAMP', 'CCDGAIN', 'CCDCHIP',
        'READNOIS', 'ATODGAIN', 'ATODGN*',
        'SUBARRAY', 'SUBSTRT1', 'SUBSTRT2', 'SUBSIZE1', 'SUBSIZE2',
        'NRS_NORM', 'NRS_FULL',
    ],
}


# =============================================================================
# HEADER EXTRACTION
# =============================================================================


def categorize_keyword(keyword: str) -> str:
    """
    Map a FITS header keyword to a semantic category.
    
    Matches the given keyword (case-insensitive) against patterns in KEYWORD_CATEGORIES:
    - exact matches or patterns ending with `*` (prefix match) are supported.
    - If no configured pattern matches, applies simple prefix-based fallbacks (e.g., `PA_/V2_/V3_` → `wcs_spatial`, `S_/R_` or `D*...DARK` → `calibration`).
    
    Parameters:
        keyword (str): The header keyword to categorize (case-insensitive).
    
    Returns:
        str: The category name from KEYWORD_CATEGORIES or `"other"` when no category applies.
    """
    keyword_upper = keyword.upper()
    
    for category, patterns in KEYWORD_CATEGORIES.items():
        for pattern in patterns:
            if pattern.endswith('*'):
                if keyword_upper.startswith(pattern[:-1]):
                    return category
            elif keyword_upper == pattern:
                return category
    
    # Fallback categories based on common prefixes
    if keyword_upper.startswith('PA_') or keyword_upper.startswith('V2_') or keyword_upper.startswith('V3_'):
        return 'wcs_spatial'
    if keyword_upper.startswith('S_') or keyword_upper.startswith('R_'):
        return 'calibration'
    if keyword_upper.startswith('D') and 'DARK' in keyword_upper:
        return 'calibration'
    
    return 'other'


def extract_header(header: fits.Header) -> dict:
    """
    Extracts and categorizes informational keywords from a FITS header and returns per-key metadata plus simple statistics.
    
    Parameters:
        header (astropy.io.fits.Header): FITS header to inspect.
    
    Returns:
        dict: {
            "keywords": {
                "<category>": [
                    {
                        "keyword": "<KEYNAME>",        # header keyword name
                        "value": <value_or_string>,    # header value (NaNs/inf converted to string)
                        "comment": "<comment>",        # comment associated with the keyword or empty string
                        "dtype": "<py_type_name>"      # inferred Python type name (e.g., "int", "float", "str", "bool")
                    },
                    ...
                ],
                ...
            },
            "stats": {
                "total_keywords": int,            # total number of extracted keywords across all categories
                "by_category": { "<category>": int, ... }  # counts per category
            }
        }
    """
    categorized: dict[str, list[dict]] = defaultdict(list)
    
    for keyword in header.keys():
        if keyword in ('', 'HISTORY', 'COMMENT', 'HIERARCH'):
            continue
        
        try:
            value = header[keyword]
            # header.comments supports dict-style access; use try/except for robustness
            try:
                comment = str(header.comments[keyword])
            except (KeyError, TypeError):
                comment = ''
            
            # Determine Python type
            if isinstance(value, bool):
                dtype = 'bool'
            elif isinstance(value, int):
                dtype = 'int'
            elif isinstance(value, float):
                dtype = 'float'
            elif isinstance(value, str):
                dtype = 'str'
            else:
                dtype = type(value).__name__
            
            category = categorize_keyword(keyword)
            
            categorized[category].append({
                'keyword': keyword,
                'value': value if not isinstance(value, float) or np.isfinite(value) else str(value),
                'comment': comment,
                'dtype': dtype,
            })
        except Exception:
            pass
    
    # Sort keywords within each category
    for cat in categorized:
        categorized[cat] = sorted(categorized[cat], key=lambda x: x['keyword'])
    
    stats = {
        'total_keywords': sum(len(v) for v in categorized.values()),
        'by_category': {k: len(v) for k, v in categorized.items()},
    }
    
    return {'keywords': dict(categorized), 'stats': stats}


# =============================================================================
# DATA ARRAY ANALYSIS
# =============================================================================


def analyze_image_data(data: np.ndarray) -> dict:
    """
    Produce descriptive metadata and finite-value statistics for a 2D image or 3D spectral cube.
    
    Returns:
        dict: Analysis results. On success, the dictionary includes:
            - shape (list[int]): array shape.
            - ndim (int): number of dimensions.
            - dtype (str): numpy dtype string.
            - size_bytes (int): number of bytes used by the array.
            - interpretation (str): 'image' for 2D or 'spectral_cube' for 3D.
            - For 3D cubes: n_wavelength, n_spatial_y, n_spatial_x (ints).
            - For 2D images: n_rows, n_cols (ints).
            - n_finite (int): count of finite values.
            - n_nan (int): count of non-finite (NaN/Inf) values.
            - nan_fraction (float): fraction of non-finite values (0.0–1.0).
            - statistics (dict, optional): present if at least one finite value exists; contains
              min, max, mean, median, std, percentile_01, percentile_99 (all floats).
        If `data` is None, returns {'error': 'No data'}.
    """
    if data is None:
        return {'error': 'No data'}
    
    result: dict = {
        'shape': list(data.shape),
        'ndim': data.ndim,
        'dtype': str(data.dtype),
        'size_bytes': int(data.nbytes),
    }
    
    # Handle multi-dimensional data
    if data.ndim == 3:
        result['interpretation'] = 'spectral_cube'
        result['n_wavelength'] = data.shape[0]
        result['n_spatial_y'] = data.shape[1]
        result['n_spatial_x'] = data.shape[2]
    elif data.ndim == 2:
        result['interpretation'] = 'image'
        result['n_rows'] = data.shape[0]
        result['n_cols'] = data.shape[1]
    
    # Compute statistics on finite values
    finite_mask = np.isfinite(data)
    n_finite = int(np.sum(finite_mask))
    n_total = data.size
    
    result['n_finite'] = n_finite
    result['n_nan'] = int(n_total - n_finite)
    result['nan_fraction'] = float((n_total - n_finite) / n_total) if n_total > 0 else 0.0
    
    if n_finite > 0:
        finite_data = data[finite_mask]
        result['statistics'] = {
            'min': float(np.min(finite_data)),
            'max': float(np.max(finite_data)),
            'mean': float(np.mean(finite_data)),
            'median': float(np.median(finite_data)),
            'std': float(np.std(finite_data)),
            'percentile_01': float(np.percentile(finite_data, 1)),
            'percentile_99': float(np.percentile(finite_data, 99)),
        }
    
    return result


def analyze_table_data(table_hdu: fits.BinTableHDU) -> dict:
    """
    Produce metadata and basic numeric statistics for columns in a FITS binary table HDU.
    
    Parameters:
        table_hdu (astropy.io.fits.BinTableHDU): Binary table HDU to analyze.
    
    Returns:
        dict: Summary with keys:
            - n_rows (int): Number of rows in the table.
            - n_columns (int): Number of defined columns.
            - columns (list): Per-column metadata objects. Each object contains:
                - name (str): Column name.
                - format (str): FITS column format string.
                - dtype (str|None): NumPy dtype derived from column data when available.
                - unit (str, optional): Column unit if present.
                - dim (str, optional): Column dimension descriptor if present.
                - null_value (any, optional): Column null value if defined.
                - sample_stats (dict, optional): Basic numeric stats for finite sampled values with keys:
                    - min (float): Minimum finite value.
                    - max (float): Maximum finite value.
                    - n_finite (int): Count of finite values.
    """
    columns = table_hdu.columns
    table_data = table_hdu.data
    
    n_columns = len(columns) if columns is not None else 0
    n_rows = 0
    if table_data is not None:
        n_rows = int(table_data.shape[0])
    
    result: dict = {
        'n_rows': n_rows,
        'n_columns': n_columns,
        'columns': [],
    }
    
    if columns is None:
        return result
    
    for col in columns:
        col_name = str(col.name)
        col_info: dict = {
            'name': col_name,
            'format': str(col.format),
            'dtype': None,
        }
        
        # Get dtype from actual data if available
        if table_data is not None:
            col_data = np.asarray(table_data[col_name])
            col_info['dtype'] = str(col_data.dtype)
        
        if col.unit:
            col_info['unit'] = str(col.unit)
        if col.dim:
            col_info['dim'] = col.dim
        if col.null is not None:
            col_info['null_value'] = col.null
        
        # Sample data statistics for numeric columns
        if table_data is not None and len(table_data) > 0:
            try:
                col_data = np.asarray(table_data[col_name])
                if np.issubdtype(col_data.dtype, np.number):
                    if col_data.ndim == 1:
                        finite_mask = np.isfinite(col_data)
                        finite = col_data[finite_mask]
                        if len(finite) > 0:
                            col_info['sample_stats'] = {
                                'min': float(np.min(finite)),
                                'max': float(np.max(finite)),
                                'n_finite': int(len(finite)),
                            }
            except Exception:
                pass
        
        result['columns'].append(col_info)
    
    return result


# =============================================================================
# WAVELENGTH GRID EXTRACTION
# =============================================================================


def extract_wavelength_grid(hdul: fits.HDUList, sci_header: fits.Header) -> dict | None:
    """
    Extract wavelength coverage and grid information from a FITS HDU list using either WCS keywords (for spectral cubes) or a table wavelength column (for 1D spectra).
    
    Parameters:
        hdul (fits.HDUList): Open FITS HDU list to examine for a wavelength-producing HDU (cube or binary table).
        sci_header (fits.Header): Science header (typically from the science extension or primary) used to read WCS spectral keywords.
    
    Returns:
        dict | None: A dictionary describing the wavelength grid when found, otherwise `None`. Possible keys:
            - source (str): 'wcs_cube' when derived from WCS keywords or 'table_column' when derived from a WAVELENGTH table column.
            - n_channels (int): Number of spectral channels / wavelength samples.
            - wave_min (float): Minimum wavelength in the unit given by `wave_unit`.
            - wave_max (float): Maximum wavelength in the unit given by `wave_unit`.
            - wave_unit (str): Unit of `wave_min`/`wave_max` (e.g., 'um', 'Angstrom', or 'unknown').
            - wave_step (float): Spectral increment from WCS `CDELT3` when `source` is 'wcs_cube'.
            - wave_min_angstrom (float), wave_max_angstrom (float): Angstrom-converted min/max when WCS method is used.
            - wave_step_median (float): Median step between sorted wavelength values (when derived from a table column and multiple samples exist).
            - wave_step_std (float): Standard deviation of steps between sorted wavelength values (when derived from a table column).
    """
    result: dict = {}
    
    # Method 1: WCS keywords for cubes
    if 'CRVAL3' in sci_header:
        crval3 = sci_header.get('CRVAL3')
        cdelt3 = sci_header.get('CDELT3', sci_header.get('CD3_3'))
        crpix3 = sci_header.get('CRPIX3', 1)
        naxis3 = sci_header.get('NAXIS3')
        
        if all(v is not None for v in [crval3, cdelt3, naxis3]):
            # Explicit casts for type checker - these are numeric header values
            crval3_f = float(crval3)  # type: ignore[arg-type]
            cdelt3_f = float(cdelt3)  # type: ignore[arg-type]
            crpix3_f = float(crpix3)  # type: ignore[arg-type]
            naxis3_i = int(naxis3)  # type: ignore[arg-type]
            
            wavelengths = crval3_f + (np.arange(naxis3_i) - crpix3_f + 1) * cdelt3_f
            
            # Detect units (microns vs Angstroms)
            # AI NOTE: The threshold of 100 distinguishes microns from Angstroms based on
            # typical NIR wavelength ranges. JWST NIRSpec G140M covers ~1-2 μm, so CRVAL3
            # values will be ~1. If this script is extended to UV/optical data where
            # wavelengths in Angstroms could be < 100, this heuristic will misclassify.
            unit = str(sci_header.get('CUNIT3', 'um'))
            if crval3_f < 100:  # Likely microns
                unit = 'um'
                wavelengths_angstrom = wavelengths * 1e4
            else:
                unit = 'Angstrom'
                wavelengths_angstrom = wavelengths
            
            result['source'] = 'wcs_cube'
            result['n_channels'] = naxis3_i
            result['wave_min'] = float(wavelengths.min())
            result['wave_max'] = float(wavelengths.max())
            result['wave_step'] = cdelt3_f
            result['wave_unit'] = unit
            result['wave_min_angstrom'] = float(wavelengths_angstrom.min())
            result['wave_max_angstrom'] = float(wavelengths_angstrom.max())
    
    # Method 2: Table column for X1D
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU):
            columns = hdu.columns
            if columns is None:
                continue
            
            # Build list of uppercase column names
            col_names_upper = [str(c.name).upper() for c in columns]
            if 'WAVELENGTH' not in col_names_upper:
                continue
            
            # Find the actual column name (preserving case)
            wave_col: str | None = None
            for c in columns:
                if str(c.name).upper() == 'WAVELENGTH':
                    wave_col = str(c.name)
                    break
            
            if wave_col is None or hdu.data is None or len(hdu.data) == 0:
                continue
            
            wavelengths_data = np.asarray(hdu.data[wave_col])
            if wavelengths_data.ndim > 1:
                wavelengths_data = wavelengths_data.flatten()
            
            finite_mask = np.isfinite(wavelengths_data)
            finite_wave = wavelengths_data[finite_mask]
            if len(finite_wave) > 0:
                result['source'] = 'table_column'
                result['n_channels'] = int(len(finite_wave))
                result['wave_min'] = float(np.min(finite_wave))
                result['wave_max'] = float(np.max(finite_wave))
                
                # Get unit from column definition
                wave_col_def = columns[wave_col]
                result['wave_unit'] = str(wave_col_def.unit) if wave_col_def.unit else 'unknown'
                
                # Compute step if regular grid
                if len(finite_wave) > 1:
                    diffs = np.diff(np.sort(finite_wave))
                    if len(diffs) > 0:
                        result['wave_step_median'] = float(np.median(diffs))
                        result['wave_step_std'] = float(np.std(diffs))
            break
    
    return result if result else None


# =============================================================================
# FILE INTROSPECTION
# =============================================================================


def introspect_fits_file(filepath: Path) -> dict:
    """
    Produce a detailed metadata dictionary describing the contents and structure of a FITS file for data-model and schema design.
    
    Parameters:
        filepath (Path): Path to the FITS file to inspect.
    
    Returns:
        result (dict): Dictionary with keys:
            - filename: base filename
            - filepath: string path
            - file_size_bytes: file size in bytes
            - n_hdus: number of HDUs (when available)
            - hdus: list of per-HDU metadata objects (index, name, type, optional 'header' and 'data' analyses)
            - wavelength_grid: extracted wavelength information or None
            - errors: list of error messages encountered during introspection
    """
    result: dict = {
        'filename': filepath.name,
        'filepath': str(filepath),
        'file_size_bytes': filepath.stat().st_size,
        'hdus': [],
        'wavelength_grid': None,
        'errors': [],
    }
    
    try:
        with fits.open(filepath) as hdul:
            result['n_hdus'] = len(hdul)
            
            sci_header: fits.Header | None = None
            
            for i, hdu in enumerate(hdul):
                hdu_info: dict = {
                    'index': i,
                    'name': hdu.name,
                    'type': type(hdu).__name__,
                }
                
                # Extract header
                if hdu.header:
                    hdu_info['header'] = extract_header(hdu.header)
                    
                    # Capture SCI header for wavelength extraction
                    if hdu.name == 'SCI' or (i == 1 and sci_header is None):
                        sci_header = hdu.header
                
                # Analyze data
                if isinstance(hdu, fits.BinTableHDU):
                    hdu_info['data'] = analyze_table_data(hdu)
                elif hdu.data is not None:
                    hdu_info['data'] = analyze_image_data(hdu.data)
                
                result['hdus'].append(hdu_info)
            
            # Extract wavelength grid for spectral data
            if sci_header is not None:
                wave_info = extract_wavelength_grid(hdul, sci_header)
                if wave_info:
                    result['wavelength_grid'] = wave_info
    
    except Exception as e:
        result['errors'].append(str(e))
    
    return result


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_file_type(data_path: Path, file_type: str, config: dict) -> dict:
    """
    Collect a sample of FITS files for a given file type, run introspection on each sampled file, and return a summary for that type.
    
    Parameters:
        data_path (Path): Root path containing the configured subdirectory for the file type.
        file_type (str): Key identifying the file type being processed.
        config (dict): Configuration for the file type. Required keys:
            - 'subdir' (str): Subdirectory under data_path to search.
            - 'pattern' (str): Glob pattern to match files.
            - 'description' (str): Human-readable description.
            Optional:
            - 'sample_count' (int): Number of files to sample (defaults to 2).
    
    Returns:
        dict: Summary for the file type containing:
            - 'file_type': the provided file_type key.
            - 'description': description from config.
            - 'pattern': glob pattern used.
            - 'total_files': number of matching files found.
            - 'sampled_files': list of per-file introspection dicts (may be empty).
            - 'common_structure': dict of inferred common structure across samples, or None.
            - 'error' (optional): error message when no files were found or another issue occurred.
    """
    subdir = data_path / config['subdir']
    pattern = config['pattern']
    sample_count = config.get('sample_count', 2)
    
    files = sorted(subdir.glob(pattern))
    
    result: dict = {
        'file_type': file_type,
        'description': config['description'],
        'pattern': pattern,
        'total_files': len(files),
        'sampled_files': [],
        'common_structure': None,
    }
    
    if not files:
        result['error'] = f'No files matching {pattern} in {subdir}'
        return result
    
    # Sample files (first N and last if different)
    sample_files = files[:sample_count]
    if len(files) > sample_count:
        sample_files.append(files[-1])
    
    # Introspect each sampled file
    introspections = []
    for f in sample_files:
        print(f"    Introspecting: {f.name}")
        intro = introspect_fits_file(f)
        introspections.append(intro)
        result['sampled_files'].append(intro)
    
    # Identify common structure across samples
    if introspections:
        result['common_structure'] = identify_common_structure(introspections)
    
    return result


def identify_common_structure(introspections: list) -> dict:
    """
    Determine HDUs present in every introspected file and collect reference header keyword names from the first sample.
    
    Parameters:
        introspections (list): Sequence of per-file introspection dictionaries produced by introspect_fits_file.
    
    Returns:
        dict: {
            "common_hdu_names": list of HDU names present in all provided introspections,
            "reference_keywords": mapping from HDU name to a dict of header categories to lists of keyword names taken from the first introspection
        }
    """
    if not introspections:
        return {}
    
    # Common HDU names
    hdu_names_sets = [set(h['name'] for h in intro['hdus']) for intro in introspections]
    common_hdus = set.intersection(*hdu_names_sets) if hdu_names_sets else set()
    
    # Common keywords per HDU (using first file as reference)
    common_keywords: dict = {}
    ref = introspections[0]
    
    for hdu in ref['hdus']:
        if 'header' not in hdu:
            continue
        
        hdu_name = hdu['name']
        keywords_by_cat: dict = {}
        
        for cat, keywords in hdu['header']['keywords'].items():
            keywords_by_cat[cat] = [kw['keyword'] for kw in keywords]
        
        common_keywords[hdu_name] = keywords_by_cat
    
    return {
        'common_hdu_names': list(common_hdus),
        'reference_keywords': common_keywords,
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================


def generate_markdown_summary(data_dict: dict, output_path: Path) -> None:
    """
    Create a human-readable Markdown report describing the FITS data dictionary and write it to disk.
    
    Generates a multi-section Markdown file that includes:
    - document metadata (generation time and source path),
    - a File Type Summary table with counts and example HDUs,
    - per-file-type sections with HDU structure (index, name, type, shape/rows),
    - wavelength coverage summary when available,
    - prioritized header keywords by semantic category (target, observation, instrument, calibration, wcs_spatial, wcs_spectral),
    - binary table column summaries with sample ranges.
    
    Parameters:
        data_dict (dict): The data dictionary produced by the FITS introspection pipeline; expected keys include 'metadata' and 'file_types' with sampled file introspections.
        output_path (Path): Filesystem path where the generated Markdown file will be written.
    """
    lines = [
        "# RBH-1 FITS Data Dictionary",
        "",
        f"**Generated:** {data_dict['metadata']['generated_at']}",
        f"**Data Path:** `{data_dict['metadata']['data_path']}`",
        "",
        "---",
        "",
        "## File Type Summary",
        "",
        "| Type | Description | Count | HDUs |",
        "|------|-------------|-------|------|",
    ]
    
    for ft_name, ft_data in data_dict['file_types'].items():
        if 'error' in ft_data:
            lines.append(f"| {ft_name} | {ft_data['description']} | 0 | ERROR |")
            continue
        
        n_files = ft_data['total_files']
        
        # Get HDU names from first sample
        if ft_data['sampled_files']:
            hdus = [h['name'] for h in ft_data['sampled_files'][0]['hdus']]
            hdu_str = ', '.join(hdus[:5])
            if len(hdus) > 5:
                hdu_str += f' (+{len(hdus)-5})'
        else:
            hdu_str = '-'
        
        lines.append(f"| {ft_name} | {ft_data['description']} | {n_files} | {hdu_str} |")
    
    lines.extend(["", "---", ""])
    
    # Detailed breakdown per file type
    for ft_name, ft_data in data_dict['file_types'].items():
        if 'error' in ft_data or not ft_data['sampled_files']:
            continue
        
        lines.extend([
            f"## {ft_name}",
            "",
            f"**Description:** {ft_data['description']}",
            f"**Pattern:** `{ft_data['pattern']}`",
            f"**Total Files:** {ft_data['total_files']}",
            "",
        ])
        
        # Use first sample as reference
        ref = ft_data['sampled_files'][0]
        
        lines.extend([
            "### HDU Structure",
            "",
            "| Index | Name | Type | Shape/Rows |",
            "|-------|------|------|------------|",
        ])
        
        for hdu in ref['hdus']:
            shape = '-'
            if 'data' in hdu:
                if 'shape' in hdu['data']:
                    shape = str(hdu['data']['shape'])
                elif 'n_rows' in hdu['data']:
                    shape = f"{hdu['data']['n_rows']} rows × {hdu['data']['n_columns']} cols"
            
            lines.append(f"| {hdu['index']} | {hdu['name']} | {hdu['type']} | {shape} |")
        
        lines.append("")
        
        # Wavelength grid if present
        if ref.get('wavelength_grid'):
            wg = ref['wavelength_grid']
            lines.extend([
                "### Wavelength Coverage",
                "",
                f"- **Source:** {wg.get('source', 'unknown')}",
                f"- **Channels:** {wg.get('n_channels', 'N/A')}",
                f"- **Range:** {wg.get('wave_min', 'N/A'):.4f} - {wg.get('wave_max', 'N/A'):.4f} {wg.get('wave_unit', '')}",
            ])
            if 'wave_min_angstrom' in wg:
                lines.append(f"- **Range (Å):** {wg['wave_min_angstrom']:.1f} - {wg['wave_max_angstrom']:.1f} Å")
            lines.append("")
        
        # Key header keywords by category
        lines.extend([
            "### Key Header Keywords",
            "",
        ])
        
        # Find SCI or primary header
        sci_hdu = next((h for h in ref['hdus'] if h['name'] == 'SCI'), None)
        header_source = sci_hdu or ref['hdus'][0]
        
        if 'header' in header_source:
            for cat in ['target', 'observation', 'instrument', 'calibration', 'wcs_spatial', 'wcs_spectral']:
                keywords = header_source['header']['keywords'].get(cat, [])
                if keywords:
                    lines.append(f"**{cat.replace('_', ' ').title()}:**")
                    for kw in keywords[:10]:  # Limit to 10 per category
                        val = kw['value']
                        if isinstance(val, str) and len(val) > 40:
                            val = val[:40] + '...'
                        lines.append(f"- `{kw['keyword']}` = `{val}`")
                    if len(keywords) > 10:
                        lines.append(f"- *... and {len(keywords)-10} more*")
                    lines.append("")
        
        # Table columns for binary tables
        for hdu in ref['hdus']:
            if hdu['type'] == 'BinTableHDU' and 'data' in hdu:
                lines.extend([
                    f"### Table: {hdu['name']}",
                    "",
                    "| Column | Format | Unit | Sample Range |",
                    "|--------|--------|------|--------------|",
                ])
                
                for col in hdu['data']['columns'][:20]:
                    unit = col.get('unit', '-')
                    if 'sample_stats' in col:
                        stats = col['sample_stats']
                        range_str = f"{stats['min']:.4g} - {stats['max']:.4g}"
                    else:
                        range_str = '-'
                    lines.append(f"| {col['name']} | {col['format']} | {unit} | {range_str} |")
                
                if len(hdu['data']['columns']) > 20:
                    lines.append(f"| ... | ... | ... | *{len(hdu['data']['columns'])-20} more columns* |")
                lines.append("")
        
        lines.extend(["---", ""])
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Wrote: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """
    Orchestrates command-line parsing and batch introspection of RBH-1 FITS files, producing a JSON data dictionary and a Markdown summary.
    
    Parses positional and optional CLI arguments, validates the provided data root (expects hst/ and jwst/ subdirectories), updates sampling configuration, runs per-file-type processing to collect introspection results, writes fits_data_dictionary.json to the output directory, and generates fits_data_dictionary.md.
    
    Returns:
        exit_code (int): `0` on successful completion.
    """
    parser = argparse.ArgumentParser(
        description="Deep introspection of RBH-1 FITS files for data dictionary generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("data_path", type=Path,
                        help="Root path containing hst/ and jwst/ subdirectories")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: script directory)")
    parser.add_argument("--sample-count", type=int, default=2,
                        help="Number of files to sample per type (default: 2)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.data_path.exists():
        print(f"ERROR: Data path does not exist: {args.data_path}", file=sys.stderr)
        sys.exit(1)
    
    hst_path = args.data_path / "hst"
    jwst_path = args.data_path / "jwst"
    
    if not hst_path.exists() or not jwst_path.exists():
        print(f"ERROR: Expected hst/ and jwst/ subdirectories in {args.data_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = args.output_dir or Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Update sample counts if specified
    for ft in FILE_TYPES.values():
        ft['sample_count'] = args.sample_count
    
    print("=" * 70)
    print("RBH-1 FITS Data Dictionary Generator")
    print("=" * 70)
    print(f"Data Path: {args.data_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Process all file types
    data_dict: dict = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'data_path': str(args.data_path),
            'sample_count': args.sample_count,
            'script': Path(__file__).name,
        },
        'file_types': {},
    }
    
    for ft_name, ft_config in FILE_TYPES.items():
        print(f"Processing: {ft_name}")
        data_dict['file_types'][ft_name] = process_file_type(args.data_path, ft_name, ft_config)
    
    print()
    
    # Write JSON output
    json_path = output_dir / "fits_data_dictionary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, default=str)
    print(f"Wrote: {json_path}")
    
    # Generate Markdown summary
    md_path = output_dir / "fits_data_dictionary.md"
    generate_markdown_summary(data_dict, md_path)
    
    print()
    print("Done. Use fits_data_dictionary.json for schema design and GDR prompts.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())