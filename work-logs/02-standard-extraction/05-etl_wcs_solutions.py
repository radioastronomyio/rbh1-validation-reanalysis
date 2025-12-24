#!/usr/bin/env python3
"""
Script Name  : 05-etl_wcs_solutions.py
Description  : ETL to populate wcs_solutions table with spatial WCS and footprints
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-24
Phase        : Phase 02 - Standard Extraction
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Extracts spatial WCS from observations (HST flc/drc, JWST s3d), computes PostGIS
footprint polygons, and links back to obs_id for geometric queries. Skips x1d
(no 2D footprint) and cal (detector-frame WCS). HST FLC multi-chip detectors
are merged into convex hull envelopes to satisfy schema POLYGON constraint.

Usage
-----
    python 05-etl_wcs_solutions.py [--dry-run]

Examples
--------
    python 05-etl_wcs_solutions.py
        Run full ETL, extracting WCS and computing footprints for all files.

    python 05-etl_wcs_solutions.py --dry-run
        Process files and validate without database changes.
"""

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import warnings

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from dotenv import load_dotenv

# Suppress noisy FITS WCS warnings (OBSGEO, DATE fixes are harmless)
# These warnings fire on nearly every file due to minor FITS standard deviations
# that astropy auto-corrects. They don't affect WCS accuracy.
warnings.filterwarnings('ignore', category=FITSFixedWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_FILE = "/opt/global-env/research.env"
DATA_ROOT = Path("/mnt/ai-ml/data")
DEFAULT_DATABASE = "rbh1_validation"

# File types with valid celestial WCS
# SKIP x1d: 1D extracted spectra have no meaningful 2D spatial footprint
# SKIP cal: JWST CAL files have detector-frame WCS (CRVAL=0,0), not celestial
VALID_FILE_TYPES = ('flc', 'drc', 's3d')

# Batch size for database inserts
BATCH_SIZE = 50

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def load_credentials() -> dict:
    """
    Load PostgreSQL connection settings from the environment file and return them as a dictionary.
    
    Reads ENV_FILE with dotenv and returns a mapping containing connection parameters; unspecified environment variables fall back to sensible defaults.
    
    Returns:
        dict: Connection settings with keys:
            - host: hostname or IP address for the database server.
            - port: TCP port number (integer).
            - database: database name.
            - user: username for authentication.
            - password: password for authentication (empty string if not set).
    """
    logger.info(f"Loading credentials from: {ENV_FILE}")
    load_dotenv(ENV_FILE)
    
    return {
        "host": os.getenv("PGSQL01_HOST", "10.25.20.8"),
        "port": int(os.getenv("PGSQL01_PORT", "5432")),
        "database": DEFAULT_DATABASE,
        "user": os.getenv("PGSQL01_ADMIN_USER", "clusteradmin_pg01"),
        "password": os.getenv("PGSQL01_ADMIN_PASSWORD", "")
    }


def create_pipeline_run(conn, stage_name: str) -> uuid.UUID:
    """
    Create and record a new pipeline run entry for the given ETL stage.
    
    Inserts a row into rbh1.pipeline_runs with status "RUNNING" and a generated run name, then commits the transaction.
    
    Returns:
        run_id (uuid.UUID): The generated UUID for the newly created pipeline run.
    """
    run_id = uuid.uuid4()
    run_name = f"etl-wcs-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rbh1.pipeline_runs 
                (run_id, run_name, stage_name, pipeline_version, started_at, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            str(run_id),
            run_name,
            stage_name,
            "0.1.0",
            datetime.now(timezone.utc),
            "RUNNING"
        ))
    conn.commit()
    
    return run_id


def update_pipeline_run(conn, run_id: uuid.UUID, status: str, notes: str | None = None):
    """
    Mark a pipeline run as completed and record its status and optional notes.
    
    This sets the pipeline run's `completed_at` timestamp to the current UTC time, updates `status`
    and `notes` for the specified `run_id`, and commits the change to the database.
    
    Parameters:
        run_id (uuid.UUID): Identifier of the pipeline run to update.
        status (str): New status string to store for the pipeline run.
        notes (str | None): Optional free-form notes to associate with the run.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE rbh1.pipeline_runs 
            SET completed_at = %s, status = %s, notes = %s
            WHERE run_id = %s
        """, (
            datetime.now(timezone.utc),
            status,
            notes,
            str(run_id)
        ))
    conn.commit()


def get_observations_for_wcs(conn) -> list:
    """
    Retrieve observations that are eligible for celestial WCS extraction (file types: 'flc', 'drc', 's3d').
    
    Returns:
        list: A list of dictionaries, each containing:
            - obs_id (str): Observation UUID as a string.
            - filename (str): Stored filename.
            - file_path (str): Filesystem path relative to the data root.
            - file_type (str): File type (one of 'flc', 'drc', 's3d').
            - instrument (str): Instrument name.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT obs_id, filename, file_path, file_type, instrument
            FROM rbh1.observations
            WHERE file_type IN ('flc', 'drc', 's3d')
            ORDER BY instrument, file_type, filename
        """)
        # AI NOTE: file_type filter must stay in sync with VALID_FILE_TYPES constant.
        # If adding new file types for WCS extraction, update both places.
        rows = cur.fetchall()
    
    return [
        {
            "obs_id": str(row[0]),
            "filename": row[1],
            "file_path": row[2],
            "file_type": row[3],
            "instrument": row[4]
        }
        for row in rows
    ]


# =============================================================================
# WCS EXTRACTION
# =============================================================================

def find_sci_extension(hdul: fits.HDUList, file_type: str) -> Optional[int]:
    """
    Locate the index of the SCI extension to use for WCS extraction.
    
    Searches the provided FITS HDUList for an HDU named "SCI" and returns its index. If no "SCI" extension is found, falls back to the primary HDU (index 0) if it contains data, then to extension 1 if it contains data. Returns None when no suitable data extension exists.
    
    Parameters:
        hdul (fits.HDUList): Open FITS HDUList to search.
        file_type (str): File type hint (present for caller context; not used to determine the index).
    
    Returns:
        index (Optional[int]): Extension index to use for WCS extraction, or `None` if none found.
    """
    for i, hdu in enumerate(hdul):
        if hdu.name == 'SCI':
            return i
    
    # Fallback to primary if no SCI extension
    primary_hdu = hdul[0]
    if primary_hdu.data is not None:  # type: ignore[union-attr]
        return 0
    
    # Try extension 1
    if len(hdul) > 1:
        ext1_hdu = hdul[1]
        if ext1_hdu.data is not None:  # type: ignore[union-attr]
            return 1
    
    return None


def get_all_sci_extensions(hdul: fits.HDUList) -> list:
    """
    Return the indices of all HDUList extensions named 'SCI'.
    
    Parameters:
        hdul (astropy.io.fits.HDUList): Open FITS HDUList to search.
    
    Returns:
        list[int]: Indices of extensions with EXTNAME equal to 'SCI'. Empty list if none are found.
    """
    return [i for i, hdu in enumerate(hdul) if hdu.name == 'SCI']


def extract_wcs_params(wcs: WCS) -> dict:
    """
    Extracts the primary celestial WCS parameters needed to store or compare spatial solutions.
    
    If the provided WCS has three axes, the celestial (2D) sub-WCS is used.
    
    Returns:
        dict: Mapping of WCS parameter names to values:
            - crpix1 (float): reference pixel X (1-based FITS convention converted to float).
            - crpix2 (float): reference pixel Y.
            - crval1 (float): reference coordinate value for axis 1 (usually RA, degrees).
            - crval2 (float): reference coordinate value for axis 2 (usually Dec, degrees).
            - cd1_1 (float): CD matrix element (1,1).
            - cd1_2 (float): CD matrix element (1,2).
            - cd2_1 (float): CD matrix element (2,1).
            - cd2_2 (float): CD matrix element (2,2).
            - ctype1 (str): projection type string for axis 1 (e.g., "RA---TAN").
            - ctype2 (str): projection type string for axis 2 (e.g., "DEC--TAN").
    """
    # Get the 2D celestial WCS if this is a 3D cube
    if wcs.naxis == 3:
        celestial_wcs = wcs.celestial
    else:
        celestial_wcs = wcs
    
    # Reference pixel (1-indexed in FITS, astropy uses 0-indexed internally)
    crpix = celestial_wcs.wcs.crpix
    crval = celestial_wcs.wcs.crval
    
    # CD matrix (or compute from CDELT + PC if CD not present)
    if celestial_wcs.wcs.has_cd():
        cd = celestial_wcs.wcs.cd
    elif celestial_wcs.wcs.has_pc():
        # CD = CDELT * PC
        cdelt = celestial_wcs.wcs.cdelt
        pc = celestial_wcs.wcs.pc
        cd = np.outer(cdelt, np.ones(2)) * pc
    else:
        # Fallback: identity with pixel scale guess
        cd = np.array([[1e-5, 0], [0, 1e-5]])
    
    # Projection types
    ctype = celestial_wcs.wcs.ctype
    
    return {
        "crpix1": float(crpix[0]),
        "crpix2": float(crpix[1]),
        "crval1": float(crval[0]),
        "crval2": float(crval[1]),
        "cd1_1": float(cd[0, 0]),
        "cd1_2": float(cd[0, 1]),
        "cd2_1": float(cd[1, 0]),
        "cd2_2": float(cd[1, 1]),
        "ctype1": str(ctype[0]) if len(ctype) > 0 else "RA---TAN",
        "ctype2": str(ctype[1]) if len(ctype) > 1 else "DEC--TAN"
    }


def compute_footprint_polygon(wcs: WCS, naxis1: int, naxis2: int) -> str:
    """
    Compute the image footprint as a closed WKT POLYGON in sky coordinates.
    
    Parameters:
        wcs (WCS): A 2D celestial WCS or a 3D WCS (the celestial sub-WCS will be used).
        naxis1 (int): Image width in pixels (number of columns).
        naxis2 (int): Image height in pixels (number of rows).
    
    Returns:
        wkt (str): WKT `POLYGON` string representing the four image corners in RA/DEC order (closed by repeating the first vertex).
    """
    # Get 2D celestial WCS
    if wcs.naxis == 3:
        celestial_wcs = wcs.celestial
    else:
        celestial_wcs = wcs
    
    # Image corners (0-indexed pixel coordinates)
    # Order: bottom-left, bottom-right, top-right, top-left (counter-clockwise)
    corners_pix = np.array([
        [0, 0],
        [naxis1 - 1, 0],
        [naxis1 - 1, naxis2 - 1],
        [0, naxis2 - 1]
    ])
    
    # Transform to sky coordinates
    corners_sky = celestial_wcs.pixel_to_world_values(corners_pix[:, 0], corners_pix[:, 1])
    ra = corners_sky[0]
    dec = corners_sky[1]
    
    # Build WKT polygon (closed - repeat first point)
    coords = [f"{ra[i]} {dec[i]}" for i in range(4)]
    coords.append(coords[0])  # Close polygon
    
    wkt = f"POLYGON(({', '.join(coords)}))"
    return wkt


def compute_multichip_footprint(hdul: fits.HDUList, sci_indices: list) -> str:
    """
    Compute a convex-hull POLYGON WKT covering all SCI extension corners for multi-chip detectors.
    
    This produces a single POLYGON WKT that is the convex hull of all chip corner sky coordinates (RA, DEC). Using a convex hull avoids multipart geometries for detectors with separated chips.
    
    Parameters:
        hdul (fits.HDUList): Open FITS HDUList containing the SCI extensions.
        sci_indices (list): Sequence of integer indices identifying SCI extensions to include.
    
    Returns:
        str: WKT `POLYGON` string (closed) representing the convex hull of all provided chip corners.
    """
    all_corners_ra = []
    all_corners_dec = []
    
    for idx in sci_indices:
        hdu = hdul[idx]
        # Astropy type stubs don't fully model HDUList indexing
        hdu_header = hdu.header  # type: ignore[union-attr]
        hdu_data = hdu.data  # type: ignore[union-attr]
        wcs = WCS(hdu_header, fobj=hdul, naxis=2)
        
        naxis1 = hdu_header.get('NAXIS1', hdu_data.shape[1] if hdu_data is not None else 4096)
        naxis2 = hdu_header.get('NAXIS2', hdu_data.shape[0] if hdu_data is not None else 2048)
        
        # Image corners
        corners_pix = np.array([
            [0, 0],
            [naxis1 - 1, 0],
            [naxis1 - 1, naxis2 - 1],
            [0, naxis2 - 1]
        ])
        
        corners_sky = wcs.pixel_to_world_values(corners_pix[:, 0], corners_pix[:, 1])
        all_corners_ra.extend(corners_sky[0])
        all_corners_dec.extend(corners_sky[1])
    
    # Compute convex hull
    from scipy.spatial import ConvexHull
    
    points = np.column_stack([all_corners_ra, all_corners_dec])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Build WKT polygon (closed)
    coords = [f"{hull_points[i, 0]} {hull_points[i, 1]}" for i in range(len(hull_points))]
    coords.append(coords[0])  # Close polygon
    
    wkt = f"POLYGON(({', '.join(coords)}))"
    return wkt


def extract_wcs_from_fits(filepath: Path, file_type: str) -> Optional[dict]:
    """
    Extract a validated celestial WCS solution and footprint from a FITS file.
    
    Parameters:
        filepath (Path): Path to the FITS file to read.
        file_type (str): File type hint used to choose WCS handling (typical values: 'flc', 'drc', 's3d').
    
    Returns:
        dict | None: A dictionary containing WCS metadata and a footprint suitable for insertion into wcs_solutions, or `None` if extraction or validation failed. The dictionary contains:
            - crpix1 (float), crpix2 (float)
            - crval1 (float), crval2 (float)
            - cd1_1 (float), cd1_2 (float), cd2_1 (float), cd2_2 (float)
            - ctype1 (str), ctype2 (str)
            - footprint_wkt (str): WKT POLYGON in longitude/latitude order (suitable for ST_GeomFromText with SRID 4326)
            - has_spectral_axis (bool)
            - wavelength_unit (str | None)
            - naxis1 (int), naxis2 (int)
            - naxis3 (int | None)
    """
    with fits.open(filepath) as hdul:
        # Find SCI extension(s)
        sci_indices = get_all_sci_extensions(hdul)
        
        if not sci_indices:
            # Try primary or first extension
            primary_idx = find_sci_extension(hdul, file_type)
            if primary_idx is None:
                logger.warning(f"No valid WCS extension found in {filepath.name}")
                return None
            sci_indices = [primary_idx]
        
        # Use first SCI extension for WCS parameters
        primary_idx = sci_indices[0]
        hdu = hdul[primary_idx]
        # Astropy type stubs don't fully model HDUList indexing
        hdu_header = hdu.header  # type: ignore[union-attr]
        hdu_data = hdu.data  # type: ignore[union-attr]
        
        # Build WCS object - pass full HDUList for lookup table distortion support
        try:
            if file_type == 's3d':
                # 3D cube - use all axes
                wcs = WCS(hdu_header, fobj=hdul)
                has_spectral = True
                naxis3 = hdu_header.get('NAXIS3', hdu_data.shape[0] if hdu_data is not None else None)
            else:
                # 2D image - force 2 axes, pass HDUList for distortion lookup tables
                wcs = WCS(hdu_header, fobj=hdul, naxis=2)
                has_spectral = False
                naxis3 = None
        except Exception as e:
            logger.warning(f"Failed to parse WCS for {filepath.name}: {e}")
            return None
        
        # Get dimensions
        naxis1 = hdu_header.get('NAXIS1', hdu_data.shape[-1] if hdu_data is not None else None)
        naxis2 = hdu_header.get('NAXIS2', hdu_data.shape[-2] if hdu_data is not None else None)
        
        if naxis1 is None or naxis2 is None:
            logger.warning(f"Missing NAXIS dimensions in {filepath.name}")
            return None
        
        # Extract WCS parameters
        try:
            wcs_params = extract_wcs_params(wcs)
        except Exception as e:
            logger.warning(f"Failed to extract WCS params for {filepath.name}: {e}")
            return None
        
        # =================================================================
        # GATE CHECKS: Validate celestial WCS semantics
        # AI NOTE: These gates reject files with technically-valid WCS that
        # would produce meaningless or corrupt footprints. Each gate logs
        # why it rejected a file. If a gate rejects unexpectedly, check
        # whether the file type should be in VALID_FILE_TYPES at all.
        # =================================================================
        
        # Gate 1: CTYPE sanity - require celestial axes
        ctype1, ctype2 = wcs_params['ctype1'], wcs_params['ctype2']
        if not (ctype1.startswith('RA--') or ctype1.startswith('RA---')):
            logger.warning(f"Skipping {filepath.name}: CTYPE1 '{ctype1}' is not celestial RA")
            return None
        if not (ctype2.startswith('DEC-') or ctype2.startswith('DEC--')):
            logger.warning(f"Skipping {filepath.name}: CTYPE2 '{ctype2}' is not celestial DEC")
            return None
        
        # Gate 2: CRVAL present, finite, and not both exactly zero
        crval1, crval2 = wcs_params['crval1'], wcs_params['crval2']
        if not (np.isfinite(crval1) and np.isfinite(crval2)):
            logger.warning(f"Skipping {filepath.name}: CRVAL not finite ({crval1}, {crval2})")
            return None
        if abs(crval1) < 0.01 and abs(crval2) < 0.01:
            logger.warning(f"Skipping {filepath.name}: CRVAL near origin (0,0) - likely detector-frame")
            return None
        
        # Compute footprint
        try:
            if file_type == 'flc' and len(sci_indices) > 1:
                # Multi-chip: compute convex hull
                footprint_wkt = compute_multichip_footprint(hdul, sci_indices)
            else:
                # Single chip/cube: compute from corners
                footprint_wkt = compute_footprint_polygon(wcs, naxis1, naxis2)
        except Exception as e:
            logger.warning(f"Failed to compute footprint for {filepath.name}: {e}")
            return None
        
        # Gate 3: Corner transform sanity - validate lat/lon bounds
        # Parse the WKT to extract coordinates for validation
        try:
            # Extract coordinates from WKT: POLYGON((ra1 dec1, ra2 dec2, ...))
            coords_str = footprint_wkt.split('((')[1].split('))')[0]
            coord_pairs = [c.strip().split() for c in coords_str.split(',')]
            ras = [float(p[0]) for p in coord_pairs]
            decs = [float(p[1]) for p in coord_pairs]
            
            # Check latitude bounds
            if min(decs) < -90 or max(decs) > 90:
                logger.warning(f"Skipping {filepath.name}: Footprint Dec out of bounds [{min(decs):.2f}, {max(decs):.2f}]")
                return None
            
            # Check longitude bounds (allow 0-360 wrap)
            if min(ras) < -180 or max(ras) > 360:
                logger.warning(f"Skipping {filepath.name}: Footprint RA out of bounds [{min(ras):.2f}, {max(ras):.2f}]")
                return None
            
            # Gate 4: Span sanity - reject absurd footprints
            # Max reasonable FOV: ~1 degree = 3600 arcsec (HST/JWST are much smaller)
            MAX_SPAN_ARCSEC = 7200  # 2 degrees, generous threshold
            dra_arcsec = (max(ras) - min(ras)) * 3600
            ddec_arcsec = (max(decs) - min(decs)) * 3600
            
            if dra_arcsec > MAX_SPAN_ARCSEC or ddec_arcsec > MAX_SPAN_ARCSEC:
                logger.warning(f"Skipping {filepath.name}: Footprint span too large ({dra_arcsec:.0f}x{ddec_arcsec:.0f} arcsec)")
                return None
                
        except Exception as e:
            logger.warning(f"Failed footprint validation for {filepath.name}: {e}")
            return None
        
        return {
            **wcs_params,
            "footprint_wkt": footprint_wkt,
            "has_spectral_axis": has_spectral,
            "wavelength_unit": "um" if has_spectral else None,
            "naxis1": int(naxis1),
            "naxis2": int(naxis2),
            "naxis3": int(naxis3) if naxis3 else None
        }


# =============================================================================
# DATABASE INSERT
# =============================================================================

def insert_wcs_solutions(conn, records: list) -> tuple[int, int, int]:
    """
    Insert or update multiple WCS solution records for observations in rbh1.wcs_solutions.
    
    Parameters:
        records (list): List of dictionaries, each containing WCS data for a single observation.
            Required keys: "obs_id", "crpix1", "crpix2", "crval1", "crval2",
            "cd1_1", "cd1_2", "cd2_1", "cd2_2", "ctype1", "ctype2",
            "footprint_wkt", "has_spectral_axis", "wavelength_unit",
            "naxis1", "naxis2", "naxis3".
    
    Returns:
        tuple[int, int, int]: (inserted_count, updated_count, error_count)
            inserted_count: number of records inserted,
            updated_count: number of records that conflicted on obs_id and were updated,
            error_count: number of records that failed (always 0 for current implementation).
    
    Notes:
        Commits the transaction; on conflict by obs_id the existing row is updated with the new WCS parameters.
    """
    if not records:
        return 0, 0, 0
    
    # Check for existing records
    obs_ids = [rec["obs_id"] for rec in records]
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT obs_id FROM rbh1.wcs_solutions
            WHERE obs_id = ANY(%s::uuid[])
        """, (obs_ids,))
        existing = {str(row[0]) for row in cur.fetchall()}
    
    insert_sql = """
        INSERT INTO rbh1.wcs_solutions (
            wcs_id, obs_id, crpix1, crpix2, crval1, crval2,
            cd1_1, cd1_2, cd2_1, cd2_2, ctype1, ctype2,
            footprint, has_spectral_axis, wavelength_unit,
            naxis1, naxis2, naxis3
        ) VALUES %s
        ON CONFLICT (obs_id) DO UPDATE SET
            crpix1 = EXCLUDED.crpix1,
            crpix2 = EXCLUDED.crpix2,
            crval1 = EXCLUDED.crval1,
            crval2 = EXCLUDED.crval2,
            cd1_1 = EXCLUDED.cd1_1,
            cd1_2 = EXCLUDED.cd1_2,
            cd2_1 = EXCLUDED.cd2_1,
            cd2_2 = EXCLUDED.cd2_2,
            ctype1 = EXCLUDED.ctype1,
            ctype2 = EXCLUDED.ctype2,
            footprint = EXCLUDED.footprint,
            has_spectral_axis = EXCLUDED.has_spectral_axis,
            wavelength_unit = EXCLUDED.wavelength_unit,
            naxis1 = EXCLUDED.naxis1,
            naxis2 = EXCLUDED.naxis2,
            naxis3 = EXCLUDED.naxis3
    """
    
    # Prepare values with PostGIS geometry conversion
    values = []
    for rec in records:
        values.append((
            str(uuid.uuid4()),
            rec["obs_id"],
            rec["crpix1"],
            rec["crpix2"],
            rec["crval1"],
            rec["crval2"],
            rec["cd1_1"],
            rec["cd1_2"],
            rec["cd2_1"],
            rec["cd2_2"],
            rec["ctype1"],
            rec["ctype2"],
            rec["footprint_wkt"],  # Will be converted via template
            rec["has_spectral_axis"],
            rec["wavelength_unit"],
            rec["naxis1"],
            rec["naxis2"],
            rec["naxis3"]
        ))
    
    # Custom template for PostGIS geometry conversion
    # AI NOTE: ST_GeomFromText(%s, 4326) converts WKT string to PostGIS geometry
    # with SRID 4326 (WGS84). The template must match the VALUES tuple order exactly.
    # If adding/removing columns, update both insert_sql and this template.
    template = """(
        %s, %s::uuid, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        ST_GeomFromText(%s, 4326), %s, %s,
        %s, %s, %s
    )"""
    
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values, template=template, page_size=BATCH_SIZE)
    
    conn.commit()
    
    inserted = len(records) - len(existing)
    updated = len(existing)
    
    return inserted, updated, 0


# =============================================================================
# MAIN ETL PIPELINE
# =============================================================================

def run_etl(dry_run: bool = False):
    """
    Run the ETL pipeline that extracts WCS parameters and footprints from observation FITS files and upserts them into the database.
    
    Parameters:
        dry_run (bool): If True, perform a validation run without making any changes to the database.
    
    Returns:
        bool: `True` if the ETL completed with no errors, `False` otherwise.
    """
    
    logger.info("=" * 60)
    logger.info("RBH-1 WCS SOLUTIONS ETL")
    logger.info("=" * 60)
    
    # Connect to database
    creds = load_credentials()
    logger.info(f"Connecting to: {creds['host']}:{creds['port']}/{creds['database']}")
    
    conn = psycopg2.connect(**creds)
    
    if dry_run:
        logger.info("DRY RUN MODE - no database changes will be made")
        run_id = uuid.uuid4()
    else:
        run_id = create_pipeline_run(conn, "02-etl-wcs-solutions")
        logger.info(f"Pipeline run ID: {run_id}")
    
    # Get observations to process
    observations = get_observations_for_wcs(conn)
    logger.info(f"Found {len(observations)} observations for WCS extraction")
    
    # Process files
    records = []
    errors = []
    skipped = 0
    
    for i, obs in enumerate(observations, 1):
        filename = obs["filename"]
        file_type = obs["file_type"]
        # instrument available in obs dict if needed for future filtering
        
        # Build filepath from DB file_path (source of truth)
        filepath = DATA_ROOT / obs["file_path"]
        
        logger.info(f"[{i}/{len(observations)}] Processing: {filename}")
        
        if not filepath.exists():
            error_msg = f"File not found: {filepath}"
            logger.warning(f"  {error_msg}")
            errors.append({"filename": filename, "error": error_msg})
            continue
        
        try:
            wcs_data = extract_wcs_from_fits(filepath, file_type)
            
            if wcs_data is None:
                skipped += 1
                continue
            
            # Add obs_id linkage
            wcs_data["obs_id"] = obs["obs_id"]
            records.append(wcs_data)
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"  Error: {error_msg}")
            errors.append({"filename": filename, "error": error_msg})
            continue
    
    # Insert to database
    if dry_run:
        logger.info(f"DRY RUN: Would insert {len(records)} WCS solutions")
        inserted, updated = len(records), 0
    else:
        logger.info(f"Inserting {len(records)} WCS solutions to database...")
        inserted, updated, _ = insert_wcs_solutions(conn, records)
    
    # Update pipeline run status
    status = "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS"
    notes = f"Inserted: {inserted}, Updated: {updated}, Skipped: {skipped}, Errors: {len(errors)}"
    
    if not dry_run:
        update_pipeline_run(conn, run_id, status, notes)
    
    conn.close()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("ETL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total observations queried: {len(observations)}")
    logger.info(f"Successfully processed:     {len(records)}")
    logger.info(f"New WCS records inserted:   {inserted}")
    logger.info(f"Existing records updated:   {updated}")
    logger.info(f"Skipped (no valid WCS):     {skipped}")
    logger.info(f"Errors:                     {len(errors)}")
    
    if errors:
        logger.info("-" * 60)
        logger.info("ERRORS:")
        for err in errors[:10]:
            logger.info(f"  {err['filename']}: {err['error']}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")
    
    logger.info("=" * 60)
    
    return len(errors) == 0


def main():
    """
    Run the ETL pipeline for extracting WCS solutions from FITS files and exit with an appropriate status code.
    
    Parses the command-line option `--dry-run` to run the pipeline without writing to the database, invokes the ETL, and exits with status code 0 on success or 1 on failure.
    """
    parser = argparse.ArgumentParser(description="ETL WCS solutions from FITS files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process files without database changes")
    args = parser.parse_args()
    
    success = run_etl(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()