#!/usr/bin/env python3
"""
Script Name  : 04-etl_observations.py
Description  : ETL to populate observations table from Phase-01 FITS manifest
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-24
Phase        : Phase 02 - Standard Extraction
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Reads the data manifest CSV from Phase-01, opens each FITS file to extract
full headers, computes SHA-256 checksums, and inserts into rbh1.observations
with UPSERT semantics. Populates Zone 0 of the database schema.

Usage
-----
    python 04-etl_observations.py [--dry-run]

Examples
--------
    python 04-etl_observations.py
        Run full ETL, inserting/updating all observations in database.

    python 04-etl_observations.py --dry-run
        Process files and validate without database changes.
"""

import argparse
import csv
import hashlib
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json, execute_values
from astropy.io import fits
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_FILE = "/opt/global-env/research.env"
DATA_ROOT = Path("/mnt/ai-ml/data")
MANIFEST_PATH = DATA_ROOT / "data_manifest.csv"
DEFAULT_DATABASE = "rbh1_validation"

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
# FILE TYPE CLASSIFICATION
# =============================================================================

def classify_file(filename: str, manifest_instrument: str) -> dict:
    """
    Determine the FITS file classification and instrument type from its filename and a manifest instrument hint.
    
    Returns:
        dict: Mapping with keys:
            - file_type: One of "flc", "drc", "cal", "s3d", "x1d", or "unknown".
            - data_level: "CAL" for calibrated products or "RESAMPLED" for drizzled/combined products.
            - instrument_type: "HST_WFC3" or "JWST_NIRSPEC".
    
    Raises:
        ValueError: If the instrument cannot be determined from the filename or normalized manifest value.
    """
    fn_lower = filename.lower()
    
    # Determine file type from suffix
    # AI NOTE: data_level values must match rbh1.data_level enum in 02-schema_ddl.sql.
    # CAL = calibrated but not resampled, RESAMPLED = drizzled/combined products.
    if fn_lower.endswith("_flc.fits"):
        file_type = "flc"
        data_level = "CAL"
    elif fn_lower.endswith("_drc.fits"):
        file_type = "drc"
        data_level = "RESAMPLED"
    elif fn_lower.endswith("_cal.fits"):
        file_type = "cal"
        data_level = "CAL"
    elif fn_lower.endswith("_s3d.fits"):
        file_type = "s3d"
        data_level = "RESAMPLED"
    elif fn_lower.endswith("_x1d.fits"):
        file_type = "x1d"
        data_level = "CAL"
    else:
        file_type = "unknown"
        data_level = "CAL"
    
    # Infer instrument from filename pattern (robust to manifest drift)
    # HST patterns: if3x*, hst_*, hst_skycell*
    # JWST patterns: jw*
    if fn_lower.startswith(("if3x", "hst_")):
        instrument_type = "HST_WFC3"
    elif fn_lower.startswith("jw"):
        instrument_type = "JWST_NIRSPEC"
    else:
        # Fall back to manifest value with normalization
        norm = manifest_instrument.upper().replace("_", "").replace("-", "")
        if norm in ("HST", "HSTWFC3", "WFC3"):
            instrument_type = "HST_WFC3"
        elif norm in ("JWST", "JWSTNIRSPEC", "NIRSPEC"):
            instrument_type = "JWST_NIRSPEC"
        else:
            raise ValueError(f"Cannot determine instrument for {filename}")
    
    return {
        "file_type": file_type,
        "data_level": data_level,
        "instrument_type": instrument_type
    }

# =============================================================================
# HEADER EXTRACTION
# =============================================================================

def extract_header_value(header: fits.Header, keys: list, default: Any = None) -> Any:
    """
    Return the first non-blank value found in a FITS header for the provided keys, or the default if none are present.
    
    Parameters:
        header (fits.Header): FITS header to search.
        keys (list): Ordered list of header keywords to check.
        default (Any): Value to return if none of the keys yield a usable value.
    
    Returns:
        Any: The first header value that is not None or an empty/whitespace-only string, otherwise `default`.
    """
    for key in keys:
        if key in header:
            val = header[key]
            # Handle FITS undefined values
            if val is None or (isinstance(val, str) and val.strip() == ""):
                continue
            return val
    return default


def sanitize_header_for_json(header: fits.Header) -> dict:
    """
    Convert a FITS header into a JSON-serializable dictionary suitable for storage.
    
    - Skips blank keywords and keys with undefined (`None`) values.
    - Aggregates `COMMENT` and `HISTORY` cards into lists under `_COMMENTS` and `_HISTORY`.
    - Preserves booleans and numeric values; represents NaN and infinite numeric values as their string forms.
    - Trims string values and converts other non-serializable types to strings.
    
    Parameters:
        header (fits.Header): FITS header to sanitize.
    
    Returns:
        dict: JSON-safe mapping of header keywords to values, including optional `_COMMENTS` and `_HISTORY` lists when present.
    """
    result = {}
    comments = []
    history = []
    
    for card in header.cards:
        key = card.keyword
        val = card.value
        
        # Skip blank keywords
        if not key or key.strip() == "":
            continue
        
        # Aggregate COMMENT and HISTORY
        if key == "COMMENT":
            if val and str(val).strip():
                comments.append(str(val).strip())
            continue
        if key == "HISTORY":
            if val and str(val).strip():
                history.append(str(val).strip())
            continue
        
        # Skip undefined values
        if val is None:
            continue
        
        # Convert to JSON-safe types
        if isinstance(val, bool):
            result[key] = val
        elif isinstance(val, (int, float)):
            # NaN/Inf check: val != val is the standard Python idiom for NaN detection
            # that works without importing math and handles numpy floats correctly
            if val != val or val == float('inf') or val == float('-inf'):
                result[key] = str(val)
            else:
                result[key] = val
        elif isinstance(val, str):
            if val.strip():
                result[key] = val.strip()
        else:
            result[key] = str(val)
    
    # Add aggregated fields
    # AI NOTE: Underscore prefix distinguishes these synthetic keys from actual FITS
    # keywords. Downstream code (e.g., header_json queries) may filter on this convention.
    if comments:
        result["_COMMENTS"] = comments
    if history:
        result["_HISTORY"] = history
    
    return result


def extract_fits_metadata(filepath: Path, classification: dict) -> dict:
    """
    Extracts observation metadata from a FITS file for insertion into the observations table.
    
    Parameters:
        filepath (Path): Path to the FITS file to read.
        classification (dict): Classification produced by classify_file; must include "instrument_type" (e.g., "HST_WFC3" or "JWST_NIRSPEC") to select instrument-specific extraction.
    
    Returns:
        dict: Mapping of observation fields:
            - program_id (str): Program identifier (prefixed with "GO-").
            - target_name (str): Target name or "RBH-1" if absent.
            - date_obs (datetime): Observation timestamp with UTC timezone.
            - exp_time (float): Exposure time in seconds (coerced to > 0, defaults to 1.0).
            - filter (str): Filter name (instrument-dependent default applied).
            - disperser (str | None): Disperser/grating for JWST or None for HST.
            - aperture (str | None): Aperture or None if not present.
            - crds_context (str | None): CRDS context/version if present.
            - cal_version (str | None): Calibration/version string if present.
            - header_json (dict): JSON-serializable representation of the FITS header.
    
    Raises:
        ValueError: If DATE-OBS is missing or cannot be parsed.
    """
    with fits.open(filepath) as hdul:
        primary_hdu = hdul[0]
        # Astropy's type stubs don't fully model HDUList indexing; all HDUs have headers
        header: fits.Header = primary_hdu.header  # type: ignore[union-attr]
        
        # Instrument-specific extraction
        is_hst = classification["instrument_type"] == "HST_WFC3"
        
        # Program ID
        if is_hst:
            proposid = extract_header_value(header, ["PROPOSID"], "")
            program_id = f"GO-{proposid}" if proposid else "GO-17301"
        else:
            program = extract_header_value(header, ["PROGRAM"], "3149")
            program_id = f"GO-{program}"
        
        # AI NOTE: We intentionally raise on missing/unparsable DATE-OBS rather than
        # defaulting to current time. A silent fallback would poison provenance by
        # making ingestion time appear to be observation time. Failed files get logged
        # to the errors list and can be manually reviewed.
        date_obs_str = extract_header_value(header, ["DATE-OBS", "DATE_OBS", "DATE"], None)
        time_obs_str = extract_header_value(header, ["TIME-OBS", "TIME_OBS"], "00:00:00")
        
        if not date_obs_str:
            raise ValueError(f"No DATE-OBS found in header for {filepath.name}")
        
        try:
            # Combine date and time
            dt_str = f"{date_obs_str}T{time_obs_str}"
            date_obs = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if date_obs.tzinfo is None:
                date_obs = date_obs.replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(f"Unparsable DATE-OBS '{date_obs_str}' in {filepath.name}: {e}")
        
        # Exposure time
        exp_time = extract_header_value(header, ["EXPTIME", "TEXPTIME", "EFFEXPTM"], 1.0)
        if exp_time is None or exp_time <= 0:
            exp_time = 1.0
        
        # Filter (instrument-specific)
        if is_hst:
            filter_name = extract_header_value(header, ["FILTER"], "UNKNOWN")
        else:
            # JWST NIRSpec: combine grating and filter
            filter_name = extract_header_value(header, ["FILTER"], "F100LP")
        
        # Disperser (JWST only)
        if is_hst:
            disperser = None
        else:
            disperser = extract_header_value(header, ["GRATING", "DISPERSER"], "G140M")
        
        # Aperture
        aperture = extract_header_value(header, ["APERTURE", "APERNAME"], None)
        
        # Target name
        target_name = extract_header_value(header, ["TARGNAME", "TARGETID", "OBJECT"], "RBH-1")
        
        # Calibration provenance
        crds_context = extract_header_value(header, ["CRDS_CTX", "CRDS_VER"], None)
        cal_version = extract_header_value(header, ["CAL_VER", "OPUS_VER"], None)
        
        # Full header as JSON
        header_json = sanitize_header_for_json(header)
        
        return {
            "program_id": program_id,
            "target_name": target_name if target_name else "RBH-1",
            "date_obs": date_obs,
            "exp_time": float(exp_time),
            "filter": filter_name,
            "disperser": disperser,
            "aperture": aperture,
            "crds_context": crds_context,
            "cal_version": cal_version,
            "header_json": header_json
        }

# =============================================================================
# FILE UTILITIES
# =============================================================================

def compute_md5(filepath: Path) -> str:
    """
    Compute the MD5 checksum of a file.
    
    Returns:
        str: Hexadecimal MD5 digest of the file contents.
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_size(filepath: Path) -> int:
    """Get file size in bytes."""
    return filepath.stat().st_size

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def load_credentials() -> dict:
    """
    Load database connection credentials by reading the module ENV_FILE and environment variables.
    
    Returns:
        dict: Mapping with keys:
            host (str): Database host; defaults to "10.25.20.8" if PGSQL01_HOST is not set.
            port (int): Database port; defaults to 5432 if PGSQL01_PORT is not set.
            database (str): Database name from DEFAULT_DATABASE.
            user (str): Database user; defaults to "clusteradmin_pg01" if PGSQL01_ADMIN_USER is not set.
            password (str): Database password; defaults to an empty string if PGSQL01_ADMIN_PASSWORD is not set.
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
    Register a new pipeline run record in rbh1.pipeline_runs and persist it to the database.
    
    Inserts a pipeline run row with the provided stage name, a generated run identifier and name, timestamps the start, and commits the transaction.
    
    @returns uuid.UUID: The newly generated run identifier for the registered pipeline run.
    """
    run_id = uuid.uuid4()
    run_name = f"etl-observations-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    
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
    Mark a pipeline run completed by setting its completion timestamp, status, and optional notes.
    
    This updates the rbh1.pipeline_runs row identified by `run_id`, setting `completed_at` to the current UTC time, `status` to the provided value, and `notes` to the provided text, then commits the transaction.
    
    Parameters:
        run_id (uuid.UUID): Identifier of the pipeline run to update.
        status (str): New status to record for the run (e.g., "COMPLETED", "COMPLETED_WITH_ERRORS").
        notes (str | None): Optional explanatory notes to store with the run.
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


def insert_observations(conn, records: list, run_id: uuid.UUID) -> tuple[int, int, int]:
    """
    Upsert a batch of observation records into rbh1.observations and record which pipeline run ingested them.
    
    Performs a batch INSERT ... ON CONFLICT (filename) DO UPDATE: preserves existing `obs_id`, updates mutable metadata (file_size_bytes, checksum_md5, header_json, crds_context, cal_version, ingested_by_run) and resets `ingested_at` to NOW(). Logs a warning when an existing file's checksum changes.
    
    Parameters:
        conn: Active database connection (cursor is obtained from this connection).
        records (list): List of observation dictionaries. Each record must contain the keys:
            "program_id", "instrument_type", "target_name", "date_obs", "exp_time",
            "filter", "disperser", "aperture", "data_level", "filename", "file_type",
            "file_path", "file_size_bytes", "checksum_md5", "crds_context",
            "cal_version", "header_json".
        run_id (uuid.UUID): UUID of the pipeline run ingesting these records.
    
    Returns:
        tuple[int, int, int]: (inserted_count, updated_count, unchanged_count)
            inserted_count: number of new rows added to the table;
            updated_count: number of records that matched existing filenames (were updated);
            unchanged_count: number of matched records left unchanged (always 0 with current behavior).
    """
    if not records:
        return 0, 0, 0
    
    # First, check for checksum changes on existing files
    filenames = [rec["filename"] for rec in records]
    checksum_map = {rec["filename"]: rec["checksum_md5"] for rec in records}
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT filename, checksum_md5 FROM rbh1.observations
            WHERE filename = ANY(%s)
        """, (filenames,))
        existing = {row[0]: row[1] for row in cur.fetchall()}
    
    # Checksum change detection: if a file was reprocessed (e.g., new CRDS context),
    # the checksum will differ. We log this as a warning for audit purposes but proceed
    # with the update. This is expected behavior, not an error condition.
    for fn, old_checksum in existing.items():
        new_checksum = checksum_map.get(fn)
        if new_checksum and old_checksum != new_checksum:
            logger.warning(f"Checksum changed for {fn}: {old_checksum[:8]}... -> {new_checksum[:8]}... (file reprocessed)")
    
    # AI NOTE: UPSERT semantics — on conflict (filename exists):
    # - obs_id is PRESERVED (stable identifier for foreign keys in other tables)
    # - Mutable fields update: file_size, checksum, header_json, crds_context, cal_version
    # - ingested_at resets to NOW() to track last refresh
    # If adding new columns, decide whether they should update or preserve on conflict.
    insert_sql = """
        INSERT INTO rbh1.observations (
            obs_id, program_id, instrument, target_name, date_obs, exp_time,
            filter, disperser, aperture, data_level, filename, file_type,
            file_path, file_size_bytes, checksum_md5, crds_context, cal_version,
            header_json, ingested_by_run
        ) VALUES %s
        ON CONFLICT (filename) DO UPDATE SET
            file_size_bytes = EXCLUDED.file_size_bytes,
            checksum_md5 = EXCLUDED.checksum_md5,
            header_json = EXCLUDED.header_json,
            crds_context = EXCLUDED.crds_context,
            cal_version = EXCLUDED.cal_version,
            ingested_by_run = EXCLUDED.ingested_by_run,
            ingested_at = NOW()
    """
    
    # Prepare values
    values = []
    for rec in records:
        values.append((
            str(uuid.uuid4()),
            rec["program_id"],
            rec["instrument_type"],
            rec["target_name"],
            rec["date_obs"],
            rec["exp_time"],
            rec["filter"],
            rec["disperser"],
            rec["aperture"],
            rec["data_level"],
            rec["filename"],
            rec["file_type"],
            rec["file_path"],
            rec["file_size_bytes"],
            rec["checksum_md5"],
            rec["crds_context"],
            rec["cal_version"],
            Json(rec["header_json"]),
            str(run_id)
        ))
    
    with conn.cursor() as cur:
        # Get count before
        cur.execute("SELECT COUNT(*) FROM rbh1.observations")
        count_before = cur.fetchone()[0]
        
        # Execute batch upsert
        execute_values(cur, insert_sql, values, page_size=BATCH_SIZE)
        
        # Get count after
        cur.execute("SELECT COUNT(*) FROM rbh1.observations")
        count_after = cur.fetchone()[0]
    
    conn.commit()
    
    inserted = count_after - count_before
    updated = len(existing)  # Files that existed before
    unchanged = 0  # We always update if exists, so this is 0
    
    return inserted, updated, unchanged

# =============================================================================
# MAIN ETL PIPELINE
# =============================================================================

def run_etl(dry_run: bool = False):
    """
    Orchestrates the end-to-end ETL from the FITS manifest into the observations database table and returns whether processing completed without errors.
    
    Performs: manifest loading, per-file classification and FITS metadata extraction, file checksum and size calculation, and batch upsert into the observations table. Supports a dry-run mode that performs all processing without making database changes and records a synthetic pipeline run.
    
    Parameters:
        dry_run (bool): If True, do not write to the database or create a persistent pipeline run; processing is simulated and a synthetic run ID is used.
    
    Returns:
        bool: `true` if no file-level errors were encountered during processing, `false` otherwise.
    """
    
    logger.info("=" * 60)
    logger.info("RBH-1 OBSERVATIONS ETL")
    logger.info("=" * 60)
    
    # Validate paths
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
    
    # Load manifest
    logger.info(f"Loading manifest: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)
    
    logger.info(f"Manifest contains {len(manifest_rows)} files")
    
    # Connect to database
    creds = load_credentials()
    logger.info(f"Connecting to: {creds['host']}:{creds['port']}/{creds['database']}")
    
    if dry_run:
        logger.info("DRY RUN MODE - no database changes will be made")
        conn = None
        run_id = uuid.uuid4()
    else:
        conn = psycopg2.connect(**creds)
        run_id = create_pipeline_run(conn, "02-etl-observations")
        logger.info(f"Pipeline run ID: {run_id}")
    
    # Process files
    records = []
    errors = []
    
    for i, row in enumerate(manifest_rows, 1):
        filename = row["filename"]
        instrument = row["instrument"]
        
        # Determine subdirectory
        subdir = "hst" if instrument.upper() == "HST" else "jwst"
        filepath = DATA_ROOT / subdir / filename
        
        logger.info(f"[{i}/{len(manifest_rows)}] Processing: {filename}")
        
        if not filepath.exists():
            error_msg = f"File not found: {filepath}"
            logger.warning(error_msg)
            errors.append({"filename": filename, "error": error_msg})
            continue
        
        try:
            # Classify file
            classification = classify_file(filename, instrument)
            
            # Extract FITS metadata
            metadata = extract_fits_metadata(filepath, classification)
            
            # Compute file properties
            file_size = get_file_size(filepath)
            checksum = compute_md5(filepath)
            
            # Build record
            record = {
                **classification,
                **metadata,
                "filename": filename,
                "file_path": f"{subdir}/{filename}",
                "file_size_bytes": file_size,
                "checksum_md5": checksum
            }
            
            records.append(record)
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"  Error: {error_msg}")
            errors.append({"filename": filename, "error": error_msg})
            continue
    
    # Insert to database
    if dry_run:
        logger.info(f"DRY RUN: Would insert {len(records)} records")
        inserted, updated, unchanged = len(records), 0, 0
    else:
        logger.info(f"Inserting {len(records)} records to database...")
        inserted, updated, unchanged = insert_observations(conn, records, run_id)
    
    # Update pipeline run status
    status = "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS"
    notes = f"Inserted: {inserted}, Updated: {updated}, Errors: {len(errors)}"
    
    if not dry_run and conn is not None:
        update_pipeline_run(conn, run_id, status, notes)
        conn.close()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("ETL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files in manifest: {len(manifest_rows)}")
    logger.info(f"Successfully processed:  {len(records)}")
    logger.info(f"New records inserted:    {inserted}")
    logger.info(f"Existing records updated:{updated}")
    logger.info(f"Errors:                  {len(errors)}")
    
    if errors:
        logger.info("-" * 60)
        logger.info("ERRORS:")
        for err in errors[:10]:  # Show first 10
            logger.info(f"  {err['filename']}: {err['error']}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")
    
    logger.info("=" * 60)
    
    return len(errors) == 0


def main():
    """
    Parse command-line arguments, run the ETL pipeline, and exit the process with status 0 on success or 1 on failure.
    
    This function accepts a --dry-run flag to process files without performing database writes, invokes run_etl with that flag, and terminates the program with an exit code reflecting the pipeline outcome.
    """
    parser = argparse.ArgumentParser(description="ETL observations from FITS manifest")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Process files without database changes")
    args = parser.parse_args()
    
    success = run_etl(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()