#!/usr/bin/env python3
"""
Script Name  : 06-etl_spectral_grids.py
Description  : ETL to populate spectral_grids table from JWST s3d IFU cubes
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-24
Phase        : Phase 02 - Standard Extraction
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Extracts wavelength arrays from JWST s3d IFU cubes using WCS spectral axis,
validates monotonicity and sanity, compares each grid to a canonical reference,
and stores one row per wcs_id. Computes mean_resolution as sampling proxy.

Usage
-----
    python 06-etl_spectral_grids.py [--dry-run]

Examples
--------
    python 06-etl_spectral_grids.py
        Run full ETL, extracting spectral grids from all s3d cubes.

    python 06-etl_spectral_grids.py --dry-run
        Process files and validate without database changes.
"""

import argparse
import hashlib
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import warnings

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from dotenv import load_dotenv

# Suppress noisy FITS WCS warnings (DATE fixes, OBSGEO, etc.)
warnings.filterwarnings("ignore", category=FITSFixedWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_FILE = "/opt/global-env/research.env"
DATA_ROOT = Path("/mnt/ai-ml/data")
DEFAULT_DATABASE = "rbh1_validation"
BATCH_SIZE = 50

# Spectral grid comparison tolerance: relative to channel spacing (robust to float noise)
# Match criterion: max_abs_diff_um <= (REL_TOL_SPACING * median_dlambda_um) + ABS_FLOOR_UM
# AI NOTE: These tolerances determine whether two wavelength grids are "the same".
# Changing these affects grid deduplication logic and canonical grid warnings.
REL_TOL_SPACING = 1e-3          # 0.1% of channel spacing
ABS_FLOOR_UM = 1e-10            # absolute floor in microns

# Monotonicity tolerance: allow tiny numerical jitter from WCS float operations
# AI NOTE: JWST wavelength grids should always be monotonic increasing. This
# epsilon catches IEEE-754 rounding artifacts, not actual non-monotonic data.
MONO_EPS_UM = 1e-12

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE
# =============================================================================


def load_credentials() -> dict:
    logger.info(f"Loading credentials from: {ENV_FILE}")
    load_dotenv(ENV_FILE)
    return {
        "host": os.getenv("PGSQL01_HOST", "10.25.20.8"),
        "port": int(os.getenv("PGSQL01_PORT", "5432")),
        "database": DEFAULT_DATABASE,
        "user": os.getenv("PGSQL01_ADMIN_USER", "clusteradmin_pg01"),
        "password": os.getenv("PGSQL01_ADMIN_PASSWORD", ""),
    }


def create_pipeline_run(conn, stage_name: str) -> uuid.UUID:
    run_id = uuid.uuid4()
    run_name = f"etl-spectral-grids-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rbh1.pipeline_runs
                (run_id, run_name, stage_name, pipeline_version, started_at, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(run_id),
                run_name,
                stage_name,
                "0.1.0",
                datetime.now(timezone.utc),
                "RUNNING",
            ),
        )
    conn.commit()
    return run_id


def update_pipeline_run(conn, run_id: uuid.UUID, status: str, notes: Optional[str] = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE rbh1.pipeline_runs
            SET completed_at = %s, status = %s, notes = %s
            WHERE run_id = %s
            """,
            (datetime.now(timezone.utc), status, notes, str(run_id)),
        )
    conn.commit()


def get_s3d_targets(conn) -> list:
    """
    Fetch (wcs_id, filename, file_path, naxis3) for s3d cubes.

    We use wcs_solutions as the authoritative registry for valid cubes.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT w.wcs_id, o.filename, o.file_path, w.naxis3
            FROM rbh1.wcs_solutions w
            JOIN rbh1.observations o ON o.obs_id = w.obs_id
            WHERE o.file_type = 's3d'
            ORDER BY o.filename
            """
        )
        rows = cur.fetchall()

    out = []
    for wcs_id, filename, file_path, naxis3 in rows:
        out.append(
            {
                "wcs_id": str(wcs_id),
                "filename": filename,
                "file_path": file_path,
                "naxis3": int(naxis3) if naxis3 is not None else None,
            }
        )
    return out


# =============================================================================
# FITS / WAVELENGTH EXTRACTION
# =============================================================================


def find_sci_extension(hdul: fits.HDUList) -> Optional[int]:
    """Return first SCI extension index, else fallback to primary or ext 1 if data exists."""
    for i, hdu in enumerate(hdul):
        if hdu.name == "SCI" and hdu.data is not None:
            return i
    # Astropy type stubs don't fully model HDUList indexing
    primary_hdu = hdul[0]
    if primary_hdu.data is not None:  # type: ignore[union-attr]
        return 0
    if len(hdul) > 1:
        ext1_hdu = hdul[1]
        if ext1_hdu.data is not None:  # type: ignore[union-attr]
            return 1
    return None


def _to_um(values: np.ndarray, unit_str: Optional[str]) -> np.ndarray:
    """
    Convert numeric spectral values into microns using astropy units when possible.
    
    AI NOTE: If unit_str is missing or unparsable, we assume meters. This is correct
    for JWST NIRSpec cubes (CUNIT3='m' in WCS). If extending to other instruments
    where default units differ (e.g., Angstroms for optical), this fallback must change.
    """
    if unit_str is None or unit_str.strip() == "":
        # JWST cubes are commonly in meters in FITS WCS; if unit missing, assume meters conservatively.
        unit = u.m
    else:
        try:
            unit = u.Unit(unit_str)
        except Exception:
            # Fallback: assume meters if parsing fails
            unit = u.m

    q = np.asarray(values, dtype=np.float64) * unit
    # Astropy Quantity.to_value() returns ndarray; stubs don't model this fully
    return q.to_value(u.um)  # type: ignore[union-attr, return-value]


def extract_wavelength_array_um(filepath: Path) -> Tuple[np.ndarray, dict]:
    """
    Extract wavelength array (microns) from a JWST s3d cube.

    Primary path:
        - Build full WCS from SCI header
        - Use wcs.sub(['spectral']) (or wcs.spectral) to compute wavelength for k=0..N-1

    Fallback path:
        - Use full 3D WCS at the spatial center and vary spectral pixel coordinate
    """
    with fits.open(filepath) as hdul:
        sci_idx = find_sci_extension(hdul)
        if sci_idx is None:
            raise RuntimeError("No SCI/primary extension with data found")

        hdu = hdul[sci_idx]
        header = hdu.header

        naxis1 = header.get("NAXIS1")
        naxis2 = header.get("NAXIS2")
        naxis3 = header.get("NAXIS3")

        if naxis3 is None:
            # Try shape fallback (data is usually (naxis3, naxis2, naxis1))
            if hdu.data is None or hdu.data.ndim < 3:
                raise RuntimeError("Missing NAXIS3 and data shape is not 3D")
            naxis3 = int(hdu.data.shape[0])

        # Build WCS (full 3D)
        wcs = WCS(header, fobj=hdul)

        # Try spectral-only WCS first (axis-order safe)
        wave_native = None
        unit_str = None
        used_mode = "spectral_subwcs"

        try:
            spec_wcs = getattr(wcs, "spectral", None)
            if spec_wcs is None:
                spec_wcs = wcs.sub(["spectral"])

            k = np.arange(naxis3, dtype=np.float64)
            wave_native = spec_wcs.pixel_to_world_values(k)

            # Get unit from spectral WCS
            # spec_wcs.wcs.cunit is 1-element array-like
            try:
                unit_str = str(spec_wcs.wcs.cunit[0]) if spec_wcs.wcs.cunit is not None else None
            except Exception:
                unit_str = None

            wave_um = _to_um(np.asarray(wave_native), unit_str)

        except Exception as e:
            # Fallback: evaluate full WCS at spatial center
            used_mode = f"full_wcs_center_fallback({type(e).__name__})"

            # Infer dims if missing
            if naxis1 is None or naxis2 is None:
                if hdu.data is None or hdu.data.ndim < 3:
                    raise RuntimeError("Missing NAXIS1/2 and cannot infer from data")
                # data shape: (naxis3, naxis2, naxis1)
                naxis2 = int(hdu.data.shape[1])
                naxis1 = int(hdu.data.shape[2])

            x0 = (naxis1 - 1) / 2.0
            y0 = (naxis2 - 1) / 2.0
            k = np.arange(naxis3, dtype=np.float64)

            x = np.full_like(k, x0)
            y = np.full_like(k, y0)

            world = wcs.pixel_to_world_values(x, y, k)

            # Find spectral world axis by matching CTYPE
            ctype = [c.upper() for c in wcs.wcs.ctype]
            spec_idx = None
            for i, ct in enumerate(ctype):
                if "WAVE" in ct or "FREQ" in ct or "VELO" in ct:
                    spec_idx = i
                    break
            if spec_idx is None:
                raise RuntimeError(f"Could not identify spectral axis from CTYPE={wcs.wcs.ctype}")

            wave_native = np.asarray(world[spec_idx], dtype=np.float64)
            try:
                unit_str = str(wcs.wcs.cunit[spec_idx]) if wcs.wcs.cunit is not None else None
            except Exception:
                unit_str = None

            wave_um = _to_um(wave_native, unit_str)

        meta = {
            "naxis3": int(naxis3),
            "unit_str": unit_str,
            "mode": used_mode,
            "sci_ext": sci_idx,
        }
        return wave_um, meta


def validate_wavelength_array(wave_um: np.ndarray, expected_n: Optional[int]) -> Tuple[bool, str, dict]:
    """Run basic sanity checks and return (ok, reason, stats)."""
    if expected_n is not None and len(wave_um) != expected_n:
        return False, f"length mismatch: got {len(wave_um)} expected {expected_n}", {}

    if not np.all(np.isfinite(wave_um)):
        return False, "non-finite values in wavelength array", {}

    d = np.diff(wave_um)
    if np.any(d < -MONO_EPS_UM):
        return False, "wavelength array not monotonic increasing", {}

    # If there are tiny non-positive diffs, it's numerical jitter; still compute stats safely
    d_pos = d[d > 0]
    if d_pos.size == 0:
        return False, "no positive channel spacing detected", {}

    dl_med = float(np.median(d_pos))
    wmin = float(np.min(wave_um))
    wmax = float(np.max(wave_um))

    # Sampling-resolution proxy: median(lambda / dlambda)
    R = wave_um[1:] / np.where(d > 0, d, np.nan)
    mean_R = float(np.nanmedian(R))

    stats = {
        "n_channels": int(len(wave_um)),
        "min_wavelength": wmin,
        "max_wavelength": wmax,
        "median_dlambda_um": dl_med,
        "mean_resolution": mean_R,
    }
    return True, "ok", stats


def quantized_sha256(wave_um: np.ndarray, quantum_um: float) -> str:
    """
    Quantize wavelength array to a grid and hash for stable comparisons.
    
    Direct float comparison is fragile due to IEEE-754 representation differences.
    Quantizing to ~1e-9 µm bins before hashing produces stable fingerprints that
    survive round-trip through different code paths while remaining sensitive to
    meaningful wavelength differences.
    """
    q = np.round(wave_um / quantum_um).astype(np.int64)
    h = hashlib.sha256(q.tobytes()).hexdigest()
    return h


# =============================================================================
# INSERT/UPSERT
# =============================================================================


def insert_spectral_grids(conn, rows: list) -> Tuple[int, int]:
    """
    Upsert spectral grids by UNIQUE(wcs_id).

    Returns (inserted_count, updated_count) estimated by pre-check.
    """
    if not rows:
        return 0, 0

    wcs_ids = [r["wcs_id"] for r in rows]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT wcs_id
            FROM rbh1.spectral_grids
            WHERE wcs_id = ANY(%s::uuid[])
            """,
            (wcs_ids,),
        )
        existing = {str(x[0]) for x in cur.fetchall()}

    # AI NOTE: UPSERT on wcs_id. On conflict:
    # - wavelength_array, n_channels, min/max_wavelength, mean_resolution all UPDATE
    # - grid_id is auto-generated and preserved (not in EXCLUDED)
    # - systemic_redshift is a user-supplied column, not touched by ETL
    insert_sql = """
        INSERT INTO rbh1.spectral_grids (
            wcs_id,
            wavelength_array,
            n_channels,
            min_wavelength,
            max_wavelength,
            mean_resolution
        ) VALUES %s
        ON CONFLICT (wcs_id) DO UPDATE SET
            wavelength_array = EXCLUDED.wavelength_array,
            n_channels = EXCLUDED.n_channels,
            min_wavelength = EXCLUDED.min_wavelength,
            max_wavelength = EXCLUDED.max_wavelength,
            mean_resolution = EXCLUDED.mean_resolution
    """

    values = []
    for r in rows:
        values.append(
            (
                r["wcs_id"],
                r["wavelength_array"],  # python list[float]
                r["n_channels"],
                r["min_wavelength"],
                r["max_wavelength"],
                r["mean_resolution"],
            )
        )

    template = "(%s::uuid, %s::double precision[], %s, %s, %s, %s)"

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values, template=template, page_size=BATCH_SIZE)

    conn.commit()

    inserted = len(rows) - len(existing)
    updated = len(existing)
    return inserted, updated


# =============================================================================
# MAIN
# =============================================================================


def run_etl(dry_run: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("RBH-1 SPECTRAL GRIDS ETL")
    logger.info("=" * 60)

    creds = load_credentials()
    logger.info(f"Connecting to: {creds['host']}:{creds['port']}/{creds['database']}")
    conn = psycopg2.connect(**creds)

    if dry_run:
        logger.info("DRY RUN MODE - no database changes will be made")
        run_id = uuid.uuid4()
    else:
        run_id = create_pipeline_run(conn, "02-etl-spectral-grids")
        logger.info(f"Pipeline run ID: {run_id}")

    targets = get_s3d_targets(conn)
    logger.info(f"Found {len(targets)} s3d cubes with WCS solutions")

    canonical_wave = None
    canonical_hash = None
    canonical_dl = None

    out_rows = []
    errors = []
    skipped = 0

    for i, t in enumerate(targets, 1):
        filename = t["filename"]
        wcs_id = t["wcs_id"]
        naxis3 = t["naxis3"]

        filepath = DATA_ROOT / t["file_path"]

        logger.info(f"[{i}/{len(targets)}] Processing: {filename}")
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            logger.warning(f"  {msg}")
            errors.append({"filename": filename, "error": msg})
            continue

        try:
            wave_um, meta = extract_wavelength_array_um(filepath)
            ok, reason, stats = validate_wavelength_array(wave_um, expected_n=naxis3)
            if not ok:
                logger.warning(f"  Skipping {filename}: {reason}")
                skipped += 1
                continue

            # Canonical compare/log
            dl_med = stats["median_dlambda_um"]
            tol_um = (REL_TOL_SPACING * dl_med) + ABS_FLOOR_UM

            # Stable-ish fingerprint for log/debug (quantize to 1e-9 µm by default)
            h = quantized_sha256(wave_um, quantum_um=1e-9)

            # Canonical grid: first successfully extracted grid becomes the reference.
            # All subsequent grids are compared against it to detect wavelength drift
            # between observations (should not happen for same grating/filter config).
            if canonical_wave is None:
                canonical_wave = wave_um
                canonical_hash = h
                canonical_dl = dl_med
                logger.info(f"  Canonical grid set from {filename} | hash={canonical_hash[:16]}.. | dl_med={canonical_dl:.3e} µm | mode={meta['mode']}")
            else:
                max_abs = float(np.max(np.abs(wave_um - canonical_wave)))
                max_rel = float(np.max(np.abs((wave_um - canonical_wave) / np.maximum(canonical_wave, 1e-30))))
                match = max_abs <= tol_um
                logger.info(
                    f"  Grid compare | max_abs={max_abs:.3e} µm | tol={tol_um:.3e} µm | max_rel={max_rel:.3e} | hash={h[:16]}.. | match={match}"
                )
                if not match:
                    logger.warning(f"  WARNING: {filename} wavelength grid differs from canonical beyond tolerance")

            # Prepare DB row (store 1:1 per wcs_id)
            out_rows.append(
                {
                    "wcs_id": wcs_id,
                    "wavelength_array": wave_um.astype(float).tolist(),
                    "n_channels": stats["n_channels"],
                    "min_wavelength": stats["min_wavelength"],
                    "max_wavelength": stats["max_wavelength"],
                    "mean_resolution": stats["mean_resolution"],
                }
            )

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            logger.error(f"  Error: {msg}")
            errors.append({"filename": filename, "error": msg})
            continue

    # Insert/upsert
    if dry_run:
        logger.info(f"DRY RUN: Would upsert {len(out_rows)} spectral grids")
        inserted, updated = len(out_rows), 0
    else:
        logger.info(f"Upserting {len(out_rows)} spectral grids...")
        inserted, updated = insert_spectral_grids(conn, out_rows)

    status = "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS"
    notes = f"Inserted: {inserted}, Updated: {updated}, Skipped: {skipped}, Errors: {len(errors)}"
    if canonical_hash is not None:
        notes += f", CanonicalHash: {canonical_hash[:16]}.."

    if not dry_run:
        update_pipeline_run(conn, run_id, status, notes)

    conn.close()

    logger.info("=" * 60)
    logger.info("ETL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total s3d targets:          {len(targets)}")
    logger.info(f"Successfully processed:     {len(out_rows)}")
    logger.info(f"New grids inserted:         {inserted}")
    logger.info(f"Existing grids updated:     {updated}")
    logger.info(f"Skipped (invalid grids):    {skipped}")
    logger.info(f"Errors:                     {len(errors)}")
    if canonical_hash is not None:
        logger.info(f"Canonical grid hash:        {canonical_hash}")

    if errors:
        logger.info("-" * 60)
        logger.info("ERRORS (first 10):")
        for err in errors[:10]:
            logger.info(f"  {err['filename']}: {err['error']}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")

    logger.info("=" * 60)
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="ETL spectral_grids from JWST s3d cubes")
    parser.add_argument("--dry-run", action="store_true", help="Process files without database changes")
    args = parser.parse_args()

    ok = run_etl(dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
