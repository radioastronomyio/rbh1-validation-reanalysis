#!/usr/bin/env python3
"""
===============================================================================
Script       : 01-acquire_data.py
Phase        : Phase 01 - Data Acquisition
Project      : RBH-1 Independent Validation Study
Repository   : rbh1-validation-reanalysis
-------------------------------------------------------------------------------
Author       : CrainBramp / VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-22
===============================================================================

SCIENTIFIC CONTEXT
==================
This script acquires the primary observational data for independent validation
of the RBH-1 candidate hypervelocity supermassive black hole (van Dokkum et al.
2025). RBH-1 is interpreted as an SMBH ejected via gravitational wave recoil,
traveling at ~1000 km/s and creating a 62 kpc ionized wake.

We download from two observing programs:

HST GO-17301 (van Dokkum, PI)
-----------------------------
WFC3/UVIS imaging in F200LP and F350LP long-pass filters. These UV/optical
images revealed the linear feature interpreted as the bow shock wake. The
F200LP filter is sensitive to young stellar populations and shock-heated gas,
while F350LP provides continuum reference.

- 36 exposures across 6 visits
- ~30,000 seconds total integration
- 0.04"/pixel plate scale
- FLC: Flat-fielded, CTE-corrected individual exposures
- DRC: Distortion-corrected, drizzled combined images

JWST GO-3149 (van Dokkum, PI)
-----------------------------
NIRSpec IFU spectroscopy with G140M grating and F100LP filter, providing
R~1000 spectroscopy from 0.97-1.84 μm. This covers key emission lines at
z=0.964 including [O III] λλ4959,5007 and Hα for kinematic analysis.

- 2 target positions along the wake (tip and mid-wake)
- 4 dithers per position for bad pixel mitigation
- CAL: Calibrated 2D spectra
- S3D: Reconstructed 3D spectral cubes
- X1D: Extracted 1D spectra

PROVENANCE TRACKING
===================
This script generates a manifest CSV recording every downloaded file with
timestamps for full reproducibility. The script is idempotent—re-running
skips existing files, enabling recovery from interrupted downloads.

TECHNICAL NOTES
===============
Dependencies: astroquery >= 0.4.6
Data volume: ~45 GB total
Runtime: 1-4 hours depending on network/MAST load

Usage:
    python 01-acquire_data.py

Output:
    data/hst/           - HST FLC and DRC products
    data/jwst/          - JWST CAL, S3D, and X1D products
    data/data_manifest.csv - Provenance tracking

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import csv
from datetime import datetime, timezone
from pathlib import Path

# astroquery provides the MAST interface
# https://astroquery.readthedocs.io/en/latest/mast/mast.html
from astroquery.mast import Observations

# =============================================================================
# CONFIGURATION
# -----------------------------------------------------------------------------
# These values correspond to the observing programs described in van Dokkum
# et al. (2025). Any changes here would indicate analysis of different data.
# =============================================================================

# Data directory relative to script location
# In production, this may be symlinked to cluster storage (e.g., /mnt/ai-ml/data)
DATA_DIR = Path(__file__).parent.parent / "data"

# ---- HST Configuration ----
# GO-17301: Deep imaging of the RBH-1 field
HST_PROPOSAL_ID = "17301"
HST_PRODUCT_TYPES = [
    "FLC",  # Flat-fielded, CTE-corrected exposures (photometric analysis)
    "DRC"   # Distortion-corrected drizzled images (morphological analysis)
]

# ---- JWST Configuration ----
# GO-3149: NIRSpec IFU follow-up spectroscopy
JWST_PROPOSAL_ID = "3149"
JWST_PRODUCT_TYPES = [
    "CAL",  # Calibrated 2D spectra (Level 2b)
    "S3D",  # 3D spectral cubes - primary data product for kinematic analysis
    "X1D"   # 1D extracted spectra - useful for quick-look and integrated fluxes
]

# =============================================================================
# FUNCTIONS
# =============================================================================


def download_program(
    proposal_id: str,
    collection: str,
    product_types: list[str],
    subdir: str
) -> list | None:
    """
    Download data products for a single observing program from MAST.

    This function queries the Mikulski Archive for Space Telescopes (MAST) for
    all observations associated with a proposal, filters to the requested
    product types, and downloads to a local directory.

    Parameters
    ----------
    proposal_id : str
        MAST proposal ID. For HST this is the GO/GTO/AR program number
        (e.g., "17301"). For JWST this is the program number (e.g., "3149").
    collection : str
        Observatory collection identifier: "HST" or "JWST".
    product_types : list[str]
        Product type codes to filter. Common values:
        - HST: "FLC", "FLT", "DRC", "DRZ"
        - JWST: "CAL", "S3D", "X1D", "RATE", "UNCAL"
    subdir : str
        Subdirectory name under DATA_DIR for organizing output.

    Returns
    -------
    list or None
        astroquery download manifest table, or None if no products found.
        Manifest contains 'Local Path' column with downloaded file locations.

    Notes
    -----
    MAST delivers multiple product versions:
    - HST: Original pipeline + HAP (Hubble Advanced Products) reprocessing
    - JWST: Individual dither + combined products

    The flat=True option places all files in a single directory rather than
    the nested MAST structure, simplifying downstream processing.
    """
    print(f"\n{'='*60}")
    print(f"Querying {collection} proposal {proposal_id}...")
    print(f"Product types: {product_types}")

    # ---- Step 1: Query observations by proposal ID ----
    # This returns the parent observation records (visits/exposures)
    obs = Observations.query_criteria(
        proposal_id=proposal_id,
        obs_collection=collection
    )
    print(f"Found {len(obs)} observations")

    if len(obs) == 0:
        print("WARNING: No observations found - verify proposal ID is correct")
        return None

    # ---- Step 2: Get all data products associated with observations ----
    # Each observation can have multiple associated products (raw, cal, etc.)
    products = Observations.get_product_list(obs)
    print(f"Total products available: {len(products)}")

    # ---- Step 3: Filter to requested product types ----
    # productSubGroupDescription contains the product level code
    filtered = Observations.filter_products(
        products,
        productSubGroupDescription=product_types
    )
    print(f"After filtering: {len(filtered)} products")

    if len(filtered) == 0:
        print("WARNING: No products match the requested types")
        return None

    # ---- Step 4: Download with flat directory structure ----
    out_dir = DATA_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {out_dir}")
    print("(This may take a while for large datasets...)")

    # flat=True prevents creation of nested mastDownload/ structure
    # Existing files are automatically skipped (idempotent)
    manifest = Observations.download_products(
        filtered,
        download_dir=str(out_dir),
        flat=True
    )

    print(f"Download complete: {len(manifest)} files")
    return manifest


def write_manifest(hst_manifest, jwst_manifest) -> None:
    """
    Write a combined manifest CSV for provenance tracking.

    The manifest records every downloaded file with metadata, enabling
    verification that the correct data was used in the analysis. This is
    a key component of reproducibility for the validation study.

    Parameters
    ----------
    hst_manifest : astropy.table.Table or None
        Download manifest from HST acquisition.
    jwst_manifest : astropy.table.Table or None
        Download manifest from JWST acquisition.

    Output
    ------
    Writes to DATA_DIR/data_manifest.csv with columns:
    - filename: Base filename of downloaded product
    - instrument: HST or JWST
    - download_timestamp: ISO 8601 UTC timestamp

    Notes
    -----
    This manifest is distinct from the MAST-provided checksums. It records
    *when* we downloaded the data, which matters because MAST periodically
    reprocesses products with updated calibrations.
    """
    manifest_path = DATA_DIR / "data_manifest.csv"
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "instrument", "download_timestamp"])

        for m, inst in [(hst_manifest, "HST"), (jwst_manifest, "JWST")]:
            if m is not None:
                for row in m:
                    # Extract just the filename from the full path
                    fname = Path(row["Local Path"]).name
                    writer.writerow([fname, inst, timestamp])

    print(f"\nManifest written: {manifest_path}")
    print(f"Timestamp: {timestamp}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Main entry point for RBH-1 data acquisition.

    Downloads all HST and JWST data products required for the validation
    study and generates a provenance manifest.
    """
    print("=" * 60)
    print("RBH-1 VALIDATION STUDY - DATA ACQUISITION")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # ---- HST WFC3/UVIS Imaging (GO-17301) ----
    # Discovery imaging that revealed the 62 kpc linear feature
    # F200LP + F350LP filters, ~30,000s integration
    hst = download_program(
        proposal_id=HST_PROPOSAL_ID,
        collection="HST",
        product_types=HST_PRODUCT_TYPES,
        subdir="hst"
    )

    # ---- JWST NIRSpec IFU Spectroscopy (GO-3149) ----
    # Follow-up spectroscopy for kinematic analysis
    # G140M/F100LP, two positions along the wake
    jwst = download_program(
        proposal_id=JWST_PROPOSAL_ID,
        collection="JWST",
        product_types=JWST_PRODUCT_TYPES,
        subdir="jwst"
    )

    # ---- Generate provenance manifest ----
    write_manifest(hst, jwst)

    print("\n" + "=" * 60)
    print("ACQUISITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
