#!/usr/bin/env python3
"""
===============================================================================
Script       : 03-visualize_data.py
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
This script generates QA (Quality Assurance) visualizations for Phase 01 data
validation. These figures serve two purposes:

1. Verify data sanity before proceeding to analysis phases
2. Provide visual documentation for the validation study

The figures focus on "did we get the right data?" rather than scientific
analysis. Science figures (PV diagrams, line maps, etc.) are deferred to
later phases.

FIGURE INVENTORY
================

01_pointing_sanity.png
----------------------
Scatter plot of HST and JWST pointing offsets from the target coordinates.
Verifies all observations targeted the RBH-1 field within tolerance.

Scientific context: Pointing errors could cause us to analyze the wrong
field or miss critical parts of the 62 kpc wake structure.

02_footprint_overlay.png
------------------------
JWST NIRSpec IFU footprints overlaid on HST DRC image.
Shows where the spectroscopic observations sampled the wake.

Scientific context: The IFU pointings were chosen to capture the "tip"
(leading edge of the wake) and a mid-wake position for kinematic comparison.

03_context_zoom.png
-------------------
Three-panel zoom: full field → RBH-1 region → tip region.
Provides spatial context and orientation for the linear feature.

Scientific context: The 62 kpc wake is faint and requires careful background
subtraction. This zoom sequence helps identify the feature visually.

04_wavelength_coverage.png
--------------------------
Expected emission lines at z=0.964 vs G140M/F100LP bandpass.
Shows which diagnostic lines fall within the spectral coverage.

Scientific context: The kinematic analysis relies on [O III] and Hα, both
within the bandpass. [O II] (often used for redshift confirmation) is NOT
covered—this is a known limitation.

05_noise_vs_wavelength.png
--------------------------
Per-channel noise from the S3D ERR extension.
Identifies wavelength regions with elevated noise (thermal, detector edges).

Scientific context: The rising noise above 1.5 μm is thermal background.
Line measurements in this region require larger uncertainties.

06_dq_heatmaps.png
------------------
NaN and DQ flagged pixel fractions by file type.
Verifies data quality flags are consistent and expected.

Scientific context: High flagged fractions in JWST IFU data are NORMAL—the
detector area outside the 3"×3" IFU aperture is masked. Only ~20% of pixels
contain science data.

07_inventory_bars.png
---------------------
File counts by product type for HST and JWST.
Visual summary of acquired data inventory.

08_acquisition_timeline.png
---------------------------
Observation dates showing HST (2023) and JWST (2024) campaigns.
Documents the temporal baseline between discovery and follow-up.

Scientific context: The ~1 year gap means proper motion could be detectable
if the feature is truly moving at ~1000 km/s (though at z=0.964, this is
negligible in angular terms).

TECHNICAL NOTES
===============
Dependencies: matplotlib, seaborn, astropy, numpy
Runtime: ~2-5 minutes depending on file I/O

Some figures may generate warnings (WCS parsing, etc.) that are non-fatal.
Check the output directory for all 8 figures.

Usage:
    python 03-visualize_data.py /path/to/data --config validation_config.yaml

Output:
    figures/01_pointing_sanity.png
    figures/02_footprint_overlay.png
    ... (8 figures total)

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LogNorm
import seaborn as sns
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval, ImageNormalize

# =============================================================================
# PLOT STYLING
# -----------------------------------------------------------------------------
# Consistent style across all figures for professional appearance.
# =============================================================================

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_config(config_path: Path) -> dict:
    """Load validation configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_header_value(header, *keys):
    """
    Get first available header value from list of possible keys.

    FITS headers vary between instruments—this provides robust extraction.
    """
    for key in keys:
        if key in header:
            return header[key]
    return None


# =============================================================================
# FIGURE 01: POINTING SANITY
# -----------------------------------------------------------------------------
# Verifies all observations targeted the correct sky position.
# =============================================================================


def fig_pointing_sanity(hst_path: Path, jwst_path: Path, config: dict, output_dir: Path):
    """
    Generate scatter plot of pointing offsets from target coordinates.

    This figure provides immediate visual confirmation that all data comes
    from the RBH-1 field. Points should cluster near the origin, well within
    the tolerance circle.

    Scientific interpretation:
    - Tight clustering = consistent pointing across all visits
    - Systematic offset = possible coordinate system issue
    - Scatter outside tolerance = wrong field or acquisition failure
    """
    target_ra = config['target']['ra_deg']
    target_dec = config['target']['dec_deg']
    target_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)

    hst_offsets = {'ra': [], 'dec': [], 'file': []}
    jwst_offsets = {'ra': [], 'dec': [], 'file': []}

    # ---- Extract HST pointings ----
    for fpath in hst_path.glob("*_flc.fits"):
        try:
            with fits.open(fpath) as hdul:
                h = hdul[0].header
                ra = get_header_value(h, 'RA_TARG', 'CRVAL1')
                dec = get_header_value(h, 'DEC_TARG', 'CRVAL2')
                if ra and dec:
                    # Convert to arcmin offset, accounting for cos(dec) in RA
                    dra = (float(ra) - target_ra) * 60 * np.cos(np.radians(target_dec))
                    ddec = (float(dec) - target_dec) * 60
                    hst_offsets['ra'].append(dra)
                    hst_offsets['dec'].append(ddec)
                    hst_offsets['file'].append(fpath.name)
        except Exception:
            pass

    # ---- Extract JWST pointings ----
    for fpath in jwst_path.glob("*_cal.fits"):
        try:
            with fits.open(fpath) as hdul:
                h = hdul[0].header
                ra = get_header_value(h, 'TARG_RA', 'RA_TARG', 'CRVAL1')
                dec = get_header_value(h, 'TARG_DEC', 'DEC_TARG', 'CRVAL2')
                if ra and dec:
                    dra = (float(ra) - target_ra) * 60 * np.cos(np.radians(target_dec))
                    ddec = (float(dec) - target_dec) * 60
                    jwst_offsets['ra'].append(dra)
                    jwst_offsets['dec'].append(ddec)
                    jwst_offsets['file'].append(fpath.name)
        except Exception:
            pass

    # ---- Create figure ----
    fig, ax = plt.subplots(figsize=(8, 8))

    # Target position at origin
    ax.scatter([0], [0], marker='+', s=200, c='black', linewidths=2,
               label='Target', zorder=10)

    # HST observations
    if hst_offsets['ra']:
        ax.scatter(hst_offsets['ra'], hst_offsets['dec'],
                   alpha=0.6, s=30, c='tab:blue',
                   label=f"HST (n={len(hst_offsets['ra'])})")

    # JWST observations
    if jwst_offsets['ra']:
        ax.scatter(jwst_offsets['ra'], jwst_offsets['dec'],
                   alpha=0.6, s=50, c='tab:orange', marker='s',
                   label=f"JWST (n={len(jwst_offsets['ra'])})")

    # Tolerance circle from config
    tol = config['target']['pointing_tolerance_arcmin']
    circle = plt.Circle((0, 0), tol, fill=False, linestyle='--',
                         color='gray', label=f'{tol} arcmin tolerance')
    ax.add_patch(circle)

    # Annotate maximum offsets
    all_offsets = np.sqrt(np.array(hst_offsets['ra'])**2 +
                          np.array(hst_offsets['dec'])**2)
    if len(all_offsets) > 0:
        hst_max = np.max(all_offsets)
        ax.annotate(f'HST max: {hst_max:.3f}\'', xy=(0.02, 0.98),
                    xycoords='axes fraction', ha='left', va='top', fontsize=9)

    jwst_all = np.sqrt(np.array(jwst_offsets['ra'])**2 +
                       np.array(jwst_offsets['dec'])**2)
    if len(jwst_all) > 0:
        jwst_max = np.max(jwst_all)
        ax.annotate(f'JWST max: {jwst_max:.3f}\'', xy=(0.02, 0.94),
                    xycoords='axes fraction', ha='left', va='top', fontsize=9)

    ax.set_xlabel('ΔRA (arcmin)')
    ax.set_ylabel('ΔDec (arcmin)')
    ax.set_title('Pointing Offsets from Target')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='gray', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    outpath = output_dir / "01_pointing_sanity.png"
    plt.savefig(outpath)
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 02: FOOTPRINT OVERLAY
# -----------------------------------------------------------------------------
# Shows JWST IFU positions on HST imaging.
# =============================================================================


def fig_footprint_overlay(hst_path: Path, jwst_path: Path, config: dict, output_dir: Path):
    """
    Overlay JWST NIRSpec IFU footprints on HST DRC image.

    This figure shows the spatial relationship between imaging (HST) and
    spectroscopy (JWST). The IFU positions were chosen to capture the
    kinematically interesting regions of the wake.

    Known limitation: At full-field scale, the 3" IFU boxes may be too small
    to see clearly. A zoomed version would be more informative.

    TODO: [Phase01] Fix IFU boxes not visible - need zoom or larger box size
    """
    # ---- Find a suitable HST DRC for background ----
    # Prefer HAP products for better astrometry
    drc_files = list(hst_path.glob("hst_*_drc.fits"))
    if not drc_files:
        drc_files = list(hst_path.glob("*_drc.fits"))

    if not drc_files:
        print("  WARNING: No DRC files found for footprint overlay")
        return

    # ---- Load DRC with valid WCS ----
    drc_file = None
    drc_data = None
    drc_wcs = None

    for f in drc_files:
        try:
            with fits.open(f) as hdul:
                # Try SCI extension first (standard), then primary
                for ext in ['SCI', 0, 1]:
                    try:
                        if isinstance(ext, str):
                            data = hdul[ext].data
                            wcs = WCS(hdul[ext].header)
                        else:
                            data = hdul[ext].data
                            wcs = WCS(hdul[ext].header)
                        if data is not None and data.ndim == 2:
                            drc_file = f
                            drc_data = data
                            drc_wcs = wcs
                            break
                    except Exception:
                        continue
                if drc_data is not None:
                    break
        except Exception:
            continue

    if drc_data is None:
        print("  WARNING: Could not load DRC image data")
        return

    # ---- Extract JWST IFU positions from S3D headers ----
    s3d_files = list(jwst_path.glob("*_s3d.fits"))
    ifu_footprints = []

    for f in s3d_files:
        try:
            with fits.open(f) as hdul:
                h = hdul['SCI'].header if 'SCI' in hdul else hdul[1].header
                ra_ref = get_header_value(h, 'CRVAL1')
                dec_ref = get_header_value(h, 'CRVAL2')
                if ra_ref and dec_ref:
                    ifu_footprints.append({
                        'ra': float(ra_ref),
                        'dec': float(dec_ref),
                        'file': f.name
                    })
        except Exception:
            pass

    # ---- Create WCS-aware figure ----
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=drc_wcs)

    # Display with zscale stretch (standard astronomical visualization)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(drc_data[np.isfinite(drc_data)])
    ax.imshow(drc_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

    # Mark target position
    target_ra = config['target']['ra_deg']
    target_dec = config['target']['dec_deg']
    target_pix = drc_wcs.world_to_pixel(
        SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg))
    ax.scatter(target_pix[0], target_pix[1], marker='+', s=200, c='red',
               linewidths=2, label='Target')

    # ---- Draw IFU footprints ----
    # NIRSpec IFU is approximately 3" × 3"
    ifu_size_arcsec = 3.0
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ifu_footprints), 1)))

    for i, fp in enumerate(ifu_footprints[:4]):  # Limit to 4 for clarity
        center = SkyCoord(ra=fp['ra']*u.deg, dec=fp['dec']*u.deg)
        center_pix = drc_wcs.world_to_pixel(center)

        # Estimate pixel scale from WCS
        try:
            pixscale = np.abs(drc_wcs.wcs.cdelt[0]) * 3600 if hasattr(drc_wcs.wcs, 'cdelt') else 0.04
        except:
            pixscale = 0.04  # WFC3/UVIS default: 0.04"/pixel

        box_size_pix = ifu_size_arcsec / pixscale

        rect = Rectangle(
            (center_pix[0] - box_size_pix/2, center_pix[1] - box_size_pix/2),
            box_size_pix, box_size_pix,
            fill=False, edgecolor=colors[i], linewidth=2,
            label=f'IFU {i+1}'
        )
        ax.add_patch(rect)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title(f'JWST IFU Footprints on HST DRC\n{drc_file.name}')
    ax.legend(loc='upper right')

    plt.tight_layout()
    outpath = output_dir / "02_footprint_overlay.png"
    plt.savefig(outpath)
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 03: CONTEXT ZOOM
# -----------------------------------------------------------------------------
# Three-panel zoom sequence showing spatial context.
# =============================================================================


def fig_context_zoom(hst_path: Path, config: dict, output_dir: Path):
    """
    Generate three-panel zoom: full field → RBH-1 region → tip region.

    This figure helps orient viewers to the scale and appearance of the
    linear feature. The 62 kpc wake is subtle and requires careful examination.
    """
    # ---- Find suitable DRC ----
    drc_files = list(hst_path.glob("hst_*_drc.fits"))
    if not drc_files:
        drc_files = list(hst_path.glob("*_drc.fits"))

    if not drc_files:
        print("  WARNING: No DRC files found for context zoom")
        return

    drc_data = None
    drc_wcs = None

    for f in drc_files:
        try:
            with fits.open(f) as hdul:
                for ext in ['SCI', 0, 1]:
                    try:
                        if isinstance(ext, str):
                            data = hdul[ext].data
                            wcs = WCS(hdul[ext].header)
                        else:
                            data = hdul[ext].data
                            wcs = WCS(hdul[ext].header)
                        if data is not None and data.ndim == 2:
                            drc_data = data
                            drc_wcs = wcs
                            break
                    except:
                        continue
                if drc_data is not None:
                    break
        except:
            continue

    if drc_data is None:
        print("  WARNING: Could not load DRC for context zoom")
        return

    # ---- Convert target to pixel coordinates ----
    target_ra = config['target']['ra_deg']
    target_dec = config['target']['dec_deg']
    target_pix = drc_wcs.world_to_pixel(
        SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg))

    # ---- Compute display stretch ----
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(drc_data[np.isfinite(drc_data)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- Panel 1: Full field ----
    ax1 = axes[0]
    ax1.imshow(drc_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax1.scatter(target_pix[0], target_pix[1], marker='+', s=100, c='red',
                linewidths=1.5)
    ax1.set_title('Full Field')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    # ---- Panel 2: Region zoom (±500 pixels) ----
    ax2 = axes[1]
    cx, cy = int(target_pix[0]), int(target_pix[1])
    half = 500
    y1, y2 = max(0, cy-half), min(drc_data.shape[0], cy+half)
    x1, x2 = max(0, cx-half), min(drc_data.shape[1], cx+half)

    ax2.imshow(drc_data[y1:y2, x1:x2], origin='lower', cmap='gray',
               vmin=vmin, vmax=vmax)
    ax2.scatter(cx-x1, cy-y1, marker='+', s=100, c='red', linewidths=1.5)
    ax2.set_title(f'RBH-1 Region (±{half} pix)')
    ax2.set_xlabel('X (pixels)')

    # Draw region box on panel 1
    rect1 = Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                       edgecolor='cyan', linewidth=1)
    ax1.add_patch(rect1)

    # ---- Panel 3: Tip zoom (±150 pixels) ----
    ax3 = axes[2]
    half2 = 150
    y1b, y2b = max(0, cy-half2), min(drc_data.shape[0], cy+half2)
    x1b, x2b = max(0, cx-half2), min(drc_data.shape[1], cx+half2)

    ax3.imshow(drc_data[y1b:y2b, x1b:x2b], origin='lower', cmap='gray',
               vmin=vmin, vmax=vmax)
    ax3.scatter(cx-x1b, cy-y1b, marker='+', s=100, c='red', linewidths=1.5)
    ax3.set_title(f'Tip Region (±{half2} pix)')
    ax3.set_xlabel('X (pixels)')

    # Draw tip box on panel 2
    rect2 = Rectangle((x1b-x1, y1b-y1), x2b-x1b, y2b-y1b, fill=False,
                       edgecolor='yellow', linewidth=1)
    ax2.add_patch(rect2)

    plt.tight_layout()
    outpath = output_dir / "03_context_zoom.png"
    plt.savefig(outpath)
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 04: WAVELENGTH COVERAGE
# -----------------------------------------------------------------------------
# Shows which emission lines fall within the spectral bandpass.
# =============================================================================


def fig_wavelength_coverage(config: dict, output_dir: Path):
    """
    Show expected emission lines at z=0.964 vs G140M/F100LP bandpass.

    This figure is critical for understanding what diagnostics are available.
    The G140M grating provides R~1000 spectroscopy from 0.97-1.84 μm.

    Key findings:
    - [O III] 4959,5007 and Hα 6563 are IN BAND (kinematic analysis possible)
    - [O II] 3727,3729 is OUT OF BAND (no independent redshift from [O II])
    - [S II] doublet is IN BAND (shock diagnostics possible)
    """
    z = config['target']['redshift']

    # ---- Rest-frame emission lines (vacuum wavelengths in Angstroms) ----
    # These are the primary diagnostic lines for shock-heated gas
    lines = {
        '[O II] 3727': 3727,   # Redshift indicator (often strongest in star-forming)
        '[O II] 3729': 3729,
        'Hγ 4340': 4340,       # Balmer series
        'Hβ 4861': 4861,       # Balmer series, extinction indicator
        '[O III] 4959': 4959,  # Collisionally excited, shock tracer
        '[O III] 5007': 5007,  # Strongest optical forbidden line
        '[N II] 6548': 6548,   # Nitrogen abundance, shock indicator
        'Hα 6563': 6563,       # Strongest Balmer line
        '[N II] 6584': 6584,   # [N II]/Hα ratio is AGN/shock diagnostic
        '[S II] 6717': 6717,   # Sulfur doublet, density diagnostic
        '[S II] 6731': 6731,
    }

    # Compute observed wavelengths
    obs_lines = {name: wave * (1 + z) for name, wave in lines.items()}

    # ---- G140M/F100LP coverage (approximate, in Angstroms) ----
    # This combination provides 0.97-1.84 μm coverage
    g140m_min = 9700   # Blue cutoff
    g140m_max = 18400  # Red cutoff

    fig, ax = plt.subplots(figsize=(14, 6))

    # Shade bandpass region
    ax.axvspan(g140m_min, g140m_max, alpha=0.2, color='blue',
               label='G140M/F100LP coverage')

    # ---- Plot emission lines ----
    colors = {'O': 'green', 'H': 'red', 'N': 'orange', 'S': 'purple'}

    for name, obs_wave in obs_lines.items():
        # Color by element
        element = name[1] if name.startswith('[') else name[0]
        color = colors.get(element, 'gray')

        # Solid if in band, dashed if out
        in_band = g140m_min <= obs_wave <= g140m_max
        linestyle = '-' if in_band else '--'
        alpha = 1.0 if in_band else 0.4

        ax.axvline(obs_wave, color=color, linestyle=linestyle,
                   alpha=alpha, linewidth=1.5)

        # Label
        y_pos = 0.85 if in_band else 0.5
        ax.annotate(name, xy=(obs_wave, y_pos), xycoords=('data', 'axes fraction'),
                    rotation=90, ha='right', va='bottom', fontsize=8, alpha=alpha)

    # ---- Legend ----
    handles = [mpatches.Patch(color=c, label=f'[{e}]' if e in ['O','N','S'] else 'H lines')
               for e, c in colors.items()]
    handles.append(mpatches.Patch(color='blue', alpha=0.2, label='G140M/F100LP'))
    ax.legend(handles=handles, loc='upper right')

    ax.set_xlim(5000, 20000)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Observed Wavelength (Å)')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title(f'Expected Emission Lines at z={z:.3f} vs NIRSpec G140M/F100LP Coverage')

    # Secondary axis in microns (more common for NIR)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0]/10000, ax.get_xlim()[1]/10000)
    ax2.set_xlabel('Wavelength (μm)')

    plt.tight_layout()
    outpath = output_dir / "04_wavelength_coverage.png"
    plt.savefig(outpath)
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 05: NOISE VS WAVELENGTH
# -----------------------------------------------------------------------------
# Shows spectral noise properties from the ERR extension.
# =============================================================================


def fig_noise_vs_wavelength(jwst_path: Path, output_dir: Path):
    """
    Plot per-channel noise from S3D ERR extension.

    This figure reveals wavelength-dependent noise properties:
    - Thermal background rises above ~1.5 μm
    - Detector artifacts may appear as noise spikes
    - Low-noise regions are optimal for faint line detection

    Used for planning integration time and setting error bars on measurements.
    """
    s3d_files = list(jwst_path.glob("*_s3d.fits"))
    if not s3d_files:
        print("  WARNING: No S3D files found for noise analysis")
        return

    # Prefer combined products over individual dithers
    combined = [f for f in s3d_files if 't001' in f.name or 't002' in f.name]
    target_file = combined[0] if combined else s3d_files[0]

    try:
        with fits.open(target_file) as hdul:
            # ---- Get ERR extension ----
            if 'ERR' in hdul:
                err_cube = hdul['ERR'].data
            elif len(hdul) > 2:
                err_cube = hdul[2].data
            else:
                print("  WARNING: No ERR extension found in S3D")
                return

            # ---- Construct wavelength array from WCS ----
            sci_header = hdul['SCI'].header if 'SCI' in hdul else hdul[1].header
            nwave = err_cube.shape[0]
            crval3 = sci_header.get('CRVAL3', 1.0)
            cdelt3 = sci_header.get('CDELT3', sci_header.get('CD3_3', 0.001))
            crpix3 = sci_header.get('CRPIX3', 1)

            wavelength = crval3 + (np.arange(nwave) - crpix3 + 1) * cdelt3
            # Convert to Angstroms if in microns
            wavelength_ang = wavelength * 1e4 if wavelength[0] < 100 else wavelength

            # ---- Compute median error per channel ----
            # Median is robust to outliers from bad pixels
            median_err = np.nanmedian(err_cube, axis=(1, 2))

    except Exception as e:
        print(f"  WARNING: Error reading S3D for noise analysis: {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(wavelength_ang, median_err, 'b-', linewidth=0.5, alpha=0.7)
    ax.fill_between(wavelength_ang, 0, median_err, alpha=0.3)

    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Median Error (per channel)')
    ax.set_title(f'Noise vs Wavelength\n{target_file.name}')
    ax.set_xlim(wavelength_ang.min(), wavelength_ang.max())
    ax.set_ylim(0, None)

    # Mark 95th percentile as reference for high-noise regions
    threshold = np.nanpercentile(median_err, 95)
    high_noise = median_err > threshold
    if np.any(high_noise):
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5,
                   label='95th percentile')
        ax.legend()

    plt.tight_layout()
    outpath = output_dir / "05_noise_vs_wavelength.png"
    plt.savefig(outpath)
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 06: DQ/NaN HEATMAPS
# -----------------------------------------------------------------------------
# Shows flagged pixel fractions to verify data quality is nominal.
# =============================================================================


def fig_dq_heatmaps(hst_path: Path, jwst_path: Path, output_dir: Path):
    """
    Show NaN and DQ flagged pixel fractions by file type.

    High flagged fractions in JWST IFU data are EXPECTED—the detector area
    outside the 3"×3" aperture is masked. This figure documents that the
    flagging is consistent and nominal.

    Interpretation:
    - HST FLC: ~8% DQ flagged (cosmic rays, bad pixels) is typical
    - HST DRC: ~28% NaN (drizzle footprint edges) is normal
    - JWST CAL: ~70% NaN is NORMAL for IFU (detector vs aperture)
    - JWST S3D: ~50% NaN is NORMAL for reconstructed cubes
    """
    results = {'HST FLC': [], 'HST DRC': [], 'JWST CAL': [], 'JWST S3D': []}

    # ---- Process HST FLC ----
    for f in sorted(hst_path.glob("hst_*_flc.fits"))[:20]:
        try:
            with fits.open(f) as hdul:
                sci = hdul['SCI'].data if 'SCI' in hdul else hdul[1].data
                dq = hdul['DQ'].data if 'DQ' in hdul else None

                nan_frac = np.sum(~np.isfinite(sci)) / sci.size if sci is not None else 0
                dq_frac = np.sum(dq > 0) / dq.size if dq is not None else 0

                results['HST FLC'].append({
                    'file': f.name[:30],
                    'nan_frac': nan_frac,
                    'dq_frac': dq_frac
                })
        except:
            pass

    # ---- Process HST DRC ----
    for f in sorted(hst_path.glob("hst_*_drc.fits"))[:10]:
        try:
            with fits.open(f) as hdul:
                sci = hdul['SCI'].data if 'SCI' in hdul else hdul[1].data
                nan_frac = np.sum(~np.isfinite(sci)) / sci.size if sci is not None else 0
                results['HST DRC'].append({
                    'file': f.name[:30],
                    'nan_frac': nan_frac,
                    'dq_frac': 0
                })
        except:
            pass

    # ---- Process JWST CAL ----
    for f in sorted(jwst_path.glob("*_cal.fits")):
        try:
            with fits.open(f) as hdul:
                sci = hdul['SCI'].data if 'SCI' in hdul else hdul[1].data
                dq = hdul['DQ'].data if 'DQ' in hdul else None

                nan_frac = np.sum(~np.isfinite(sci)) / sci.size if sci is not None else 0
                dq_frac = np.sum(dq > 0) / dq.size if dq is not None else 0

                results['JWST CAL'].append({
                    'file': f.name[:30],
                    'nan_frac': nan_frac,
                    'dq_frac': dq_frac
                })
        except:
            pass

    # ---- Process JWST S3D ----
    for f in sorted(jwst_path.glob("*_s3d.fits")):
        try:
            with fits.open(f) as hdul:
                sci = hdul['SCI'].data if 'SCI' in hdul else hdul[1].data
                dq = hdul['DQ'].data if 'DQ' in hdul else None

                nan_frac = np.sum(~np.isfinite(sci)) / sci.size if sci is not None else 0
                dq_frac = np.sum(dq > 0) / dq.size if dq is not None else 0

                results['JWST S3D'].append({
                    'file': f.name[:30],
                    'nan_frac': nan_frac,
                    'dq_frac': dq_frac
                })
        except:
            pass

    # ---- Create 2×2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (cat, data) in zip(axes, results.items()):
        if not data:
            ax.text(0.5, 0.5, f'No {cat} data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(cat)
            continue

        files = [d['file'] for d in data]
        nan_fracs = [d['nan_frac'] * 100 for d in data]
        dq_fracs = [d['dq_frac'] * 100 for d in data]

        x = np.arange(len(files))
        width = 0.35

        ax.bar(x - width/2, nan_fracs, width, label='NaN %', alpha=0.8)
        ax.bar(x + width/2, dq_fracs, width, label='DQ flagged %', alpha=0.8)

        ax.set_ylabel('Fraction (%)')
        ax.set_title(cat)
        ax.set_xticks(x)
        ax.set_xticklabels(files, rotation=45, ha='right', fontsize=6)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, max(max(nan_fracs + [1]), max(dq_fracs + [1])) * 1.2)

    plt.suptitle('Data Quality: NaN and DQ Flagged Pixel Fractions', y=1.02)
    plt.tight_layout()
    outpath = output_dir / "06_dq_heatmaps.png"
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 07: INVENTORY BARS
# -----------------------------------------------------------------------------
# Visual summary of file counts by type.
# =============================================================================


def fig_inventory_bars(hst_path: Path, jwst_path: Path, output_dir: Path):
    """
    Stacked bar chart of file counts by type.

    Simple visual confirmation that the inventory matches expectations.
    Complements the validation report with a graphical summary.
    """
    inventory = {
        'HST': {
            'Original FLC': len(list(hst_path.glob("if3x*_flc.fits"))),
            'Original DRC': len(list(hst_path.glob("if3x*_drc.fits"))),
            'HAP FLC': len(list(hst_path.glob("hst_*_flc.fits"))),
            'HAP DRC': len(list(hst_path.glob("hst_*_drc.fits"))),
            'Skycell': len(list(hst_path.glob("hst_skycell*"))),
        },
        'JWST': {
            'CAL': len(list(jwst_path.glob("*_cal.fits"))),
            'S3D': len(list(jwst_path.glob("*_s3d.fits"))),
            'X1D': len(list(jwst_path.glob("*_x1d.fits"))),
        }
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- HST inventory ----
    ax1 = axes[0]
    hst_types = list(inventory['HST'].keys())
    hst_counts = list(inventory['HST'].values())
    colors_hst = sns.color_palette("Blues", len(hst_types))

    bars = ax1.barh(hst_types, hst_counts, color=colors_hst)
    ax1.bar_label(bars, padding=3)
    ax1.set_xlabel('File Count')
    ax1.set_title(f'HST GO-17301\n(Total: {sum(hst_counts)} files)')
    ax1.set_xlim(0, max(hst_counts) * 1.2)

    # ---- JWST inventory ----
    ax2 = axes[1]
    jwst_types = list(inventory['JWST'].keys())
    jwst_counts = list(inventory['JWST'].values())
    colors_jwst = sns.color_palette("Oranges", len(jwst_types))

    bars = ax2.barh(jwst_types, jwst_counts, color=colors_jwst)
    ax2.bar_label(bars, padding=3)
    ax2.set_xlabel('File Count')
    ax2.set_title(f'JWST GO-3149\n(Total: {sum(jwst_counts)} files)')
    ax2.set_xlim(0, max(jwst_counts) * 1.2)

    plt.suptitle('Data Inventory by Product Type', y=1.02)
    plt.tight_layout()
    outpath = output_dir / "07_inventory_bars.png"
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# FIGURE 08: ACQUISITION TIMELINE
# -----------------------------------------------------------------------------
# Shows when observations were taken.
# =============================================================================


def fig_acquisition_timeline(hst_path: Path, jwst_path: Path, output_dir: Path):
    """
    Timeline of observation dates from FITS headers.

    Documents the temporal baseline between HST discovery imaging (2023)
    and JWST spectroscopic follow-up (2024).

    Scientific context: The ~1 year gap theoretically allows proper motion
    detection, but at z=0.964, even 1000 km/s produces negligible angular
    motion over this baseline.
    """
    dates = {'HST': [], 'JWST': []}

    # ---- Extract HST dates ----
    for f in hst_path.glob("hst_*_flc.fits"):
        try:
            with fits.open(f) as hdul:
                h = hdul[0].header
                date_obs = get_header_value(h, 'DATE-OBS', 'TDATEOBS')
                if date_obs:
                    dates['HST'].append(
                        datetime.fromisoformat(date_obs.replace('T', ' ').split()[0]))
        except:
            pass

    # ---- Extract JWST dates ----
    for f in jwst_path.glob("*_cal.fits"):
        try:
            with fits.open(f) as hdul:
                h = hdul[0].header
                date_obs = get_header_value(h, 'DATE-OBS', 'DATE-BEG')
                if date_obs:
                    dates['JWST'].append(
                        datetime.fromisoformat(date_obs.split('T')[0]))
        except:
            pass

    if not dates['HST'] and not dates['JWST']:
        print("  WARNING: No observation dates found")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    # ---- Plot HST dates ----
    if dates['HST']:
        hst_unique = sorted(set(dates['HST']))
        hst_counts = [dates['HST'].count(d) for d in hst_unique]
        ax.scatter(hst_unique, [1] * len(hst_unique),
                   s=[c*30 for c in hst_counts],
                   alpha=0.7, label='HST', c='tab:blue')
        for d, c in zip(hst_unique, hst_counts):
            ax.annotate(f'{c}', (d, 1.1), ha='center', fontsize=8)

    # ---- Plot JWST dates ----
    if dates['JWST']:
        jwst_unique = sorted(set(dates['JWST']))
        jwst_counts = [dates['JWST'].count(d) for d in jwst_unique]
        ax.scatter(jwst_unique, [2] * len(jwst_unique),
                   s=[c*30 for c in jwst_counts],
                   alpha=0.7, label='JWST', c='tab:orange', marker='s')
        for d, c in zip(jwst_unique, jwst_counts):
            ax.annotate(f'{c}', (d, 2.1), ha='center', fontsize=8)

    ax.set_yticks([1, 2])
    ax.set_yticklabels(['HST', 'JWST'])
    ax.set_ylim(0.5, 2.5)
    ax.set_xlabel('Observation Date')
    ax.set_title('Acquisition Timeline (bubble size = file count)')
    ax.legend(loc='upper left')

    fig.autofmt_xdate()

    plt.tight_layout()
    outpath = output_dir / "08_acquisition_timeline.png"
    plt.savefig(outpath)
    plt.close()
    print(f"  Wrote: {outpath}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main entry point for Phase 01 visualization.

    Generates all 8 QA figures and saves to the output directory.
    """
    parser = argparse.ArgumentParser(
        description="Generate Phase 01 data validation visualizations for RBH-1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_path", type=Path,
                        help="Root path containing hst/ and jwst/ subdirectories")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "validation_config.yaml",
                        help="Path to validation config YAML (default: ./validation_config.yaml)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures (default: ./figures/)")

    args = parser.parse_args()

    # ---- Validate paths ----
    if not args.data_path.exists():
        print(f"ERROR: Data path does not exist: {args.data_path}")
        return 1

    hst_path = args.data_path / "hst"
    jwst_path = args.data_path / "jwst"

    if not hst_path.exists() or not jwst_path.exists():
        print(f"ERROR: Expected hst/ and jwst/ subdirectories in {args.data_path}")
        return 1

    # ---- Load configuration ----
    config = load_config(args.config)

    # ---- Setup output directory ----
    output_dir = args.output_dir or Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Generating Phase 01 visualizations...")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {output_dir}")
    print()

    # ---- Generate all figures ----
    print("Figure 1: Pointing sanity scatter...")
    fig_pointing_sanity(hst_path, jwst_path, config, output_dir)

    print("Figure 2: Footprint overlay...")
    fig_footprint_overlay(hst_path, jwst_path, config, output_dir)

    print("Figure 3: Context zoom ladder...")
    fig_context_zoom(hst_path, config, output_dir)

    print("Figure 4: Wavelength coverage...")
    fig_wavelength_coverage(config, output_dir)

    print("Figure 5: Noise vs wavelength...")
    fig_noise_vs_wavelength(jwst_path, output_dir)

    print("Figure 6: DQ/NaN heatmaps...")
    fig_dq_heatmaps(hst_path, jwst_path, output_dir)

    print("Figure 7: Inventory bars...")
    fig_inventory_bars(hst_path, jwst_path, output_dir)

    print("Figure 8: Acquisition timeline...")
    fig_acquisition_timeline(hst_path, jwst_path, output_dir)

    print()
    print(f"Done. {len(list(output_dir.glob('*.png')))} figures written to {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
