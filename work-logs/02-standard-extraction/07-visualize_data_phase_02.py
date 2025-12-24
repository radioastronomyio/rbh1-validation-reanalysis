#!/usr/bin/env python3
"""
Script Name  : 07-visualize_data_phase_02.py
Description  : Generate Phase-02 validation plots from Zone 1 database tables
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-24
Phase        : Phase 02 - Standard Extraction
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Generates validation dashboard from wcs_solutions and spectral_grids tables.
Includes footprint overlay with IFU inset, span regimes scatter, CRVAL offsets,
pixel scale distribution, and spectral grid characterization (Δλ and R_grid).

Usage
-----
    python 07-visualize_data_phase_02.py --outdir <dir> [--show] [--standalone]

Examples
--------
    python 07-visualize_data_phase_02.py --outdir ./phase02_plots
        Generate dashboard PNG to specified directory.

    python 07-visualize_data_phase_02.py --outdir ./phase02_plots --show --standalone
        Generate dashboard plus individual plots, display interactively.
"""

# cspell:ignore dotenv clusteradmin skycell skycells xlabel ylabel ddec scatterplot xscale yscale pixscale histplot Rgrid creds figsize suptitle fontsize inset_axes

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# -----------------------------
# Config
# -----------------------------
DEFAULT_ENV_FILE = "/opt/global-env/research.env"
DEFAULT_DB = "rbh1_validation"


# -----------------------------
# DB helpers
# -----------------------------
def load_credentials(env_file: str, database: str) -> dict[str, str | int]:
    load_dotenv(env_file)
    return {
        "host": os.getenv("PGSQL01_HOST", "10.25.20.8"),
        "port": int(os.getenv("PGSQL01_PORT", "5432")),
        "database": database,
        "user": os.getenv("PGSQL01_ADMIN_USER", "clusteradmin_pg01"),
        "password": os.getenv("PGSQL01_ADMIN_PASSWORD", ""),
    }


def query_df(conn: psycopg2.extensions.connection, sql: str) -> pd.DataFrame:
    # pandas type stubs expect SQLAlchemy connection; psycopg2 works fine at runtime
    return pd.read_sql_query(sql, conn)  # type: ignore[arg-type]


# -----------------------------
# Geometry parsing (simple WKT POLYGON only)
# -----------------------------
def parse_polygon_wkt(wkt: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse WKT like POLYGON((x1 y1,x2 y2,...)) to coordinate arrays.
    
    Returns (xs, ys) as numpy arrays. Only supports simple POLYGON geometry,
    matching the schema constraint (footprint is NOT NULL POLYGON, not MULTI).
    Will raise ValueError for MULTIPOLYGON or other geometry types.
    """
    s = wkt.strip()
    if not s.startswith("POLYGON((") or not s.endswith("))"):
        raise ValueError(f"Unsupported WKT (expected POLYGON): {s[:64]}...")

    inner = s[len("POLYGON((") : -2]
    pts: list[tuple[float, float]] = []
    for pair in inner.split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Malformed coordinate pair in WKT: {pair!r}")
        x_str, y_str = parts[0], parts[1]
        pts.append((float(x_str), float(y_str)))

    xs = np.asarray([p[0] for p in pts], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts], dtype=np.float64)
    return xs, ys


def polygon_bounds_from_wkt(wkt: str) -> Tuple[float, float, float, float]:
    xs, ys = parse_polygon_wkt(wkt)
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


# -----------------------------
# Plot builders
# -----------------------------
def _classify_footprint_row(filename: str, file_type: str) -> str:
    """
    Classify footprint for visualization layering.
    
    Skycell mosaics (hst_skycell-*) are distinguished from individual exposures
    because they have much larger footprints and need different draw styling.
    """
    if filename.startswith("hst_skycell-"):
        return "skycell"
    return file_type


def plot_footprint_overlay_with_inset(ax: Axes, wcs_df: pd.DataFrame) -> None:
    """
    Main footprint overlay plus inset zoom around the JWST s3d union region.
    """
    df = wcs_df.copy()
    df["filename"] = df["filename"].astype(str)
    df["file_type"] = df["file_type"].astype(str)
    df["class"] = [
        _classify_footprint_row(fn, ft) for fn, ft in zip(df["filename"], df["file_type"])
    ]

    # Draw order: large/background first, small/foreground last
    # This ensures IFU footprints (tiny) are visible on top of mosaics (huge)
    order = ["skycell", "flc", "drc", "s3d"]

    def draw(ax_: Axes, subset: pd.DataFrame, emphasize_s3d: bool = True) -> None:
        for cls in order:
            sub = subset[subset["class"] == cls]
            if sub.empty:
                continue
            lw = 2.2 if (cls == "s3d" and emphasize_s3d) else (1.1 if cls in ("drc", "flc") else 0.8)
            alpha = 0.95 if (cls == "s3d" and emphasize_s3d) else (0.7 if cls in ("drc", "flc") else 0.25)

            for _, r in sub.iterrows():
                xs, ys = parse_polygon_wkt(str(r["footprint_wkt"]))
                ax_.plot(xs, ys, linewidth=lw, alpha=alpha)

    # Main overlay
    draw(ax, df, emphasize_s3d=True)
    ax.set_title("WCS Footprint Overlay (Zone 1)")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.invert_xaxis()

    # Compute bounds of JWST s3d union region (approx by bounding boxes)
    s3d = df[df["class"] == "s3d"]
    if s3d.empty:
        return

    min_ra, max_ra, min_dec, max_dec = (np.inf, -np.inf, np.inf, -np.inf)
    for wkt in s3d["footprint_wkt"]:
        a, b, c, d = polygon_bounds_from_wkt(str(wkt))
        min_ra = min(min_ra, a)
        max_ra = max(max_ra, b)
        min_dec = min(min_dec, c)
        max_dec = max(max_dec, d)

    # Add margin (in degrees)
    dra = max_ra - min_ra
    ddec = max_dec - min_dec
    margin_ra = max(dra * 6.0, 0.002)   # generous but not insane
    margin_dec = max(ddec * 6.0, 0.002)

    zmin_ra = min_ra - margin_ra
    zmax_ra = max_ra + margin_ra
    zmin_dec = min_dec - margin_dec
    zmax_dec = max_dec + margin_dec

    axins = inset_axes(ax, width="45%", height="45%", loc="lower left", borderpad=1.1)
    draw(axins, df[df["class"].isin(["flc", "drc", "s3d"])], emphasize_s3d=True)
    axins.set_xlim(zmax_ra, zmin_ra)  # inverted RA
    axins.set_ylim(zmin_dec, zmax_dec)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Zoom: IFU + HST", fontsize=10)


def plot_span_scatter(ax: Axes, spans_df: pd.DataFrame) -> None:
    """dra_arcsec vs ddec_arcsec scatter; log axes + y=x reference."""
    df = spans_df.copy()
    df["filename"] = df["filename"].astype(str)
    df["file_type"] = df["file_type"].astype(str)
    df["is_skycell"] = df["filename"].str.startswith("hst_skycell-")

    sns.scatterplot(
        data=df,
        x="dra_arcsec",
        y="ddec_arcsec",
        hue="file_type",
        style="is_skycell",
        ax=ax,
        s=40,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    # y=x reference line in data range
    lo = float(min(df["dra_arcsec"].min(), df["ddec_arcsec"].min()))
    hi = float(max(df["dra_arcsec"].max(), df["ddec_arcsec"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_title("Footprint Span Regimes (arcsec)")
    ax.set_xlabel("ΔRA span (arcsec)")
    ax.set_ylabel("ΔDec span (arcsec)")


def plot_crval_offsets_arcsec(ax: Axes, wcs_df: pd.DataFrame, annotate_skycells: bool = True) -> None:
    """
    CRVAL offsets (arcsec) relative to median pointing for legibility.
    """
    df = wcs_df.copy()
    df["filename"] = df["filename"].astype(str)
    df["file_type"] = df["file_type"].astype(str)
    df["is_skycell"] = df["filename"].str.startswith("hst_skycell-")

    ra = df["crval1"].to_numpy(dtype=np.float64)
    dec = df["crval2"].to_numpy(dtype=np.float64)

    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))

    # RA arcsec needs cos(dec) scaling to be closer to true angular offsets
    cosdec = float(np.cos(np.deg2rad(dec0)))
    df["dra_arcsec"] = (df["crval1"] - ra0) * 3600.0 * cosdec
    df["ddec_arcsec"] = (df["crval2"] - dec0) * 3600.0

    sns.scatterplot(
        data=df,
        x="dra_arcsec",
        y="ddec_arcsec",
        hue="file_type",
        style="is_skycell",
        ax=ax,
        s=45,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_title("Pointing Coherence (CRVAL offsets)")
    ax.set_xlabel("ΔRA * cos(Dec) (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")

    if annotate_skycells:
        sky = df[df["is_skycell"]]
        for _, r in sky.iterrows():
            # short label
            label = r["filename"].replace("hst_skycell-", "skycell-")
            ax.annotate(label, (r["dra_arcsec"], r["ddec_arcsec"]), xytext=(5, 5), textcoords="offset points", fontsize=8)


def compute_pixscale_arcsec(df: pd.DataFrame) -> pd.Series:
    """
    Approximate pixel scale from CD matrix.
    
    Computes sqrt(CD1_1^2 + CD2_1^2) for x-axis and sqrt(CD1_2^2 + CD2_2^2) for y-axis,
    then averages. This is an approximation that assumes near-orthogonal axes and
    ignores shear/rotation, which is fine for visualization of scale regimes.
    
    Returns a pandas Series of floats in arcsec/pix.
    """
    cd1_1 = df["cd1_1"].to_numpy(dtype=np.float64)
    cd2_1 = df["cd2_1"].to_numpy(dtype=np.float64)
    cd1_2 = df["cd1_2"].to_numpy(dtype=np.float64)
    cd2_2 = df["cd2_2"].to_numpy(dtype=np.float64)

    scale_x = np.hypot(cd1_1, cd2_1) * 3600.0  # type: ignore[operator]
    scale_y = np.hypot(cd1_2, cd2_2) * 3600.0  # type: ignore[operator]
    return pd.Series(0.5 * (scale_x + scale_y), index=df.index, dtype="float64")


def plot_pixel_scale_strip(ax: Axes, wcs_df: pd.DataFrame) -> None:
    """
    Pixel scale visualization: strip plot + median markers.
    """
    df = wcs_df.copy()
    df["file_type"] = df["file_type"].astype(str)
    df["pixscale_arcsec"] = compute_pixscale_arcsec(df)

    sns.stripplot(
        data=df,
        x="file_type",
        y="pixscale_arcsec",
        ax=ax,
        jitter=0.15,
        alpha=0.65,
        size=5,
    )

    # Add median markers per type
    meds = df.groupby("file_type", as_index=False)["pixscale_arcsec"].median()
    for _, r in meds.iterrows():
        ax.plot([r["file_type"]], [r["pixscale_arcsec"]], marker="D", markersize=7, alpha=0.95)

    ax.set_title("Pixel Scale by File Type")
    ax.set_xlabel("file_type")
    ax.set_ylabel("Approx pixel scale (arcsec/pix)")


def plot_dlambda_vs_lambda(ax: Axes, wave_um: np.ndarray) -> None:
    dl = np.diff(wave_um)
    lam = wave_um[1:]
    ax.plot(lam, dl, linewidth=2.0)
    ax.set_title("JWST s3d Grid Spacing Δλ")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Δλ (µm)")


def plot_Rgrid_vs_lambda(ax: Axes, wave_um: np.ndarray) -> None:
    dl = np.diff(wave_um)
    lam = wave_um[1:]
    R = lam / dl
    ax.plot(lam, R, linewidth=2.0)
    ax.set_title("Sampling R = λ/Δλ (proxy)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("R_grid (sampling proxy)")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=DEFAULT_ENV_FILE, help="Path to env file with DB creds")
    ap.add_argument("--db", default=DEFAULT_DB, help="Database name")
    ap.add_argument("--outdir", default="./phase02_plots", help="Output directory")
    ap.add_argument("--format", default="png", choices=["png", "pdf"], help="Output format")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    ap.add_argument("--standalone", action="store_true", help="Also write standalone plots (in addition to dashboard)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    creds = load_credentials(args.env, args.db)
    conn = psycopg2.connect(**creds)  # type: ignore[arg-type]

    wcs_sql = """
    SELECT o.filename, o.file_type, o.instrument,
           w.crval1, w.crval2,
           w.cd1_1, w.cd1_2, w.cd2_1, w.cd2_2,
           ST_AsText(w.footprint) AS footprint_wkt
    FROM rbh1.wcs_solutions w
    JOIN rbh1.observations o ON o.obs_id = w.obs_id;
    """

    spans_sql = """
    SELECT o.filename, o.file_type,
           (ST_XMax(w.footprint)-ST_XMin(w.footprint))*3600 AS dra_arcsec,
           (ST_YMax(w.footprint)-ST_YMin(w.footprint))*3600 AS ddec_arcsec
    FROM rbh1.wcs_solutions w
    JOIN rbh1.observations o ON o.obs_id = w.obs_id;
    """

    spectral_sql = """
    SELECT g.wavelength_array
    FROM rbh1.spectral_grids g
    JOIN rbh1.wcs_solutions w ON w.wcs_id = g.wcs_id
    JOIN rbh1.observations o ON o.obs_id = w.obs_id
    WHERE o.file_type = 's3d'
    ORDER BY o.filename
    LIMIT 1;
    """

    wcs_df = query_df(conn, wcs_sql)
    spans_df = query_df(conn, spans_sql)
    spec_df = query_df(conn, spectral_sql)

    conn.close()

    if spec_df.empty:
        raise RuntimeError("No spectral_grids row found for s3d; run Phase-02 ETL first.")

    wave_um = np.asarray(spec_df.iloc[0]["wavelength_array"], dtype=np.float64)

    print(f"WCS rows: {len(wcs_df)}")
    print(f"Span rows: {len(spans_df)}")
    print(f"Spectral channels: {wave_um.size}, λmin={wave_um.min():.6f} µm, λmax={wave_um.max():.6f} µm")

    sns.set_theme(context="talk", style="ticks")

    # AI NOTE: Dashboard layout - 2x3 grid
    # Row 0: [0,0] Footprints + inset | [0,1] Span scatter | [0,2] CRVAL offsets
    # Row 1: [1,0] Pixel scale strip  | [1,1] Δλ vs λ     | [1,2] R_grid vs λ
    # If adding panels, update figsize and consider constrained_layout behavior.
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

    plot_footprint_overlay_with_inset(axes[0, 0], wcs_df)
    plot_span_scatter(axes[0, 1], spans_df)
    plot_crval_offsets_arcsec(axes[0, 2], wcs_df, annotate_skycells=True)

    plot_pixel_scale_strip(axes[1, 0], wcs_df)
    plot_dlambda_vs_lambda(axes[1, 1], wave_um)
    plot_Rgrid_vs_lambda(axes[1, 2], wave_um)

    fig.suptitle("RBH-1 Phase 02 Validation (Zone 1: WCS + Spectral Grids)", y=1.02, fontsize=18)
    dash_path = outdir / f"phase02_validation_dashboard.{args.format}"
    fig.savefig(dash_path, dpi=200)
    print(f"Wrote: {dash_path}")

    if args.standalone:
        fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
        plot_footprint_overlay_with_inset(ax2, wcs_df)
        p = outdir / f"phase02_footprints_inset.{args.format}"
        fig2.savefig(p, dpi=200)
        print(f"Wrote: {p}")

        fig3, ax3 = plt.subplots(figsize=(10, 8), constrained_layout=True)
        plot_span_scatter(ax3, spans_df)
        p = outdir / f"phase02_span_regimes.{args.format}"
        fig3.savefig(p, dpi=200)
        print(f"Wrote: {p}")

        fig4, ax4 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        plot_dlambda_vs_lambda(ax4, wave_um)
        p = outdir / f"phase02_dlambda_vs_lambda.{args.format}"
        fig4.savefig(p, dpi=200)
        print(f"Wrote: {p}")

        fig5, ax5 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        plot_Rgrid_vs_lambda(ax5, wave_um)
        p = outdir / f"phase02_Rgrid_vs_lambda.{args.format}"
        fig5.savefig(p, dpi=200)
        print(f"Wrote: {p}")

        fig6, ax6 = plt.subplots(figsize=(10, 6), constrained_layout=True)
        plot_pixel_scale_strip(ax6, wcs_df)
        p = outdir / f"phase02_pixel_scale_strip.{args.format}"
        fig6.savefig(p, dpi=200)
        print(f"Wrote: {p}")

        fig7, ax7 = plt.subplots(figsize=(10, 6), constrained_layout=True)
        plot_crval_offsets_arcsec(ax7, wcs_df, annotate_skycells=True)
        p = outdir / f"phase02_crval_offsets.{args.format}"
        fig7.savefig(p, dpi=200)
        print(f"Wrote: {p}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
