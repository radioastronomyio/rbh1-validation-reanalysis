# RBH-1 FITS Data Dictionary

Generated: 2025-12-24T04:48:47.864772+00:00
Data Path: `/mnt/ai-ml/data`

---

## File Type Summary

| Type | Description | Count | HDUs |
|------|-------------|-------|------|
| hst_original_flc | HST WFC3/UVIS flat-fielded CTE-corrected (original pipeline) | 36 | PRIMARY, SCI, ERR, DQ, SCI (+10) |
| hst_hap_flc | HST WFC3/UVIS flat-fielded CTE-corrected (HAP reprocessed) | 36 | PRIMARY, SCI, ERR, DQ, SCI (+11) |
| hst_original_drc | HST WFC3/UVIS drizzle-combined (original pipeline) | 12 | PRIMARY, SCI, WHT, CTX, HDRTAB |
| hst_hap_drc | HST WFC3/UVIS drizzle-combined (HAP reprocessed) | 54 | PRIMARY, SCI, WHT, CTX, HDRTAB |
| hst_skycell | HST HAP skycell mosaic | 4 | PRIMARY, SCI, WHT, CTX, HDRTAB |
| jwst_cal | JWST NIRSpec IFU calibrated 2D spectra | 16 | PRIMARY, SCI, ERR, DQ, REGIONS (+7) |
| jwst_s3d | JWST NIRSpec IFU reconstructed 3D spectral cube | 18 | PRIMARY, SCI, ERR, DQ, WMAP (+2) |
| jwst_x1d | JWST NIRSpec IFU extracted 1D spectrum | 18 | PRIMARY, EXTRACT1D, ASDF |

---

## hst_original_flc

Description: HST WFC3/UVIS flat-fielded CTE-corrected (original pipeline)
Pattern: `if3x*_flc.fits`
Total Files: 36

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [2051, 4096] |
| 2 | ERR | ImageHDU | [2051, 4096] |
| 3 | DQ | ImageHDU | [2051, 4096] |
| 4 | SCI | ImageHDU | [2051, 4096] |
| 5 | ERR | ImageHDU | [2051, 4096] |
| 6 | DQ | ImageHDU | [2051, 4096] |
| 7 | D2IMARR | ImageHDU | [32, 64] |
| 8 | D2IMARR | ImageHDU | [32, 64] |
| 9 | D2IMARR | ImageHDU | [32, 64] |
| 10 | D2IMARR | ImageHDU | [32, 64] |
| 11 | HDRLET | NonstandardExtHDU | [60480] |
| 12 | WCSCORR | BinTableHDU | 14 rows × 24 cols |
| 13 | HDRLET | NonstandardExtHDU | [60480] |
| 14 | HDRLET | NonstandardExtHDU | [63360] |

### Key Header Keywords

### Table: WCSCORR

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| WCS_ID | 40A | - | - |
| EXTVER | I | - | 0 - 2 |
| WCS_key | A | - | - |
| HDRNAME | 24A | - | - |
| SIPNAME | 24A | - | - |
| NPOLNAME | 24A | - | - |
| D2IMNAME | 24A | - | - |
| CRVAL1 | D | - | 0 - 40.46 |
| CRVAL2 | D | - | -8.353 - 0 |
| CRPIX1 | D | - | 0 - 2048 |
| CRPIX2 | D | - | 0 - 1026 |
| CD1_1 | D | - | 2.277e-06 - 1 |
| CD1_2 | D | - | 0 - 1.095e-05 |
| CD2_1 | D | - | 0 - 1.083e-05 |
| CD2_2 | D | - | -1.592e-06 - 1 |
| CTYPE1 | 24A | - | - |
| CTYPE2 | 24A | - | - |
| ORIENTAT | D | - | 0 - 0 |
| PA_V3 | D | - | 0 - 0 |
| RMS_RA | D | - | 0 - 0 |
| ... | ... | ... | *4 more columns* |

---

## hst_hap_flc

Description: HST WFC3/UVIS flat-fielded CTE-corrected (HAP reprocessed)
Pattern: `hst_*_flc.fits`
Total Files: 36

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [2051, 4096] |
| 2 | ERR | ImageHDU | [2051, 4096] |
| 3 | DQ | ImageHDU | [2051, 4096] |
| 4 | SCI | ImageHDU | [2051, 4096] |
| 5 | ERR | ImageHDU | [2051, 4096] |
| 6 | DQ | ImageHDU | [2051, 4096] |
| 7 | D2IMARR | ImageHDU | [32, 64] |
| 8 | D2IMARR | ImageHDU | [32, 64] |
| 9 | D2IMARR | ImageHDU | [32, 64] |
| 10 | D2IMARR | ImageHDU | [32, 64] |
| 11 | HDRLET | NonstandardExtHDU | [60480] |
| 12 | WCSCORR | BinTableHDU | 14 rows × 24 cols |
| 13 | HDRLET | NonstandardExtHDU | [60480] |
| 14 | HDRLET | NonstandardExtHDU | [63360] |
| 15 | HDRLET | NonstandardExtHDU | [63360] |

### Key Header Keywords

### Table: WCSCORR

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| WCS_ID | 40A | - | - |
| EXTVER | I | - | 0 - 2 |
| WCS_key | A | - | - |
| HDRNAME | 24A | - | - |
| SIPNAME | 24A | - | - |
| NPOLNAME | 24A | - | - |
| D2IMNAME | 24A | - | - |
| CRVAL1 | D | - | 0 - 40.46 |
| CRVAL2 | D | - | -8.353 - 0 |
| CRPIX1 | D | - | 0 - 2048 |
| CRPIX2 | D | - | 0 - 1026 |
| CD1_1 | D | - | 2.277e-06 - 1 |
| CD1_2 | D | - | 0 - 1.095e-05 |
| CD2_1 | D | - | 0 - 1.083e-05 |
| CD2_2 | D | - | -1.592e-06 - 1 |
| CTYPE1 | 24A | - | - |
| CTYPE2 | 24A | - | - |
| ORIENTAT | D | - | 0 - 0 |
| PA_V3 | D | - | 0 - 0 |
| RMS_RA | D | - | 0 - 0 |
| ... | ... | ... | *4 more columns* |

---

## hst_original_drc

Description: HST WFC3/UVIS drizzle-combined (original pipeline)
Pattern: `if3x*_drc.fits`
Total Files: 12

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [4389, 4129] |
| 2 | WHT | ImageHDU | [4389, 4129] |
| 3 | CTX | ImageHDU | [4389, 4129] |
| 4 | HDRTAB | BinTableHDU | 6 rows × 276 cols |

### Key Header Keywords

### Table: HDRTAB

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| ROOTNAME | 9A | - | - |
| EXTNAME | 3A | - | - |
| EXTVER | K | - | 1 - 2 |
| A_0_2 | D | - | -1.065e-08 - 9.462e-08 |
| A_0_3 | D | - | 2.36e-12 - 1.904e-11 |
| A_0_4 | D | - | -1.637e-14 - 5.077e-15 |
| A_1_1 | D | - | -2.971e-06 - -2.921e-06 |
| A_1_2 | D | - | 1.853e-11 - 1.993e-11 |
| A_1_3 | D | - | 3.091e-15 - 1.599e-14 |
| A_2_0 | D | - | 2.844e-06 - 2.874e-06 |
| A_2_1 | D | - | -1.559e-11 - 5.3e-12 |
| A_2_2 | D | - | -1.781e-14 - -1.269e-16 |
| A_3_0 | D | - | 7.749e-12 - 2.042e-11 |
| A_3_1 | D | - | -8.389e-16 - 7.694e-15 |
| A_4_0 | D | - | 5.349e-16 - 2.976e-15 |
| A_ORDER | K | - | 4 - 4 |
| APERTURE | 5A | - | - |
| ASN_ID | 9A | - | - |
| ASN_MTYP | 7A | - | - |
| ASN_TAB | 18A | - | - |
| ... | ... | ... | *256 more columns* |

---

## hst_hap_drc

Description: HST WFC3/UVIS drizzle-combined (HAP reprocessed)
Pattern: `hst_17301*_drc.fits`
Total Files: 54

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [4733, 4951] |
| 2 | WHT | ImageHDU | [4733, 4951] |
| 3 | CTX | ImageHDU | [4733, 4951] |
| 4 | HDRTAB | BinTableHDU | 6 rows × 277 cols |

### Key Header Keywords

### Table: HDRTAB

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| ROOTNAME | 9A | - | - |
| EXTNAME | 3A | - | - |
| EXTVER | K | - | 1 - 2 |
| A_0_2 | D | - | -1.065e-08 - 9.462e-08 |
| A_0_3 | D | - | 2.36e-12 - 1.904e-11 |
| A_0_4 | D | - | -1.637e-14 - 5.077e-15 |
| A_1_1 | D | - | -2.971e-06 - -2.921e-06 |
| A_1_2 | D | - | 1.853e-11 - 1.993e-11 |
| A_1_3 | D | - | 3.091e-15 - 1.599e-14 |
| A_2_0 | D | - | 2.844e-06 - 2.874e-06 |
| A_2_1 | D | - | -1.559e-11 - 5.3e-12 |
| A_2_2 | D | - | -1.781e-14 - -1.269e-16 |
| A_3_0 | D | - | 7.749e-12 - 2.042e-11 |
| A_3_1 | D | - | -8.389e-16 - 7.694e-15 |
| A_4_0 | D | - | 5.349e-16 - 2.976e-15 |
| A_ORDER | K | - | 4 - 4 |
| APERTURE | 5A | - | - |
| ASN_ID | 9A | - | - |
| ASN_MTYP | 7A | - | - |
| ASN_TAB | 18A | - | - |
| ... | ... | ... | *257 more columns* |

---

## hst_skycell

Description: HST HAP skycell mosaic
Pattern: `hst_skycell*_drc.fits`
Total Files: 4

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [18398, 21953] |
| 2 | WHT | ImageHDU | [18398, 21953] |
| 3 | CTX | ImageHDU | - |
| 4 | HDRTAB | BinTableHDU | 24 rows × 277 cols |

### Key Header Keywords

### Table: HDRTAB

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| ROOTNAME | 9A | - | - |
| EXTNAME | 3A | - | - |
| EXTVER | K | - | 1 - 2 |
| A_0_2 | D | - | -1.065e-08 - 9.462e-08 |
| A_0_3 | D | - | 2.36e-12 - 1.904e-11 |
| A_0_4 | D | - | -1.637e-14 - 5.077e-15 |
| A_1_1 | D | - | -2.971e-06 - -2.921e-06 |
| A_1_2 | D | - | 1.853e-11 - 1.993e-11 |
| A_1_3 | D | - | 3.091e-15 - 1.599e-14 |
| A_2_0 | D | - | 2.844e-06 - 2.874e-06 |
| A_2_1 | D | - | -1.559e-11 - 5.3e-12 |
| A_2_2 | D | - | -1.781e-14 - -1.269e-16 |
| A_3_0 | D | - | 7.749e-12 - 2.042e-11 |
| A_3_1 | D | - | -8.389e-16 - 7.694e-15 |
| A_4_0 | D | - | 5.349e-16 - 2.976e-15 |
| A_ORDER | K | - | 4 - 4 |
| APERTURE | 5A | - | - |
| ASN_ID | 9A | - | - |
| ASN_MTYP | 7A | - | - |
| ASN_TAB | 18A | - | - |
| ... | ... | ... | *257 more columns* |

---

## jwst_cal

Description: JWST NIRSpec IFU calibrated 2D spectra
Pattern: `*_cal.fits`
Total Files: 16

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [2048, 2048] |
| 2 | ERR | ImageHDU | [2048, 2048] |
| 3 | DQ | ImageHDU | [2048, 2048] |
| 4 | REGIONS | ImageHDU | [2048, 2048] |
| 5 | VAR_POISSON | ImageHDU | [2048, 2048] |
| 6 | VAR_RNOISE | ImageHDU | [2048, 2048] |
| 7 | VAR_FLAT | ImageHDU | [2048, 2048] |
| 8 | WAVELENGTH | ImageHDU | [2048, 2048] |
| 9 | PATHLOSS_PS | ImageHDU | [2048, 2048] |
| 10 | PATHLOSS_UN | ImageHDU | [2048, 2048] |
| 11 | ASDF | BinTableHDU | 1 rows × 1 cols |

### Key Header Keywords

### Table: ASDF

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| ASDF_METADATA | 35197319B | - | - |

---

## jwst_s3d

Description: JWST NIRSpec IFU reconstructed 3D spectral cube
Pattern: `*_s3d.fits`
Total Files: 18

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | SCI | ImageHDU | [1447, 53, 49] |
| 2 | ERR | ImageHDU | [1447, 53, 49] |
| 3 | DQ | ImageHDU | [1447, 53, 49] |
| 4 | WMAP | ImageHDU | [1447, 53, 49] |
| 5 | HDRTAB | BinTableHDU | 4 rows × 250 cols |
| 6 | ASDF | BinTableHDU | 1 rows × 1 cols |

### Wavelength Coverage

- Source: wcs_cube
- Channels: 1447
- Range: 0.9703 - 1.8900 um
- Range (Å): 9703.2 - 18899.7 Å

### Key Header Keywords

### Table: HDRTAB

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| DATE | 23A | - | - |
| ORIGIN | 5A | - | - |
| TIMESYS | 3A | - | - |
| FILENAME | 39A | - | - |
| SDP_VER | 7A | - | - |
| PRD_VER | 13A | - | - |
| OSS_VER | 3A | - | - |
| GSC_VER | 5A | - | - |
| CAL_VER | 6A | - | - |
| CAL_VCS | 7A | - | - |
| DATAMODL | 13A | - | - |
| TELESCOP | 4A | - | - |
| HGA_MOVE | L | - | - |
| PWFSEET | D | - | 6.051e+04 - 6.051e+04 |
| NWFSEST | D | - | 6.052e+04 - 6.052e+04 |
| ASNPOOL | 32A | - | - |
| ASNTABLE | 49A | - | - |
| TITLE | 85A | - | - |
| PI_NAME | 18A | - | - |
| CATEGORY | 2A | - | - |
| ... | ... | ... | *230 more columns* |

### Table: ASDF

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| ASDF_METADATA | 93378B | - | - |

---

## jwst_x1d

Description: JWST NIRSpec IFU extracted 1D spectrum
Pattern: `*_x1d.fits`
Total Files: 18

### HDU Structure

| Index | Name | Type | Shape/Rows |
|-------|------|------|------------|
| 0 | PRIMARY | PrimaryHDU | - |
| 1 | EXTRACT1D | BinTableHDU | 1447 rows × 18 cols |
| 2 | ASDF | BinTableHDU | 1 rows × 1 cols |

### Wavelength Coverage

- Source: table_column
- Channels: 1447
- Range: 0.9703 - 1.8900 um

### Key Header Keywords

### Table: EXTRACT1D

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| WAVELENGTH | D | um | 0.9703 - 1.89 |
| FLUX | D | Jy | -0.03756 - 0.0003047 |
| FLUX_ERROR | D | Jy | 1.119e-06 - 0.0007909 |
| FLUX_VAR_POISSON | D | Jy^2 | 0 - 0 |
| FLUX_VAR_RNOISE | D | Jy^2 | 0 - 0 |
| FLUX_VAR_FLAT | D | Jy^2 | 0 - 0 |
| SURF_BRIGHT | D | MJy/sr | -114.7 - 0.9292 |
| SB_ERROR | D | MJy/sr | 0.003417 - 2.416 |
| SB_VAR_POISSON | D | (MJy/sr)^2 | 0 - 0 |
| SB_VAR_RNOISE | D | (MJy/sr)^2 | 0 - 0 |
| SB_VAR_FLAT | D | (MJy/sr)^2 | 0 - 0 |
| DQ | J | - | 0 - 0 |
| BACKGROUND | D | MJy/sr | 0.3395 - 2.214 |
| BKGD_ERROR | D | MJy/sr | 0.001557 - 0.009007 |
| BKGD_VAR_POISSON | D | (MJy/sr)^2 | 0 - 0 |
| BKGD_VAR_RNOISE | D | (MJy/sr)^2 | 0 - 0 |
| BKGD_VAR_FLAT | D | (MJy/sr)^2 | 0 - 0 |
| NPIXELS | D | - | 1291 - 1406 |

### Table: ASDF

| Column | Format | Unit | Sample Range |
|--------|--------|------|--------------|
| ASDF_METADATA | 100257B | - | - |

---
