-- =============================================================================
-- RBH-1 Validation Database Schema
-- PostgreSQL 16 DDL for Analysis-Ready Dataset
-- =============================================================================
-- 
-- Schema: rbh1
-- Target: pgsql01 (research data lake)
-- 
-- Architecture: Four-zone scalar materialization
--   Zone 0: Observation metadata (provenance anchor)
--   Zone 1: WCS solutions (geometric materialization)
--   Zone 2: Spaxel data (spectral materialization)
--   Zone 3: Derived properties (physical materialization)
--
-- Design Sources:
--   - GDR schema research (2024-12-24)
--   - MCMC storage research (DESI/SDSS patterns)
--   - GPT5.2 review recommendations
--
-- Note: Partitioning removed from observations table. For RBH-1's ~180 files,
-- partitioning adds FK complexity without meaningful performance benefit.
-- Instrument-based filtering uses standard B-tree index instead.
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- EXTENSIONS AND SCHEMA SETUP
-- -----------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS postgis;        -- Spatial indexing
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";    -- Unique identifiers
CREATE EXTENSION IF NOT EXISTS btree_gin;      -- GIN indexing on scalars

CREATE SCHEMA IF NOT EXISTS rbh1;
SET search_path TO rbh1, public;

-- -----------------------------------------------------------------------------
-- ENUMERATED TYPES
-- -----------------------------------------------------------------------------

CREATE TYPE instrument_type AS ENUM (
    'HST_WFC3',
    'JWST_NIRSPEC'
);

CREATE TYPE data_level AS ENUM (
    'RAW',
    'CAL',
    'RESAMPLED',
    'MOSAIC'
);

CREATE TYPE geometric_region AS ENUM (
    'CORE',
    'WAKE',
    'BOW_SHOCK',
    'BACKGROUND',
    'HOST_GALAXY'
);

CREATE TYPE artifact_type AS ENUM (
    'MCMC_CHAIN',
    'COVARIANCE_MATRIX',
    'SYNTHETIC_CUBE',
    'DIFFERENCE_IMAGE',
    'MASK'
);

CREATE TYPE quality_flag AS ENUM (
    'GOOD',
    'WARNING',
    'BAD',
    'REJECTED'
);

-- =============================================================================
-- ZONE 0: OBSERVATION METADATA (PROVENANCE ANCHOR)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Pipeline Runs (provenance tracking)
-- -----------------------------------------------------------------------------

CREATE TABLE pipeline_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Identification
    run_name VARCHAR(128) NOT NULL,
    stage_name VARCHAR(64) NOT NULL,           -- e.g., 'phase-02-etl', 'phase-04-fitting'
    
    -- Versioning
    pipeline_version VARCHAR(64) NOT NULL,     -- Semantic version or git hash
    config_hash CHAR(64),                      -- SHA-256 of config file
    
    -- Environment
    python_version VARCHAR(32),
    key_packages JSONB,                        -- {'astropy': '6.0', 'emcee': '3.1', ...}
    
    -- Execution
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(32) DEFAULT 'RUNNING',      -- RUNNING, COMPLETED, FAILED
    
    -- Artifacts
    log_file_path VARCHAR(512),
    config_file_path VARCHAR(512),
    
    notes TEXT
);

CREATE INDEX idx_pipeline_runs_stage ON pipeline_runs(stage_name);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);

-- -----------------------------------------------------------------------------
-- Observations (root of all data)
-- -----------------------------------------------------------------------------

CREATE TABLE observations (
    obs_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Program identification
    program_id VARCHAR(32) NOT NULL,           -- e.g., 'GO-17301', 'GO-3149'
    instrument instrument_type NOT NULL,
    target_name VARCHAR(64) NOT NULL DEFAULT 'RBH-1',
    
    -- Temporal anchors
    date_obs TIMESTAMP WITH TIME ZONE NOT NULL,
    exp_time FLOAT NOT NULL CHECK (exp_time > 0),
    
    -- Configuration
    filter VARCHAR(32) NOT NULL,               -- e.g., 'F200LP', 'F100LP'
    disperser VARCHAR(32),                     -- e.g., 'G140M', NULL for imaging
    aperture VARCHAR(32),                      -- e.g., 'NRS_FULL_MSA', 'UVIS-CENTER'
    data_level data_level NOT NULL DEFAULT 'CAL',
    
    -- File management
    filename VARCHAR(255) NOT NULL UNIQUE,
    file_type VARCHAR(32) NOT NULL,            -- e.g., 's3d', 'flc', 'x1d', 'drc'
    file_path VARCHAR(512) NOT NULL,           -- Relative to data root
    file_size_bytes BIGINT,
    checksum_md5 CHAR(32),
    
    -- Calibration provenance
    crds_context VARCHAR(64),                  -- JWST calibration reference context
    cal_version VARCHAR(64),                   -- Pipeline calibration version
    
    -- Flexible metadata (instrument-specific headers)
    header_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Tracking
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ingested_by_run UUID REFERENCES pipeline_runs(run_id)
);

-- Indexes (instrument index replaces partitioning for filtering)
CREATE INDEX idx_obs_instrument ON observations(instrument);
CREATE INDEX idx_obs_program ON observations(program_id);
CREATE INDEX idx_obs_config ON observations(instrument, filter, disperser);
CREATE INDEX idx_obs_date ON observations(date_obs);
CREATE INDEX idx_obs_header ON observations USING GIN (header_json);

-- =============================================================================
-- ZONE 1: WCS SOLUTIONS (GEOMETRIC MATERIALIZATION)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- WCS Solutions (spatial anchors)
-- -----------------------------------------------------------------------------

CREATE TABLE wcs_solutions (
    wcs_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    obs_id UUID NOT NULL UNIQUE REFERENCES observations(obs_id) ON DELETE CASCADE,
    
    -- Spatial WCS (standard FITS keywords)
    crpix1 FLOAT NOT NULL,
    crpix2 FLOAT NOT NULL,
    crval1 FLOAT NOT NULL,                     -- RA reference (degrees)
    crval2 FLOAT NOT NULL,                     -- Dec reference (degrees)
    
    -- CD matrix (linear transformation + rotation)
    cd1_1 FLOAT NOT NULL,
    cd1_2 FLOAT NOT NULL,
    cd2_1 FLOAT NOT NULL,
    cd2_2 FLOAT NOT NULL,
    
    -- Projection type
    ctype1 VARCHAR(16) NOT NULL DEFAULT 'RA---TAN',
    ctype2 VARCHAR(16) NOT NULL DEFAULT 'DEC--TAN',
    
    -- Spatial footprint (PostGIS polygon for spatial queries)
    footprint GEOMETRY(POLYGON, 4326) NOT NULL,
    
    -- Spectral WCS metadata (for 3D data)
    has_spectral_axis BOOLEAN DEFAULT FALSE,
    wavelength_unit VARCHAR(16) DEFAULT 'um',
    
    -- Dimensions
    naxis1 INT,
    naxis2 INT,
    naxis3 INT                                 -- Spectral dimension for cubes
);

CREATE INDEX idx_wcs_obs ON wcs_solutions(obs_id);
CREATE INDEX idx_wcs_footprint ON wcs_solutions USING GIST (footprint);

-- -----------------------------------------------------------------------------
-- Spectral Grids (wavelength materialization for GWA tilt correction)
-- -----------------------------------------------------------------------------

CREATE TABLE spectral_grids (
    grid_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wcs_id UUID NOT NULL REFERENCES wcs_solutions(wcs_id) ON DELETE CASCADE,
    
    -- Materialized wavelength array (absorbs GWA tilt variation)
    -- Extracted from WCS-TABLE extension of s3d files
    wavelength_array FLOAT[] NOT NULL,
    
    -- Grid statistics
    n_channels INT NOT NULL,
    min_wavelength FLOAT NOT NULL,
    max_wavelength FLOAT NOT NULL,
    mean_resolution FLOAT,                     -- Mean spectral resolution (R)
    
    -- Reference wavelength for velocity calculations
    systemic_redshift FLOAT DEFAULT 0.964     -- RBH-1 systemic z
);

CREATE INDEX idx_spectral_grid_wcs ON spectral_grids(wcs_id);

-- =============================================================================
-- ZONE 2: SPAXEL DATA (SPECTRAL MATERIALIZATION)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- IFU Spaxels (decomposed cube data)
-- -----------------------------------------------------------------------------

CREATE TABLE ifu_spaxels (
    spaxel_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    obs_id UUID NOT NULL REFERENCES observations(obs_id) ON DELETE CASCADE,
    grid_id UUID REFERENCES spectral_grids(grid_id),
    
    -- Spatial coordinates (pre-computed from WCS)
    sky_location GEOMETRY(POINT, 4326) NOT NULL,
    ra_deg FLOAT NOT NULL,
    dec_deg FLOAT NOT NULL,
    
    -- Detector coordinates (provenance for artifact tracing)
    pixel_x INT NOT NULL,
    pixel_y INT NOT NULL,
    
    -- Spectral data (TOAST-compressed arrays)
    flux_spectrum FLOAT[] NOT NULL,            -- Units: MJy/sr
    error_spectrum FLOAT[] NOT NULL,           -- Units: MJy/sr
    dq_spectrum INT[],                         -- Data quality bitmask per channel
    
    -- Materialized integrated properties
    total_flux FLOAT,                          -- Integrated flux
    median_snr FLOAT,                          -- Median signal-to-noise
    
    -- Geometric classification
    region geometric_region,
    dist_along_feature_kpc FLOAT,              -- Distance along wake axis
    dist_perp_feature_kpc FLOAT,               -- Perpendicular distance from axis
    
    -- Quality
    is_valid BOOLEAN DEFAULT TRUE,
    quality_notes TEXT
);

CREATE INDEX idx_spaxel_obs ON ifu_spaxels(obs_id);
CREATE INDEX idx_spaxel_loc ON ifu_spaxels USING GIST (sky_location);
CREATE INDEX idx_spaxel_region ON ifu_spaxels(region);
CREATE INDEX idx_spaxel_valid ON ifu_spaxels(is_valid) WHERE is_valid = TRUE;

-- -----------------------------------------------------------------------------
-- Regions of Interest (spatial gates)
-- -----------------------------------------------------------------------------

CREATE TABLE regions_of_interest (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(64) UNIQUE NOT NULL,
    region_type geometric_region NOT NULL,
    
    -- Geometric definition
    boundary GEOMETRY(POLYGON, 4326) NOT NULL,
    
    -- Metadata
    description TEXT,
    defined_by VARCHAR(128),                   -- Source of region definition
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_roi_boundary ON regions_of_interest USING GIST (boundary);

-- =============================================================================
-- ZONE 3: DERIVED PROPERTIES (PHYSICAL MATERIALIZATION)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Emission Line Measurements
-- -----------------------------------------------------------------------------

CREATE TABLE emission_lines (
    line_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    spaxel_id UUID NOT NULL REFERENCES ifu_spaxels(spaxel_id) ON DELETE CASCADE,
    
    -- Line identification
    line_name VARCHAR(32) NOT NULL,            -- e.g., 'OIII_5007', 'Ha_6563', 'Hb_4861'
    rest_wavelength_um FLOAT NOT NULL,
    
    -- Gaussian fit parameters
    flux FLOAT NOT NULL,                       -- Integrated flux (erg/s/cm^2)
    flux_err FLOAT NOT NULL,
    velocity FLOAT NOT NULL,                   -- km/s relative to systemic z=0.964
    velocity_err FLOAT NOT NULL,
    sigma FLOAT NOT NULL,                      -- Velocity dispersion (km/s)
    sigma_err FLOAT NOT NULL,
    
    -- Equivalent width
    ew FLOAT,                                  -- Equivalent width (Angstroms)
    ew_err FLOAT,
    
    -- Higher moments (if fitted)
    h3 FLOAT,                                  -- Skewness
    h4 FLOAT,                                  -- Kurtosis
    
    -- Quality gates
    snr FLOAT NOT NULL,                        -- Signal-to-noise ratio
    quality_flag INT DEFAULT 0,                -- 0=good, 1=low_sn, 2=bad_fit, 3=edge
    fit_chi2 FLOAT,
    fit_dof INT,
    
    -- Provenance
    fitted_by_run UUID REFERENCES pipeline_runs(run_id),
    
    UNIQUE(spaxel_id, line_name)
);

CREATE INDEX idx_line_spaxel ON emission_lines(spaxel_id);
CREATE INDEX idx_line_name ON emission_lines(line_name);
CREATE INDEX idx_line_quality ON emission_lines(quality_flag) WHERE quality_flag = 0;
CREATE INDEX idx_line_snr ON emission_lines(snr);

-- -----------------------------------------------------------------------------
-- Physical Properties (inference results)
-- -----------------------------------------------------------------------------

CREATE TABLE physical_properties (
    prop_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    spaxel_id UUID NOT NULL REFERENCES ifu_spaxels(spaxel_id) ON DELETE CASCADE,
    
    -- Geometric anchors
    dist_from_shock_tip_kpc FLOAT,
    
    -- Kinematic properties
    shock_velocity FLOAT,                      -- Inferred shock velocity (km/s)
    shock_velocity_err FLOAT,
    mach_number FLOAT,
    
    -- Chemical properties
    metallicity_12_log_oh FLOAT,               -- 12 + log(O/H)
    metallicity_err FLOAT,
    metallicity_method VARCHAR(32),            -- e.g., 'O3N2', 'R23', 'N2'
    
    -- Ionization properties
    ionization_parameter_log_u FLOAT,
    electron_density FLOAT,                    -- cm^-3
    electron_temperature FLOAT,                -- K
    
    -- Hypothesis testing (falsification gate)
    ln_evidence_shock FLOAT,
    ln_evidence_galaxy FLOAT,
    bayes_factor FLOAT GENERATED ALWAYS AS (ln_evidence_shock - ln_evidence_galaxy) STORED,
    
    -- Provenance
    inferred_by_run UUID REFERENCES pipeline_runs(run_id),
    model_version VARCHAR(64),                 -- MAPPINGS V version or similar
    
    UNIQUE(spaxel_id)
);

CREATE INDEX idx_props_spaxel ON physical_properties(spaxel_id);
CREATE INDEX idx_props_bayes ON physical_properties(bayes_factor);

-- =============================================================================
-- ARTIFACT STORAGE (HYBRID DB/FILESYSTEM)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- MCMC Chain Metadata
-- -----------------------------------------------------------------------------

CREATE TABLE mcmc_metadata (
    chain_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    spaxel_id UUID NOT NULL REFERENCES ifu_spaxels(spaxel_id) ON DELETE CASCADE,
    line_name VARCHAR(32) NOT NULL,
    
    -- Chain configuration
    n_walkers INT NOT NULL DEFAULT 32,
    n_steps_total INT NOT NULL,
    n_burn_in INT NOT NULL,
    n_steps_stored INT NOT NULL,
    
    -- Convergence diagnostics (DESI pattern)
    mean_acceptance_fraction FLOAT CHECK (mean_acceptance_fraction > 0 AND mean_acceptance_fraction <= 1),
    max_autocorrelation_time FLOAT,
    gelman_rubin_r_hat FLOAT[],                -- Per-parameter convergence
    
    -- File reference
    file_format VARCHAR(20) NOT NULL DEFAULT 'HDF5',
    file_path VARCHAR(512) NOT NULL,           -- Relative to data root
    file_size_bytes BIGINT,
    checksum_md5 CHAR(32),
    
    -- Quality
    quality_flag INT DEFAULT 0,
    quality_notes TEXT,
    
    -- Provenance
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_run UUID REFERENCES pipeline_runs(run_id),
    
    UNIQUE(spaxel_id, line_name)
);

CREATE INDEX idx_mcmc_spaxel ON mcmc_metadata(spaxel_id);
CREATE INDEX idx_mcmc_quality ON mcmc_metadata(quality_flag) WHERE quality_flag = 0;

-- -----------------------------------------------------------------------------
-- MCMC Parameter Summaries (fast queries without file access)
-- -----------------------------------------------------------------------------

CREATE TABLE mcmc_parameter_summary (
    summary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id UUID NOT NULL REFERENCES mcmc_metadata(chain_id) ON DELETE CASCADE,
    
    parameter_name VARCHAR(64) NOT NULL,       -- 'velocity', 'sigma', 'flux'
    
    -- Posterior statistics
    posterior_mean FLOAT NOT NULL,
    posterior_median FLOAT NOT NULL,
    posterior_std FLOAT NOT NULL,
    credible_16th FLOAT NOT NULL,              -- 16th percentile
    credible_84th FLOAT NOT NULL,              -- 84th percentile
    credible_2_5th FLOAT,                      -- 2.5th percentile (95% CI)
    credible_97_5th FLOAT,                     -- 97.5th percentile (95% CI)
    
    -- Convergence per parameter
    effective_sample_size FLOAT,
    geweke_z FLOAT,
    
    UNIQUE(chain_id, parameter_name)
);

CREATE INDEX idx_mcmc_summary_chain ON mcmc_parameter_summary(chain_id);
CREATE INDEX idx_mcmc_summary_param ON mcmc_parameter_summary(parameter_name);

-- -----------------------------------------------------------------------------
-- Noise Artifacts (covariance matrices, noise models)
-- -----------------------------------------------------------------------------

CREATE TABLE noise_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Scope (what this artifact covers)
    obs_id UUID REFERENCES observations(obs_id),
    region_name VARCHAR(64),                   -- NULL = full field
    
    -- Type and description
    artifact_type artifact_type NOT NULL,
    description TEXT,
    
    -- Dimensions (for matrices)
    n_spaxels INT,
    matrix_shape INT[],                        -- e.g., [100, 100] for 100x100 covariance
    is_sparse BOOLEAN DEFAULT FALSE,
    sparsity_fraction FLOAT,                   -- Fraction of non-zero elements
    
    -- File reference
    file_format VARCHAR(20) NOT NULL,          -- 'HDF5', 'ASDF', 'NPZ'
    file_path VARCHAR(512) NOT NULL,
    dataset_name VARCHAR(128),                 -- Path within file (e.g., '/cov_matrix')
    file_size_bytes BIGINT,
    checksum_md5 CHAR(32),
    
    -- Summary statistics (for quick reference)
    summary_stats JSONB,                       -- e.g., {'correlation_length_arcsec': 0.3}
    
    -- Provenance
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_run UUID REFERENCES pipeline_runs(run_id)
);

CREATE INDEX idx_noise_obs ON noise_artifacts(obs_id);
CREATE INDEX idx_noise_type ON noise_artifacts(artifact_type);

-- =============================================================================
-- ANALYSIS VIEWS (ENFORCING GATES)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Gold Sample: High-confidence shock spaxels
-- -----------------------------------------------------------------------------

CREATE VIEW view_gold_sample AS
SELECT 
    s.spaxel_id,
    s.sky_location,
    s.ra_deg,
    s.dec_deg,
    s.region,
    s.dist_along_feature_kpc,
    p.dist_from_shock_tip_kpc,
    p.shock_velocity,
    p.metallicity_12_log_oh,
    p.bayes_factor,
    l_o3.flux AS oiii_5007_flux,
    l_o3.velocity AS oiii_5007_velocity,
    l_ha.flux AS ha_flux,
    l_hb.flux AS hb_flux,
    -- Pre-computed line ratios for BPT
    LOG(l_o3.flux / NULLIF(l_hb.flux, 0)) AS log_o3_hb,
    LOG(l_n2.flux / NULLIF(l_ha.flux, 0)) AS log_n2_ha
FROM ifu_spaxels s
JOIN physical_properties p ON s.spaxel_id = p.spaxel_id
LEFT JOIN emission_lines l_o3 ON s.spaxel_id = l_o3.spaxel_id AND l_o3.line_name = 'OIII_5007'
LEFT JOIN emission_lines l_ha ON s.spaxel_id = l_ha.spaxel_id AND l_ha.line_name = 'Ha_6563'
LEFT JOIN emission_lines l_hb ON s.spaxel_id = l_hb.spaxel_id AND l_hb.line_name = 'Hb_4861'
LEFT JOIN emission_lines l_n2 ON s.spaxel_id = l_n2.spaxel_id AND l_n2.line_name = 'NII_6584'
WHERE 
    s.is_valid = TRUE
    AND l_o3.quality_flag = 0
    AND l_o3.snr > 5.0
    AND p.bayes_factor > 10.0;

-- -----------------------------------------------------------------------------
-- Kinematic Gradient: Velocity field along wake
-- -----------------------------------------------------------------------------

CREATE VIEW view_kinematic_gradient AS
SELECT 
    s.spaxel_id,
    s.ra_deg,
    s.dec_deg,
    s.dist_along_feature_kpc,
    p.dist_from_shock_tip_kpc,
    l.velocity AS velocity_kms,
    l.velocity_err,
    l.sigma AS dispersion_kms,
    l.snr,
    s.region
FROM ifu_spaxels s
JOIN physical_properties p ON s.spaxel_id = p.spaxel_id
JOIN emission_lines l ON s.spaxel_id = l.spaxel_id
WHERE 
    l.line_name = 'OIII_5007'
    AND l.quality_flag = 0
    AND s.is_valid = TRUE
    AND p.dist_from_shock_tip_kpc BETWEEN -5 AND 65
ORDER BY p.dist_from_shock_tip_kpc;

-- -----------------------------------------------------------------------------
-- Metallicity Profile: Chemical gradient along wake
-- -----------------------------------------------------------------------------

CREATE VIEW view_metallicity_profile AS
SELECT
    s.spaxel_id,
    s.dist_along_feature_kpc,
    p.dist_from_shock_tip_kpc,
    p.metallicity_12_log_oh,
    p.metallicity_err,
    p.metallicity_method,
    s.region,
    p.bayes_factor
FROM ifu_spaxels s
JOIN physical_properties p ON s.spaxel_id = p.spaxel_id
WHERE
    s.is_valid = TRUE
    AND p.metallicity_12_log_oh IS NOT NULL
ORDER BY p.dist_from_shock_tip_kpc;

-- -----------------------------------------------------------------------------
-- MCMC Chain Inventory: Available posteriors
-- -----------------------------------------------------------------------------

CREATE VIEW view_mcmc_inventory AS
SELECT
    m.chain_id,
    s.spaxel_id,
    s.ra_deg,
    s.dec_deg,
    s.region,
    m.line_name,
    m.n_steps_stored,
    m.mean_acceptance_fraction,
    m.gelman_rubin_r_hat,
    m.file_path,
    m.quality_flag
FROM mcmc_metadata m
JOIN ifu_spaxels s ON m.spaxel_id = s.spaxel_id
WHERE m.quality_flag = 0;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Get spaxels within a region of interest
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION get_spaxels_in_region(p_region_name VARCHAR)
RETURNS TABLE (
    spaxel_id UUID,
    ra_deg FLOAT,
    dec_deg FLOAT,
    dist_along_feature_kpc FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.spaxel_id,
        s.ra_deg,
        s.dec_deg,
        s.dist_along_feature_kpc
    FROM ifu_spaxels s
    JOIN regions_of_interest r ON ST_Within(s.sky_location, r.boundary)
    WHERE r.region_name = p_region_name
      AND s.is_valid = TRUE;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Get emission line measurements for a spaxel
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION get_spaxel_lines(p_spaxel_id UUID)
RETURNS TABLE (
    line_name VARCHAR,
    flux FLOAT,
    flux_err FLOAT,
    velocity FLOAT,
    velocity_err FLOAT,
    sigma FLOAT,
    snr FLOAT,
    quality_flag INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        l.line_name,
        l.flux,
        l.flux_err,
        l.velocity,
        l.velocity_err,
        l.sigma,
        l.snr,
        l.quality_flag
    FROM emission_lines l
    WHERE l.spaxel_id = p_spaxel_id
    ORDER BY l.rest_wavelength_um;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS (DOCUMENTATION)
-- =============================================================================

COMMENT ON SCHEMA rbh1 IS 'RBH-1 Runaway Black Hole Validation Dataset - Analysis-Ready Schema';

COMMENT ON TABLE observations IS 'Zone 0: Root provenance anchor for all HST and JWST observations';
COMMENT ON TABLE wcs_solutions IS 'Zone 1: Materialized WCS solutions with spatial footprints';
COMMENT ON TABLE spectral_grids IS 'Zone 1: Exposure-specific wavelength arrays (GWA tilt correction)';
COMMENT ON TABLE ifu_spaxels IS 'Zone 2: Decomposed IFU cube data with pre-computed sky coordinates';
COMMENT ON TABLE emission_lines IS 'Zone 3: Materialized Gaussian fit parameters per spaxel per line';
COMMENT ON TABLE physical_properties IS 'Zone 3: Inferred physical quantities and Bayesian model evidence';
COMMENT ON TABLE mcmc_metadata IS 'Artifact reference: MCMC chain files with convergence diagnostics';
COMMENT ON TABLE mcmc_parameter_summary IS 'Artifact summary: Posterior statistics for fast queries';
COMMENT ON TABLE noise_artifacts IS 'Artifact reference: Covariance matrices and noise models';
COMMENT ON TABLE pipeline_runs IS 'Provenance: Pipeline execution history with versions and configs';

COMMENT ON VIEW view_gold_sample IS 'Gate-enforced selection of high-confidence shock spaxels (S/N>5, BF>10)';
COMMENT ON VIEW view_kinematic_gradient IS 'Velocity field along wake for discontinuity analysis';
COMMENT ON VIEW view_metallicity_profile IS 'Chemical abundance profile for CGM origin test';

-- =============================================================================
-- END OF SCHEMA DDL
-- =============================================================================
