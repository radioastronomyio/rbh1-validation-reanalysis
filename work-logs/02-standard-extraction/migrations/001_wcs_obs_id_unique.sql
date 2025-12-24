-- Migration: Add UNIQUE constraint to wcs_solutions.obs_id
-- Reason: Enable UPSERT semantics in ETL (1:1 relationship with observations)
-- Date: 2025-12-24

ALTER TABLE rbh1.wcs_solutions 
ADD CONSTRAINT wcs_solutions_obs_id_unique UNIQUE (obs_id);
