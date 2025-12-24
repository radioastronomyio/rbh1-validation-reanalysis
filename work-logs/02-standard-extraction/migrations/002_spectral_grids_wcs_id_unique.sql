-- Migration: 002_spectral_grids_wcs_id_unique.sql
-- Date: 2024-12-24
-- Purpose: Add UNIQUE constraint on spectral_grids.wcs_id for UPSERT support
-- 
-- Rationale: Enables rerunnable ETL with ON CONFLICT (wcs_id) DO UPDATE
-- Models 1:1 relationship between spectral grid and WCS solution (one grid per cube)

ALTER TABLE rbh1.spectral_grids
ADD CONSTRAINT spectral_grids_wcs_id_unique UNIQUE (wcs_id);

-- Verification:
-- SELECT constraint_name, constraint_type 
-- FROM information_schema.table_constraints 
-- WHERE table_schema='rbh1' AND table_name='spectral_grids';
