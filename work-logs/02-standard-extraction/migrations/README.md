<!--
---
title: "Phase 02 Schema Migrations"
description: "Database migrations for RBH-1 validation schema"
author: "CrainBramp / VintageDon"
date: "2024-12-24"
version: "1.0"
status: "Current"
phase: phase-02
tags:
  - domain: data-engineering
  - type: schema
  - type: migrations
related_documents:
  - "[Phase 02 README](../README.md)"
  - "[Schema DDL](../02-schema_ddl.sql)"
---
-->

# Schema Migrations

This directory contains incremental schema modifications applied after initial deployment of `02-schema_ddl.sql`.

## Migration Philosophy

Migrations are applied to support ETL requirements discovered during development. Each migration:

- Is idempotent where possible (safe to re-run)
- Includes a verification query in comments
- Documents the rationale for the change

## Applied Migrations

| Migration | Date | Purpose |
|-----------|------|---------|
| [001_wcs_obs_id_unique.sql](./001_wcs_obs_id_unique.sql) | 2024-12-24 | UPSERT support for WCS ETL |
| [002_spectral_grids_wcs_id_unique.sql](./002_spectral_grids_wcs_id_unique.sql) | 2024-12-24 | UPSERT support for spectral grids ETL |

---

## Migration Details

### 001_wcs_obs_id_unique.sql

Problem: The `05-etl_wcs_solutions.py` script uses `ON CONFLICT (obs_id) DO UPDATE` for idempotent reruns. PostgreSQL requires a UNIQUE constraint on the conflict target.

Solution:

```sql
ALTER TABLE rbh1.wcs_solutions
ADD CONSTRAINT wcs_solutions_obs_id_unique UNIQUE (obs_id);
```

Rationale: Enforces 1:1 relationship between observations and WCS solutions. Each observation should have at most one WCS record (though some observations like x1d are excluded entirely).

---

### 002_spectral_grids_wcs_id_unique.sql

Problem: The `06-etl_spectral_grids.py` script uses `ON CONFLICT (wcs_id) DO UPDATE` for idempotent reruns. Same constraint requirement as above.

Solution:

```sql
ALTER TABLE rbh1.spectral_grids
ADD CONSTRAINT spectral_grids_wcs_id_unique UNIQUE (wcs_id);
```

Rationale: Enforces 1:1 relationship between WCS solutions and spectral grids. Each s3d cube's WCS record has exactly one wavelength grid.

---

## Constraint Summary

After migrations, the `spectral_grids` table has:

| Constraint | Type | Column |
|------------|------|--------|
| `spectral_grids_pkey` | PRIMARY KEY | grid_id |
| `spectral_grids_wcs_id_fkey` | FOREIGN KEY | wcs_id → wcs_solutions(wcs_id) |
| `spectral_grids_wcs_id_unique` | UNIQUE | wcs_id |

The `wcs_solutions` table has:

| Constraint | Type | Column |
|------------|------|--------|
| `wcs_solutions_pkey` | PRIMARY KEY | wcs_id |
| `wcs_solutions_obs_id_fkey` | FOREIGN KEY | obs_id → observations(obs_id) |
| `wcs_solutions_obs_id_unique` | UNIQUE | obs_id |

---

## Applying Migrations

Migrations should be applied in order on pgsql01:

```bash
# From proj-gpu01
psql -h 10.25.20.8 -U clusteradmin_pg01 -d rbh1_validation \
  -f /mnt/ai-ml/rbh1/phase02/migrations/001_wcs_obs_id_unique.sql

psql -h 10.25.20.8 -U clusteradmin_pg01 -d rbh1_validation \
  -f /mnt/ai-ml/rbh1/phase02/migrations/002_spectral_grids_wcs_id_unique.sql
```

Or interactively:

```sql
\i migrations/001_wcs_obs_id_unique.sql
\i migrations/002_spectral_grids_wcs_id_unique.sql
```

---

## Future Migrations

As analysis progresses, additional migrations may be needed for:

- Zone 2 table constraints (`ifu_spaxels`, etc.)
- Index additions for query optimization
- View modifications for analysis convenience

Follow the naming convention: `NNN_descriptive_name.sql`

---
