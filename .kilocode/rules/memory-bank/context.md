# RBH-1 Validation & Reanalysis Context

## Current State

Last Updated: 2024-12-23

### Recent Accomplishments

- Completed Phase 00: repository scaffolding, methodology documentation, infrastructure allocation
- Completed data acquisition overnight: 142 HST products (~44 GB), 52 JWST products
- HST GO-17301: FLC and DRC products for 6 visits, F200LP and F350LP filters, plus HAP skycell mosaics
- JWST GO-3149: CAL, S3D, X1D products for NIRSpec IFU G140M/F100LP configuration
- Data manifest written to `/mnt/ai-ml/data/data_manifest.csv`
- Phase 00 worklog completed and documented

### Current Phase

We are currently in Phase 01: Data Acquisition which involves validating downloaded data integrity, generating visualization suite, and documenting the acquisition process.

### Active Work

Currently working on:

1. Data validation: Run validate_data.py, extend with comprehensive FITS integrity checks
2. Visualization suite: Matplotlib/seaborn plots for HST quick-looks, JWST cube slices, inventory summaries
3. Phase 01 worklog: Document acquisition and validation process
4. Memory bank population: Complete context files for AI agent continuity

## Next Steps

### Immediate (This Session)

1. Populate remaining memory bank files (architecture.md, tech.md)
2. Commit Phase 00 worklog and memory bank updates
3. Run and extend data validation script
4. Generate validation figures (HST thumbnails, JWST spectra)

### Near-Term (Next Few Sessions)

- Complete Phase 01 worklog with validation results and figures
- Create `validation/phase-01-data-integrity/` directory with outputs
- Begin Phase 02: Standard extraction baseline
- Compare extracted spectra to van Dokkum et al. published figures

### Future / Backlog

- Phase 03-08: Core analysis pipeline implementation
- Phase 09: ARD materialization and Zenodo upload
- Paper draft preparation

## Active Decisions

### Pending Decisions

- JWST cube combination strategy: Whether to use Level 3 combined products or work from individual dithers for systematic characterization. Will evaluate after initial cube inspection.

### Recent Decisions

- 2024-12-23 — Phase worklog dating: Use compilation date rather than backdating. Worklogs are synthesis documents, dating reflects when synthesis occurred.
- 2024-12-23 — Validation figure scope: Include HST quick-looks, JWST cube slices, inventory summaries. Match DESI project validation pattern.

## Blockers and Dependencies

### Current Blockers

None.

### External Dependencies

- MAPPINGS V grids: Will need to acquire from 3MdB database for Phase 06
- galpy installation: Required for Phase 08 galaxy falsification dynamics

## Notes and Observations

### Recent Insights

- HAP skycell mosaics are ~3 GB each — these provide wide-field context but may not be needed for core analysis
- JWST products include two target positions (t001, t002) — these correspond to positions along the linear feature
- Data acquisition took ~4 hours overnight with ~1 minute per file average

### Context for Next Session

Data is staged at `/mnt/ai-ml/data/hst/` and `/mnt/ai-ml/data/jwst/`. Validation script exists but needs extension for comprehensive checks. Next priority is validation figures and Phase 01 worklog completion.
