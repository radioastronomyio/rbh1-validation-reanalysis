# RBH-1 Validation & Reanalysis Technology Stack

## Technology Stack

### Primary Technologies

- Python: 3.12 - Core analysis and pipeline orchestration
- JAX: GPU-accelerated array operations, likelihood evaluation, MCMC
- Astropy: FITS I/O, WCS handling, coordinate transformations
- galpy: Galactic dynamics for edge-on galaxy falsification

### Supporting Technologies

- astroquery: MAST data retrieval
- numpy/scipy: Numerical operations
- matplotlib/seaborn: Visualization and validation figures
- DragonFlyDB: Ephemeral task brokering for producer-consumer pattern
- PostgreSQL: Durable MCMC chain storage

## Dependencies

### Required Dependencies

```
astropy>=6.0          # FITS handling, WCS
astroquery>=0.4       # MAST data retrieval
numpy>=1.26           # Array operations
scipy>=1.12           # Optimization, interpolation
matplotlib>=3.8       # Visualization
seaborn>=0.13         # Statistical visualization
jax>=0.4              # GPU acceleration (Phase 05+)
jaxlib>=0.4           # JAX backend
galpy>=1.9            # Galactic dynamics (Phase 08)
```

### Optional Dependencies

```
dragonfly-db          # Task brokering (Phase 08)
psycopg2>=2.9         # PostgreSQL connection (Phase 08)
arviz>=0.17           # MCMC diagnostics
corner>=2.2           # Posterior visualization
```

## Development Environment

### Prerequisites

- Python 3.12 installation
- NVIDIA GPU with CUDA support (for JAX acceleration)
- Access to Proxmox Astronomy Lab cluster

### Setup Instructions

```bash
# On proj-gpu01, activate shared environment
source /mnt/ai-ml/venvs/venv-ml-py312/bin/activate

# Verify key packages
python -c "import astropy; print(astropy.__version__)"
python -c "import jax; print(jax.devices())"

# Clone repository (if not present)
cd /mnt/ai-ml/rbh1
git clone <repo-url> rbh1-validation-reanalysis

# Data is staged at
ls /mnt/ai-ml/data/hst/
ls /mnt/ai-ml/data/jwst/
```

### Environment Variables

```bash
# JAX configuration for A4000
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Data paths
RBH1_DATA_DIR=/mnt/ai-ml/data
RBH1_REPO_DIR=/mnt/ai-ml/rbh1/rbh1-validation-reanalysis
```

## Infrastructure

### Compute Cluster

| Node | Role | Specs | Access |
|------|------|-------|--------|
| proj-gpu01 | Primary compute, GPU inference | AMD 5950X, 48GB, A4000 16GB | SSH |
| proj-cpu01 | MCMC walker pool | Intel 12900K, 48GB | SSH |
| proj-cpu02 | MCMC walker pool | Intel 12900K, 48GB | SSH |
| proj-cpu03 | MCMC walker pool | Intel 12900K, 48GB | SSH |

### Storage

| Path | Purpose | Type |
|------|---------|------|
| `/mnt/ai-ml/data/` | HST/JWST observations | NFS shared |
| `/mnt/ai-ml/rbh1/` | Working directory | NFS shared |
| `/mnt/ai-ml/venvs/` | Shared Python environments | NFS shared |

### External Services

- MAST: Mikulski Archive for Space Telescopes — data retrieval
- 3MdB: Mexican Million Models database — MAPPINGS V grids
- Zenodo: ARD publication and DOI assignment
- GitHub: Repository hosting

## Technical Constraints

### Performance Requirements

- Synthetic cube generation must complete in <1 minute per cube for tractable falsification campaign
- MCMC chains must achieve Rhat < 1.01 for convergence

### GPU Constraints

- A4000 VRAM: 16GB — limits simultaneous cube size
- Batch synthetic cube generation to avoid OOM

### Data Volume

- HST: ~44 GB (142 files)
- JWST: ~5 GB (52 files)
- Expected MCMC chains: ~10 GB
- ARD total: ~20 GB estimated

## Development Workflow

### Version Control

- Repository: GitHub (Proxmox-Astronomy-Lab organization)
- Branching: Main branch for stable, feature branches for development
- Commits: Conventional commits (feat:, fix:, docs:, etc.)

### Testing

- Validation scripts in `validation/` directory
- FITS integrity checks before analysis
- Posterior convergence diagnostics (Rhat, ESS)

### Execution Pattern

```bash
# Typical phase execution
cd /mnt/ai-ml/rbh1/rbh1-validation-reanalysis
source /mnt/ai-ml/venvs/venv-ml-py312/bin/activate

# Run phase script
python scripts/01-data-acquisition/validate_data.py

# Or from repo scripts directory
python scripts/validate_data.py
```

## Troubleshooting

### Common Issues

#### JAX GPU Not Detected
Problem: JAX falls back to CPU  
Solution: Verify CUDA installation, check `nvidia-smi`, ensure `jaxlib[cuda]` installed

#### FITS File Corruption
Problem: astropy.io.fits raises error on open  
Solution: Re-download from MAST, verify checksums

#### Memory Exhaustion During Cube Generation
Problem: OOM on GPU during synthetic cube creation  
Solution: Reduce batch size, enable memory preallocation limits

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Verify JAX devices
python -c "import jax; print(jax.devices())"

# Check data integrity
python -c "from astropy.io import fits; fits.open('/mnt/ai-ml/data/jwst/jw03149-o001_t001_nirspec_g140m-f100lp_s3d.fits').info()"

# Disk usage
du -sh /mnt/ai-ml/data/*
```

## Technical Documentation

- Astropy FITS: https://docs.astropy.org/en/stable/io/fits/
- JAX: https://jax.readthedocs.io/
- galpy: https://docs.galpy.org/
- JWST Pipeline: https://jwst-pipeline.readthedocs.io/
- MAPPINGS V: https://3mdb.astro.unam.mx/
