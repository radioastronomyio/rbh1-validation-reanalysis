# RBH-1 Validation & Reanalysis Architecture

## Overview

RBH-1 Validation & Reanalysis is architected as a 9-phase pipeline transforming raw HST and JWST observations into validated scientific conclusions and a reusable Analysis-Ready Dataset. The architecture emphasizes cube-native likelihood evaluation (models meet data in IFU space), heterogeneous compute distribution (CPU dynamics, GPU inference), and artifact preservation for community reuse.

The pipeline follows a Producer-Consumer pattern where CPU-intensive galactic dynamics calculations feed GPU-accelerated convolution and likelihood evaluation. This decoupling ensures neither resource type sits idle during the computationally intensive falsification campaign.

## Core Components

### Data Layer
Purpose: Raw and processed observational data  
Location: `/mnt/ai-ml/data/` on cluster storage  
Key Characteristics: HST WFC3/UVIS imaging (GO-17301), JWST NIRSpec IFU spectroscopy (GO-3149), MAPPINGS V shock model grids

### Pipeline Phases
Purpose: Sequential processing stages with clear handoffs  
Location: `scripts/00-*` through `scripts/09-*`  
Key Characteristics: Each phase produces artifacts consumed by subsequent phases, enabling modular development and clear provenance

### Inference Engine
Purpose: Bayesian parameter estimation  
Location: `src/inference/`  
Key Characteristics: MCMC sampling with JAX acceleration, cube-native likelihood, full marginalization over nuisance parameters

### Falsification Module
Purpose: Edge-on galaxy hypothesis testing  
Location: `src/falsification/`  
Key Characteristics: galpy dynamics, Toomre Q stability filtering, forward modeling to IFU space

### ARD Package
Purpose: Preserved computational artifacts  
Location: `rbh1-ard-v1/`  
Key Characteristics: Four layers (likelihood interface, inference, representative sample, validation), Zenodo-ready structure

## Structure

```
rbh1-validation-reanalysis/
├── .kilocode/rules/memory-bank/  # AI agent context persistence
├── data/                          # Data manifest (actual data on cluster)
├── docs/                          # Methodology specifications
│   └── documentation-standards/   # Templates and conventions
├── notebooks/                     # Exploratory analysis
├── rbh1-ard-v1/                  # ARD package staging
├── scripts/                       # Phase-organized execution
│   ├── 00-ideation-and-setup/
│   ├── 01-data-acquisition/
│   ├── 02-standard-extraction/
│   ├── 03-cube-differencing/
│   ├── 04-noise-model/
│   ├── 05-kinematic-fitting/
│   ├── 06-mappings-inference/
│   ├── 07-robustness-tests/
│   ├── 08-galaxy-falsification/
│   └── 09-ard-materialization/
├── src/                          # Reusable library code
│   ├── extraction/               # Spectral extraction
│   ├── inference/                # MCMC and likelihood
│   ├── falsification/            # Galaxy hypothesis testing
│   └── visualization/            # Figure generation
├── validation/                   # Data validation outputs
└── work-logs/                    # Phase documentation
```

## Design Patterns and Principles

### Key Patterns

- Producer-Consumer: CPU nodes produce velocity fields and dynamics, GPU node consumes for likelihood evaluation. DragonFlyDB brokers tasks.
- Cube-Native Evaluation: Forward models generate synthetic IFU cubes compared directly to observations after instrument effects applied. No reduction to 1D/2D before fitting.
- Artifact Preservation: Every computed product potentially useful for reanalysis is captured in ARD layers rather than discarded.

### Design Principles

1. Provenance First: Every artifact traces back to source data with documented transformations
2. Modular Phases: Clear boundaries enable independent development and validation
3. Heterogeneous Optimization: Match computation to hardware (dynamics→CPU, convolution→GPU)
4. Reuse Over Regenerate: Preserve artifacts; regeneration cost >> packaging cost

## Integration Points

### Internal Integrations

- scripts/ → src/: Scripts import library code, don't duplicate logic
- phases → validation/: Each phase deposits validation artifacts
- all phases → work-logs/: Documentation synthesis after phase completion

### External Integrations

- MAST: HST and JWST data retrieval via astroquery
- 3MdB: MAPPINGS V shock model grids
- Zenodo: ARD publication with DOI

## Data Flow

```
MAST (HST/JWST) → Acquisition → Validation → Extraction → Cube Differencing
                                                              ↓
Falsification ← Robustness ← MAPPINGS Inference ← Kinematic Fitting ← Noise Model
      ↓
ARD Materialization → Zenodo
```

## Architectural Decisions

### Decision: Cube-Native Likelihood
Date: 2024-12  
Decision: Compare models to data in IFU cube space, not reduced 1D spectra  
Rationale: Preserves spatial-spectral covariance, enables proper nuisance marginalization  
Implications: Requires GPU acceleration for tractable computation

### Decision: Producer-Consumer Task Distribution
Date: 2024-12  
Decision: Separate dynamics (CPU) from likelihood (GPU) via message broker  
Rationale: Heterogeneous cluster with 3 CPU nodes, 1 GPU node. Neither should idle.  
Implications: DragonFlyDB for ephemeral task queue, PostgreSQL for durable chain storage

### Decision: ARD as Co-Equal Deliverable
Date: 2024-12  
Decision: Design ARD upfront, not retrofit after analysis  
Rationale: Packaging cost trivial vs regeneration cost. Enables community reanalysis.  
Implications: All pipeline outputs structured for preservation from start

## Constraints and Limitations

- GPU Memory: A4000 has 16GB VRAM — synthetic cube generation must batch appropriately
- Network Bandwidth: Cluster interconnect limits producer-consumer throughput for large velocity fields
- MAPPINGS Grid Resolution: Pre-computed grids constrain shock parameter space granularity

## Future Considerations

### Planned Improvements

- Potential JAX-based dynamics if galpy CPU performance becomes bottleneck
- Adaptive MCMC stepping based on posterior geometry

### Scalability Considerations

- Current architecture supports ~30 representative synthetic cubes
- Full grid exploration would require cloud burst or longer runtime

### Known Technical Debt

- None yet — architecture designed before implementation
