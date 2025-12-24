# Phase Worklog Template

> Template Version: 1.0  
> Applies To: All phase completion documentation in rbh1-validation-reanalysis  
> Last Updated: 2024-12-21

---

## Template

```markdown
<!--
---
title: "Phase X: [Phase Name]"
description: "One-line summary of phase outcome"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "[Full AI Model Name/Version]"
date: "YYYY-MM-DD"
version: "X.Y"
phase: [phase-X]
tags:
  - domain: [primary domain]
  - type: methodology
  - tech: [hst/jwst/mappings-v if applicable]
related_documents:
  - "[Validation Spec](../validation-spec.md)"
  - "[Previous Phase](phase-X-worklog.md)"
---
-->

# Phase X: [Phase Name]

> Compiled from: X sessions | ~Y hours  
> Status: Complete | In Progress  
> Key Outcome: [One-line technical achievement]

---

## 1. Objective

[2-3 sentences: What this phase set out to accomplish and why it matters for the validation.]

---

## 2. Validation Progress

| Criterion | Status | Evidence |
|-----------|--------|----------|
| [Criterion from README] | Advanced / No Change | [Brief description or link to artifact] |

Summary: [1-2 sentences on how this phase moved validation forward.]

---

## 3. Artifacts Produced

| Artifact | Purpose | Location |
|----------|---------|----------|
| `script-name.py` | [Brief purpose] | `src/folder/` |
| `config.yml` | [Brief purpose] | `src/folder/` |
| `posteriors.nc` | [Brief purpose] | `data/outputs/` |

Total: X scripts | X configs | X data products

---

## 4. ARD Materialization

| ARD Layer | Artifact | Status |
|-----------|----------|--------|
| Likelihood Interface | [artifact] | Ready / Pending |
| Inference Layer | [artifact] | Ready / Pending |
| Representative Sample | [artifact] | Ready / Pending |
| Validation Layer | [artifact] | Ready / Pending |

*None — this phase does not produce ARD artifacts.*

---

## 5. Technical Approach

### Key Decisions

[Decision]: [Why this approach, what trade-offs]

[Decision]: [Why this approach, what trade-offs]

### Implementation Notes

[Brief summary of how the work was executed. Not step-by-step—high-level approach and any notable patterns.]

---

## 6. Lessons Learned

| Challenge | Resolution |
|-----------|------------|
| [Problem encountered] | [How it was solved] |

Key Insight: [Most important technical or process learning from this phase.]

---

## 7. Next Phase

Enables: [What this phase hands off to the next]

Dependencies resolved: [What blockers this phase cleared]

Open items: [Anything deferred or requiring follow-up]

---

## 8. Provenance

| Item | Value |
|------|-------|
| Python | X.Y.Z |
| Key packages | `package==version`, `package==version` |
| Commit | `abc1234` |
| Data versions | [MAST retrieval date, file versions] |
| Random seeds | [If applicable] |

---
```

---

## Style Guide

### Document Purpose

Phase worklogs are synthesis documents. They compile internal session work into an outcome-focused summary. The audience is someone who wants to understand what Phase X accomplished, not reconstruct every decision.

Include: What was built, what it enables, where artifacts live  
Reference: Full scripts, logs, detailed outputs (don't embed)  
Omit: Session-by-session granularity, internal iteration details

### Header Block

The compiled-from line provides context without detail:

```markdown
> Compiled from: 4 sessions | ~12 hours  
```

This acknowledges the work scope without promising session-level breakdown.

### Validation Progress Section

This section is always substantive. Every phase should advance at least one validation criterion.

Validation Criteria (from main README):

*Confirmation conditions:*

- 95% CI of shock velocity from line ratios overlaps spatial gradient measurement
- Difference-of-pointings artifacts account for <20% of wake flux
- Inferred metallicity consistent with CGM (Z < 0.5 Z☉)
- Velocity discontinuity persists across all jackknife subsets

*Tension conditions:*

- Marginalizing over magnetic parameter broadens v_s posterior to include virial velocities
- Standard Level 3 extraction significantly diminishes wake signal
- Jackknife analysis shows exposure/region dependence

Reference specific criteria and note how the phase advanced them.

### Artifacts Table

Inventory everything produced. Link to location, don't embed content.

Good:

| Artifact | Purpose | Location |
|----------|---------|----------|
| `extract_spectra.py` | Extracts 1D spectra from NIRSpec cubes | `src/extraction/` |

Avoid: Full script content, even in collapsible sections. The script lives in the repo—link to it.

### ARD Materialization Section

Explicitly map phase outputs to ARD layers. If the phase doesn't produce ARD artifacts, say so—the section stays present.

### Provenance Section

Reproducibility anchor. Capture everything needed to re-execute:

- Exact package versions (not ranges)
- Commit hash at phase completion
- Data retrieval dates and versions
- Random seeds if stochastic processes involved

---

## Relationship to Internal Logs

Internal session logs (compiled by Gemini or similar) contain:

- Full script text
- Complete execution logs
- Detailed decision rationale
- Iteration history

Phase worklogs synthesize these into outcomes. Reference internal logs if they're preserved, but don't reproduce their detail level.

---

## Validation Checklist

- [ ] Front-matter wrapped in HTML comments
- [ ] Header block shows compilation scope
- [ ] Validation Progress references specific criteria
- [ ] All artifacts listed with locations
- [ ] ARD Materialization section addresses all four layers
- [ ] Provenance captures reproducibility requirements
- [ ] No embedded scripts or full logs (reference only)
