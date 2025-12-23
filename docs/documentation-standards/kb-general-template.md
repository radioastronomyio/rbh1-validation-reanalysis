# General KB Article Template

> Template Version: 1.0  
> Applies To: All standalone knowledge base articles in rbh1-validation-reanalysis  
> Last Updated: 2024-12-21

---

## Template

```markdown
<!--
---
title: "[Article Title]"
description: "One-line summary of what this article covers"
author: "VintageDon - https://github.com/vintagedon"
ai_contributor: "[Full AI Model Name/Version]"
date: "YYYY-MM-DD"
version: "X.Y"
phase: [phase-1, phase-2, ...]
tags:
  - domain: [extraction/inference/falsification/validation/documentation/data]
  - type: [methodology/reference/guide/decision-record/specification]
related_documents:
  - "[Related Doc 1](path/to/doc.md)"
  - "[Related Doc 2](path/to/doc.md)"
---
-->

# [Article Title]

## 1. Purpose

[1-2 sentences: What question does this article answer? Why does it exist?]

---

## 2. Scope

[1-2 sentences: What this article covers and, if helpful, what it explicitly does not cover.]

---

## 3. [Body Content]

[Main article content. Structure as needed for the topic—subsections, tables, code blocks, etc.]

---

## 4. References

| Reference | Link |
|-----------|------|
| [Source name] | [URL or internal link] |

*None — this article is self-contained.*

---
```

---

## Style Guide

### Fixed Semantic Numbering

| Section | Content | Required |
|---------|---------|----------|
| 1 | Purpose | Yes |
| 2 | Scope | Yes |
| 3 | Body | Yes (title varies) |
| 4 | References | Yes (may be empty) |

### Front-Matter

The front-matter block is wrapped in HTML comments (`<!-- ... -->`) to hide from rendering while remaining parseable for RAG import.

Required fields:
- `title` — Article title
- `description` — One-line summary
- `author` — Primary author with GitHub link
- `ai_contributor` — AI model/version if applicable
- `date` — Creation date (YYYY-MM-DD)
- `version` — Semantic version (X.Y)
- `phase` — Pipeline phases this relates to
- `tags` — Domain and type classification
- `related_documents` — Links to connected articles

Phase Tags — From main README:

| Tag | Description |
|-----|-------------|
| `phase-1` | Data acquisition + provenance |
| `phase-2` | Standard extraction baseline |
| `phase-3` | Cube-space differencing |
| `phase-4` | Empirical noise model |
| `phase-5` | Tied kinematic fitting |
| `phase-6` | MAPPINGS V inference |
| `phase-7` | Robustness tests |
| `phase-8` | Edge-on galaxy falsification |
| `ard` | Analysis-Ready Dataset materialization |

Type Tags — KB article classification:

| Type Tag | Use For |
|----------|---------|
| `methodology` | How we do something (shock fitting approach, MCMC configuration) |
| `reference` | Lookup information (parameter definitions, file formats) |
| `guide` | Step-by-step procedures (running extraction, interpreting outputs) |
| `decision-record` | Why we chose X over Y (prior selection rationale, software choices) |
| `specification` | Formal requirements (ARD schema, validation criteria) |

### Purpose Section

One question, one answer. No background, no context-setting.

Good example (MAPPINGS V usage article):
> This article documents how RBH-1 validation uses MAPPINGS V shock model grids, including version, parameter ranges, and interpolation method.

Weak example:
> MAPPINGS V is a shock modeling code developed by... [3 paragraphs of background]

Background belongs in the body if needed, not Purpose.

### Scope Section

Define boundaries. Helps readers (and RAG systems) know if this article answers their question.

Good example:
> Covers MAPPINGS V grid selection and likelihood integration. Does not cover shock physics fundamentals—see [Shock Physics Primer](shock-physics-primer.md).

### Body Section

Rename Section 3's header to match content. Structure freely—this is where the actual knowledge lives.

Examples:
- `## 3. MAPPINGS V Integration`
- `## 3. Background Subtraction Method`
- `## 3. Jackknife Procedure`

### References Section

Link to sources: papers, external docs, internal specs. If self-contained, use the empty placeholder.

---

## Length Guidelines

No fixed limit. Let content dictate length. However:

- Under 50 lines → Consider if this needs to be standalone or belongs in another article
- Over 300 lines → Consider splitting into multiple focused articles

---

## Validation Checklist

- [ ] Front-matter wrapped in HTML comments
- [ ] All required front-matter fields populated
- [ ] Purpose is one concrete statement
- [ ] Scope defines boundaries
- [ ] Body section has descriptive header (not "Body")
- [ ] References section present (even if empty)
