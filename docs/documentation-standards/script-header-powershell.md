# PowerShell Script Header Template

> Template Version: 1.0  
> Applies To: All `.ps1` files in rbh1-validation-reanalysis  
> Last Updated: 2024-12-23

---

## Template

```powershell
<#
.SYNOPSIS
    [One-line description of what the script does]

.DESCRIPTION
    [2-4 sentences explaining the script's purpose, what it operates on,
    and what outputs it produces. Include any important behavioral notes.]

.NOTES
    Repository  : rbh1-validation-reanalysis
    Author      : VintageDon (https://github.com/vintagedon)
    ORCID       : 0009-0008-7695-4093
    Created     : YYYY-MM-DD
    Phase       : [Phase X - Phase Name]

.EXAMPLE
    .\script-name.ps1

    [Description of what this invocation does]

.EXAMPLE
    .\script-name.ps1 -Parameter Value

    [Description of what this invocation does]

.LINK
    https://github.com/radioastronomyio/rbh1-validation-reanalysis
#>

# =============================================================================
# Configuration
# =============================================================================

# [Configuration variables with inline comments]

# =============================================================================
# Functions
# =============================================================================

# [Function definitions if needed]

# =============================================================================
# Execution
# =============================================================================

# [Main script logic]
```

---

## Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `.SYNOPSIS` | Yes | Single line, verb-led description |
| `.DESCRIPTION` | Yes | Expanded explanation of purpose and behavior |
| `.NOTES` | Yes | Static metadata (repository, author, ORCID, dates, phase) |
| `.EXAMPLE` | Yes | At least one usage example with description |
| `.LINK` | Yes | Repository URL |
| `.PARAMETER` | If applicable | Document any script parameters |

---

## Section Comments

Use banner comments to separate logical sections:

```powershell
# =============================================================================
# Section Name
# =============================================================================
```

Standard sections (in order):

1. **Configuration** — Variables, paths, settings
2. **Functions** — Helper function definitions (if any)
3. **Execution** — Main script logic

---

## Example: Minimal Script

```powershell
<#
.SYNOPSIS
    Validates FITS file headers in the HST data directory.

.DESCRIPTION
    Scans all FITS files in the configured HST data path and validates
    required header keywords are present. Outputs a summary of missing
    or malformed headers.

.NOTES
    Repository  : rbh1-validation-reanalysis
    Author      : VintageDon (https://github.com/vintagedon)
    ORCID       : 0009-0008-7695-4093
    Created     : 2024-12-23
    Phase       : Phase 1 - Data Acquisition

.EXAMPLE
    .\validate-hst-headers.ps1

    Validates all FITS files in the default HST data directory.

.LINK
    https://github.com/radioastronomyio/rbh1-validation-reanalysis
#>

# =============================================================================
# Configuration
# =============================================================================

$dataPath = "/mnt/ai-ml/data/hst"

# =============================================================================
# Execution
# =============================================================================

Get-ChildItem -Path $dataPath -Filter "*.fits" | ForEach-Object {
    # Validation logic here
}
```

---

## Notes

- PowerShell comment-based help (`.SYNOPSIS`, `.DESCRIPTION`, etc.) enables `Get-Help script-name.ps1`
- Keep `.SYNOPSIS` under 80 characters
- Use present tense, active voice ("Validates..." not "This script validates...")
- Phase field should match work-log phase names exactly
