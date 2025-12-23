# Python Script Header Template

> Template Version: 1.0  
> Applies To: All `.py` files in rbh1-validation-reanalysis  
> Last Updated: 2024-12-23

---

## Template

```python
#!/usr/bin/env python3
"""
Script Name  : script_name.py
Description  : [One-line description of what the script does]
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : YYYY-MM-DD
Phase        : [Phase X - Phase Name]
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
[2-4 sentences explaining the script's purpose, what it operates on,
and what outputs it produces. Include any important behavioral notes.]

Usage
-----
    python script_name.py [options]

Examples
--------
    python script_name.py
        [Description of what this invocation does]

    python script_name.py --verbose
        [Description of what this invocation does]
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# [Configuration constants with inline comments]

# =============================================================================
# Functions
# =============================================================================


def main() -> None:
    """Entry point for script execution."""
    pass


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
```

---

## Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| Script Name | Yes | Filename for reference (snake_case) |
| Description | Yes | Single line, verb-led description |
| Repository | Yes | Repository name |
| Author | Yes | Name with GitHub profile link |
| ORCID | Yes | Author ORCID identifier |
| Created | Yes | Creation date (YYYY-MM-DD) |
| Phase | Yes | Pipeline phase this script belongs to |
| Link | Yes | Full repository URL |
| Description section | Yes | Expanded multi-line explanation |
| Usage section | Yes | Command syntax |
| Examples section | Yes | At least one usage example |

---

## Section Comments

Use banner comments to separate logical sections:

```python
# =============================================================================
# Section Name
# =============================================================================
```

Standard sections (in order):
1. **Imports** — Standard library, third-party, local imports (in that order)
2. **Configuration** — Constants, paths, settings
3. **Functions** — Function and class definitions
4. **Entry Point** — `if __name__ == "__main__":` block

---

## Docstring Style

Use NumPy-style docstrings for functions:

```python
def extract_spectrum(cube_path: Path, aperture_radius: float = 0.5) -> np.ndarray:
    """
    Extract a 1D spectrum from an IFU cube at the specified aperture.

    Parameters
    ----------
    cube_path : Path
        Path to the FITS cube file.
    aperture_radius : float, optional
        Extraction aperture radius in arcseconds. Default is 0.5.

    Returns
    -------
    np.ndarray
        Extracted 1D spectrum with shape (n_wavelengths,).

    Raises
    ------
    FileNotFoundError
        If the cube file does not exist.
    ValueError
        If aperture_radius is not positive.

    Examples
    --------
    >>> spectrum = extract_spectrum(Path("cube.fits"), aperture_radius=0.3)
    >>> spectrum.shape
    (2048,)
    """
    pass
```

---

## Example: Minimal Script

```python
#!/usr/bin/env python3
"""
Script Name  : validate_data.py
Description  : Validates FITS file integrity for HST and JWST observations
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-23
Phase        : Phase 1 - Data Acquisition
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Scans all FITS files in the configured data directories and validates:
- File readability (not corrupted)
- Required header keywords present
- WCS information consistency

Outputs a validation report to stdout and optionally to a JSON file.

Usage
-----
    python validate_data.py [--output report.json]

Examples
--------
    python validate_data.py
        Validates all data and prints summary to stdout.

    python validate_data.py --output validation_report.json
        Validates all data and writes detailed report to JSON.
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

from astropy.io import fits

# =============================================================================
# Configuration
# =============================================================================

HST_DATA_PATH = Path("/mnt/ai-ml/data/hst")
JWST_DATA_PATH = Path("/mnt/ai-ml/data/jwst")

# =============================================================================
# Functions
# =============================================================================


def validate_fits_file(filepath: Path) -> dict:
    """
    Validate a single FITS file.

    Parameters
    ----------
    filepath : Path
        Path to the FITS file.

    Returns
    -------
    dict
        Validation result with keys: 'valid', 'errors', 'warnings'.
    """
    result = {"valid": True, "errors": [], "warnings": []}

    try:
        with fits.open(filepath) as hdul:
            # Validation logic here
            pass
    except Exception as e:
        result["valid"] = False
        result["errors"].append(str(e))

    return result


def main() -> None:
    """Entry point for script execution."""
    print(f"Validating HST data in {HST_DATA_PATH}")
    print(f"Validating JWST data in {JWST_DATA_PATH}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
```

---

## Type Hints

Always use type hints for function signatures:

```python
from pathlib import Path
from typing import Optional

def process_cube(
    cube_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False
) -> dict[str, float]:
    """Process a cube and return statistics."""
    pass
```

---

## Notes

- Use `#!/usr/bin/env python3` for portability
- Module docstring goes immediately after shebang
- Keep Description line under 80 characters
- Use present tense, active voice ("Validates..." not "This script validates...")
- Use `pathlib.Path` instead of string paths
- Use type hints for all function parameters and return values
- Follow PEP 8 style guide
