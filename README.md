# KHOSTspec

A file-level parallel pipeline for optical spectra taken at KHAO. Starting from
raw FITS frames, it produces extracted spectra, wavelength-calibrated spectra,
and response-corrected relative-flux spectra.


## Repository contents

| File | Purpose |
| --- | --- |
| [spec_reduction_parallel.py](spec_reduction_parallel.py) | Command-line entry point and compatible worker aliases |
| [speclibparallel.py](speclibparallel.py) | Calibration, tracing, extraction, wavelength and response processing, and run controller |
| [config.example.yaml](config.example.yaml) | Portable example configuration |
| [NeAr.csv](NeAr.csv) | Reference arc wavelengths in Angstrom |
| [requirements.txt](requirements.txt) | Direct dependency versions used for validation |
| [requirements-dev.txt](requirements-dev.txt) | Additional dependency for the bundled tests |
| [requirements-docs.txt](requirements-docs.txt) | Sphinx documentation dependency |
| [docs/index.rst](docs/index.rst) | User guide and API reference source |
| [tests/test_entrypoint.py](tests/test_entrypoint.py) | Portable command-line and delegation tests |

Observation data, master darks, standard-star template spectra, and local
validation archives are not bundled.

## Quick start

The validated environment uses **Python 3.13.9**. Other Python versions have not
been validated. The dependency pins describe the tested direct dependencies;
they are not a complete environment lock.

Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

Copy `config.example.yaml` to `config.local.yaml`, then set the paths, detector
parameters, observatory position, and extraction settings for your observations.
The example values must be checked against the actual instrument and data.
Provide the required standard-star templates and check the object catalogs
described below.

```bash
python spec_reduction_parallel.py observations/night_001 config.local.yaml
```

The two arguments are the observation directory and the YAML configuration.
Both accept absolute or relative paths. **All relative paths, including paths
inside the YAML file, are resolved from the current working directory**, not
from the configuration file's directory. Quote paths containing spaces.

The pipeline can also be called from Python:

```python
from speclibparallel import main

if __name__ == "__main__":
    report = main("observations/night_001", "config.local.yaml")
    print(report["status"])
```

Keep the call inside the `__main__` guard when using worker processes,
particularly on Windows.

## Input data

### Observation directory and FITS headers

Place all frames at the top level of the observation directory.
Group discovery recognizes `.fit`, `.fits`, and `.fts`.
Bias frames are required; dark correction can use a compatible library master
when no suitable same-night master is available.

| Header | Use |
| --- | --- |
| `IMAGETYP` | Frame classification: `Bias Frame`, `Dark Frame`, `Flat Field`, or `Light Frame` |
| `OBJECT` | `Arc` for lamp frames; otherwise a configured target or reference-star name |
| `JD`, `JD-OBS`, `MJD`, or `MJD-OBS` | Time ordering and group discovery; `JD` has priority |
| `EXPTIME` | Exposure time in seconds for calibration and dark selection |
| `CCD-TEMP`, `SET-TEMP`, or `CCDTEMP` | Detector temperature for calibration compatibility; measured temperature is preferred |
| `DATE-OBS` | Observation time for airmass, velocity correction, and dark-library selection |
| `OBJCTRA` / `OBJCTDEC` | Optional coordinates: RA as hour angle, Dec in degrees |
| `RA` / `DEC` | Alternative coordinates, both interpreted as numeric degrees |

Group discovery skips unreadable, unrecognized, or undated frames. `DATE-OBS`
alone does not replace a Julian-date keyword for grouping. Coordinate settings
in the YAML file take precedence over header coordinates.

### Object catalogs and grouping

Review `TARGETS` and `REFERENCES` in `speclibparallel.py` before reducing a new
field. `TARGETS` contains science-object names; `REFERENCES` maps each standard
star to a template library and filename. Names are matched without regard to
case or spaces. Unrecognized science objects are ignored.

The current grouping algorithm uses consecutive runs of the same **frame
class**, ordered by time. A change of `OBJECT` alone does not split a run.
Keep each consecutive target or reference run limited to one object and inspect
`grouping_summary.json` when processing mixed-object observations. Reference
groups contain at most three frames.

### Standard-star templates

Templates must be supplied at:

```text
<refpath>/<library>/<template_file>.fits
```

The `REFERENCES` entry provides `<library>` and `<template_file>` without the
`.fits` suffix. Supported libraries are:

- **Pickles:** FITS binary tables with `WAVELENGTH` and `FLUX` columns. (https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas)
- **MILES:** one-dimensional FITS images with wavelength coordinates in the header. (https://research.iac.es/proyecto/miles/pages/stellar-libraries/miles-library.php)
- **CALSPEC:** FITS binary tables with `WAVELENGTH` and `FLUX` columns. (https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec)

Set the template wavelength convention and resolution from the template's
documented metadata. `Templates.libraries.<library>` can provide
`wavelength_medium` (`air` or `vacuum`) and either `fwhm_angstrom` or
`resolving_power`. `Templates.files.<template_file>` provides file-specific
overrides. Vacuum templates are converted to air with the corresponding flux
density transformation. An unknown wavelength convention is retained and
reported as a warning; the code does not guess a conversion.

### Arc line list and dark library

`Wavecal.refpath` points to a text file containing one reference wavelength in
Angstrom per line, with no header. The supplied `NeAr.csv` uses this format.

Library darks use names such as `Mdark_T<temp>_E<exptime>_<YYYYMMDD>.fit`.
Selection uses FITS metadata and detector compatibility, rather than assuming
the filename is authoritative. The default temperature tolerance is 0.5 degrees C.
The normal preference is an exact-exposure same-night master, an exact-exposure
library master, then a compatible scaled same-night or library master.

If a same-exposure local dark was rejected as incompatible, an exact compatible
replacement is required; scaling does not bypass that rejection. New local
master darks are written to `darkpath` unless `Runtime.update_dark_library`
is `false`.

## Processing workflow

1. **Group frames and build calibration masters.** Combine bias frames and
   compatible dark frames, grouped by exposure time. Associate target and
   reference groups with their neighboring calibration observations.
2. **Build arc masters.** Each distinct associated set of arc frames is combined
   once and shared by the applicable reduction tasks.
3. **Reduce each science and reference frame.** Apply bias and dark correction,
   trace the aperture, extract the spectrum, and extract its associated arc
   along the same trace. Independent tasks run in separate worker processes.
4. **Calibrate wavelengths.** Fit the arc solution and apply dispersion
   correction. A heliocentric velocity correction is attempted; a failed
   correction is recorded and the uncorrected wavelength grid is retained.
   Check the output wavelength-frame metadata and warnings.
5. **Fit reference responses.** Compare reference-star spectra with their
   templates, matching spectral resolution when the required information is
   available, and propagate the response uncertainty.
6. **Correct target spectra.** Interpolate the bracketing reference groups'
   log responses in airmass, divide the target spectrum by that response, and
   normalize in a fixed wavelength window.

Flat frames are recognized and grouped, but **flat division is not applied**.
The operating pipeline uses one aperture. In A+C extraction, the signed anchor
is built once after the final cosmic-ray refinement pass. Insufficient
independent anchor support retains the frame-level failure policy.

## Relative flux and uncertainties

By default, the nominal response-corrected spectrum is divided by its median
over **5400-5600 Angstrom** on the output science wavelength grid. Every Monte
Carlo realization uses its own median over the same selected wavelength bins.
`FLUXERR` is the sample standard deviation of those normalized realizations.

The normalization requires at least 20 valid bins and 90% support across the
configured window. Coverage is measured by supported bin widths within the
window, not simply by counting bins. Insufficient support produces an error;
the pipeline does not substitute the full available spectrum or another window.

The output is **relative flux, not absolute spectrophotometry**. Uncertainties
include modeled spectral correlations, response uncertainty, and normalization
uncertainty. They are conditional statistical errors: template mismatch and
other unmodeled systematic errors, and cross-frame covariance from shared
calibration masters, are not included. If a template has no statistical error
information, its flux is treated as fixed and a warning is recorded.

## Configuration

Start from [config.example.yaml](config.example.yaml). Units and key meanings
are documented there; the main settings are summarized below.

| Section / key | Meaning |
| --- | --- |
| `workpath`, `savepath` | Parent directories for per-run work and published products |
| `darkpath`, `refpath` | Master-dark and standard-template directories |
| `memory_limit` | Worker memory budget in decimal GB; also constrains parallelism |
| `longitude`, `latitude`, `elevation` | Observatory position: degrees east/north and metres |
| `CCD.gain`, `CCD.rdnoise` | Electrons per ADU and read noise in electrons |
| `CCD.datay_start`, `CCD.datay_end` | Detector rows to retain, start inclusive and end exclusive |
| `CCD.saturation` | Optional saturation threshold in ADU; otherwise determined from available detector metadata |
| `Trace.naper` | Use `1` for the operating pipeline |
| `Trace.dW`, `Trace.dH` | Dispersion median-slice half-width and spatial search-box half-width, in pixels |
| `Trace.fwhm` | Legacy name for the spatial peak-detection filter width; not a fitted Gaussian FWHM |
| `Trace.threshold` | Traced peak excess above the local median; the initial central search uses its own fixed threshold |
| `Trace.order`, `Wavecal.order` | Polynomial degrees for trace and wavelength fitting |
| `Wavecal.lindisp` | Initial dispersion estimate in Angstrom per pixel |
| `Wavecal.window` | Gaussian peak-fitting half-window in pixels |
| `Wavecal.wmin`, `Wavecal.wmax` | Accepted Gaussian sigma range in pixels |
| `Wavecal.prominence` | Arc peak prominence threshold |
| `Trace.plotyn`, `Wavecal.plotyn`, `Resgen.plotyn` | Write diagnostic plots when enabled |

### Extraction

| `Extract` key | Meaning |
| --- | --- |
| `method` | `ac` by default; `horne` retains the earlier extraction path |
| `raper` | Aperture half-width in pixels |
| `bgin`, `bgout` | Background-sideband radii; require `raper < bgin < bgout` |
| `crreject`, `niter` | Positive cosmic-ray rejection threshold in sigma and maximum rejections per column; `niter: 0` disables rejection |
| `nk`, `xbin`, `xsmooth` | Profile basis count, initial training-bin width, and smoothing scale |
| `profile_floor` | Positive profile-weight floor |
| `ac_training_bin` | Training columns per independent fold; default `64` |
| `ac_window` | Anchor pooling **half-width** in original wavelength columns; default `128` |
| `ac_cr_refinements` | Maximum final cosmic-ray refinement passes; default `3` |
| `mc_samples` | Optional extraction draw count; otherwise uses `Resgen.mc_samples` |
| `mc_seed` | Optional extraction Monte Carlo seed |

Batching does not bin or merge wavelength samples. The 512-column computation
blocks are separate from the profile-training bins and Monte Carlo draw batches.

### Response and normalization

| `Resgen` key | Meaning |
| --- | --- |
| `mc_samples` | Monte Carlo draw count; default `256`, minimum `8` |
| `order` | Response spline interval setting, not the polynomial degree; the spline is cubic |
| `lines` | Optional masked regions as `name: {center, width}` in Angstrom; defaults include Balmer and selected telluric bands |
| `clip_sigma`, `fit_iterations`, `n_shifts` | Response clipping threshold, clipping iterations, and shifted knot grids; defaults `2.5`, `5`, and `3` |
| `seed` | Response Monte Carlo base seed; default `42` |
| `match_resolution` | Match observation and template resolution when supported; default `true` |
| `observed_fwhm_angstrom` | Optional fixed observed spectral FWHM instead of the arc estimate |
| `relflux_norm_window` | Fixed science-grid normalization interval; default `[5400.0, 5600.0]` Angstrom |
| `relflux_norm_min_coverage` | Minimum supported wavelength-width fraction; default `0.9` |
| `relflux_norm_min_bins` | Minimum valid bins within the window; default `20` |

Use consistent extraction and response draw counts; downstream propagation uses
their common sample IDs. Legacy `Resgen.lambda`, `pvalue`, and `dualyn` settings
are not used by the current response-file pipeline.

### Runtime and coordinates

Optional runtime controls can be added to the YAML configuration:

```yaml
Runtime:
    max_workers: 4
    offline: true
    update_dark_library: false
```

The default CPU-based worker limit is `max(1, CPU_count // 2 - 2)`. Job count and
the memory budget further limit the number of workers, including when
`max_workers` is set. Numerical threads are limited within the worker stages.

Set `Coordinates.<object>.ra_deg` and `Coordinates.<object>.dec_deg` for explicit
coordinates in degrees. Resolution order is configuration, `OBJCTRA`/`OBJCTDEC`,
`RA`/`DEC`, then online name lookup. With `Runtime.offline: true`, online name
lookup and IERS automatic downloads are disabled for those processing stages;
coordinates must be available from configuration or headers.

## Outputs and run status

Each run writes to `<workpath>/run-<UTC timestamp>-<id>/`. When all target
reductions finish without failed stage events, the selected products are
published to `<savepath>/<observation-directory-name>/`, verified, and the
working directory is removed on successful cleanup. Incomplete runs retain
their working products and do not publish a new completed product set.
Publication and cleanup failures are recorded in the manifest.

Use separate observation, working, and output directories. Existing output
collisions are handled through the controller's staged publication policy.

| Product | Content |
| --- | --- |
| `T<group>_<target>_relflux_<suffix>.fits` | Final relative-flux spectrum and uncertainty samples |
| `*.ec.fits` | Extracted spectra on native detector columns |
| `*.ec_dc.fits` | Dispersion-corrected spectra |
| `*.res.fits` | Reference-star response functions |
| `*.png`, `*.pdf` | Enabled trace, wavelength, and response diagnostics |
| `run_manifest.json`, `pipeline.log` | Run status, stage events, warnings, and recorded provenance |
| `effective_config.yaml`, `grouping_summary.json` | Applied configuration and frame associations |

The final `SPECTRUM` table contains `WAVELENGTH`, `FLUX`, `FLUXERR`, `MASK`,
`RESP_SCATTER`, `QUALITY`, `TARGQUAL`, and `REFQUAL`. Wavelengths use the air
convention in Angstrom; inspect `SPECSYS` and correction metadata for the
velocity frame. `MASK` marks invalid bins. Quality flags can also describe
retained data, so a nonzero quality value is not automatically a fatal error.
`RESP_SCATTER` describes reference-frame log-response scatter and is distinct
from `FLUXERR`.

`MC_RELFLUX` contains the normalized Monte Carlo realizations; `MC_INDEX`
identifies them. Normalization and uncertainty assumptions are recorded in
headers including `NORMTYPE`, `NORMWLO`, `NORMWHI`, `NORMNPIX`, `NORMCOV`,
`ERRMODEL`, `SYSERR`, and `CROSSCAL`.

`main` returns the run manifest as a dictionary. Its status is `completed`,
`completed_with_warnings`, `partial_success`, or `failed`. Inspect that status
and the recorded events rather than relying only on the presence of a file.

## Documentation

The Sphinx documentation includes a user guide and a searchable API reference
for all 210 module-level functions and classes. API descriptions and signatures
are read directly from the current source and docstrings.

Build the HTML site from the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-docs.txt
python docs/build.py
```

Open `docs/_build/html/index.html` to read it locally. Navigation, search,
and source-code links are included. The build treats warnings as errors and
imports the real runtime, so the scientific dependencies must be installed.

Maintained pages live under `docs/`; generated HTML and build caches are ignored
by Git. For static hosting, publish only `docs/_build/html/`, without the
separate `docs/_build/doctrees/` cache.

### GitHub Pages

[The documentation workflow](.github/workflows/documentation.yml) runs on
pushes to `main` and pull requests targeting `main`.
It installs the real scientific dependencies, runs the portable tests, and
builds Sphinx with warnings treated as errors. Only successful builds from
`main` pushes publish to GitHub Pages. Merging a pull request into `main`
creates such a push and updates the site. Pull requests are checked without
publishing their content; pushes to `develop` do not run this workflow.
There is no manual deployment trigger. A local `git pull` downloads commits
and does not trigger a server-side build or deployment.

For the initial repository setup, use `main` as the default branch, select
**Settings > Pages > Build and deployment > Source: GitHub Actions**, and
ensure the `github-pages` environment permits deployments only from `main`.
No personal access token is needed by the workflow; GitHub supplies its
job-scoped deployment token.

For a new personal public repository, the optional
[PowerShell publishing helper](tools/publish_github.ps1) creates `KHOSTspec`
from a clean source export, pushes `main`, creates a `develop` branch for
ongoing work, and configures Pages. Install
GitHub CLI and run
`gh auth login --web --hostname github.com --git-protocol https --scopes repo,workflow`
first. The helper refuses an existing local Git repository or an existing
remote `KHOSTspec`, so it cannot publish the workspace's historical commits.
Run it from the source export after reviewing the files:

```powershell
powershell -NoProfile -File tools/publish_github.ps1
```

The helper requires a PowerShell policy that permits the reviewed local script.
It was syntax-checked, but this workstation's script execution policy prevented
an execution test. If your policy blocks it, use an approved terminal or perform
the documented repository and Pages setup through GitHub rather than disabling
your security policy.

The site URL is normally `https://<your-login>.github.io/KHOSTspec/`. The
deployment job reports the actual URL. See the documentation's development
guide for setup and deployment troubleshooting.

## Tests and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

The public test suite contains six tests for path handling and the entry
module's delegation API. It does not require observation files.

The pre-release local review also passed 308 regression tests. Refactoring
was checked on two real observation strips, each 61 by 6,216 pixels, with 256 Monte Carlo
realizations: all 30 compared output arrays and 41 diagnostic values per frame
were identical before and after the refactor. Those observation-dependent
fixtures and historical validation archives are not part of this repository.

## Local data and publication

The included `.gitignore` allows the maintained source, example configuration,
dependency files, README, documentation sources, arc line list, and portable tests. Local configuration
files, observation data, products, and study archives are excluded.

Generated FITS headers, manifests, and logs can retain original observation
metadata and local paths. Review them separately before sharing reduced data.
Ignore rules do not remove information from existing Git history; a source-only
export contains no repository history.
