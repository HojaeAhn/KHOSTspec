"""Command-line entry point and compatible worker aliases for spectral reduction.

Run ``python spec_reduction_parallel.py OBSERVATIONS CONFIG.yaml``. Relative
paths are interpreted from the current working directory. Scientific stages
and process-pool orchestration are implemented in ``speclibparallel``.
"""

from __future__ import annotations

import argparse
import warnings
from scipy.optimize import OptimizeWarning
from astropy.wcs import FITSFixedWarning

# Keep the public stage aliases available to existing import-based callers.
from speclibparallel import (
    file_sort, combine_flat, combine_arc, calibrate_science_frames, run_trace,
    get_ref_template, als_continuum, als_continuum_ref_Touptek,
    als_continuum_ref_QSI, make_response, generate_response_spline,
    response_correction, optimal_extract_fits, get_wavelength_solution, dispcor)

__all__ = [
    "file_sort", "combine_flat", "combine_arc", "calibrate_science_frames",
    "run_trace", "get_ref_template", "als_continuum", "als_continuum_ref_Touptek",
    "als_continuum_ref_QSI", "make_response", "generate_response_spline",
    "response_correction", "optimal_extract_fits", "get_wavelength_solution",
    "dispcor", "make_group_masters_worker", "process_single_frame_worker",
    "main", "cli",
]

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings(
    "ignore", message=r"^'datfix' made the change.*", category=FITSFixedWarning)


def make_group_masters_worker(task):
    """Build the task's shared arc master and return its structured stage result."""
    from speclibparallel import make_group_masters_worker as implementation
    return implementation(task)


def process_single_frame_worker(task):
    """Reduce one task frame and return output paths, warnings, and stage status."""
    from speclibparallel import process_single_frame_worker as implementation
    return implementation(task)


def main(obspath, configpath):
    """Reduce an observation directory with a YAML configuration.

    Parameters
    ----------
    obspath : str or pathlib.Path
        Directory containing the observation FITS files.
    configpath : str or pathlib.Path
        YAML configuration file. Relative paths use the working directory.

    Returns
    -------
    dict
        Run status, stage outcomes, warnings, and published product paths.
        Working products and a run manifest are written by the controller.
    """
    from speclibparallel import main as implementation
    return implementation(obspath, configpath)


def cli(argv=None):
    """Parse optional argument strings and return the controller's run report.

    ``argv=None`` reads the process command line. Observation and configuration
    paths are forwarded unchanged, preserving absolute paths and paths with
    spaces. Missing arguments produce argparse's standard usage error.
    """
    parser = argparse.ArgumentParser(
        description="KHAO spectral reduction pipeline"
    )
    parser.add_argument(
        "obspath",
        help="Path of data directory"
    )
    parser.add_argument(
        "configpath",
        help="Path of YAML config file"
    )

    args = parser.parse_args(argv)
    return main(args.obspath, args.configpath)


if __name__ == "__main__":
    cli()
