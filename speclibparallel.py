"""KHAO spectral reduction library, including calibration, A+C and execution.

The controller and scientific stages live in this ordinary Python module.
Independent column kernels are included here with the extraction stages.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')

import ccdproc as ccdp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy import constants, units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData, VarianceUncertainty
from astropy.stats import mad_std, sigma_clip
from astropy.time import Time, TimeDelta
from astropy.utils import iers
from astropy.wcs import FITSFixedWarning
from numba import jit
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy import sparse
from scipy.interpolate import CubicSpline, LSQUnivariateSpline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit, nnls
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import NMF
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
from specutils.utils.wcs_utils import vac_to_air
from threadpoolctl import threadpool_limits


# =============================================================================
# Trace, wavelength and support functions
# =============================================================================

PROFILE_SMOOTH_SIGMA = 1


@jit(nopython=True, cache=True)
def gaussian_jit(x, amplitude, center, sigma, offset):
    """
    JIT-compiled Gaussian function for faster fitting.
    """
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + offset


def gaussian(x, amplitude, center, sigma, offset):
    """
    Wrapper for gaussian function compatible with scipy.optimize.
    """
    return gaussian_jit(x, amplitude, center, sigma, offset)


def inv_median(a):
    """
    Inverse median function used for scaling in ccdproc.combine.
    """
    return 1 / np.median(a)


def fit_gaussian_peak(integer_pixel, df, window=3, sig_min=0.1, sig_max=3):
    """
    Refines peak position using Gaussian fitting.
    Uses the JIT-accelerated 'gaussian' wrapper defined above.
    """
    try:
        p_idx = integer_pixel
        if p_idx < window or p_idx >= len(df) - window:
            return np.nan  # edge

        window_indices = np.arange(p_idx - window, p_idx + window + 1)
        x_data = df['pixel'].iloc[window_indices].values.astype(float)
        y_data = df['flux_adu'].iloc[window_indices].values.astype(float)

        offset_guess = min(y_data[0], y_data[-1])
        amplitude_guess = y_data[window] - offset_guess
        p0 = [amplitude_guess, float(integer_pixel), 1.0, offset_guess]

        popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0)

        gaussian_center = popt[1]
        gaussian_sigma = popt[2]

        if abs(gaussian_center - integer_pixel) > window:
            return np.nan
        if gaussian_sigma > sig_max:
            return np.nan
        if gaussian_sigma <= sig_min:
            return np.nan

        return gaussian_center

    except (RuntimeError, ValueError) as e:
        return np.nan


def get_ccd_temp(hdr) -> float:
    """Read the preferred detector temperature in degrees Celsius.

    ``SET-TEMP`` takes precedence over measured-temperature alternatives.
    Missing or unparsable values return the legacy sentinel ``999.0``.
    """
    val = get_header_any(hdr, "SET-TEMP", "CCD-TEMP", "TEMP", "TELESCOP_TEMP")
    if val is not None:
        try:
            return float(val)
        except:
            return 999.0
    return 999.0


def parse_library_filename(filename):
    """Parse temperature, exposure seconds, and date from a master-dark name.

    Names follow ``Mdark_T<temperature>_E<exposure>_<date>.fit``. Return
    ``(None, None, None)`` when the filename does not match that convention.
    """

    pattern = r"Mdark_T([-+]?\d+\.?\d*)_E([-+]?\d+\.?\d*)_(\d+)\.fit"

    match = re.search(pattern, filename)
    if match:
        try:
            temp = float(match.group(1))
            exptime = float(match.group(2))
            date_str = match.group(3)  # YYYYMMDD
            return temp, exptime, date_str
        except ValueError:
            return None, None, None

    return None, None, None


def find_nearest_dark_exposure(image_exposure_time, dark_exposure_times, tolerance=None):
    """
    Find the nearest exposure time of a dark frame to the exposure time of the image,
    raising an error if the difference in exposure time is more than tolerance.
    
    Parameters
    ----------
    
    image_exposure_time : float
        Exposure time of the image, in seconds.
    
    dark_exposure_times : list
        Exposure times for which there are darks.
    
    tolerance : float or ``None``, optional
        Maximum difference, in seconds, between the image and the closest dark. Set
        to ``None`` to skip the tolerance test.
    
    Returns
    -------
    
    int
        Index of the closest exposure in ``dark_exposure_times``.
    """

    idx = np.argmin(np.abs(np.array(dark_exposure_times) - image_exposure_time))
    closest_dark_exposure = dark_exposure_times[idx]

    if (tolerance is not None and 
        np.abs(image_exposure_time - closest_dark_exposure) > tolerance):
        
        raise RuntimeError('Closest dark exposure time is {} for flat of exposure '
                           'time {}.'.format(closest_dark_exposure, image_exposure_time))
        
    
    return idx


def norm_obj(s: Optional[str]) -> str:
    """Normalize an object name by removing spaces and ignoring letter case."""
    return (s or "").strip().replace(" ", "").lower()


def get_header_any(hdr, *keys, default=None):
    """Return the first matching header value, then try case-insensitive keys."""
    for k in keys:
        if k in hdr:
            return hdr[k]
    low = {k.lower(): k for k in hdr.keys()}
    for k in keys:
        kl = k.lower()
        if kl in low:
            return hdr[low[kl]]
    return default


def get_jd(hdr) -> Optional[float]:
    """Read a Julian date, converting supported modified-Julian-date headers.

    Prefer ``JD``, then ``JD-OBS``, ``MJD``, and ``MJD-OBS``. Return ``None``
    when no supported value can be parsed; an invalid preferred ``JD`` also
    returns ``None`` without falling through to other keywords.
    """
    jd = get_header_any(hdr, "JD")
    if jd is not None:
        try:
            return float(jd)
        except Exception:
            return None
    for key in ("JD-OBS", "MJD", "MJD-OBS"):
        val = get_header_any(hdr, key)
        if val is not None:
            try:
                v = float(val)
                if key.startswith("MJD"):
                    return v + 2400000.5
                return v
            except Exception:
                continue
    return None


def get_exptime(hdr) -> Optional[float]:
    """Return exposure seconds from supported FITS keywords, or ``None``."""
    val = get_header_any(hdr, "EXPTIME", "EXPOSURE", "ITIME", "DARKTIME")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def classify(hdr, TARGET_NAMES, REFERENCE_NAMES) -> str:
    """Classify a FITS header as calibration, target, reference, or other.

    Match target and reference names case-insensitively after removing spaces.
    Supported classes are bias, dark, flat, arc, target, reference, and other.
    """
    imagetype = get_header_any(hdr, "IMAGETYP", "IMAGETYPE", default="")
    obj = get_header_any(hdr, "OBJECT", default="")
    it = str(imagetype).strip().lower()
    o = norm_obj(obj)

    if it == "bias frame":
        return "bias"
    if it == "dark frame":
        return "dark"
    if it == "flat field":
        return "flat"
    if it == "light frame":
        if o == norm_obj("Arc"):
            return "arc"
        if o in [norm_obj(x) for x in TARGET_NAMES]:
            return "target"
        if o in [norm_obj(x) for x in REFERENCE_NAMES]:
            return "reference"
    return "other"


@dataclass
class FileRec:
    """Store one input filename, Julian date, exposure seconds, and frame class."""

    path: str
    jd: float
    imagetype: str
    obj: str
    exptime: Optional[float]
    kind: str


@dataclass
class Group:
    """Collect a contiguous frame run and its associated calibration group IDs."""

    gid: int
    kind: str  # 'flat'|'arc'|'target'|'reference'
    files: List[str] = field(default_factory=list)
    start_jd: float = math.nan
    end_jd: float = math.nan
    center_jd: float = math.nan
    attach: Dict[str, List[int]] = field(default_factory=lambda: {"reference": [], "flat": [], "arc": []})

    def finalize(self, jd_list: List[float]):
        """Set the run's time bounds and arithmetic-mean JD from a nonempty list."""
        self.start_jd = min(jd_list)
        self.end_jd = max(jd_list)
        self.center_jd = sum(jd_list) / len(jd_list)


@dataclass
class Summary:
    """Collect bias files, exposure-indexed darks, and classified frame groups."""

    bias_files: List[str] = field(default_factory=list)
    darks_by_exptime: Dict[str, List[str]] = field(default_factory=dict)
    groups: Dict[str, List[Group]] = field(default_factory=lambda: {"flat": [], "arc": [], "target": [], "reference": []})


def scan_fits(root: Path, TARGET_NAMES, REFERENCE_NAMES) -> List[FileRec]:
    """Recursively read FITS headers and return recognized frames sorted by JD.

    Unreadable headers, unclassified frames, and frames without a usable
    Julian date are skipped. Image arrays are not loaded.
    """
    if not(isinstance(root, Path)):
        root = Path(root)
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".fits", ".fit", ".fts"):
            continue
        try:
            hdr = fits.getheader(p, ignore_missing_end=True)
        except Exception:
            continue
        kind = classify(hdr, TARGET_NAMES, REFERENCE_NAMES)
        if kind == "other":
            continue
        jd = get_jd(hdr)
        if jd is None:
            continue
        files.append(
            FileRec(
                path=str(p),
                jd=float(jd),
                imagetype=str(get_header_any(hdr, "IMAGETYP", "IMAGETYPE", default="")),
                obj=str(get_header_any(hdr, "OBJECT", default="")),
                exptime=get_exptime(hdr),
                kind=kind,
            )
        )
    files.sort(key=lambda r: r.jd)
    return files


def build_runs(recs: List[FileRec]) -> Dict[str, List[Group]]:
    """Group time-ordered records into consecutive runs of the same frame class.

    Reference runs contain at most three frames. Other supported runs split
    only when the frame class changes; object names do not define boundaries.
    """
    groups: Dict[str, List[Group]] = {"flat": [], "arc": [], "target": [], "reference": []}
    next_gid = 1

    current: Optional[Group] = None
    jd_list_for_current: List[float] = []

    def close_current():
        """Finalize the active run and reset its accumulated frame times."""
        nonlocal current, jd_list_for_current
        if current is None:
            return
        current.finalize(jd_list_for_current)
        groups[current.kind].append(current)
        current = None
        jd_list_for_current = []

    for rec in recs:
        if rec.kind not in groups:
            close_current()
            continue

        if (current is None) or (rec.kind != current.kind):
            close_current()
            current = Group(gid=next_gid, kind=rec.kind)
            next_gid += 1
            current.files.append(rec.path)
            jd_list_for_current = [rec.jd]

        else:
            if rec.kind == "reference" and len(current.files) >= 3:
                close_current()
                current = Group(gid=next_gid, kind=rec.kind)
                next_gid += 1
                current.files.append(rec.path)
                jd_list_for_current = [rec.jd]
            else:
                current.files.append(rec.path)
                jd_list_for_current.append(rec.jd)

    close_current()
    for k in groups:
        groups[k].sort(key=lambda g: g.center_jd)
    return groups


def pick_before_after(base_center: float, cand_centers: List[float]) -> Tuple[Optional[int], Optional[int]]:
    """Return bracketing indices in sorted candidate times, using ``None`` at ends.

    A candidate exactly at ``base_center`` belongs to the after side.
    """
    if not cand_centers:
        return None, None
    i = bisect.bisect_left(cand_centers, base_center)
    before_idx = i-1 if i-1 >= 0 else None
    after_idx = i if i < len(cand_centers) else None
    return before_idx, after_idx


def attach_nearest(summary: Summary):
    """Attach calibration and standard groups to science runs in place.

    Flats and arcs cannot cross another science run on the time-ordered
    group timeline. Targets use the preceding and following standards only
    when their object names agree, or the following standard when none precedes.
    """
    centers = {k: [g.center_jd for g in summary.groups[k]] for k in summary.groups}

    # Calibration adjacency is defined on the shared timeline of all groups.
    all_groups: List[Group] = []
    for kind in ("flat", "arc", "target", "reference"):
        all_groups.extend(summary.groups[kind])
    all_groups.sort(key=lambda g: g.center_jd)

    gid_to_index: Dict[int, int] = {g.gid: idx for idx, g in enumerate(all_groups)}

    science_kinds = {"target", "reference"}

    def find_before_after_calib(science_group: Group, calib_kind: str) -> Tuple[Optional[int], Optional[int]]:
        """Find bracketing calibration group IDs without crossing a science run.

        Return ``(before_gid, after_gid)`` with ``None`` for an absent neighbor.
        """
        idx = gid_to_index[science_group.gid]

        before_gid = None
        j = idx - 1
        while j >= 0:
            g = all_groups[j]
            if g.kind in science_kinds:
                break
            if g.kind == calib_kind:
                before_gid = g.gid
                break
            j -= 1

        after_gid = None
        j = idx + 1
        while j < len(all_groups):
            g = all_groups[j]
            if g.kind in science_kinds:
                break
            if g.kind == calib_kind:
                after_gid = g.gid
                break
            j += 1

        return before_gid, after_gid

    def get_group_object_name(grp: Group) -> str:
        """Read the first frame's normalized object name, or return ``unknown``."""
        try:
            with fits.open(grp.files[0]) as hdul:
                return norm_obj(hdul[0].header.get("OBJECT", ""))
        except:
            return "unknown"

    for tgt in summary.groups["target"]:
        before_idx, after_idx = pick_before_after(tgt.center_jd, centers["reference"])

        name_before = None

        if before_idx is not None:
            ref_group_before = summary.groups["reference"][before_idx]
            tgt.attach["reference"].append(ref_group_before.gid)
            name_before = get_group_object_name(ref_group_before)

        if after_idx is not None and after_idx != before_idx:
            ref_group_after = summary.groups["reference"][after_idx]

            if name_before is not None:
                name_after = get_group_object_name(ref_group_after)

                if name_before == name_after:
                    tgt.attach["reference"].append(ref_group_after.gid)
            else:
                tgt.attach["reference"].append(ref_group_after.gid)

        for k in ("flat", "arc"):
            before_gid, after_gid = find_before_after_calib(tgt, k)
            if before_gid is not None:
                tgt.attach[k].append(before_gid)
            if after_gid is not None and after_gid != before_gid:
                tgt.attach[k].append(after_gid)

    # Standards retain their own flat and arc associations.
    for refg in summary.groups["reference"]:
        for k in ("flat", "arc"):
            before_gid, after_gid = find_before_after_calib(refg, k)
            if before_gid is not None:
                refg.attach[k].append(before_gid)
            if after_gid is not None and after_gid != before_gid:
                refg.attach[k].append(after_gid)


def expand_target_results(summary: Summary) -> List[dict]:
    """Expand group IDs into per-target filenames and associated reference blocks.

    Return dictionaries sorted by target-center JD. Each target and each of
    its standards retains separate flat and arc file lists. Read object names
    from the first FITS header in each run, falling back to ``Unknown``.
    """
    gid_map: Dict[int, Group] = {}
    for kind in summary.groups:
        for g in summary.groups[kind]:
            gid_map[g.gid] = g

    results = []

    # Cache each standard's calibration files for targets sharing that group.
    ref_to_flat_arc: Dict[int, Dict[str, List[str]]] = {}
    for refg in summary.groups["reference"]:
        try:
            hdr = fits.getheader(refg.files[0], ignore_missing_end=True)
            reference_name = str(get_header_any(hdr, "OBJECT", default="Unknown")).strip()
        except Exception:
            reference_name = "Unknown"

        flat_files, arc_files = [], []

        for gid in refg.attach.get("flat", []):
            gg = gid_map.get(gid)
            if gg:
                flat_files.extend(gg.files)
        for gid in refg.attach.get("arc", []):
            gg = gid_map.get(gid)
            if gg:
                arc_files.extend(gg.files)

        ref_to_flat_arc[refg.gid] = {
            "reference_name": reference_name,
            "reference_flat_files": sorted(set(flat_files)),
            "reference_arc_files": sorted(set(arc_files)),
        }

    for tgt in summary.groups["target"]:
        try:
            hdr = fits.getheader(tgt.files[0], ignore_missing_end=True)
            target_name = str(get_header_any(hdr, "OBJECT", default="Unknown")).strip()
        except Exception:
            target_name = "Unknown"

        target_files = list(tgt.files)

        target_flat_files, target_arc_files = [], []
        for gid in tgt.attach.get("flat", []):
            g = gid_map.get(gid)
            if g:
                target_flat_files.extend(g.files)
        for gid in tgt.attach.get("arc", []):
            g = gid_map.get(gid)
            if g:
                target_arc_files.extend(g.files)

        references_block = []
        for ref_gid in tgt.attach.get("reference", []):
            refg = gid_map.get(ref_gid)
            if not refg:
                continue

            ref_name = ref_to_flat_arc.get(ref_gid, {}).get("reference_name", "Unknown")
            reference_files = list(refg.files)

            ref_flat_files = ref_to_flat_arc.get(ref_gid, {}).get("reference_flat_files", [])
            ref_arc_files  = ref_to_flat_arc.get(ref_gid, {}).get("reference_arc_files", [])

            references_block.append({
                "reference_name": ref_name,
                "reference_files": sorted(set(reference_files)),
                "reference_flat_files": ref_flat_files,
                "reference_arc_files": ref_arc_files,
            })

        results.append({
            "target_name": target_name,
            "target_center_jd": tgt.center_jd,
            "target_files": target_files,
            "target_flat_files": sorted(set(target_flat_files)),
            "target_arc_files":  sorted(set(target_arc_files)),
            "references": references_block,
        })

    results.sort(key=lambda x: x["target_center_jd"])
    return results


def file_sort(filepath, targets, references):
    """Scan an observation directory and return target/standard calibration plans.

    ``targets`` and ``references`` are object-name sequences. The result is a
    list of per-target dictionaries from :func:`expand_target_results`; no
    observation files are moved or modified.
    """
    ROOT = filepath
    TARGET_NAMES = targets
    REFERENCE_NAMES = references
    
    recs = scan_fits(ROOT, TARGET_NAMES, REFERENCE_NAMES)
    
    summary = Summary()
    for r in recs:
        if r.kind == "bias":
            summary.bias_files.append(r.path)
        elif r.kind == "dark":
            key = f"{r.exptime:.6f}" if r.exptime is not None else "unknown"
            summary.darks_by_exptime.setdefault(key, []).append(r.path)
    
    run_candidates = [r for r in recs if r.kind in ("flat", "arc", "target", "reference")]
    summary.groups = build_runs(run_candidates)
    
    attach_nearest(summary)
    
    target_results = expand_target_results(summary)
    
    return target_results


def run_trace(dat, NAP, CHEV_ORDER, APT_FIT_PLOT, fname, APTdW, APTdH, APTFWHM, APTTHRES, WORKPATH):
    """Fit spatial trace positions along a two-dimensional detector image.

    ``dat`` is indexed by spatial row and dispersion column; widths and trace
    positions are in pixels. Return ascending Chebyshev coefficients for the
    final fitted aperture. The pipeline uses one aperture. When requested,
    write trace-finding and fit diagnostics beneath ``WORKPATH``.
    """
    H, W = dat.shape
    halfW = int(W/2) 
    # Initialize the trace from a median slice near the detector midpoint.
    ys = np.arange(H)
    cols = np.median(dat[:,(halfW-APTdW):(halfW+APTdW)], axis=1)
    fcols = cr_reject(cols, nsigma=10, npix=3)
    cys0, cpeaks0 = find_emission(ys, fcols, thres=200, width=APTFWHM)

    NPEAKS = len(cys0)
    cys, cpeaks = [], []
    for k in range(NPEAKS): 
        if cpeaks0[k] < (cpeaks0[k-1]+cpeaks0[-(NPEAKS-k-1)])/10.0:
            print('SIDE-CHECK: %i, %i, %i, %i' % \
                   (cys0[k], cpeaks0[k], cpeaks0[k-1], cpeaks0[-(NPEAKS-k-1)]))
            continue
        cys.append(cys0[k])
        cpeaks.append(cpeaks0[k])
    cys, cpeaks = np.array(cys), np.array(cpeaks)
    
    cys, cpeaks = cys[:NAP], cpeaks[:NAP]
    cap = np.arange(NAP)+1
    
    if APT_FIT_PLOT: 
        fig, ax = plt.subplots(num=1, figsize=(12,6))
        ax.plot(ys, cols, 'g-', lw=4, alpha=0.4)
        ax.plot(ys, fcols, 'b-', lw=1)
        ax.plot(cys, cpeaks, 'ro', ms=4, alpha=0.7)
        ax.plot(cys-APTdH, cpeaks, 'r|', ms=15)
        ax.plot(cys+APTdH, cpeaks, 'r|', ms=15)
        ax.set_title('APFIND in the middle plane')
        fig.savefig(WORKPATH / f"APFIND_{fname.split('.')[0]}")
        plt.close('all')
    
    if APT_FIT_PLOT:
        fig, ax = plt.subplots(num=2, figsize=(20,10))
        ldat = np.arcsinh(dat)
        z1, z2 = np.percentile(ldat,60), np.percentile(ldat,95)
        ax.imshow(ldat, vmin=z1, vmax=z2, cmap='Blues')
        cxs = np.zeros_like(cys)+halfW
        for ap, xo, yo in zip(cap, cxs, cys):
            ax.plot(xo, yo, 'r.')
            ax.text(xo, yo, ap, fontsize=10, color='r')
        ax.grid()

    for j in range(NAP)[:]:
        xc, yc = [], [] 
        
        for k in [-1, 1]:
            nerr = 0
            # Follow each trace outward from the midpoint in both directions.
            y, _ = cys[j], cpeaks[j]
            for i in range(halfW, int(W*(1-k)/2)+k*APTdW, -k*2*APTdW):
                if len(yc) > (CHEV_ORDER+1)*2:
                    ctmp = chebfit(xc, yc, CHEV_ORDER)
                    y = chebval(i, ctmp)
                ind1, ind2 = int(y-APTdH), int(y+APTdH)
                ycut = np.arange(ind1, ind2)
                vcut = np.median(dat[ind1:ind2,(i-APTdW):(i+APTdW)], axis=1)
                y_tmp, peak_tmp = find_emission(ycut, vcut, thres=APTTHRES, width=APTFWHM)
                if len(y_tmp) == 0: 
                    nerr += 1
                    if nerr > 5: break
                    continue
               
                mm = np.argmin((y_tmp-y)**2)
                y_tmp, peak_tmp = y_tmp[mm], peak_tmp[mm]
                xc.append(i)
                yc.append(y_tmp)
           
        xc = np.array(xc)
        yc = np.array(yc)
        ss = np.argsort(xc)
        xc, yc = xc[ss], yc[ss]
        coeffs = chebfit(xc, yc, CHEV_ORDER)

        if APT_FIT_PLOT:
            xfit = np.arange(W)
            yfit = chebval(xfit, coeffs)
            ax.plot(xfit, yfit, 'r-', lw=5, alpha=0.3)
         
        if APT_FIT_PLOT:
            fsub = plt.figure(num=99, figsize=(8,4))
            asub = fsub.add_subplot(111)
            yfit = chebval(xc, coeffs)
            rms = np.std(yc - yfit)
            asub.plot(xc, yc, 'k+', mew=1, ms=10)
            asub.plot(xc, yfit, 'r-', lw=3, alpha=0.5, label=f'RMS = {rms:.5f}')
            asub.set_xlabel('X [pixel]')
            asub.set_ylabel('Y [pixel]')
            asub.set_title('AP%02i Trace FIT' % (cap[j],))
            asub.grid()
            asub.legend()
            fsub.savefig(WORKPATH / f"TRACEFIT-AP{cap[j]:02d}_{fname.split('.')[0]}")
            fsub.clf()
    
    if APT_FIT_PLOT:
        ax.set_xlim(0,W)
        ax.set_ylim(H,0)
        ax.set_title('Diagram of APTRACE')
        fig.savefig(WORKPATH / f"diagram-aptrace_{fname.split('.')[0]}.pdf")
        plt.close('all')
    
    return coeffs


def discover_files(workdir: Path):
    """Index conventionally named calibrated FITS files by target/reference group.

    Scan only the immediate directory. Return nested dictionaries of science
    frame paths and master-arc paths; matching master-flat names can establish
    groups but their paths are not included in the result.
    """
    TARGET_RE  = re.compile(r'^(T\d{3})_zdf([^-]+)-(\d{3})\.fit$', re.IGNORECASE)
    REF_RE     = re.compile(r'^(T\d{3})_(R\d{2})_zdf([^-]+)-(\d{3})\.fit$', re.IGNORECASE)
    MARC_RE    = re.compile(r'^(T\d{3})(?:_(R\d{2}))?_marc\.fit$', re.IGNORECASE)
    MFLAT_RE   = re.compile(r'^(T\d{3})(?:_(R\d{2}))?_mflat\.fit$', re.IGNORECASE)
    groups = {}
    
    for f in sorted(workdir.glob('*.fit*')):
        name = f.name
        if TARGET_RE.match(name):
            tgrp, tname, fno = TARGET_RE.findall(name)[0]
            G = groups.setdefault(tgrp, {'science': [], 'marc': None, 'refs': {}})
            G['science'].append(f)
        elif REF_RE.match(name):
            tgrp, rid, rname, fno = REF_RE.findall(name)[0]
            G = groups.setdefault(tgrp, {'science': [], 'marc': None, 'refs': {}})
            R = G['refs'].setdefault(rid, {'science': [], 'marc': None})
            R['science'].append(f)
        elif MARC_RE.match(name):
            tgrp, rid = MARC_RE.findall(name)[0]
            G = groups.setdefault(tgrp, {'science': [], 'marc': None, 'refs': {}})
            if rid:
                R = G['refs'].setdefault(rid, {'science': [], 'marc': None})
                R['marc'] = f
            else:
                G['marc'] = f
        elif MFLAT_RE.match(name):
            tgrp, rid = MFLAT_RE.findall(name)[0]
            G = groups.setdefault(tgrp, {'science': [], 'marc': None, 'refs': {}})
            if rid:
                R = G['refs'].setdefault(rid, {'science': [], 'marc': None})
    return groups


def parse_trace_coeffs(header, order):
    """Read ``TRACE0`` through ``TRACE<order>`` as ascending Chebyshev coefficients."""
    coeffs = []
    for i in range(order+1):
        coeffs.append(float(header[f"TRACE{i}"]))
    return np.array(coeffs, dtype=float)


def get_lambda_from_WLCAL(header, n_pix, order):
    """Evaluate ascending FITS wavelength coefficients at zero-based pixel centers.

    Return a one-dimensional array of ``n_pix`` wavelengths in the stored
    calibration units, normally Angstrom.
    """
    coeff = np.array([header[f"WLCAL{i}"] for i in range(order+1)])
    x = np.arange(n_pix, dtype=float)
    lam = np.polyval(coeff[::-1], x)
    return lam


def make_edges_from_centers(lam):
    """Return N+1 bin edges from N centers using midpoint and endpoint extrapolation.

    ``lam`` must contain at least two centers in a common wavelength unit.
    """
    lam = np.asarray(lam)
    n = lam.size
    edges = np.zeros(n+1, dtype=float)
    edges[1:-1] = 0.5*(lam[:-1] + lam[1:])
    edges[0]    = lam[0] - 0.5*(lam[1] - lam[0])
    edges[-1]   = lam[-1] + 0.5*(lam[-1] - lam[-2])
    return edges


def cr_reject(flux, nsigma=15.0, npix=3, verbose=False):
    """Replace local outliers in a one-dimensional detection profile.

    Compare each interior sample with unflagged neighbors within ``npix``
    positions on either side. Values exceeding ``nsigma`` local standard
    deviations are replaced sequentially by the neighbor median. Return only
    the filtered array; no measurement errors are accepted or propagated.

    Adapted by wskang on 2018-02-06 to operate without error arrays.
    """
    if verbose:  print (nsigma,npix)
    flux, flag = list(flux), list(np.zeros_like(flux))
    i1 = npix
    i2 = len(flux) - npix
    for i in range(i1, i2):
        # make a list of flux values used to find the median
        fl = flux[i-npix:i] + flux[i+1:i+1+npix]
        er = flag[i-npix:i] + flag[i+1:i+1+npix]
        fl = [f for f,e in zip(fl,er) if e == 0]
        medfl = np.median(fl)
        medsg = np.std(fl)
        if np.abs((flux[i] - medfl) / medsg) > nsigma:
            flux[i] = medfl
            flag[i] = flag[i]+1
            if verbose:  print (len(fl), len(er))

    return np.array(flux)


def find_emission(fx, fy, thres=200, width=9):
    """Locate emission peaks by zero crossings of a sawtooth-filtered profile.

    ``fx`` and ``fy`` give matching one-dimensional coordinates and intensities.
    Retain interpolated crossings above the median background plus ``thres``;
    return peak coordinates and intensities, or two empty lists for short input.
    """
    bg_level = np.nanmedian(fy)

    sawtooth = np.array([0,-1,-2,-1,0,1,2,1,0])
    w0 = len(sawtooth)
    npix = len(fy)

    if npix < width:
        return [], []

    if (width % 2) == 0:
        width += 1
    if w0 >= width:
        width = w0
    else:
        d0 = np.arange(w0)/w0
        dp = np.arange(width)/width
        sawtooth = np.interp(dp, d0, sawtooth)
    # Reflect the profile at both boundaries before filtering.
    sfy = np.r_[fy[(width-1):0:-1],fy,fy[-2:(-width-1):-1]]
    pcov = np.convolve(sawtooth, sfy, 'valid')[int(width/2):-int(width/2)]
    cx, cy = [], []

    for i in range(1, npix - 1):
        p1, p2 = pcov[i - 1], pcov[i]
        py = fy[i - 1] + (fy[i] - fy[i - 1]) * (p1) / (p1 - p2)
        px = fx[i - 1] + (fx[i] - fx[i - 1]) * (p1) / (p1 - p2)

        if (p1 < 0) & (p2 > 0) & (py > thres+bg_level):
            cx.append(px)
            cy.append(py)

    return np.array(cx), np.array(cy)


def robust_sigma_logspec(log_spec: np.ndarray) -> float:
    """Calculate robust sigma of log-spectrum for weighting (using SavGol resid)."""
    m = log_spec.copy()
    # Fill NaNs
    nans = np.isnan(m)
    if nans.any():
        x = np.arange(m.size)
        m[nans] = np.interp(x[nans], x[~nans], m[~nans])
        
    # Dynamic Window Size
    win = max(51, (len(m)//50)*2+1)
    win = min(win, len(m) if len(m)%2!=0 else len(m)-1)
    
    smooth = savgol_filter(m, window_length=win, polyorder=2)
    resid = m - smooth
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    return float(max(1.4826 * mad, 1e-6))


def get_variable_mask(wave, lines):
    """Return True outside the configured open intervals around spectral lines.

    Each line supplies ``center`` and half-width ``width`` in the same units
    as ``wave``. Lines with centers outside the wavelength span are ignored.
    """
    mask = np.ones_like(wave, dtype=bool)
    
    for name, params in lines.items():
        center = params['center']
        width = params['width']    
        
        if wave.min() < center < wave.max():
            mask &= ~((wave > center - width) & (wave < center + width))
            
    return mask


def get_adaptive_p(wave, p_high, p_low=0.5, transition_wave=4000, transition_width=500):
    """
    Calculate adaptive asymmetry parameter (p) based on wavelength.

    Generates a sigmoid transition from p_low (at blue end) to p_high (at red end)
    to mitigate noise bias in low-SNR regions (e.g., UV/Blue).

    Args:
        wave (np.array): Wavelength array.
        p_high (float): p-value for high SNR region (e.g., 0.001).
        p_low (float): p-value for low SNR region (default 0.5).
        transition_wave (float): Central wavelength of the transition region.
        transition_width (float): Width of the transition region.

    Returns:
        np.array: Array of p-values with the same shape as wave.
    """
    # Sigmoid function for smooth transition
    # The factor 10 ensures the transition is mostly contained within +/- transition_width/2
    k = 10 / transition_width
    weight_map = 1 / (1 + np.exp(-k * (wave - transition_wave)))

    # Interpolate between p_low and p_high
    p_vector = p_low * (1 - weight_map) + p_high * weight_map
    return p_vector


def als_continuum(filepath, balmer_lines, WORKPATH, lam=1e6, p=0.005, niter=10, RC_PLOT=False,
                  p_low=0.5, trans_wave=3700, trans_width=200):
    """Fit a masked asymmetric-least-squares continuum to a legacy FITS spectrum.

    Read the primary flux array and linear wavelength keywords from
    ``filepath``. ``balmer_lines`` defines excluded wavelength intervals;
    ``lam``, ``p``, and ``niter`` control smoothing, asymmetry, and iterations.
    Return a ``Spectrum`` with Angstrom coordinates and count-density units;
    uncertainties are not propagated by this helper. If ``RC_PLOT`` is true,
    save a continuum plot beneath ``WORKPATH``.

    ``p_low``, ``trans_wave``, and ``trans_width`` are retained for call
    compatibility but are not used; the fit uses the supplied constant ``p``.
    """

    with fits.open(filepath) as hdul:
        header = hdul[0].header
        flux = hdul[0].data

    crpix = header['CRPIX1']
    crval = header['CRVAL1']
    cdelt = header['CDELT1']
    wave = np.arange(len(flux))
    wave = crval + (wave + crpix) * cdelt

    mask = get_variable_mask(wave, balmer_lines)
    y = -flux
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    DTD = D.T.dot(D)
    w = np.ones(L)
    w[~mask] = 0

    for i in range(niter):
        W = sparse.diags([w], [0], shape=(L, L))
        Z = (W + lam * DTD).tocsc()
        z = spsolve(Z, w * y)
        w_new = p * (y > z) + (1 - p) * (y < z)
        w_new[~mask] = 0
        w = w_new

    continuum = -z

    spec = Spectrum(
        spectral_axis=wave * u.AA,
        flux=continuum * (u.ct / u.AA)
    )

    if RC_PLOT:
        fig = plt.figure(figsize=(12, 6))

        plt.plot(wave, flux, 'k-', lw=1, alpha=0.8, label='Flux')
        plt.plot(wave, continuum, 'r-', lw=1, label='Continuum')
        for name, params in balmer_lines.items():
            center = params['center']
            width = params['width']
            if wave.min() < center < wave.max():
                plt.axvspan(center - width, center + width, color='red', alpha=0.15)
        plt.ylabel('Flux')
        plt.legend()
        plt.grid(alpha=0.2)

        file_basename = os.path.basename(filepath)
        plot_filename = file_basename.rsplit('.', 2)[0]
        plt.tight_layout()
        plt.savefig(WORKPATH / f"{plot_filename}_continuum.pdf")
        plt.close(fig)

    return spec


def baseline_als(y, lam, p, mask, niter=10):
    """Fit an asymmetric penalized baseline, excluding samples where mask is False.

    ``y`` and ``mask`` are matching one-dimensional arrays. Return the fitted
    baseline after ``niter`` weight updates with smoothness ``lam`` and
    asymmetry ``p``.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    DTD = D.T.dot(D)
    w = np.ones(L)
    w[~mask] = 0

    for i in range(niter):
        W = sparse.diags([w], [0], shape=(L, L))
        Z = (W + lam * DTD).tocsc()
        z = spsolve(Z, w * y)
        w_new = p * (y > z) + (1 - p) * (y < z)
        w_new[~mask] = 0
        w = w_new
    return z


def dual_als_continuum_core(wave, flux, mask, lam=1e6, p=0.01, niter=10,
                            lam_blue=1e8, p_blue=0.0005, overlap_width=300):
    """Fit separate blue/red ALS continua around the smoothed flux maximum.

    Blend the fits linearly over ``overlap_width`` in the units of ``wave``.
    Input wavelengths must have uniform spacing. Return the continuum array
    and the wavelength at the selected split peak.
    """
    y = -flux

    # 1. Detect Peak (Smooth slightly to avoid noise peaks)
    flux_smooth = gaussian_filter1d(flux, sigma=10)
    peak_idx = np.argmax(flux_smooth)

    # 2. Define Split Indices with Overlap
    dw = np.abs(wave[1] - wave[0])
    overlap_pix = int(overlap_width / dw)
    half_overlap = overlap_pix // 2

    idx_split_start = max(0, peak_idx - half_overlap)
    idx_split_end = min(len(y), peak_idx + half_overlap)

    # 3. Blue Side Fit (Stronger constraint for line blanketing)
    y_blue = y[:idx_split_end]
    mask_blue = mask[:idx_split_end]
    z_blue = baseline_als(y_blue, lam=lam_blue, p=p_blue, mask=mask_blue, niter=niter)

    # 4. Red Side Fit (Standard constraint)
    y_red = y[idx_split_start:]
    mask_red = mask[idx_split_start:]
    z_red = baseline_als(y_red, lam=lam, p=p, mask=mask_red, niter=niter)

    # 5. Stitching (Blending)
    z_final = np.zeros_like(y)

    # Non-overlapping parts
    z_final[:idx_split_start] = z_blue[:idx_split_start]
    z_final[idx_split_end:] = z_red[(idx_split_end - idx_split_start):]

    # Overlapping part (Linear blending)
    len_overlap = idx_split_end - idx_split_start
    if len_overlap > 0:
        weights = np.linspace(0, 1, len_overlap)
        overlap_blue = z_blue[idx_split_start:]
        overlap_red = z_red[:len_overlap]
        z_final[idx_split_start:idx_split_end] = (1 - weights) * overlap_blue + weights * overlap_red

    return -z_final, wave[peak_idx]


def als_continuum_ref_QSI(spec_ref, balmer_lines, lam=1e6, p=0.001, niter=10, RC_PLOT=False, plot_file=None):
    """Fit a single masked ALS continuum to a reference ``Spectrum``.

    Return an Angstrom/count-density ``Spectrum`` without propagated errors.
    Write the optional diagnostic only when both ``RC_PLOT`` and ``plot_file``
    are supplied.
    """
    wave = spec_ref.spectral_axis.value
    flux = spec_ref.flux.value

    mask = get_variable_mask(wave, balmer_lines)

    y = -flux
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    DTD = D.T.dot(D)
    w = np.ones(L)
    w[~mask] = 0

    for i in range(niter):
        W = sparse.diags([w], [0], shape=(L, L))
        Z = (W + lam * DTD).tocsc()
        z = spsolve(Z, w * y)
        w_new = p * (y > z) + (1 - p) * (y < z)
        w_new[~mask] = 0
        w = w_new

    continuum = -z

    spec = Spectrum(
        spectral_axis=wave * u.AA,
        flux=continuum * (u.ct / u.AA)
    )

    if RC_PLOT and plot_file:
        try:
            fig = plt.figure(figsize=(12, 6))
            plt.plot(wave, flux, 'k-', lw=1, alpha=0.6, label='Template Flux')
            plt.plot(wave, continuum, 'r-', lw=1, label='Extracted Continuum')

            for name, params in balmer_lines.items():
                c, wd = params['center'], params['width']
                if wave.min() < c < wave.max():
                    plt.axvspan(c - wd, c + wd, color='blue', alpha=0.1, label='Mask' if name == 'H-alpha' else None)

            plt.legend()
            plt.title(f"Reference Continuum Fit: {Path(plot_file).name}")
            plt.grid(True, alpha=0.3)
            plt.xlabel("Wavelength (A)")
            plt.ylabel("Flux")

            plt.savefig(plot_file)
            plt.close(fig)
        except Exception as e:
            print(f"Ref plotting failed: {e}")

    return spec


def als_continuum_ref_Touptek(spec_ref, balmer_lines, WORKPATH, lam=1e6, p=0.005, niter=10, RC_PLOT=False, plot_file=None):
    """Fit a blended blue/red ALS continuum to a reference ``Spectrum``.

    Exclude ``balmer_lines`` and return an Angstrom/count-density spectrum
    without propagated errors. If plotting is enabled, use ``plot_file`` or
    ``WORKPATH/ref_continuum.pdf``.
    """

    wave = spec_ref.spectral_axis.value
    flux = spec_ref.flux.value

    mask = get_variable_mask(wave, balmer_lines)

    # The blue fit uses distinct smoothing and asymmetry for line blanketing.
    continuum, peak_wave = dual_als_continuum_core(
        wave, flux, mask,
        lam=lam, p=p, niter=niter,
        lam_blue=lam*5e-2, p_blue=0.005, overlap_width=300
    )

    spec = Spectrum(
        spectral_axis=wave * u.AA,
        flux=continuum * (u.ct / u.AA)
    )

    if RC_PLOT:
        if plot_file is None:
            plot_file = WORKPATH / "ref_continuum.pdf"

        fig = plt.figure(figsize=(12, 6))
        plt.plot(wave, flux, 'k-', lw=1, alpha=0.8, label='Reference Flux')
        plt.plot(wave, continuum, 'r-', lw=1, label='Dual-ALS Continuum')
        plt.axvline(peak_wave, color='b', ls=':', alpha=0.5, label='Split Point')

        for name, params in balmer_lines.items():
            center = params['center']
            width = params['width']
            if wave.min() < center < wave.max():
                plt.axvspan(center - width, center + width, color='red', alpha=0.15)

        plt.legend()
        plt.grid(alpha=0.2)
        plt.title("Reference Continuum Fit (Dual-ALS)")
        plt.savefig(plot_file)
        plt.close(fig)

    return spec


def generate_response_polyfit(obs_spec: Spectrum, ref_spec: Spectrum,
                              poly_order: int = 7,
                              clip_sigma: float = 2.5,
                              n_iter: int = 5,
                              RC_PLOT: bool = False,
                              plot_file: str = None) -> Spectrum:
    """
    Generates a response function by dividing observed spectrum by reference,
    clipping outliers (absorption lines), and fitting a polynomial.

    Parameters
    ----------
    obs_spec : specutils.Spectrum
        The observed spectrum.
    ref_spec : specutils.Spectrum
        The reference spectrum template.
    poly_order : int
        Order of Chebyshev polynomial.
    clip_sigma : float
        Sigma threshold for clipping.
    n_iter : int
        Number of clipping iterations.
    RC_PLOT : bool
        Whether to save a plot.
    plot_file : str
        Output path for the plot file.

    Returns
    -------
    specutils.Spectrum
        The smoothed response function.
    """

    # 1. Resample Reference
    resampler = FluxConservingResampler()
    ref_resampled = resampler(ref_spec, obs_spec.spectral_axis)

    # 2. Calculate Raw Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_flux = obs_spec.flux.value / ref_resampled.flux.value

    wave = obs_spec.spectral_axis.value

    # Initial Mask (Finite & Positive)
    valid_mask = np.isfinite(ratio_flux) & (ratio_flux > 0)

    # Data for fitting
    fit_wave = wave[valid_mask]
    fit_ratio = ratio_flux[valid_mask]

    keep_mask = np.ones(len(fit_wave), dtype=bool)

    # 3. Iterative Sigma Clipping
    for i in range(n_iter):
        if np.sum(keep_mask) < poly_order + 5:
            break

        # Fit current subset
        w_sub = fit_wave[keep_mask]
        r_sub = fit_ratio[keep_mask]

        poly = Chebyshev.fit(w_sub, r_sub, poly_order, domain=[wave.min(), wave.max()])
        model = poly(w_sub)

        residuals = r_sub - model
        std_dev = np.std(residuals)

        # Reevaluate all valid samples so clipped points can reenter the fit.
        full_model_on_fit_data = poly(fit_wave)
        full_residuals = fit_ratio - full_model_on_fit_data

        # New mask based on the latest fit
        keep_mask = np.abs(full_residuals) < (clip_sigma * std_dev)

    # 4. Final Fit
    final_w = fit_wave[keep_mask]
    final_r = fit_ratio[keep_mask]

    final_poly = Chebyshev.fit(final_w, final_r, poly_order, domain=[wave.min(), wave.max()])

    # Evaluate over full grid
    smooth_response = final_poly(wave)

    # Normalize
    norm_factor = np.nanmedian(smooth_response)
    if norm_factor == 0 or np.isnan(norm_factor):
        norm_factor = 1.0
    smooth_response /= norm_factor

    # 5. Plotting
    if RC_PLOT and plot_file:
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)

            # Plot Raw Ratio (Grey background)
            # Normalize raw ratio by the same factor for visual consistency
            ax.plot(fit_wave, fit_ratio / norm_factor, 'k.', ms=2, alpha=0.3, label='Raw Ratio')

            if np.sum(~keep_mask) > 0:
                ax.plot(fit_wave[~keep_mask], fit_ratio[~keep_mask] / norm_factor,
                        'rx', ms=4, alpha=0.6, label='Masked/Clipped')

            # Plot Final Fit (Blue line)
            ax.plot(wave, smooth_response, 'b-', lw=2, label='Fitted Response')

            ax.set_xlabel(r'Wavelength ($\AA$)')
            ax.set_ylabel('Normalized Response')
            ax.set_title(f'Response Generation: {Path(plot_file).name}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close(fig)

        except Exception as e:
            print(f"[Warning] Failed to plot response: {e}")

    # 6. Return Spectrum
    response_spec = Spectrum(
        spectral_axis=obs_spec.spectral_axis,
        flux=smooth_response * u.dimensionless_unscaled
    )

    return response_spec


def make_response(spec_cont, spec_ref):
    """Return a median-normalized observed/template continuum ratio.

    Rebin ``spec_ref`` onto the observed continuum's spectral axis with the
    flux-conserving resampler before dividing the two ``Spectrum`` objects.
    """
    resampler = FluxConservingResampler()
    spec_ref_resampled = resampler(spec_ref, spec_cont.spectral_axis)
    res=spec_cont/spec_ref_resampled
    return res/np.nanmedian(res.flux.value)


def _wavelength_vote(obs_pixels, brightest_pixels, ref_waves, dispersion_grid,
                     ref_tree, max_batch_elements=262144):
    """Score the unchanged wavelength grid in bounded, ordered batches.

    Only nearest distances are needed for voting, so sorted one-dimensional
    searches replace repeated KD-tree calls. Other nearest-neighbor backends
    retain their own distance calculation, including its rounding behavior.
    The final line assignment remains in ``robust_wavelength_solution`` and
    continues to use sklearn, preserving its handling of equally close lines.
    """
    best_score, best_shift, best_dispersion = -1, 0, 0
    n_dispersion = len(dispersion_grid)
    n_hypotheses = len(ref_waves) * n_dispersion
    if not np.isfinite(obs_pixels).all():
        # Preserve rejection of nonfinite coordinates by the original backend.
        ref_tree.kneighbors(np.asarray(obs_pixels).reshape(-1, 1))
    batch_size = max(1, int(max_batch_elements) // max(len(obs_pixels), 1))
    use_sorted_search = getattr(ref_tree, "_fit_method", None) == "kd_tree"
    for anchor_pix in brightest_pixels[:5]:
        offsets = obs_pixels - anchor_pix
        for start in range(0, n_hypotheses, batch_size):
            hypotheses = np.arange(start, min(start + batch_size, n_hypotheses))
            reference_indices = hypotheses // n_dispersion
            dispersion_indices = hypotheses % n_dispersion
            predicted = (ref_waves[reference_indices, None] +
                         offsets[None, :] * dispersion_grid[dispersion_indices, None])
            if use_sorted_search:
                upper = np.searchsorted(ref_waves, predicted)
                lower = np.maximum(upper - 1, 0)
                upper = np.minimum(upper, len(ref_waves) - 1)
                distances = np.minimum(np.abs(predicted - ref_waves[lower]),
                                       np.abs(predicted - ref_waves[upper]))
            else:
                distances, _ = ref_tree.kneighbors(predicted.reshape(-1, 1))
                distances = distances.reshape(predicted.shape)
            scores = np.sum(distances < 2.5, axis=1)
            winner = int(np.argmax(scores))
            score = int(scores[winner])
            if score > best_score:
                # Ordered batches and strict improvement retain the first tie.
                best_score = score
                anchor_wave = ref_waves[reference_indices[winner]]
                best_dispersion = dispersion_grid[dispersion_indices[winner]]
                best_shift = anchor_wave - (anchor_pix * best_dispersion)
    return best_score, best_shift, best_dispersion


def robust_wavelength_solution(obs_pixels, obs_fluxes, ref_waves,
                               estimated_dispersion=0.89,
                               search_window=0.1,
                               poly_order=3):
    """
    Wavelength Solution Function:

    1. Ensures edge stability by using Chebyshev polynomials instead of standard polynomials.
    2. Improves precision through RANSAC followed by iterative Sigma Clipping.
    3. Converts final coefficients back to standard Polynomial form for compatibility.

    Args:
        obs_pixels (array): Pixel coordinates of observed peaks.
        obs_fluxes (array): Flux values of observed peaks.
        ref_waves (array): List of reference wavelengths (e.g., Arc lamp lines).
        estimated_dispersion (float): Estimated dispersion (Angstrom/pixel).
        search_window (float): Search range for dispersion grid.
        poly_order (int): Order of the polynomial fitting (Recommend 3 or 4).

    Returns:
        dict: A dictionary containing coefficients, RMS error, matched points, etc.
              Returns None if calibration fails.
    """

    # --- 1. Initial Matching (Voting Method) ---
    sorted_indices = np.argsort(obs_fluxes)[::-1]
    brightest_pixels = obs_pixels[sorted_indices][:25]

    ref_waves = np.sort(ref_waves)
    ref_tree = NearestNeighbors(n_neighbors=1).fit(ref_waves.reshape(-1, 1))

    # Grid Search for dispersion
    dispersion_grid = np.linspace(estimated_dispersion - search_window,
                                  estimated_dispersion + search_window,
                                  40)

    best_score, best_shift, best_dispersion = _wavelength_vote(
        obs_pixels, brightest_pixels, ref_waves, dispersion_grid, ref_tree)

    if best_score < 5:
        print("  [Warning] Initial matching score is too low.")
        return None

    # --- 2. First Refinement using RANSAC ---
    rough_predicted_waves = obs_pixels * best_dispersion + best_shift
    dists, indices = ref_tree.kneighbors(rough_predicted_waves.reshape(-1, 1))

    mask = dists.flatten() < 10
    candidate_pixels = obs_pixels[mask]
    candidate_waves = ref_waves[indices.flatten()[mask]]

    if len(candidate_pixels) < poly_order + 3:
        return None

    ransac = RANSACRegressor(min_samples=poly_order + 2,
                             residual_threshold=2.0,
                             random_state=42)
    try:
        ransac.fit(candidate_pixels.reshape(-1, 1), candidate_waves)
        inlier_mask = ransac.inlier_mask_
    except Exception:
        return None

    final_pixels = candidate_pixels[inlier_mask]
    final_waves = candidate_waves[inlier_mask]

    # --- 3. Chebyshev Fitting & Iterative Sigma Clipping ---
    # Fix the domain to the data range (or full CCD width if known) for stability.
    fit_domain = [np.min(final_pixels), np.max(final_pixels)]

    # Initial fitting (Chebyshev)
    cheb_fit = Chebyshev.fit(final_pixels, final_waves, poly_order, domain=fit_domain)

    # Clipping Loop (Repeat 3 times)
    for i in range(3):
        residuals = final_waves - cheb_fit(final_pixels)
        std_res = np.std(residuals)

        if std_res < 0.05: break  # Break if result is sufficiently good

        # sigma clipping
        threshold = 2.5 * std_res

        good_mask = np.abs(residuals) < threshold

        # Break if too few data points remain
        if np.sum(good_mask) < poly_order + 5:
            break

        final_pixels = final_pixels[good_mask]
        final_waves = final_waves[good_mask]

        # Re-fitting with cleaned data
        cheb_fit = Chebyshev.fit(final_pixels, final_waves, poly_order, domain=fit_domain)

    # --- 4. Return Final Result (Convert to standard Polynomial) ---
    fitted_waves = cheb_fit(final_pixels)
    residuals = final_waves - fitted_waves
    rms = np.sqrt(np.mean(residuals ** 2))

    # Convert Chebyshev object to standard Polynomial (c0 + c1*x + ...) coefficients.
    poly_std = cheb_fit.convert(kind=Polynomial)
    coeffs_standard = poly_std.coef[::-1]

    # Create poly1d object (for plotting and verification purposes)
    poly_func_standard = np.poly1d(coeffs_standard)

    return {
        'coeffs': coeffs_standard,
        'polynomial': poly_func_standard,
        'rms': rms,
        'matched_pixels': final_pixels,
        'matched_waves': final_waves,
        'residuals': residuals
    }


def get_wavelength_solution(fits_path, ref_lines_path, POLY_ORDER=5, lindisp=0.89, window=5, w_min=0.1, w_max=5, prominence=5000, save_plot=False):
    """Fit an arc's pixel-to-wavelength polynomial from a reference line list.

    Args:
        fits_path (str): Extracted FITS file containing ``FLUX_ADU``.
        ref_lines_path (str): Headerless CSV of reference wavelengths in Angstrom.
        POLY_ORDER (int): Degree of the wavelength polynomial.
        lindisp (float): Initial linear dispersion estimate in Angstrom per pixel.
        window (int): Gaussian-fit half-width in pixels.
        w_min (float): Minimum accepted Gaussian sigma in pixels.
        w_max (float): Maximum accepted Gaussian sigma in pixels.
        prominence (float): Peak prominence threshold in the input flux units.
        save_plot (bool): Save a calibration/residual PNG beside the input FITS.

    Returns:
        dict: Fit diagnostics, including RMS in Angstrom and polynomial
            coefficients in descending power order. The input FITS is not changed.

    Raises:
        ValueError: If no acceptable wavelength solution is obtained.
    """
    
    file_basename = os.path.basename(fits_path)
    with fits.open(fits_path) as hdul:
        flux_data = hdul['FLUX_ADU'].data.astype(float)
    
    num_pixels = len(flux_data)
    pixel_indices = np.arange(num_pixels)
    spec_df = pd.DataFrame({'pixel': pixel_indices, 'flux_adu': flux_data})
    
    lines_df = pd.read_csv(ref_lines_path, header=None, names=['wavelength'])
    ref_wavelengths = np.sort(lines_df['wavelength'].values)
    
    peaks_indices, _ = find_peaks(spec_df['flux_adu'], prominence=prominence, distance=5)
    
    if len(peaks_indices) == 0:
        print(f" -> [Error] No peaks found in {file_basename}")
    
    obs_pixels_int = spec_df['pixel'].iloc[peaks_indices].values 
    obs_fluxes = spec_df['flux_adu'].iloc[peaks_indices].values
    
    obs_pixels_sub = []
    obs_fluxes_sub = []
    
    for i, px in enumerate(obs_pixels_int):
        refined_px = fit_gaussian_peak(px, spec_df, window=window, sig_min=w_min, sig_max=w_max)
        if not np.isnan(refined_px):
            obs_pixels_sub.append(refined_px)
            obs_fluxes_sub.append(obs_fluxes[i])
            
    obs_pixels_sub = np.array(obs_pixels_sub)
    obs_fluxes_sub = np.array(obs_fluxes_sub)
    
    if len(obs_pixels_sub) < 5:
        print(" -> [Error] Too few peaks after Gaussian fitting.")

    result = robust_wavelength_solution(
        obs_pixels=obs_pixels_sub,
        obs_fluxes=obs_fluxes_sub,
        ref_waves=ref_wavelengths,
        estimated_dispersion=lindisp,
        search_window=0.1,
        poly_order=POLY_ORDER
    )
    
    if result is None:
        raise ValueError(f"Wavelength calibration failed for {file_basename}")

    poly_fit_func = result['polynomial']
    rms_error = result['rms']
    coeffs = result['coeffs']

    final_pixels = result['matched_pixels']
    final_fitted_waves = result['polynomial'](final_pixels)
    final_residuals = result['residuals']

    final_fluxes = []
    for fpx in final_pixels:
        idx = np.argmin(np.abs(obs_pixels_sub - fpx))
        final_fluxes.append(obs_fluxes_sub[idx])
        
    calibrated_wavelengths = poly_fit_func(pixel_indices)
    spec_df['wavelength'] = calibrated_wavelengths

    
    if save_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(spec_df['wavelength'], spec_df['flux_adu'], label='Calibrated Spectrum', color='blue')
        max_flux = spec_df['flux_adu'].max()
        
    
        ax1.vlines(ref_wavelengths, ymin=0, ymax=max_flux, 
                   label='NeAr Reference Lines', color='red', 
                   linestyle='--', alpha=0.5, linewidth=0.8)
        
        ax1.plot(final_fitted_waves, final_fluxes, 
                 'go', markersize=10, alpha=0.8, 
                 label=f'Final Matched Peaks ({len(final_pixels)} points)')
        
        ax1.set_xlim(np.min(spec_df['wavelength']), np.max(spec_df['wavelength']))
        ax1.set_title(f'Wavelength Calibration Result (Robust) - {file_basename}')
        ax1.set_ylabel('Flux (ADU)')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax1.set_ylim(bottom=-0.05 * max_flux)
        
        ax2.plot(final_fitted_waves, final_residuals, 'kx', label=f'Residuals (RMS = {rms_error:.4f} A)')
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Wavelength (A)')
        ax2.set_ylabel('Residual (A)')
        ax2.legend(loc='upper right')
        ax2.grid(True)
    
        y_limit = max(rms_error * 5, 0.5) 
        ax2.set_ylim(-y_limit, y_limit)
        
        plt.tight_layout(h_pad=0) 
        plot_filename = fits_path.rsplit('.', 1)[0] + "_calib.png"
        plt.savefig(plot_filename)
        plt.close('all')

    return {
        'file': file_basename,
        'status': 'Success',
        'rms_error_A': rms_error,
        'points_used': len(final_pixels),
        'coefficients': coeffs,
    }


# =============================================================================
# Calibration
# =============================================================================

_MISSING_METADATA_WARNED = set()


class CalibrationMismatchError(ValueError):
    """Indicate an incompatible or unavailable detector calibration."""


def _header_value(header, keys, default=None):
    """Return the first available nonempty header value."""
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _temperature(header):
    """Prefer the measured detector temperature over its requested set point."""
    value = _header_value(header, ("CCD-TEMP", "SET-TEMP", "CCDTEMP"))
    return None if value is None else float(value)


def validate_calibration(science_header, calibration_header, label="calibration",
                         temperature_tolerance=0.5, require_temperature=True):
    """Reject known detector/configuration mismatches and return missing fields.

    EGAIN is intentionally not compared: this camera's raw EGAIN is a placeholder
    and the configured physical gain overrides it. Hardware GAIN is compared when
    both headers provide it. Missing legacy metadata are reported, never invented.
    """
    missing = []
    for key in ("INSTRUME", "XBINNING", "YBINNING", "READMODE", "READOUTM",
                "GAIN", "OFFSET", "TRIM_Y0", "TRIM_Y1"):
        left, right = science_header.get(key), calibration_header.get(key)
        if left is None or right is None:
            if key in ("INSTRUME", "XBINNING", "YBINNING"):
                missing.append(key)
            continue
        try:
            equal = np.isclose(float(left), float(right), rtol=0, atol=1e-8)
        except (ValueError, TypeError):
            equal = str(left).strip().casefold() == str(right).strip().casefold()
        if not equal:
            raise CalibrationMismatchError(
                f"{label}: {key} mismatch ({left!r} versus {right!r})")
    left_temp, right_temp = _temperature(science_header), _temperature(calibration_header)
    if ((left_temp is not None and not np.isfinite(left_temp)) or
            (right_temp is not None and not np.isfinite(right_temp))):
        raise CalibrationMismatchError(f"{label}: detector temperature must be finite")
    if left_temp is None or right_temp is None:
        if require_temperature:
            raise CalibrationMismatchError(f"{label}: detector temperature is missing")
        missing.append("CCD-TEMP")
    elif abs(left_temp - right_temp) > float(temperature_tolerance):
        raise CalibrationMismatchError(
            f"{label}: temperature mismatch ({left_temp:.2f} versus {right_temp:.2f} C; "
            f"tolerance {temperature_tolerance:.2f} C)")
    if missing:
        warning_key = tuple(sorted(set(missing)))
        if warning_key not in _MISSING_METADATA_WARNED:
            warnings.warn(
                f"{label}: incomplete legacy detector metadata ({', '.join(warning_key)}); "
                "only available conditions were checked",
                RuntimeWarning, stacklevel=2)
            _MISSING_METADATA_WARNED.add(warning_key)
    return missing


def _variance(ccd):
    """Return a variance array in ADU squared, or None if it was not supplied."""
    if ccd.uncertainty is None:
        return None
    return np.asarray(ccd.uncertainty.represent_as(VarianceUncertainty).array,
                      dtype=float)


def _mask(data, header, existing=None, saturation=None, raw_dtype=None):
    """Mark nonfinite and known saturated samples without guessing a float limit."""
    result = ~np.isfinite(data)
    if existing is not None:
        result |= np.asarray(existing, dtype=bool)
    limit = saturation
    if limit is None:
        limit = _header_value(header, ("SATURATE", "SATLEVEL"))
    dtype = np.dtype(raw_dtype if raw_dtype is not None else data.dtype)
    if limit is None and np.issubdtype(dtype, np.unsignedinteger):
        limit = np.iinfo(dtype).max
    if limit is not None:
        result |= data >= float(limit)
    return result, limit


def _detector_parameters(header, gain=None, rdnoise=None):
    """Resolve calibrated detector constants and reject invalid noise models."""
    resolved_gain = float(gain if gain is not None else header.get("EGAIN", 1.0))
    resolved_noise = float(rdnoise if rdnoise is not None else header.get("RDNOISE", 0.0))
    if not np.isfinite(resolved_gain) or resolved_gain <= 0:
        raise ValueError("Detector gain must be finite and positive")
    if not np.isfinite(resolved_noise) or resolved_noise < 0:
        raise ValueError("Detector read noise must be finite and nonnegative")
    return resolved_gain, resolved_noise


def _trim(ccd, trim_i, trim_f):
    """Trim raw rows and consistently label the resulting detector coordinates."""
    start, stop = int(trim_i), int(trim_f)
    if ccd.data.ndim != 2 or not (0 <= start < stop <= ccd.data.shape[0]):
        raise ValueError(f"Invalid trim [{start}:{stop}] for image shape {ccd.data.shape}")
    result = ccd[start:stop, :].copy()
    result.meta["TRIM_Y0"] = start
    result.meta["TRIM_Y1"] = stop
    return result


def load_ccd(path, gain=None, rdnoise=None, saturation=None):
    """Read data, uncertainty, and masks while identifying raw detector saturation."""
    ccd = CCDData.read(path, unit=u.adu)
    raw_dtype = ccd.data.dtype
    ccd.mask, limit = _mask(ccd.data, ccd.meta, ccd.mask, saturation, raw_dtype)
    if limit is not None:
        ccd.meta["SATLEVEL"] = float(limit)
    if gain is not None:
        ccd.meta["EGAIN"] = float(gain)
    if rdnoise is not None:
        ccd.meta["RDNOISE"] = float(rdnoise)
    return ccd


def _master_id(files, kind, extra=""):
    """Identify the exact input files and processing identity of a master."""
    digest = hashlib.sha256()
    digest.update((kind + str(extra)).encode())
    for name in files:
        path = Path(name).resolve()
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:24]


def _read_raw_strip(path, start, stop, saturation=None):
    """Read only the requested FITS rows, including unsigned integer scaling."""
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        header = hdul[0].header.copy()
        raw = hdul[0].data[start:stop]
        data = np.asarray(raw, dtype=float)
        scale, zero = float(header.get("BSCALE", 1)), float(header.get("BZERO", 0))
        data = data * scale + zero
        limit = saturation
        if limit is None and header.get("BITPIX") == 16 and zero == 32768 and scale == 1:
            limit = 65535.0
        mask, limit = _mask(data, header, saturation=limit)
        if "MASK" in hdul:
            mask |= np.asarray(hdul["MASK"].data[start:stop], dtype=bool)
    return data, mask, header, limit


def _clipped_weights(stack, masks, sigma, robust):
    """Return surviving equal weights using the original one-pass clip policy."""
    values = np.ma.array(stack, mask=masks | ~np.isfinite(stack))
    clipped = sigma_clip(values, sigma_lower=sigma, sigma_upper=sigma,
                         cenfunc="median" if robust else "mean",
                         stdfunc=mad_std if robust else "std", maxiters=1,
                         axis=0, masked=True, copy=True)
    good = ~np.ma.getmaskarray(clipped)
    count = good.sum(axis=0)
    weights = good / np.maximum(count, 1)[None, :, :]
    return weights, count


@lru_cache(maxsize=128)
def _clipped_mean_variance_factor(sample_count):
    """Calibrate selection variance for the retained one-pass two-sigma mean.

    The deterministic simulation assumes independent equal-variance Gaussian
    samples. It corrects the extra variance caused by selecting the weights from
    the same noisy samples. Counts up to five cannot be rejected by this rule.
    This is an approximation for low-count Poisson noise and changing detector
    noise; those limitations are recorded in master metadata and tested.
    """
    sample_count = int(sample_count)
    if sample_count <= 5:
        return 1.0
    repetitions = max(2048, min(50000, 2_000_000 // sample_count))
    rng = np.random.default_rng(90477 + sample_count)
    samples = rng.normal(size=(sample_count, repetitions, 1))
    weights, _ = _clipped_weights(samples, np.zeros_like(samples, bool), 2, False)
    measured = np.sum(weights * samples, axis=0)
    conditional = np.sum(weights ** 2, axis=0)
    return max(1.0, float(np.var(measured, ddof=1) / np.mean(conditional)))


def _combine_raw_master(files, trim_i, trim_f, gain, rdnoise, memlim, bias=None,
                        saturation=None, temperature_tolerance=0.5):
    """Combine raw bias/dark samples in bounded row chunks with detector variance."""
    if not files:
        raise ValueError("No input files supplied for a master")
    headers = [fits.getheader(path) for path in files]
    header = headers[0].copy()
    start, stop = int(trim_i), int(trim_f)
    shape = (stop - start, int(header["NAXIS1"]))
    if not (0 <= start < stop <= int(header["NAXIS2"])):
        raise ValueError("Master trim lies outside the raw detector")
    missing_metadata = set()
    for candidate in headers:
        missing_metadata.update(validate_calibration(header, candidate, "master input", temperature_tolerance))
        if (candidate["NAXIS1"], candidate["NAXIS2"]) != (header["NAXIS1"], header["NAXIS2"]):
            raise CalibrationMismatchError("Master input dimensions differ")
        if not np.isclose(float(candidate.get("EXPTIME", 0)), float(header.get("EXPTIME", 0))):
            raise CalibrationMismatchError("Master input exposure times differ")
        exposure = float(candidate.get("EXPTIME", 0))
        if not np.isfinite(exposure) or exposure < 0 or (bias is not None and exposure == 0):
            raise CalibrationMismatchError("Bias exposure must be nonnegative and dark exposure positive")
    if bias is not None:
        missing_metadata.update(validate_calibration(header, bias.meta, "master bias", temperature_tolerance))
        if bias.shape != shape:
            raise CalibrationMismatchError("Bias shape differs from trimmed dark")
    gain, rdnoise = _detector_parameters(header, gain, rdnoise)
    rows = max(1, min(shape[0], int(float(memlim) / max(len(files) * shape[1] * 8 * 10, 1))))
    output = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)
    output_mask = np.zeros(shape, dtype=bool)
    counts = np.zeros(shape, dtype=np.uint16)
    minimum_count = len(files)
    clip_factors = np.array([_clipped_mean_variance_factor(count)
                             for count in range(len(files) + 1)])
    for offset in range(0, shape[0], rows):
        end = min(offset + rows, shape[0])
        data_parts, mask_parts = [], []
        for path in files:
            data, mask, _, _ = _read_raw_strip(path, start + offset, start + end, saturation)
            data_parts.append(data)
            mask_parts.append(mask)
        stack, masks = np.stack(data_parts), np.stack(mask_parts)
        if bias is not None and bias.mask is not None:
            masks |= bias.mask[offset:end][None, :, :]
        available_count = np.sum(~masks & np.isfinite(stack), axis=0)
        weights, count = _clipped_weights(stack, masks, 2.0, False)
        safe = np.where(np.isfinite(stack), stack, 0.0)
        output[offset:end] = np.sum(weights * safe, axis=0)
        signal = np.zeros_like(stack) if bias is None else np.maximum(
            safe - bias.data[offset:end][None, :, :], 0.0)
        raw_variance = signal / gain + (rdnoise / gain) ** 2
        variance[offset:end] = (np.sum(weights ** 2 * raw_variance, axis=0) *
                                clip_factors[available_count])
        output_mask[offset:end] = count == 0
        counts[offset:end] = count
        minimum_count = min(minimum_count, int(count.min()))
    output[output_mask] = np.nan
    variance[output_mask] = np.nan
    header["TRIM_Y0"], header["TRIM_Y1"] = start, stop
    header["EGAIN"], header["RDNOISE"] = gain, rdnoise
    header["NCOMBINE"], header["NCOMBMIN"] = len(files), minimum_count
    header["VARMODEL"] = "DETECTOR"
    header["VCLIPMOD"] = "GAUSS_IID_MC"
    header["VCLIPFAC"] = float(clip_factors[len(files)])
    header["VCLIPAPP"] = "Approximate for low Poisson counts or unequal variances"
    if missing_metadata:
        header["CALMISS"] = ",".join(sorted(missing_metadata))
    header["HISTORY"] = "Mean master: one-pass 2-sigma clipping; variance uses surviving weights"
    result = CCDData(output, unit=u.adu, meta=header, mask=output_mask,
                     uncertainty=VarianceUncertainty(variance))
    result._effective_count = counts
    return result


def make_bias_master(files, trim_i, trim_f, memlim=8e9, gain=1.0, rdnoise=0.0,
                     saturation=None, temperature_tolerance=0.5):
    """Build a clipped mean bias with read-noise variance and a stable bias ID."""
    result = _combine_raw_master(files, trim_i, trim_f, gain, rdnoise, memlim,
                                 saturation=saturation,
                                 temperature_tolerance=temperature_tolerance)
    result.meta["BIASID"] = _master_id(files, "bias", (trim_i, trim_f, gain, rdnoise))
    result.meta["CALTYPE"] = "BIAS"
    return result


def make_dark_master(files, mbias, trim_i, trim_f, memlim=8e9, gain=None,
                     rdnoise=None, saturation=None, temperature_tolerance=0.5):
    """Build a bias-subtracted dark, retaining its shared-bias covariance identity."""
    gain, rdnoise = _detector_parameters(mbias.meta, gain, rdnoise)
    raw = _combine_raw_master(files, trim_i, trim_f, gain, rdnoise, memlim,
                              bias=mbias, saturation=saturation,
                              temperature_tolerance=temperature_tolerance)
    bias_variance = _required_variance(mbias, gain, rdnoise, "bias")
    raw.data -= mbias.data
    raw.uncertainty = VarianceUncertainty(_variance(raw) + bias_variance)
    raw.mask |= np.asarray(mbias.mask if mbias.mask is not None else False)
    raw.meta["BIASID"] = str(mbias.meta.get("BIASID", "UNKNOWN"))
    raw.meta["DARKID"] = _master_id(files, "dark", raw.meta["BIASID"])
    raw.meta["CALTYPE"] = "DARK"
    raw.meta["DVARBIAS"] = True
    raw.meta["HISTORY"] = "Master dark variance includes its subtracted master bias"
    return raw


def _required_variance(ccd, gain, rdnoise, kind):
    """Read master variance or conservatively mark an explicit legacy estimate."""
    result = _variance(ccd)
    if result is not None:
        return result
    count = max(1, int(ccd.meta.get("NCOMBINE", 1)))
    signal = 0.0 if kind == "bias" else np.maximum(np.asarray(ccd.data, float), 0.0) / gain
    warnings.warn(f"{kind} master has no variance; detector-model estimate with N={count} used",
                  RuntimeWarning, stacklevel=2)
    return np.broadcast_to((signal + (rdnoise / gain) ** 2) / count, ccd.shape).copy()


def library_dark_candidates(directory, target_header, shape, temperature_tolerance=0.5):
    """Inspect actual library headers and rank compatible masters by date.

    Temperature and exposure are never accepted from a filename alone. The
    returned list contains lightweight headers; only the selected image is read.
    """
    candidates, rejected = [], []
    if not directory:
        return candidates, rejected
    target_date = str(target_header.get("DATE-OBS", ""))[:10]
    try:
        observed = datetime.fromisoformat(target_date)
    except ValueError:
        observed = None
    for path in sorted(Path(directory).glob("Mdark_*.fit")):
        try:
            header = fits.getheader(path)
            validate_calibration(target_header, header, f"library dark {path.name}",
                                 temperature_tolerance)
            if (int(header["NAXIS2"]), int(header["NAXIS1"])) != tuple(shape):
                raise CalibrationMismatchError("Library dark shape differs from trimmed science")
            exposure = float(header["EXPTIME"])
            if not np.isfinite(exposure) or exposure <= 0:
                raise CalibrationMismatchError("Library dark exposure must be positive")
            try:
                date = datetime.fromisoformat(str(header.get("DATE-OBS", ""))[:10])
                distance = abs((date - observed).days) if observed is not None else float("inf")
            except ValueError:
                distance = float("inf")
            candidates.append((distance, str(path), exposure, header))
        except (CalibrationMismatchError, ValueError, KeyError, OSError) as exc:
            rejected.append(f"{path.name}: {exc}")
    return sorted(candidates), rejected


def select_dark_master(target_header, shape, masters, exposures, dark_library=None,
                       temperature_tolerance=0.5, incompatible_local_exposures=()):
    """Prefer compatible exact local or library masters before allowed scaling.

    If a same-exposure local dark was incompatible, a missing exact compatible
    replacement is an explicit failure. Scaling remains available for short arc
    exposures that never had a conflicting same-exposure local dark.
    """
    if len(masters) != len(exposures):
        raise CalibrationMismatchError("Dark master/exposure list is inconsistent")
    exposure = float(target_header["EXPTIME"])
    if not np.isfinite(exposure) or exposure < 0:
        raise CalibrationMismatchError("Science exposure must be finite and nonnegative")
    exact_required = any(np.isclose(exposure, item, rtol=0, atol=1e-6)
                         for item in incompatible_local_exposures)
    compatible, reasons = [], []
    for index, dark in enumerate(masters):
        dark_exposure = float(dark.meta.get("EXPTIME", exposures[index]))
        try:
            if not np.isfinite(dark_exposure) or dark_exposure <= 0 or not np.isclose(
                    dark_exposure, exposures[index], rtol=0, atol=1e-6):
                raise CalibrationMismatchError("Dark exposure metadata are inconsistent")
            missing = validate_calibration(target_header, dark.meta, "dark", temperature_tolerance)
            if dark.shape != tuple(shape):
                raise CalibrationMismatchError("Dark shape differs from science")
            compatible.append((abs(exposure - dark_exposure), index, missing))
        except (CalibrationMismatchError, ValueError) as exc:
            reasons.append(str(exc))
            exact_required |= bool(np.isclose(exposure, dark_exposure, rtol=0, atol=1e-6))
    exact = [item for item in compatible if np.isclose(item[0], 0, rtol=0, atol=1e-6)]
    if exact:
        _, index, missing = min(exact)
        return masters[index], missing, {"source": "LOCAL", "reason": "LOCAL_EXACT",
                                         "file": str(masters[index].meta.get("DARKFILE", ""))}
    library, rejected = library_dark_candidates(dark_library, target_header, shape, temperature_tolerance)
    library_exact = [item for item in library if np.isclose(item[2], exposure, rtol=0, atol=1e-6)]
    if library_exact:
        _, name, _, header = min(library_exact)
        dark = CCDData.read(name, unit=u.adu)
        missing = validate_calibration(target_header, header, "selected library dark", temperature_tolerance)
        warnings.warn(f"Selected library dark {name}: matching detector temperature and exposure {exposure:g} s",
                      RuntimeWarning, stacklevel=2)
        return dark, missing, {"source": "LIBRARY", "reason": "LIBRARY_EXACT", "file": name}
    if exact_required:
        detail = "; ".join(reasons + rejected[:5])
        raise CalibrationMismatchError(
            f"An incompatible local dark exists at {exposure:g} s, but no library master matches "
            f"both temperature {_temperature(target_header)} C and exposure; scaling is not substituted. {detail}")
    if compatible:
        _, index, missing = min(compatible)
        return masters[index], missing, {"source": "LOCAL", "reason": "LOCAL_SCALED",
                                         "file": str(masters[index].meta.get("DARKFILE", ""))}
    if library:
        _, name, _, header = min(library, key=lambda item: (abs(exposure-item[2]), item[0], item[1]))
        dark = CCDData.read(name, unit=u.adu)
        missing = validate_calibration(target_header, header, "selected library dark", temperature_tolerance)
        return dark, missing, {"source": "LIBRARY", "reason": "LIBRARY_SCALED", "file": name}
    raise CalibrationMismatchError("No compatible dark: " + "; ".join(reasons + rejected[:5]))


def basic_calibrate(ccd, mbias, mdark, darkexptimelist, trim_i, trim_f,
                    gain=None, rdnoise=None, saturation=None, temperature_tolerance=0.5,
                    dark_library=None, incompatible_local_exposures=()):
    """Subtract compatible masters and propagate raw and shared calibration noise."""
    gain, rdnoise = _detector_parameters(mbias.meta, gain, rdnoise)
    native_mask, limit = _mask(ccd.data, ccd.meta, ccd.mask, saturation)
    ccd = ccd.copy()
    ccd.mask = native_mask
    ccd = _trim(ccd, trim_i, trim_f)
    ccd.meta["EGAIN"], ccd.meta["RDNOISE"] = gain, rdnoise
    missing = validate_calibration(ccd.meta, mbias.meta, "bias", temperature_tolerance)
    if ccd.shape != mbias.shape:
        raise CalibrationMismatchError("Science and bias dimensions differ after trim")
    exposure = float(ccd.meta["EXPTIME"])
    if not np.isfinite(exposure) or exposure < 0:
        raise CalibrationMismatchError("Science exposure must be finite and nonnegative")
    dark, dark_missing, selection = select_dark_master(
        ccd.meta, ccd.shape, mdark, darkexptimelist, dark_library,
        temperature_tolerance, incompatible_local_exposures)
    selected = str(dark.meta.get("DARKID", selection["file"] or "SELECTED"))
    missing = sorted(set(missing + dark_missing))
    dark_exposure = float(dark.meta["EXPTIME"])
    alpha = exposure / dark_exposure
    if not np.isclose(alpha, 1.0, rtol=0, atol=1e-10):
        warnings.warn(f"Dark scaled by {alpha:g}: science {exposure:g} s / dark {dark_exposure:g} s "
                      f"({selection['reason']}); linear dark-current scaling assumed",
                      RuntimeWarning, stacklevel=2)
    raw = np.asarray(ccd.data, dtype=float)
    raw_variance = _variance(ccd)
    detector_variance = raw_variance is None
    if raw_variance is None:
        raw_variance = np.maximum(raw - mbias.data, 0.0) / gain + (rdnoise / gain) ** 2
    bias_variance = _required_variance(mbias, gain, rdnoise, "bias")
    dark_variance = _required_variance(dark, gain, rdnoise, "dark")
    bias_id, dark_bias_id = mbias.meta.get("BIASID"), dark.meta.get("BIASID")
    shared = bool(bias_id and bias_id != "UNKNOWN" and bias_id == dark_bias_id
                  and dark.meta.get("DVARBIAS", False))
    result_data = raw - mbias.data - alpha * dark.data
    if shared:
        raw_dark_variance = dark_variance - bias_variance
        tolerance = 1e-10 * np.maximum(1.0, np.maximum(np.abs(dark_variance), np.abs(bias_variance)))
        valid_master = np.isfinite(raw_dark_variance)
        if mbias.mask is not None:
            valid_master &= ~mbias.mask
        if dark.mask is not None:
            valid_master &= ~dark.mask
        if np.any(valid_master & (raw_dark_variance < -tolerance)):
            raise CalibrationMismatchError(
                "Shared-bias dark variance is smaller than its bias variance; "
                "the master provenance or uncertainty is inconsistent")
        raw_dark_variance = np.maximum(raw_dark_variance, 0.0)
        result_variance = raw_variance + alpha ** 2 * raw_dark_variance + (alpha - 1) ** 2 * bias_variance
        shared_terms = {
            "bias:" + str(bias_id): (alpha - 1, bias_variance),
            "dark:" + str(dark.meta.get("DARKID", selected)): (-alpha, raw_dark_variance),
        }
        variance_status = "SHARED_BIAS"
    else:
        result_variance = raw_variance + bias_variance + alpha ** 2 * dark_variance
        shared_terms = {
            "bias:" + str(bias_id or "UNKNOWN"): (-1.0, bias_variance),
            "dark:" + str(dark.meta.get("DARKID", selected)): (-alpha, dark_variance),
        }
        variance_status = "INDEPENDENT"
        if not dark_bias_id or dark_bias_id == "UNKNOWN":
            variance_status = "LEGACY_APPROX"
            warnings.warn("Dark bias provenance is unknown; independent-master variance assumed",
                          RuntimeWarning, stacklevel=2)
    result_mask = (np.asarray(ccd.mask, bool) | ~np.isfinite(result_data) |
                   ~np.isfinite(result_variance) | (result_variance < 0) |
                   ~np.isfinite(raw_variance) | (raw_variance < 0) |
                   ~np.isfinite(bias_variance) | (bias_variance < 0) |
                   ~np.isfinite(dark_variance) | (dark_variance < 0))
    for master in (mbias, dark):
        if master.mask is not None:
            result_mask |= master.mask
    header = ccd.meta.copy()
    header["BIASID"] = str(bias_id or "UNKNOWN")
    header["DARKID"] = str(dark.meta.get("DARKID", "UNKNOWN"))
    header["DARKSCL"], header["CALVAR"] = alpha, variance_status
    header["DARKSRC"] = selection["source"]
    header["DARKWHY"] = selection["reason"]
    header["DARKEXP"] = dark_exposure
    header["DARKTEMP"] = float(_temperature(dark.meta))
    if selection["file"]:
        header["DARKFILE"] = selection["file"]
    header["FLATCOR"] = False
    if limit is not None:
        header["SATLEVEL"] = float(limit)
    if missing:
        header["CALMISS"] = ",".join(missing)
    header["HISTORY"] = "Bias/dark corrected; raw counting noise and master covariance propagated"
    result = CCDData(result_data, unit=u.adu, meta=header, mask=result_mask,
                     uncertainty=VarianceUncertainty(result_variance))
    result._independent_variance = raw_variance
    result._shared_variance = shared_terms
    if detector_variance:
        result._noise_floor = result_variance - raw_variance + (rdnoise / gain) ** 2
        result._dark_model = alpha * np.asarray(dark.data, float)
        result.meta["NOISEMOD"] = "DETECTOR"
    else:
        result.meta["NOISEMOD"] = "SUPPLIED"
    return result


def _combine_calibrated(ccds, method="average", sigma=5.0, memlim=8e9):
    """Combine calibrated samples while retaining common master-noise terms."""
    if not ccds:
        return None
    shape = ccds[0].shape
    rows = max(1, min(shape[0], int(float(memlim) / max(len(ccds) * shape[1] * 8 * 10, 1))))
    output, variance = np.empty(shape), np.empty(shape)
    mask = np.zeros(shape, bool)
    for start in range(0, shape[0], rows):
        stop = min(start + rows, shape[0])
        stack = np.stack([ccd.data[start:stop] for ccd in ccds])
        masks = np.stack([ccd.mask[start:stop] if ccd.mask is not None else np.zeros_like(ccd.data[start:stop], bool)
                          for ccd in ccds])
        independent = np.stack([getattr(ccd, "_independent_variance", _variance(ccd))[start:stop] for ccd in ccds])
        supplied_variance = np.stack([_variance(ccd)[start:stop] for ccd in ccds])
        masks |= (~np.isfinite(independent) | (independent < 0) |
                  ~np.isfinite(supplied_variance) | (supplied_variance < 0))
        weights, count = _clipped_weights(stack, masks, sigma, True)
        safe = np.where(np.isfinite(stack), stack, 0)
        if method == "median":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                output[start:stop] = np.nanmedian(np.where(weights > 0, safe, np.nan), axis=0)
        else:
            output[start:stop] = np.sum(weights * safe, axis=0)
        current_variance = np.sum(weights ** 2 * np.where(np.isfinite(independent), independent, 0), axis=0)
        common = {}
        for index, ccd in enumerate(ccds):
            for key, (coefficient, component_variance) in getattr(ccd, "_shared_variance", {}).items():
                if key not in common:
                    common[key] = [np.zeros_like(current_variance), component_variance[start:stop]]
                common[key][0] += weights[index] * coefficient
        for coefficient, component_variance in common.values():
            # A completely unused master contributes no variance, even if its
            # masked pixels have undefined uncertainty.
            contribution = np.zeros_like(current_variance)
            np.multiply(coefficient ** 2, component_variance, out=contribution,
                        where=coefficient != 0)
            current_variance += contribution
        if method == "median":
            current_variance *= np.pi / 2
        variance[start:stop] = current_variance
        mask[start:stop] = ((count == 0) | ~np.isfinite(output[start:stop]) |
                           ~np.isfinite(current_variance) | (current_variance < 0))
    output[mask], variance[mask] = np.nan, np.nan
    header = ccds[0].meta.copy()
    header["NCOMBINE"] = len(ccds)
    header["COMBINED"] = True
    header["FLATCOR"] = False
    header["HISTORY"] = "Calibrated combine retains shared master variance; flat not applied"
    return CCDData(output, unit=u.adu, meta=header, mask=mask,
                   uncertainty=VarianceUncertainty(variance))


def combine_arc(files, mbias, mdark, darkexptimelist, trim_i, trim_f,
                mflat=None, memlim=8e9, gain=None, rdnoise=None, saturation=None,
                dark_library=None, incompatible_local_exposures=()):
    """Keep the existing clipped arc mean while deliberately omitting flat division."""
    ccds = [basic_calibrate(load_ccd(path, saturation=saturation), mbias, mdark,
                            darkexptimelist, trim_i, trim_f, gain, rdnoise, saturation,
                            dark_library=dark_library,
                            incompatible_local_exposures=incompatible_local_exposures)
            for path in files]
    return _combine_calibrated(ccds, method="average", sigma=5.0, memlim=memlim)


def combine_flat(files, mbias, mdark, darkexptimelist, trim_i, trim_f, memlim=8e9):
    """Retain the legacy diagnostic median flat; it is never applied to spectra."""
    ccds = []
    gain, rdnoise = _detector_parameters(mbias.meta)
    for path in files:
        raw = _trim(load_ccd(path), trim_i, trim_f)
        validate_calibration(raw.meta, mbias.meta, "flat bias")
        values = np.asarray(raw.data, float) - mbias.data
        mask = raw.mask | np.asarray(mbias.mask if mbias.mask is not None else False)
        median = np.median(values[~mask & np.isfinite(values)])
        if not np.isfinite(median) or median <= 0:
            raise ValueError(f"Invalid diagnostic flat normalization: {path}")
        variance = (np.maximum(values, 0) / gain + (rdnoise / gain) ** 2 +
                    _required_variance(mbias, gain, rdnoise, "bias")) / median ** 2
        raw.data = values / median
        raw.uncertainty = VarianceUncertainty(variance)
        raw.mask = mask
        ccds.append(raw)
    result = _combine_calibrated(ccds, method="median", sigma=5.0, memlim=memlim)
    if result is not None:
        result.meta["CALVAR"] = "DIAGNOSTIC_APPROX"
        result.meta["HISTORY"] = "Unused flat: median uncertainty is an asymptotic approximation"
    return result


def calibrate_science_frames(file_list, mbias, mdark, darkexptimelist, trim_i,
                             trim_f, mflat, prefix, outdir, gain=None, rdnoise=None,
                             saturation=None, dark_library=None, incompatible_local_exposures=()):
    """Write calibrated science/reference frames with variance and detector masks."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for path in file_list:
        ccd = basic_calibrate(load_ccd(path, saturation=saturation), mbias, mdark,
                              darkexptimelist, trim_i, trim_f, gain, rdnoise, saturation,
                              dark_library=dark_library,
                              incompatible_local_exposures=incompatible_local_exposures)
        output = outdir / f"{prefix}_zdf{Path(path).name}"
        ccd.write(output, overwrite=True, checksum=True)
        if hasattr(ccd, "_noise_floor"):
            with fits.open(output, mode="append") as hdul:
                floor_hdu = fits.ImageHDU(ccd._noise_floor, name="NOISE_FLOOR")
                floor_hdu.header["BUNIT"] = "adu2"
                floor_hdu.header["HISTORY"] = "Read noise plus propagated master uncertainty; excludes raw shot noise"
                dark_hdu = fits.ImageHDU(ccd._dark_model, name="DARK_MODEL")
                dark_hdu.header["BUNIT"] = "adu"
                floor_hdu.add_checksum()
                dark_hdu.add_checksum()
                hdul.append(floor_hdu)
                hdul.append(dark_hdu)
        output_paths.append(str(output))
    return output_paths


# =============================================================================
# Extraction
# =============================================================================

INSUFFICIENT_SUPPORT = 1


INVALID_BACKGROUND = 2


PROFILE_MISMATCH = 4


NEGATIVE_RESIDUAL = 8


PROFILE_FALLBACK = 16


VARIANCE_FALLBACK = 32


CR_REJECTED = 64


APERTURE_OUTSIDE = 128


ANCHOR_FAILED = 512


CR_PROFILE_UNSTABLE = 1024


FATAL_QUALITY = (INSUFFICIENT_SUPPORT | INVALID_BACKGROUND | APERTURE_OUTSIDE |
                 ANCHOR_FAILED | CR_PROFILE_UNSTABLE)


QUALITY_BIT_DESCRIPTIONS = {
    INSUFFICIENT_SUPPORT: 'Insufficient support',
    INVALID_BACKGROUND: 'Invalid sky',
    PROFILE_MISMATCH: 'Profile mismatch',
    NEGATIVE_RESIDUAL: 'Negative residual retained',
    PROFILE_FALLBACK: 'Limited profile training (legacy flag)',
    VARIANCE_FALLBACK: 'Missing propagated variance (legacy)',
    CR_REJECTED: 'CR masked',
    APERTURE_OUTSIDE: 'Aperture outside detector',
    ANCHOR_FAILED: 'Anchor normalization failed',
    CR_PROFILE_UNSTABLE: 'CR/profile did not stabilize',
}


# =============================================================================
# Independent Horne fits and finite derivatives in bounded column blocks
# =============================================================================

# Native arrays are spatial pixels by wavelength columns. Each column
# retains its own convergence and CR state; chunking does not bin spectra.
_HORNE_BATCH_COLUMNS = 512


def _extraction_control(value, name, minimum=None, maximum=None, integer=False):
    """Reject non-scalar or nonfinite controls before numerical broadcasting."""
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        number = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if (not np.isfinite(number) or (integer and number != int(number))
            or (minimum is not None and number < minimum)
            or (maximum is not None and number > maximum)):
        raise ValueError(f"Invalid {name}: {value!r}")
    return int(number) if integer else number


def _batch_optional_gain(value):
    """Validate a supplied detector gain while retaining fixed-variance mode."""
    if value is None:
        return None
    number = _extraction_control(value, "gain")
    if number <= 0:
        raise ValueError("Gain must be finite and positive")
    return number


def _batch_rows(values, name, shape=None):
    """Convert a native matrix to contiguous independent spectral rows."""
    result = np.asarray(values, dtype=float)
    if (result.ndim != 2 or 0 in result.shape
            or (shape is not None and result.shape != shape)):
        raise ValueError(f"{name} must match the two-dimensional native strip")
    return np.ascontiguousarray(result.T)


def _batch_vector(values, size, name):
    """Broadcast a scalar or spectral vector without accepting spatial maps."""
    try:
        return np.broadcast_to(np.asarray(values, float), (size,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be a scalar or one value per column") from error


def _batch_masked_sum(values, usable):
    """Sum compact usable vectors with the same pairwise order as scalar NumPy."""
    count = usable.sum(axis=1)
    output = np.zeros(len(values), dtype=float)
    for number in np.unique(count):
        if number == 0:
            continue
        selected = np.flatnonzero(count == number)
        if number == values.shape[1]:
            output[selected] = values[selected].sum(axis=1)
        else:
            compact = values[selected][usable[selected]].reshape(len(selected), int(number))
            output[selected] = compact.sum(axis=1)
    return output


def _batch_weighted_rows(data, profile, variance, usable):
    """Fit independent Horne amplitudes while retaining compact reduction order."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = _batch_masked_sum(profile ** 2 / variance, usable)
        valid = np.isfinite(denominator) & (denominator > 0)
        weights = np.where(usable & valid[:, None],
                           profile / variance / denominator[:, None], 0.)
        flux = _batch_masked_sum(weights * data, usable)
    flux[~valid] = np.nan
    return flux, weights


def _batch_model_rows(data, profile, variance, usable, floor=None, gain=None,
                sky=None, dark=None):
    """Run capped photon-weight updates with independently frozen columns."""
    flux, weights = _batch_weighted_rows(data, profile, variance, usable)
    working = variance.copy()
    iterations = np.zeros(len(data), np.int16)
    converged = np.zeros(len(data), bool)
    if floor is None or gain is None:
        converged[:] = np.isfinite(flux)
        return dict(flux=flux, weights=weights, working_variance=working,
                    iterations=iterations, converged=converged)
    if (floor.shape != data.shape or dark.shape != data.shape
            or np.any(~np.isfinite(floor[usable])) or np.any(floor[usable] < 0)
            or np.any(~np.isfinite(dark[usable])) or not np.isfinite(gain) or gain <= 0):
        raise ValueError("The separated photon-variance model is invalid.")
    floor = np.where(usable, floor, 1.)
    dark = np.where(usable, np.maximum(dark, 0.), 0.)
    active = np.isfinite(flux)
    for step in range(8):
        selected = np.flatnonzero(active)
        if not len(selected):
            break
        photon = np.maximum(flux[selected, None] * profile[selected]
                            + sky[selected, None] + dark[selected], 0.)
        current = np.maximum(floor[selected] + photon / gain, 1e-10)
        updated, new_weights = _batch_weighted_rows(data[selected], profile[selected],
                                              current, usable[selected])
        stopped = np.abs(updated - flux[selected]) <= 1e-7 * np.maximum(np.abs(flux[selected]), 1.)
        working[selected] = current
        weights[selected] = new_weights
        flux[selected] = updated
        iterations[selected] = step + 1
        converged[selected] = stopped
        active[selected] = np.isfinite(updated) & ~stopped
    return dict(flux=flux, weights=weights, working_variance=working,
                iterations=iterations, converged=converged)


def batch_model_weighted_fit(data, profile, variance, usable, noise_floor=None,
                             gain=None, background_level=0., dark_model=None):
    """Vectorize scalar photon-weight fitting without sharing convergence states."""
    gain = _batch_optional_gain(gain)
    native_shape = np.shape(data)
    d = _batch_rows(data, "data")
    p = _batch_rows(profile, "profile", native_shape)
    v = _batch_rows(variance, "variance", native_shape)
    good = np.asarray(usable, bool)
    if good.shape != native_shape:
        raise ValueError("Usable pixels must match the native strip")
    good = np.ascontiguousarray(good.T)
    floor = None if noise_floor is None else _batch_rows(noise_floor, "noise_floor", native_shape)
    dark = np.zeros_like(d) if dark_model is None else _batch_rows(dark_model, "dark_model", native_shape)
    sky = _batch_vector(background_level, len(d), "background_level")
    output = _batch_model_rows(d, p, v, good, floor, gain, sky, dark)
    for key in ("weights", "working_variance"):
        output[key] = output[key].T
    output["diagnostics"] = dict(model_fit_columns=len(d),
                                 photon_iterations=int(output["iterations"].sum()),
                                 scalar_fallback_columns=0)
    return output


def _batch_physical_rows(flux, profile, variance, floor, gain, sky, dark):
    """Predict unsmoothed physical variance with unchanged photon clipping."""
    if floor is None or gain is None:
        return variance
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Gain must be finite and positive")
    return np.maximum(floor + np.maximum(flux[:, None] * profile + sky[:, None]
                                        + np.maximum(dark, 0.), 0.) / gain, 1e-10)


def _batch_rejection_rows(data, profile, usable, flux):
    """Compute resistant rejection scales without replacing measured fluxes."""
    support = usable & (profile > .03 * profile.max(axis=1)[:, None])
    selected = np.flatnonzero(support.sum(axis=1) >= 5)
    result = flux.copy()
    if len(selected):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(support[selected], data[selected] / profile[selected], np.nan)
        median = np.nanmedian(ratios, axis=1)
        valid = np.isfinite(median) & (median > 0)
        result[selected[valid]] = median[valid]
    return result


def batch_extract_physical(data, profile, variance, mask=None,
                           background_variance=0., profile_sigma=None,
                           sigma_clip=10., max_rejections=4,
                           min_profile_fraction=.5, min_core_fraction=.5,
                           min_pixels=5, core_threshold=.5, reject_cosmics=True,
                           noise_floor=None, gain=None, background_level=0.,
                           dark_model=None, *, physical_noise_floor=None,
                           physical_dark_model=None):
    """Extract native columns with independent finite fits and sequential CR masks.

    Completed columns retain their final fit when other columns continue CR
    removal. Reusing that retained fit avoids the scalar routine's duplicate
    final fit. Returned matrices retain the native spatial-by-column layout.
    """
    gain = _batch_optional_gain(gain)
    sigma_clip = _extraction_control(sigma_clip, "sigma_clip")
    if sigma_clip <= 0:
        raise ValueError("sigma_clip must be positive")
    max_rejections = _extraction_control(max_rejections, "max_rejections", 0, integer=True)
    min_pixels = _extraction_control(min_pixels, "min_pixels", 1, integer=True)
    min_profile_fraction = _extraction_control(min_profile_fraction, "min_profile_fraction", 0, 1)
    min_core_fraction = _extraction_control(min_core_fraction, "min_core_fraction", 0, 1)
    core_threshold = _extraction_control(core_threshold, "core_threshold", 0, 1)
    if not isinstance(reject_cosmics, (bool, np.bool_)):
        raise ValueError("reject_cosmics must be boolean")
    shape = np.shape(data)
    d = _batch_rows(data, "data")
    v = _batch_rows(variance, "variance", shape)
    p = _batch_rows(profile, "profile", shape)
    p = np.maximum(np.where(np.isfinite(p), p, 0.), 0.)
    total = p.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("The full-aperture profile has no finite support.")
    p /= total[:, None]
    nx, ny = d.shape
    usable = np.isfinite(d) & np.isfinite(v) & (v > 0)
    usable &= np.isfinite(p) & (p >= 0)
    if mask is not None:
        bad = np.asarray(mask, bool)
        if bad.shape != shape:
            raise ValueError("Mask must match the native strip")
        usable &= ~bad.T
    floor = None if noise_floor is None else _batch_rows(noise_floor, "noise_floor", shape)
    dark = np.zeros_like(d) if dark_model is None else _batch_rows(dark_model, "dark_model", shape)
    separated = physical_noise_floor is not None or physical_dark_model is not None
    physical_floor = floor if physical_noise_floor is None else _batch_rows(physical_noise_floor, "physical_noise_floor", shape)
    physical_dark = dark if physical_dark_model is None else _batch_rows(physical_dark_model, "physical_dark_model", shape)
    if physical_floor is not None and gain is not None:
        if (np.any(~np.isfinite(physical_floor[usable])) or np.any(physical_floor[usable] < 0)
                or np.any(~np.isfinite(physical_dark[usable]))):
            raise ValueError("Physical detector terms are invalid on usable pixels")
        physical_floor = np.where(usable, physical_floor, 1.)
        physical_dark = np.where(usable, physical_dark, 0.)
    initial_usable = usable.copy()
    rejected = np.zeros((nx, ny), bool)
    ps = np.zeros_like(p) if profile_sigma is None else np.maximum(_batch_rows(profile_sigma, "profile_sigma", shape), 0.)
    if not np.isfinite(ps).all():
        raise ValueError("Profile prediction uncertainties must be finite")
    sky = _batch_vector(background_level, nx, "background_level")
    bvar = _batch_vector(background_variance, nx, "background_variance")
    quality = np.zeros(nx, np.uint16)
    bad_sky = ~np.isfinite(bvar) | (bvar < 0)
    quality[bad_sky] |= 2
    bvar[bad_sky] = 0.
    maximum = max(0, int(max_rejections)) if reject_cosmics else 0
    active = np.ones(nx, bool)
    flux = np.full(nx, np.nan)
    weights = np.zeros_like(d)
    working = v.copy()
    fit_calls = np.zeros(nx, np.int16)
    fit_iterations = np.zeros(nx, np.int16)
    converged = np.zeros(nx, bool)
    photon_iterations = 0
    for iteration in range(maximum + 1):
        selected = np.flatnonzero(active)
        if not len(selected):
            break
        fit = _batch_model_rows(d[selected], p[selected], v[selected], usable[selected],
                          None if floor is None else floor[selected], gain,
                          sky[selected], dark[selected])
        flux[selected], weights[selected] = fit["flux"], fit["weights"]
        working[selected] = fit["working_variance"]
        fit_calls[selected] += 1
        fit_iterations[selected] = fit["iterations"]
        photon_iterations += int(fit["iterations"].sum())
        converged[selected] = fit["converged"]
        finite = np.isfinite(fit["flux"])
        quality[selected[~finite]] |= 1
        active[selected] = False
        selected = selected[finite]
        if not len(selected):
            continue
        physical = (_batch_physical_rows(flux[selected], p[selected], v[selected],
                                    None if physical_floor is None else physical_floor[selected],
                                    gain, sky[selected], physical_dark[selected])
                    if separated else working[selected])
        detection_flux = _batch_rejection_rows(d[selected], p[selected], usable[selected], flux[selected])
        detection_variance = physical + bvar[selected, None] + (detection_flux[:, None] * ps[selected]) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = (d[selected] - detection_flux[:, None] * p[selected]) / np.sqrt(detection_variance)
        candidates = usable[selected] & (residual > sigma_clip)
        has_candidate = candidates.any(axis=1)
        if not reject_cosmics:
            continue
        if iteration == maximum:
            quality[selected[has_candidate]] |= 1 | 4
            continue
        positions = np.flatnonzero(has_candidate)
        if len(positions):
            strongest = np.argmax(np.where(candidates[positions], residual[positions], -np.inf), axis=1)
            continuing = selected[positions]
            usable[continuing, strongest] = False
            rejected[continuing, strongest] = True
            active[continuing] = True
    fraction = _batch_masked_sum(p, usable)
    core = p >= core_threshold * p.max(axis=1)[:, None]
    core_total = _batch_masked_sum(p, core)
    core_fraction = np.divide(_batch_masked_sum(p, core & usable), core_total,
                              out=np.zeros(nx), where=core_total > 0)
    bad_support = ((usable.sum(axis=1) < min_pixels) | (fraction < min_profile_fraction)
                   | (core_fraction < min_core_fraction))
    quality[bad_support] |= 1
    quality[rejected.any(axis=1)] |= 64
    outvar = np.full(nx, np.nan)
    selected = np.flatnonzero(np.isfinite(flux))
    if len(selected):
        physical = (_batch_physical_rows(flux[selected], p[selected], v[selected],
                                    None if physical_floor is None else physical_floor[selected],
                                    gain, sky[selected], physical_dark[selected])
                    if separated else working[selected])
        detection_variance = physical + bvar[selected, None] + (flux[selected, None] * ps[selected]) ** 2
        negative = usable[selected] & ((d[selected] - flux[selected, None] * p[selected])
                                      < -sigma_clip * np.sqrt(detection_variance))
        quality[selected[negative.any(axis=1)]] |= 8
        bad_negative = negative.sum(axis=1) > np.maximum(2, initial_usable[selected].sum(axis=1) // 4)
        quality[selected[bad_negative]] |= 4
        outvar[selected] = (_batch_masked_sum(weights[selected] ** 2 * physical, usable[selected])
                            + weights[selected].sum(axis=1) ** 2 * bvar[selected])
    invalid = (quality & FATAL_QUALITY) != 0
    flux[invalid], outvar[invalid] = np.nan, np.nan
    return dict(flux=flux, variance=outvar, rejected=rejected.T, quality=quality,
                profile_fraction=fraction, core_fraction=core_fraction, weights=weights.T,
                fit_iterations=fit_iterations, converged=converged, model_fit_calls=fit_calls,
                diagnostics=dict(model_fit_columns=int(fit_calls.sum()),
                                 photon_iterations=photon_iterations,
                                 scalar_fallback_columns=0))



def _batch_packed(values, order):
    """Place each column's usable samples first without changing spatial order."""
    return np.take_along_axis(np.ascontiguousarray(values.T), order, axis=1)


def _batch_unpacked(values, order):
    """Restore packed weights and derivatives to native spatial coordinates."""
    result = np.zeros_like(values)
    np.put_along_axis(result, order, values, axis=1)
    return result.T


def batch_iterated_weight_jacobian(data, profile, variance, usable,
                                   noise_floor=None, gain=None,
                                   background_level=0.0, dark_model=None):
    """Differentiate each column's actual capped iteration without mixing columns.

    Inputs and matrix-valued outputs have shape ``(spatial, columns)``. The
    scalar production policy is retained: at most eight updates, tolerance
    ``1e-7 * max(abs(previous_flux), 1)``, and derivatives of the finite update
    path rather than an assumed converged fixed point. Unsupported columns
    raise, matching the scalar function; callers must preselect eligible fits.
    """
    gain = _batch_optional_gain(gain)
    data, profile, variance = (np.asarray(value, dtype=float)
                               for value in (data, profile, variance))
    good = np.asarray(usable, dtype=bool)
    if (data.ndim != 2 or 0 in data.shape
            or not (data.shape == profile.shape == variance.shape == good.shape)):
        raise ValueError("Column fit arrays must have identical two-dimensional shapes")
    if (np.any(~np.isfinite(data[good])) or np.any(~np.isfinite(variance[good]))
            or np.any(variance[good] <= 0)):
        raise ValueError("Usable column samples and variances must be finite")
    if np.any(~np.isfinite(profile[good])) or np.any(profile[good] < 0):
        raise ValueError("Usable profile samples must be finite and nonnegative")
    ny, nx = data.shape
    counts = good.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Column fit has no supported profile")
    order = np.argsort(~good.T, axis=1, kind="stable")
    packed_data, packed_profile, packed_variance = (
        _batch_packed(value, order) for value in (data, profile, variance))
    weights = np.zeros((nx, ny), dtype=float)
    flux = np.empty(nx, dtype=float)
    groups = [(int(count), np.flatnonzero(counts == count)) for count in np.unique(counts)]
    for count, indices in groups:
        p = packed_profile[indices, :count]
        v = packed_variance[indices, :count]
        denominator = np.sum(p ** 2 / v, axis=1)
        if np.any(~np.isfinite(denominator) | (denominator <= 0)):
            raise ValueError("Column fit has no supported profile")
        w = p / v / denominator[:, None]
        weights[indices, :count] = w
        flux[indices] = np.sum(w * packed_data[indices, :count], axis=1)
    gradient = weights.copy()
    previous_gradient = np.zeros_like(weights)
    weight_prime = np.zeros_like(weights)
    iterations = np.zeros(nx, dtype=np.int32)
    converged = np.ones(nx, dtype=bool)
    if noise_floor is None or gain is None:
        return dict(flux=flux, weights=_batch_unpacked(weights, order),
                    gradient=_batch_unpacked(gradient, order),
                    weight_prime=_batch_unpacked(weight_prime, order),
                    previous_gradient=_batch_unpacked(previous_gradient, order),
                    iterations=iterations, converged=converged)
    floor = np.asarray(noise_floor, dtype=float)
    dark = np.zeros_like(data) if dark_model is None else np.asarray(dark_model, dtype=float)
    if floor.shape != data.shape or dark.shape != data.shape:
        raise ValueError("Weight detector maps must match the column")
    if (not np.isfinite(gain) or gain <= 0 or np.any(~np.isfinite(floor[good]))
            or np.any(floor[good] < 0) or np.any(~np.isfinite(dark[good]))):
        raise ValueError("Invalid detector terms for photon-weight differentiation")
    packed_floor = _batch_packed(np.where(good, floor, 1.0), order)
    packed_dark = _batch_packed(np.where(good, np.maximum(dark, 0.0), 0.0), order)
    sky = _batch_vector(background_level, nx, "background_level")
    converged[:] = False
    active = np.ones(nx, dtype=bool)
    for step in range(8):
        if not active.any():
            break
        for count, group_indices in groups:
            indices = group_indices[active[group_indices]]
            if not len(indices):
                continue
            p = packed_profile[indices, :count]
            d = packed_data[indices, :count]
            old_flux = flux[indices]
            signal = old_flux[:, None] * p + sky[indices, None] + packed_dark[indices, :count]
            unclipped = packed_floor[indices, :count] + np.maximum(signal, 0.0) / gain
            working = np.maximum(unclipped, 1e-10)
            vp = np.where((signal > 0) & (unclipped > 1e-10), p / gain, 0.0)
            denominator = np.sum(p ** 2 / working, axis=1)
            denominator_prime = -np.sum(p ** 2 * vp / working ** 2, axis=1)
            w = p / working / denominator[:, None]
            wp = (-p * vp / working ** 2 / denominator[:, None]
                  - w * denominator_prime[:, None] / denominator[:, None])
            previous = gradient[indices, :count].copy()
            updated = np.sum(w * d, axis=1)
            derivative_scale = np.einsum("ij,ij->i", wp, d, optimize=False)
            weights[indices, :count] = w
            weight_prime[indices, :count] = wp
            previous_gradient[indices, :count] = previous
            gradient[indices, :count] = w + derivative_scale[:, None] * previous
            iterations[indices] = step + 1
            done = np.abs(updated - old_flux) <= 1e-7 * np.maximum(np.abs(old_flux), 1.0)
            converged[indices] = done
            flux[indices] = updated
            active[indices[done]] = False
    return dict(flux=flux, weights=_batch_unpacked(weights, order),
                gradient=_batch_unpacked(gradient, order),
                weight_prime=_batch_unpacked(weight_prime, order),
                previous_gradient=_batch_unpacked(previous_gradient, order),
                iterations=iterations, converged=converged)


def _batch_chunked(function, data, profile, variance, parameters, block_size):
    """Partition independent columns after checking every full-size input shape."""
    block_size = _extraction_control(block_size, "block_size", 1, integer=True)
    data, profile, variance = (np.asarray(value, dtype=float)
                               for value in (data, profile, variance))
    if (data.ndim != 2 or 0 in data.shape
            or not (data.shape == profile.shape == variance.shape)):
        raise ValueError("Column fit arrays must have matching nonempty two-dimensional shapes")
    nx = data.shape[1]
    matrices = {"mask", "usable", "profile_sigma", "noise_floor", "dark_model",
                "physical_noise_floor", "physical_dark_model"}
    parameters = parameters.copy()
    for name, value in parameters.items():
        if name in matrices and value is not None:
            array = np.asarray(value, dtype=bool if name in {"mask", "usable"} else float)
            if array.shape != data.shape:
                raise ValueError(f"{name} must match the native strip shape")
            parameters[name] = array
        elif name in {"background_level", "background_variance"}:
            parameters[name] = _batch_vector(value, nx, name)
    pieces = []
    for start in range(0, nx, block_size):
        stop = min(nx, start + block_size)
        subset = {name: value[..., start:stop] if isinstance(value, np.ndarray) and value.ndim
                  else value for name, value in parameters.items()}
        pieces.append(function(data[:, start:stop], profile[:, start:stop],
                               variance[:, start:stop], **subset))
    output = {name: np.concatenate([piece[name] for piece in pieces], axis=-1)
              for name in pieces[0] if name != "diagnostics"} if len(pieces) > 1 else pieces[0].copy()
    diagnostics = {"kernel_blocks": len(pieces)}
    for piece in pieces:
        for name, value in piece.get("diagnostics", {}).items():
            diagnostics[name] = diagnostics.get(name, 0) + int(value)
    output["diagnostics"] = diagnostics
    return output


def chunked_extract_physical(data, profile, variance, mask=None,
                             background_variance=0., profile_sigma=None,
                             sigma_clip=10., max_rejections=4,
                             min_profile_fraction=.5, min_core_fraction=.5,
                             min_pixels=5, core_threshold=.5, reject_cosmics=True,
                             noise_floor=None, gain=None, background_level=0.,
                             dark_model=None, *, physical_noise_floor=None,
                             physical_dark_model=None, block_size=_HORNE_BATCH_COLUMNS):
    """Extract full-resolution columns in bounded blocks with identical CR states."""
    parameters = dict(mask=mask, background_variance=background_variance,
                      profile_sigma=profile_sigma, sigma_clip=sigma_clip,
                      max_rejections=max_rejections, min_profile_fraction=min_profile_fraction,
                      min_core_fraction=min_core_fraction, min_pixels=min_pixels,
                      core_threshold=core_threshold, reject_cosmics=reject_cosmics,
                      noise_floor=noise_floor, gain=gain, background_level=background_level,
                      dark_model=dark_model, physical_noise_floor=physical_noise_floor,
                      physical_dark_model=physical_dark_model)
    return _batch_chunked(batch_extract_physical, data, profile, variance, parameters, block_size)


def chunked_iterated_weight_jacobian(data, profile, variance, usable,
                                     noise_floor=None, gain=None, background_level=0.,
                                     dark_model=None, *, block_size=_HORNE_BATCH_COLUMNS):
    """Differentiate each actual finite fit in bounded independent column blocks."""
    parameters = dict(usable=usable, noise_floor=noise_floor, gain=gain,
                      background_level=background_level, dark_model=dark_model)
    return _batch_chunked(batch_iterated_weight_jacobian, data, profile, variance, parameters, block_size)


@dataclass
class ExtractionResult:
    """One detector column's extracted signal, uncertainty, and diagnostics."""

    flux: float
    variance: float
    rejected: np.ndarray
    quality: int
    profile_fraction: float
    core_fraction: float
    weights: np.ndarray


def aperture_geometry(y_trace, half_width, height):
    """Return consistent fixed-width detector rows and subpixel offsets."""
    y_trace = np.asarray(y_trace, dtype=float)
    radius = int(np.ceil(float(half_width)))
    if radius < 1 or not np.all(np.isfinite(y_trace)):
        raise ValueError("Aperture width and trace coordinates must be valid.")
    centers = np.floor(y_trace + 0.5).astype(int)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    rows = centers[None, :] + offsets.astype(int)[:, None]
    inside = (rows >= 0) & (rows < int(height))
    return rows, inside, offsets, y_trace - centers


def estimate_background(column, y0, inner, outer, variance, mask=None,
                        noise_floor=None, gain=None, dark_model=None):
    """Estimate clipped sideband mean and its variance from usable samples.

    The returned variance describes the shared estimated mean, not the noise
    in one sky pixel.  A five-sigma selection removes gross outliers without
    truncating ordinary low-count Poisson tails.  Selection is treated as
    fixed in the mean variance, an approximation checked with simulations.
    Known photon-noise components are updated from the measured sky mean.
    """
    column = np.asarray(column, dtype=float)
    variance = np.asarray(variance, dtype=float)
    yy = np.arange(column.size, dtype=float)
    side = (np.abs(yy - y0) >= inner) & (np.abs(yy - y0) <= outer)
    side &= np.isfinite(column) & np.isfinite(variance) & (variance > 0)
    if mask is not None:
        side &= ~np.asarray(mask, dtype=bool)
    if np.count_nonzero(side) < 10:
        return np.nan, np.nan, 0
    values = column[side]
    variances = variance[side]
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale <= 0:
        scale = np.sqrt(np.median(variances))
    scale = max(scale, float(np.sqrt(np.median(variances))))
    selected = np.abs(values - median) <= 5.0 * scale
    if np.count_nonzero(selected) < 5:
        return np.nan, np.nan, 0
    values = values[selected]
    variances = variances[selected]
    count = len(values)
    mean = float(values.mean())
    if noise_floor is not None and gain is not None:
        floor_selected = np.asarray(noise_floor, dtype=float)[side][selected]
        dark_selected = (np.zeros(count) if dark_model is None
                         else np.maximum(np.asarray(dark_model, dtype=float)[side][selected], 0.0))
        if not np.all(np.isfinite(floor_selected)) or not np.all(np.isfinite(dark_selected)):
            return np.nan, np.nan, 0
        variances = floor_selected + np.maximum(mean + dark_selected, 0.0) / gain
    return mean, float(variances.sum() / count**2), count


def column_background(col, y0, inner, outer, gain, rdnoise,
                      variance=None, mask=None):
    """Provide a compatible background API with an explicit mean variance."""
    if variance is None:
        variance = np.maximum(col, 0) / gain + (rdnoise / gain)**2
    background, mean_variance, _ = estimate_background(
        col, y0, inner, outer, variance, mask)
    return background, mean_variance


def _normalise(profile, floor):
    """Normalize a full-aperture nonnegative profile before masking."""
    profile = np.where(np.isfinite(profile), profile, 0.0)
    profile = np.maximum(profile, float(floor))
    total = profile.sum()
    if total <= 0:
        raise ValueError("The full-aperture profile has no finite support.")
    return profile / total


def _weighted_fit(data, profile, variance, usable):
    """Return Horne flux and weights without renormalizing surviving profile."""
    weights = np.zeros_like(profile, dtype=float)
    denominator = np.sum(profile[usable]**2 / variance[usable])
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan, weights
    weights[usable] = profile[usable] / variance[usable] / denominator
    return float(np.sum(weights[usable] * data[usable])), weights


def _model_weighted_fit(data, profile, variance, usable, noise_floor=None,
                        gain=None, background_level=0.0, dark_model=None):
    """Iterate photon variance without discarding read/master uncertainty.

    A fixed propagated variance remains supported for inputs whose independent
    read/calibration variance is unknown.  When its components are available,
    only the science photon contribution is replaced by the fitted expectation.
    """
    flux, weights = _weighted_fit(data, profile, variance, usable)
    if noise_floor is None or gain is None:
        return flux, weights, variance
    noise_floor = np.asarray(noise_floor, dtype=float)
    dark_model = np.zeros_like(data) if dark_model is None else np.asarray(dark_model, dtype=float)
    if (noise_floor.shape != data.shape or dark_model.shape != data.shape
            or not np.all(np.isfinite(noise_floor[usable])) or np.any(noise_floor[usable] < 0)
            or not np.all(np.isfinite(dark_model[usable])) or not np.isfinite(gain) or gain <= 0):
        raise ValueError("The separated photon-variance model is invalid.")
    noise_floor = np.where(usable, noise_floor, 1.0)
    dark_model = np.where(usable, np.maximum(dark_model, 0.0), 0.0)
    model_variance = variance
    for _ in range(8):
        if not np.isfinite(flux):
            break
        photon_signal = np.maximum(flux * profile + background_level + dark_model, 0.0)
        model_variance = np.maximum(noise_floor + photon_signal / gain, 1e-10)
        updated, weights = _weighted_fit(data, profile, model_variance, usable)
        if abs(updated - flux) <= 1e-7 * max(abs(flux), 1.0):
            flux = updated
            break
        flux = updated
    return flux, weights, model_variance


def _rejection_scale(data, profile, usable, fitted_flux):
    """Estimate a resistant source scale for rejection, not flux measurement.

    Adjacent core hits can inflate the least-squares scale and then conceal
    their own residuals.  An unweighted median over appreciable profile support
    supplies a check whose influence is not dominated by two bright core pixels.
    """
    support = usable & (profile > 0.03 * profile.max())
    if support.sum() < 5:
        return fitted_flux
    scale = float(np.median(data[support] / profile[support]))
    return scale if np.isfinite(scale) and scale > 0 else fitted_flux


def extract_column(data, profile, variance, mask=None,
                   background_variance=0.0, profile_sigma=None,
                   sigma_clip=10.0, max_rejections=4,
                   min_profile_fraction=0.5, min_core_fraction=0.5,
                   min_pixels=5, core_threshold=0.5, reject_cosmics=True,
                   noise_floor=None, gain=None, background_level=0.0,
                   dark_model=None):
    """Extract native detector samples with sequential positive rejection.

    The propagated detector/calibration variance retains all read/master
    uncertainty.  When separately supplied, its photon
    term is iteratively updated from the fitted source, sky, and dark model to
    prevent noisy inverse-variance weights from biasing faint extracted fluxes.
    An empirical profile prediction error protects normal PSF-core residuals
    during CR detection only.  The estimated sky contributes common variance.
    """
    data = np.asarray(data, dtype=float)
    variance = np.asarray(variance, dtype=float)
    profile = _normalise(np.asarray(profile, dtype=float), 0.0)
    if data.shape != profile.shape or data.shape != variance.shape:
        raise ValueError("Data, profile and variance shapes must match.")
    usable = np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    usable &= np.isfinite(profile) & (profile >= 0)
    if mask is not None:
        usable &= ~np.asarray(mask, dtype=bool)
    initial_usable = usable.copy()
    rejected = np.zeros(data.size, dtype=bool)
    profile_sigma = (np.zeros_like(profile) if profile_sigma is None
                     else np.maximum(np.asarray(profile_sigma, dtype=float), 0.0))
    if profile_sigma.shape != profile.shape or not np.all(np.isfinite(profile_sigma)):
        raise ValueError("Profile prediction uncertainties must be finite.")
    quality = 0
    if not np.isfinite(background_variance) or background_variance < 0:
        quality |= INVALID_BACKGROUND
        background_variance = 0.0
    maximum = max(0, int(max_rejections)) if reject_cosmics else 0
    for iteration in range(maximum + 1):
        flux, weights, working_variance = _model_weighted_fit(
            data, profile, variance, usable, noise_floor, gain,
            background_level, dark_model)
        if not np.isfinite(flux):
            quality |= INSUFFICIENT_SUPPORT
            break
        detection_flux = _rejection_scale(data, profile, usable, flux)
        detection_variance = working_variance + background_variance + (detection_flux * profile_sigma)**2
        residual = (data - detection_flux * profile) / np.sqrt(detection_variance)
        candidates = usable & (residual > sigma_clip)
        if not reject_cosmics or not np.any(candidates):
            break
        if iteration == maximum:
            quality |= INSUFFICIENT_SUPPORT | PROFILE_MISMATCH
            break
        strongest = int(np.argmax(np.where(candidates, residual, -np.inf)))
        usable[strongest] = False
        rejected[strongest] = True

    flux, weights, working_variance = _model_weighted_fit(
        data, profile, variance, usable, noise_floor, gain,
        background_level, dark_model)
    fraction = float(profile[usable].sum())
    core = profile >= core_threshold * profile.max()
    core_total = profile[core].sum()
    core_fraction = float(profile[core & usable].sum() / core_total) if core_total > 0 else 0.0
    if (np.count_nonzero(usable) < min_pixels or fraction < min_profile_fraction
            or core_fraction < min_core_fraction):
        quality |= INSUFFICIENT_SUPPORT
    if np.any(rejected):
        quality |= CR_REJECTED
    if np.isfinite(flux):
        detection_variance = working_variance + background_variance + (flux * profile_sigma)**2
        negative = usable & ((data - flux * profile) < -sigma_clip * np.sqrt(detection_variance))
        if np.any(negative):
            quality |= NEGATIVE_RESIDUAL
        if np.count_nonzero(negative) > max(2, np.count_nonzero(initial_usable) // 4):
            quality |= PROFILE_MISMATCH
        flux_variance = float(np.sum(weights[usable]**2 * working_variance[usable])
                              + weights.sum()**2 * background_variance)
    else:
        flux_variance = np.nan
    if quality & FATAL_QUALITY:
        flux, flux_variance = np.nan, np.nan
    return ExtractionResult(flux, flux_variance, rejected, quality,
                            fraction, core_fraction, weights)


def optimal_extract_math(D, P, gain, rdnoise, bkg_scalar, F_prev,
                         has_F_prev, floor, variance=None, mask=None,
                         background_variance=0.0):
    """Retain the legacy kernel API with an optional propagated variance.

    The old signature alone cannot carry calibration errors.  The production
    FITS extractor supplies those errors through ``extract_column`` instead.
    """
    profile = _normalise(np.asarray(P, dtype=float), floor)
    data = np.asarray(D, dtype=float)
    if variance is None:
        signal = F_prev * profile if has_F_prev else data
        variance = np.maximum(signal + bkg_scalar, 0) / gain + (rdnoise / gain)**2
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-10)
    result = extract_column(data, profile, variance, mask,
                            background_variance=background_variance,
                            reject_cosmics=False, min_pixels=1,
                            min_profile_fraction=0.0, min_core_fraction=0.0)
    return result.flux, result.variance, profile, variance


def optimal_extract_one(col_data, y0, half_w, gain, rdnoise,
                        profile_hint=None, sigma_clip=7.0, floor=1e-8,
                        bkg_scalar=None, F_prev=None, variance=None,
                        mask=None, background_variance=0.0,
                        profile_sigma=None, max_rejections=4):
    """Retain the single-column API while applying actual rejection masks.

    Accurate CR detection requires an independently predicted profile.  A
    profile inferred from this single column is only a legacy fallback.
    """
    column = np.asarray(col_data, dtype=float)
    rows, inside, _, _ = aperture_geometry([y0], half_w, len(column))
    selected_rows = rows[inside[:, 0], 0]
    sl = slice(int(selected_rows[0]), int(selected_rows[-1]) + 1)
    data = column[sl]
    if profile_hint is None or len(profile_hint) != len(data):
        warnings.warn("Independent profile missing in optimal_extract_one; CR detection is provisional.", RuntimeWarning)
        positive = np.maximum(data - np.nanmedian(data), 0)
        profile_hint = gaussian_filter1d(positive, 1.0)
        if profile_hint.sum() <= 0:
            profile_hint = np.ones_like(data)
    profile = _normalise(np.asarray(profile_hint, dtype=float), floor)
    if variance is None:
        model = data if F_prev is None else F_prev * profile
        variance_slice = np.maximum(model + (bkg_scalar or 0.0), 0) / gain + (rdnoise / gain)**2
    else:
        variance = np.asarray(variance)
        variance_slice = variance[sl] if variance.shape == column.shape else variance
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        mask = mask[sl] if mask.shape == column.shape else mask
    result = extract_column(data, profile, np.maximum(variance_slice, 1e-10), mask,
                            background_variance, profile_sigma, sigma_clip,
                            max_rejections)
    return result.flux, result.variance, profile, result.rejected, sl


def _align_for_training(strip, mask, offsets, phases):
    """Align profile-training samples only, leaving extraction data untouched."""
    aligned = np.full_like(strip, np.nan, dtype=float)
    for x in range(strip.shape[1]):
        valid = ~mask[:, x] & np.isfinite(strip[:, x])
        if valid.sum() < 4:
            continue
        yy = offsets[valid] - phases[x]
        aligned[:, x] = np.interp(offsets, yy, strip[valid, x], left=np.nan, right=np.nan)
    return aligned


def _nanmedian(values, axis=None):
    """Take a NaN-aware median while allowing entirely masked model edges."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _compact_profile_model(strip, variance, bad, offsets, phases,
                           n_components=4, x_bin=64, smooth_sigma_x=20.0,
                           profile_floor=1e-5, sigma_clip=10.0,
                           nmf_max_iter=5000, nmf_tol=1e-3):
    """Fit NMF profile bases and robust masked NNLS column coefficients.

    Only model-training arrays are shifted to a trace-centered coordinate.  The
    learned basis is sampled back at the original pixel positions before fitting.
    Per-column shape coefficients are normalized before wavelength smoothing so
    bright emission lines do not distort neighboring spatial profiles.  Only
    training bins exceeding an approximate integrated S/N of five are used.
    """
    ny, nx = strip.shape
    aligned = _align_for_training(strip, bad, offsets, phases)
    training = []
    for start in range(0, nx, max(1, int(x_bin))):
        stop = min(nx, start + max(1, int(x_bin)))
        sample = _nanmedian(aligned[:, start:stop], axis=1)
        sample = np.where(np.isfinite(sample), sample, 0.0)
        training_variance = np.where(bad[:, start:stop], np.nan, variance[:, start:stop])
        typical_variance = _nanmedian(training_variance, axis=1)
        sample_count = np.sum(np.isfinite(training_variance), axis=1)
        median_variance = np.divide(
            0.5 * np.pi * typical_variance, sample_count,
            out=np.zeros(ny), where=sample_count > 0)
        noise = np.sqrt(np.nansum(median_variance))
        positive = np.maximum(sample, 0)
        if positive.sum() > 0 and sample.sum() > 5.0 * noise:
            training.append(_normalise(positive, profile_floor))
    fallback = len(training) < 2
    if not training:
        raise ValueError("No usable signal is available for the spatial profile.")
    training = np.asarray(training)
    components = min(max(1, int(n_components)), training.shape[0], ny)
    if fallback or np.allclose(training, training[0], rtol=1e-8, atol=1e-10):
        basis = _normalise(np.median(training, axis=0), profile_floor)[None, :]
        components = 1
    else:
        model = NMF(n_components=components, init="nndsvda", solver="cd",
                    max_iter=nmf_max_iter, tol=nmf_tol, random_state=0)
        model.fit(np.ascontiguousarray(training))
        basis = np.asarray([_normalise(row, profile_floor) for row in model.components_])
    interpolator = CubicSpline(offsets, basis, axis=1, bc_type="natural", extrapolate=False)
    sampled = np.empty((nx, ny, components), dtype=float)
    coefficients = np.zeros((nx, components), dtype=float)
    for x in range(nx):
        model_basis = interpolator(offsets - phases[x]).T
        model_basis = np.nan_to_num(model_basis, nan=0.0)
        model_basis = np.maximum(model_basis, profile_floor)
        model_basis /= model_basis.sum(axis=0, keepdims=True)
        sampled[x] = model_basis
        usable = ~bad[:, x] & np.isfinite(variance[:, x]) & (variance[:, x] > 0)
        usable &= np.isfinite(strip[:, x])
        if usable.sum() < max(components + 1, 5):
            continue
        for _ in range(min(4, ny // 4)):
            scale = np.sqrt(variance[usable, x])
            coefficient, _ = nnls(np.ascontiguousarray(model_basis[usable] / scale[:, None]),
                                  np.ascontiguousarray(strip[usable, x] / scale))
            prediction = model_basis @ coefficient
            residual = (strip[:, x] - prediction) / np.sqrt(variance[:, x])
            candidates = usable & (residual > sigma_clip)
            if not np.any(candidates) or usable.sum() <= max(components + 2, 6):
                break
            usable[np.argmax(np.where(candidates, residual, -np.inf))] = False
        if coefficient.sum() > 0:
            coefficients[x] = coefficient / coefficient.sum()
    good_columns = coefficients.sum(axis=1) > 0
    if not np.any(good_columns):
        raise ValueError("All spatial-profile coefficient fits failed.")
    xx = np.arange(nx)
    for k in range(components):
        coefficients[:, k] = np.interp(xx, xx[good_columns], coefficients[good_columns, k])
        if smooth_sigma_x and smooth_sigma_x > 0:
            coefficients[:, k] = gaussian_filter1d(coefficients[:, k], smooth_sigma_x, mode="nearest")
    profile = np.einsum("xyk,xk->yx", sampled, coefficients)
    profile = np.maximum(profile, profile_floor)
    profile /= profile.sum(axis=0, keepdims=True)

    normalized_residual = np.full_like(strip, np.nan, dtype=float)
    normalized_noise = np.full_like(strip, np.nan, dtype=float)
    for x in range(nx):
        usable = ~bad[:, x] & np.isfinite(strip[:, x]) & np.isfinite(variance[:, x]) & (variance[:, x] > 0)
        amplitude, _ = _weighted_fit(strip[:, x], profile[:, x], variance[:, x], usable)
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        normalized_residual[usable, x] = strip[usable, x] / amplitude - profile[usable, x]
        normalized_noise[usable, x] = variance[usable, x] / amplitude**2
    profile_sigma = np.zeros_like(strip, dtype=float)
    for start in range(0, nx, max(8, int(x_bin))):
        stop = min(nx, start + max(8, int(x_bin)))
        residuals = normalized_residual[:, start:stop]
        bias = _nanmedian(residuals, axis=1)
        scatter = 1.4826 * _nanmedian(np.abs(residuals - bias[:, None]), axis=1)
        noise = _nanmedian(normalized_noise[:, start:stop], axis=1)
        model_error = np.sqrt(np.maximum(scatter**2 - noise, 0) + bias**2)
        model_error = np.nan_to_num(model_error, nan=0.0, posinf=0.0, neginf=0.0)
        profile_sigma[:, start:stop] = model_error[:, None]
    return profile, profile_sigma, basis, coefficients, fallback


def build_profile_map(data_bs, y_trace, half_w, n_components=3, x_bin=32,
                      smooth_sigma_x=8.0, profile_floor=1e-8,
                      nmf_max_iter=5000, nmf_tol=1e-3, variance=None, mask=None):
    """Provide the legacy full-image profile API using the compact model."""
    data_bs = np.asarray(data_bs, dtype=float)
    rows, inside, offsets, phases = aperture_geometry(y_trace, half_w, data_bs.shape[0])
    clipped = np.clip(rows, 0, data_bs.shape[0] - 1)
    xx = np.arange(data_bs.shape[1])[None, :]
    strip = data_bs[clipped, xx]
    variance = np.ones_like(data_bs) if variance is None else np.asarray(variance)
    bad = ~inside | ~np.isfinite(strip)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)[clipped, xx]
    profile, _, basis, coefficients, _ = _compact_profile_model(
        strip, variance[clipped, xx], bad, offsets, phases, n_components,
        x_bin, smooth_sigma_x, profile_floor,
        nmf_max_iter=nmf_max_iter, nmf_tol=nmf_tol)
    full = np.zeros_like(data_bs, dtype=float)
    for x in range(data_bs.shape[1]):
        full[rows[inside[:, x], x], x] = profile[inside[:, x], x]
    return full, basis, coefficients


def _read_calibrated(path):
    """Read calibrated ADU data, propagated variance, and detector masks."""
    with fits.open(path, memmap=False) as hdul:
        data = np.array(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
        variance = None
        if "VARIANCE" in hdul:
            variance = np.array(hdul["VARIANCE"].data, dtype=float)
        elif "UNCERT" in hdul:
            uncertainty = hdul["UNCERT"]
            value = np.array(uncertainty.data, dtype=float)
            kind = str(uncertainty.header.get("UTYPE", header.get("UTYPE", "StdDevUncertainty")))
            if "InverseVariance" in kind:
                variance = np.divide(1.0, value, out=np.full_like(value, np.inf), where=value > 0)
            elif "VarianceUncertainty" in kind:
                variance = value
            elif "StdDevUncertainty" in kind:
                variance = value**2
            else:
                raise ValueError(f"Unsupported uncertainty type {kind!r} in {path}")
        mask = np.array(hdul["MASK"].data, dtype=bool) if "MASK" in hdul else np.zeros(data.shape, dtype=bool)
        noise_floor, dark_model = None, None
        if (header.get("NOISEMOD") == "DETECTOR" and "NOISE_FLOOR" in hdul
                and "DARK_MODEL" in hdul):
            noise_floor = np.array(hdul["NOISE_FLOOR"].data, dtype=float)
            dark_model = np.array(hdul["DARK_MODEL"].data, dtype=float)
    fallback = variance is None
    if fallback:
        gain = float(header.get("EGAIN", 1.0))
        readnoise = float(header.get("RDNOISE", 0.0))
        variance = np.maximum(data, 0) / gain + (readnoise / gain)**2
        warnings.warn(f"{Path(path).name}: no propagated variance; calibration noise is unknown.", RuntimeWarning)
    if variance.shape != data.shape or mask.shape != data.shape:
        raise ValueError("Calibrated data, uncertainty and mask dimensions disagree.")
    if noise_floor is not None and (noise_floor.shape != data.shape or dark_model.shape != data.shape):
        raise ValueError("Separated photon-model dimensions disagree with the image.")
    mask |= ~np.isfinite(data) | ~np.isfinite(variance) | (variance <= 0)
    return data, variance, mask, header, fallback, noise_floor, dark_model


def optimal_extract_fits(input_fits, order, APEX_WIDTH, APBG_IN, APBG_OUT,
                         APEX_NITER, CR_SIGMA, APEX_NK, APEX_XBIN,
                         APEX_SX, PROFILE_FLOOR, coeffs=None, num=None,
                         min_profile_fraction=0.5, min_core_fraction=0.5,
                         min_pixels=5, core_threshold=0.5, *, method="ac",
                         ac_training_bin=64, ac_window=128, ac_cr_refinements=3,
                         mc_samples=256, mc_seed=None):
    """Extract a science or arc FITS frame with variance and quality products.

    The positional API and output filenames match the previous pipeline.  Arc
    extraction retains its no-sky-subtraction/no-iterative-CR policy.  Science
    defaults to refined A+C with correlated conditional samples. Explicit
    ``method='horne'`` retains the old science algorithm for comparison only.
    The A+C return value is a diagnostic dictionary; the legacy/arc paths retain
    the trace-coefficient return value. No automatic method fallback is used.
    """
    data, variance, detector_mask, header, variance_fallback, noise_floor, dark_model = _read_calibrated(input_fits)
    if data.ndim != 2:
        raise ValueError("Extraction requires a two-dimensional calibrated frame.")
    if not (0 < APEX_WIDTH < APBG_IN < APBG_OUT):
        raise ValueError("Require aperture width < inner sky radius < outer sky radius.")
    ny, nx = data.shape
    if coeffs is None:
        coeffs = np.array([float(header[f"TRACE{k}"]) for k in range(order + 1)])
    coeffs = np.asarray(coeffs, dtype=float)
    trace = chebval(np.arange(nx), coeffs)
    rows, inside, offsets, phases = aperture_geometry(trace, APEX_WIDTH, ny)
    clipped = np.clip(rows, 0, ny - 1)
    xx = np.arange(nx)[None, :]
    background = np.zeros(nx)
    background_variance = np.zeros(nx)
    background_count = np.zeros(nx, dtype=int)
    if num is None:
        for x in range(nx):
            background[x], background_variance[x], background_count[x] = estimate_background(
                data[:, x], trace[x], APBG_IN, APBG_OUT,
                variance[:, x], detector_mask[:, x],
                noise_floor=None if noise_floor is None else noise_floor[:, x],
                gain=float(header.get("EGAIN", 1.0)) if noise_floor is not None else None,
                dark_model=None if dark_model is None else dark_model[:, x])
    strip = data[clipped, xx] - background[None, :]
    variance_strip = variance[clipped, xx]
    floor_strip = None if noise_floor is None else noise_floor[clipped, xx]
    dark_strip = None if dark_model is None else dark_model[clipped, xx]
    bad = detector_mask[clipped, xx] | ~inside | ~np.isfinite(strip)
    if num is None and str(method).lower() == "ac":
        
        return write_ac_extraction(
            input_fits, header, coeffs, trace, rows, inside, clipped, offsets, phases,
            strip, variance_strip, bad, noise_floor, dark_model,
            background, background_variance, background_count,
            aperture_width=APEX_WIDTH, sky_outer=APBG_OUT, niter=APEX_NITER,
            cr_sigma=CR_SIGMA, n_components=APEX_NK, initial_xbin=APEX_XBIN,
            x_smooth=APEX_SX, profile_floor=PROFILE_FLOOR,
            min_profile_fraction=min_profile_fraction, min_core_fraction=min_core_fraction,
            min_pixels=min_pixels, core_threshold=core_threshold,
            training_bin=ac_training_bin, window=ac_window,
            max_cr_refinements=ac_cr_refinements, mc_samples=mc_samples, mc_seed=mc_seed,
            variance_fallback=variance_fallback)
    if num is None and str(method).lower() != "horne":
        raise ValueError("Extraction method must be 'ac' or 'horne'")
    profile, model_sigma, _, _, profile_fallback = _compact_profile_model(
        strip, variance_strip, bad, offsets, phases, APEX_NK,
        APEX_XBIN, APEX_SX, PROFILE_FLOOR, CR_SIGMA)
    flux = np.full(nx, np.nan)
    flux_variance = np.full(nx, np.nan)
    quality = np.zeros(nx, dtype=np.uint16)
    rejected = np.zeros(strip.shape, dtype=bool)
    fractions = np.zeros(nx)
    core_fractions = np.zeros(nx)
    refinements = 2 if num is None and int(APEX_NITER) > 1 else 1
    for refinement in range(refinements):
        for x in range(nx):
            result = extract_column(
                strip[:, x], profile[:, x], variance_strip[:, x], bad[:, x] | rejected[:, x],
                background_variance[x], model_sigma[:, x], CR_SIGMA,
                max(0, int(APEX_NITER)), min_profile_fraction, min_core_fraction,
                min_pixels, core_threshold, reject_cosmics=(num is None),
                noise_floor=None if floor_strip is None else floor_strip[:, x],
                gain=float(header.get("EGAIN", 1.0)) if floor_strip is not None else None,
                background_level=background[x],
                dark_model=None if dark_strip is None else dark_strip[:, x])
            flux[x], flux_variance[x] = result.flux, result.variance
            quality[x] = result.quality
            if not np.all(inside[:, x]):
                quality[x] |= APERTURE_OUTSIDE
            if variance_fallback:
                quality[x] |= VARIANCE_FALLBACK
            if profile_fallback:
                quality[x] |= PROFILE_FALLBACK
            rejected[:, x] |= result.rejected
            if rejected[:, x].any():
                quality[x] |= CR_REJECTED
            fractions[x], core_fractions[x] = result.profile_fraction, result.core_fraction
        if refinement + 1 == refinements or not rejected.any():
            break
        profile, model_sigma, _, _, limited_training = _compact_profile_model(
            strip, variance_strip, bad | rejected, offsets, phases, APEX_NK,
            APEX_XBIN, APEX_SX, PROFILE_FLOOR, CR_SIGMA)
        profile_fallback = profile_fallback or limited_training
    invalid = (quality & FATAL_QUALITY) != 0
    flux[invalid] = np.nan
    flux_variance[invalid] = np.nan
    if np.any(invalid):
        warnings.warn(f"{Path(input_fits).name}: {invalid.sum()}/{nx} extraction columns are unusable.", RuntimeWarning)
    header.add_history("Mask-aware NMF/Horne extraction; no replacement of rejected detector data")
    header["HFWIDTH"] = (float(APEX_WIDTH), "Extraction aperture half width")
    header["EXTRBAD"] = (int(invalid.sum()), "Unusable extracted wavelength bins")
    header["EXTRCR"] = (int(rejected.sum()), "Rejected detector pixels")
    header["ERRMODEL"] = ("MODEL+SKY" if noise_floor is not None else "PROP+SKY",
                          "Pixel variance plus shared sky mean error")
    header["PROFERR"] = ("REJECT", "Profile scatter affects CR detection only")
    hdus = [fits.PrimaryHDU(header=header)]
    products = [("FLUX_ADU", flux), ("VAR_ADU2", flux_variance),
                ("BKG_ADU", background), ("BKG_VAR", background_variance),
                ("PROFILE_FRACTION", fractions), ("CORE_FRACTION", core_fractions)]
    for name, values in products:
        hdus.append(fits.ImageHDU(np.asarray(values, dtype=np.float32), name=name))
    qhdu = fits.ImageHDU(quality, name="QUALITY")
    qhdu.header["FATALBIT"] = FATAL_QUALITY
    for bit, label in [(1, "Insufficient support or unresolved CR"), (2, "Invalid sky estimate"),
                       (4, "Profile mismatch"), (8, "Negative residual"),
                       (16, "Limited profile training"), (32, "Missing propagated variance"),
                       (64, "CR masked and recovered"), (128, "Aperture outside detector"),
                       (512, "A+C anchor failed"), (1024, "A+C CR/profile unstable")]:
        qhdu.header[f"BIT{bit}"] = label
    hdus.extend([qhdu, fits.ImageHDU(invalid.astype(np.uint8), name="MASK"),
                 fits.ImageHDU(rejected.sum(axis=0).astype(np.int16), name="NREJECT"),
                 fits.ImageHDU(background_count.astype(np.int16), name="BKG_N"),
                 fits.ImageHDU(rejected.astype(np.uint8), name="CR_MASK"),
                 fits.ImageHDU(rows[0].astype(np.int32), name="AP_ROW0")])
    base = str(input_fits).rsplit(".", 1)[0]
    output = f"{base}.ec.fits" if num is None else f"{base}-{num:03d}.ec.fits"
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return coeffs


# =============================================================================
# Ac detector
# =============================================================================

def _ring_footprint(rows=3, cols=11):
    """Return the fixed centered-exclusion footprint used by correction A."""
    footprint = np.ones((int(rows), int(cols)), dtype=bool)
    footprint[int(rows) // 2, int(cols) // 2] = False
    return footprint


def _ring_median(work, rows, cols):
    """Ring median that never sees the pixel itself, including border pixels."""
    footprint = _ring_footprint(rows, cols)
    smooth = median_filter(work, footprint=footprint, mode="nearest")
    ry, rc = int(rows) // 2, int(cols) // 2
    ny, nx = work.shape
    border = np.zeros(work.shape, dtype=bool)
    if ry > 0:
        border[:ry] = True
        border[-ry:] = True
    if rc > 0:
        border[:, :rc] = True
        border[:, -rc:] = True
    by, bx = np.nonzero(border)
    if by.size:
        padded = np.pad(work, ((ry, ry), (rc, rc)), constant_values=np.nan)
        neighbours = [padded[by + ry + dy, bx + rc + dx]
                      for dy in range(-ry, ry + 1) for dx in range(-rc, rc + 1)
                      if footprint[dy + ry, dx + rc]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            smooth[by, bx] = np.nanmedian(np.stack(neighbours, axis=1), axis=1)
    return smooth


def expected_map(values, rows=3, cols=11, hot_sigma=5.0):
    """Return a noise-decoupled expectation of a master-derived per-pixel map.

    The ring median excludes the pixel itself, so the result is independent of
    that pixel's own master noise.  Departures larger than ``hot_sigma`` robust
    sigma (global MAD of the residual) are retained above the threshold, so real
    hot or dead pixels keep most of their variance.  Only non-finite inputs are
    treated as missing (the maps come from the masters, not from the science
    frame, so the science mask is irrelevant here).
    """
    values = np.asarray(values, dtype=float)
    invalid = ~np.isfinite(values)
    work = values.copy()
    if invalid.all():
        return work
    if invalid.any():
        work[invalid] = np.median(values[~invalid])
    smooth = _ring_median(work, rows, cols)
    residual = work - smooth
    scale = max(1.4826 * float(np.median(np.abs(residual[~invalid]))), 1e-12)
    excess = np.sign(residual) * np.maximum(np.abs(residual) - hot_sigma * scale, 0.0)
    result = smooth + excess
    result[invalid] = np.nan
    return result


def expected_detector_terms(noise_floor, dark_model, row_range=None, **kw):
    """Expected NOISE_FLOOR and DARK_MODEL, optionally only inside ``row_range``."""
    if noise_floor is None or dark_model is None:
        return noise_floor, dark_model
    nf = np.array(noise_floor, dtype=float, copy=True)
    dm = np.array(dark_model, dtype=float, copy=True)
    ny = nf.shape[0]
    lo, hi = (0, ny) if row_range is None else row_range
    lo = max(0, min(ny, int(lo)))
    hi = max(lo, min(ny, int(hi)))
    if hi > lo:
        nf[lo:hi] = np.maximum(expected_map(nf[lo:hi], **kw), 0.0)
        dm[lo:hi] = expected_map(dm[lo:hi], **kw)
    return nf, dm


def pixel_variance(flux, profile, variance, noise_floor=None, gain=None,
                   background_level=0.0, dark_model=None):
    """Predict physical variance without spatially smoothing detector terms.

    ``variance`` is the propagated fallback if separated detector terms are
    unavailable. The exact production photon clipping convention is retained.
    The source model and calibration variances are estimates, not noise-free
    truth; uncertainty in those estimates is not an additional independent
    pixel-noise term.
    """
    if noise_floor is None or gain is None:
        return np.asarray(variance, float)
    floor = np.asarray(noise_floor, float)
    dark = np.zeros_like(floor) if dark_model is None else np.asarray(dark_model, float)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Gain must be finite and positive')
    if floor.shape != np.shape(profile) or dark.shape != floor.shape:
        raise ValueError('Physical detector maps must match the profile shape')
    return np.maximum(floor + np.maximum(flux*np.asarray(profile, float) +
                       background_level + np.maximum(dark, 0.), 0.)/gain, 1e-10)


def extract_column_physical(data, profile, variance, mask=None,
                            background_variance=0.0, profile_sigma=None,
                            sigma_clip=10.0, max_rejections=4,
                            min_profile_fraction=0.5, min_core_fraction=0.5,
                            min_pixels=5, core_threshold=0.5, reject_cosmics=True,
                            noise_floor=None, gain=None, background_level=0.0,
                            dark_model=None, *, physical_noise_floor=None,
                            physical_dark_model=None):
    """Use A for fitting weights and raw detector maps for physical errors.

    ``noise_floor`` and ``dark_model`` are the weight maps, usually A-smoothed.
    ``physical_noise_floor`` and ``physical_dark_model`` are their original
    unsmoothed counterparts. If both are omitted, the original working variance
    is retained exactly for backwards compatibility. ``background_variance``
    must likewise preserve the original
    detector terms in the selected sky pixels. Positive rejection and quality
    flags follow the production algorithm; only their noise model is separated.
    Returned variance is sum(w**2 * V_physical) plus the common sky contribution.
    Profile learning and data-dependent selection uncertainty remain external.
    """
    data, variance = np.asarray(data, float), np.asarray(variance, float)
    profile = _normalise(np.asarray(profile, float), 0.)
    if data.shape != profile.shape or data.shape != variance.shape:
        raise ValueError('Data, profile and variance shapes must match')
    separated = physical_noise_floor is not None or physical_dark_model is not None
    physical_floor = noise_floor if physical_noise_floor is None else physical_noise_floor
    physical_dark = dark_model if physical_dark_model is None else physical_dark_model
    usable = np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    usable &= np.isfinite(profile) & (profile >= 0)
    if mask is not None:
        usable &= ~np.asarray(mask, bool)
    if physical_floor is not None and gain is not None:
        physical_floor = np.asarray(physical_floor, float)
        physical_dark = (np.zeros_like(data) if physical_dark is None
                         else np.asarray(physical_dark, float))
        if physical_floor.shape != data.shape or physical_dark.shape != data.shape:
            raise ValueError('Physical detector maps must match data')
        if (not np.all(np.isfinite(physical_floor[usable])) or
                np.any(physical_floor[usable] < 0) or
                not np.all(np.isfinite(physical_dark[usable]))):
            raise ValueError('Physical detector terms are invalid on usable pixels')
        physical_floor = np.where(usable, physical_floor, 1.)
        physical_dark = np.where(usable, physical_dark, 0.)
    initial_usable = usable.copy()
    rejected = np.zeros(data.size, bool)
    profile_sigma = (np.zeros_like(profile) if profile_sigma is None
                     else np.maximum(np.asarray(profile_sigma, float), 0.))
    if profile_sigma.shape != profile.shape or not np.all(np.isfinite(profile_sigma)):
        raise ValueError('Profile prediction uncertainties must be finite')
    quality = 0
    if not np.isfinite(background_variance) or background_variance < 0:
        quality |= INVALID_BACKGROUND
        background_variance = 0.
    maximum = max(0, int(max_rejections)) if reject_cosmics else 0
    for iteration in range(maximum + 1):
        flux, weights, working_variance = _model_weighted_fit(
            data, profile, variance, usable, noise_floor, gain,
            background_level, dark_model)
        if not np.isfinite(flux):
            quality |= INSUFFICIENT_SUPPORT
            break
        physical = (pixel_variance(flux, profile, variance, physical_floor, gain,
                                   background_level, physical_dark)
                    if separated else working_variance)
        detection_flux = _rejection_scale(data, profile, usable, flux)
        detection_variance = physical + background_variance + (detection_flux*profile_sigma)**2
        residual = (data - detection_flux*profile)/np.sqrt(detection_variance)
        candidates = usable & (residual > sigma_clip)
        if not reject_cosmics or not np.any(candidates):
            break
        if iteration == maximum:
            quality |= INSUFFICIENT_SUPPORT | PROFILE_MISMATCH
            break
        strongest = int(np.argmax(np.where(candidates, residual, -np.inf)))
        usable[strongest] = False
        rejected[strongest] = True
    flux, weights, working_variance = _model_weighted_fit(
        data, profile, variance, usable, noise_floor, gain,
        background_level, dark_model)
    fraction = float(profile[usable].sum())
    core = profile >= core_threshold*profile.max()
    core_total = profile[core].sum()
    core_fraction = float(profile[core & usable].sum()/core_total) if core_total > 0 else 0.
    if (np.count_nonzero(usable) < min_pixels or fraction < min_profile_fraction
            or core_fraction < min_core_fraction):
        quality |= INSUFFICIENT_SUPPORT
    if np.any(rejected):
        quality |= CR_REJECTED
    if np.isfinite(flux):
        physical = (pixel_variance(flux, profile, variance, physical_floor, gain,
                                   background_level, physical_dark)
                    if separated else working_variance)
        detection_variance = physical + background_variance + (flux*profile_sigma)**2
        negative = usable & ((data-flux*profile) < -sigma_clip*np.sqrt(detection_variance))
        if np.any(negative):
            quality |= NEGATIVE_RESIDUAL
        if np.count_nonzero(negative) > max(2, np.count_nonzero(initial_usable)//4):
            quality |= PROFILE_MISMATCH
        flux_variance = float(np.sum(weights[usable]**2*physical[usable]) +
                              weights.sum()**2*background_variance)
    else:
        flux_variance = np.nan
    if quality & FATAL_QUALITY:
        flux, flux_variance = np.nan, np.nan
    return ExtractionResult(flux, flux_variance, rejected, quality,
                               fraction, core_fraction, weights)


# =============================================================================
# Ac profile
# =============================================================================

def _pooled(values, variances, valid, columns):
    """Compute an unweighted signed spatial mean and its independent variance."""
    good=valid[:,columns]
    count=good.sum(axis=1)
    total=np.where(good,values[:,columns],0).sum(axis=1)
    var=np.where(good,variances[:,columns],0).sum(axis=1)
    mean=np.divide(total,count,out=np.zeros(len(count)),where=count>0)
    variance=np.divide(var,count**2,out=np.full(len(count),np.inf),where=count>0)
    return mean,variance,count


def _local_polynomial_pool(values,variances,columns,centre,order):
    """Predict signed native-row counts with a common local polynomial kernel."""
    indices=np.flatnonzero(columns)
    coordinate=(indices-centre)/max(np.max(abs(indices-centre)),1)
    design=np.polynomial.polynomial.polyvander(coordinate,order)
    kernel=np.linalg.pinv(design)[0]
    mean=values[:,indices]@kernel
    variance=variances[:,indices]@(kernel**2)
    return mean,variance,np.full(values.shape[0],len(indices)),indices,kernel


def feature_training_exclusions(strip,variance,bad,background_variance=None,
                                sigma=8.,guard=12):
    """Detect strong sharp features from an independent shape-training fold.

    The returned mask only excludes anchor-training columns; extracted science
    columns are untouched. Each row corresponds to one science fold. Detection
    uses the disjoint shape fold, so anchor noise does not select its own mask.
    """
    ny,nx=strip.shape
    x=np.arange(nx)
    bvar=np.zeros(nx) if background_variance is None else np.broadcast_to(background_variance,(nx,))
    column=np.sum(strip,axis=0)
    cvar=np.sum(variance,axis=0)+ny**2*bvar
    valid=np.all(~bad&np.isfinite(strip)&np.isfinite(variance),axis=0)
    excluded=np.zeros((3,nx),bool)
    for science_fold in range(3):
        indices=np.flatnonzero(valid&(x%3==(science_fold+1)%3))
        if len(indices)<15:
            continue
        size=min(51,len(indices)//2*2-1)
        baseline=median_filter(column[indices],size=size,mode='nearest')
        selected=abs(column[indices]-baseline)>sigma*np.sqrt(cvar[indices])
        selected_indices=indices[selected]
        groups=np.split(selected_indices,np.flatnonzero(np.diff(selected_indices)>6)+1)
        for group in groups:
            if not len(group):
                continue
            padding=max(int(guard),int(group[-1]-group[0])+3)
            lo=max(0,int(group[0])-padding)
            hi=min(nx,int(group[-1])+padding+1)
            excluded[science_fold,lo:hi]=True
    return excluded


def _align(strip, variance, valid, offsets, phases):
    """Shift training arrays with cubic operators and retain full covariance."""
    aligned=np.zeros_like(strip)
    aligned_var=np.zeros_like(variance)
    aligned_valid=np.zeros_like(valid)
    covariances=np.zeros((strip.shape[1],strip.shape[0],strip.shape[0]))
    interpolator=CubicSpline(offsets,np.eye(len(offsets)),axis=0,
                            bc_type='natural',extrapolate=True)
    for x in range(strip.shape[1]):
        if not valid[:,x].all():
            continue
        operator=interpolator(offsets+phases[x])
        aligned[:,x]=operator@strip[:,x]
        covariances[x]=(operator*variance[:,x][None,:])@operator.T
        aligned_var[:,x]=np.diag(covariances[x])
        aligned_valid[:,x]=True
    return aligned,aligned_var,aligned_valid,covariances,interpolator


def _anchor_linear_kernel(chosen, centre):
    """Return the local-linear intercept kernel with a guarded closed form."""
    coordinate = (np.asarray(chosen) - centre).astype(float)
    coordinate /= max(float(np.max(np.abs(coordinate))), 1.)
    count = float(len(coordinate))
    first = coordinate.sum()
    second = coordinate @ coordinate
    determinant = count * second - first * first
    # Use the original SVD when distant clustered anchors amplify cancellation.
    if not np.isfinite(determinant) or determinant <= 1e-4 * count * second:
        design = np.polynomial.polynomial.polyvander(coordinate, 1)
        return np.linalg.pinv(design)[0]
    return (second - first * coordinate) / determinant


def _anchor_fold_columns(n_columns, complete, window, excluded):
    """Select sorted calibration-fold columns without rescanning the spectrum."""
    columns = np.arange(n_columns)
    folds = [columns[columns % 3 == fold] for fold in range(3)]
    chosen = []
    for x in columns:
        candidates = folds[(x + 2) % 3]
        left = np.searchsorted(candidates, x - window)
        right = np.searchsorted(candidates, x + window, side="right")
        candidates = candidates[left:right]
        chosen.append(candidates[complete[candidates] & ~excluded[x % 3, candidates]])
    return chosen, folds


def _anchor_fast_aligned(strip, variance, valid, offsets, phases, window,
                         spatial_smooth, background_variance, excluded):
    """Build the three-fold local-linear anchor with bounded matrix batches.

    Direct local-window sums retain small signals after bright distant columns;
    subtracting cumulative sums would introduce cancellation in that case.
    Insufficient support still raises for the entire frame, as in the scalar path.
    """
    ny, nx = strip.shape
    complete = valid.all(axis=0)
    spline = CubicSpline(offsets, np.eye(ny), axis=0,
                         bc_type="natural", extrapolate=True)
    aligned = np.zeros_like(strip)
    covariance = np.zeros((nx, ny, ny))
    selected = np.flatnonzero(complete)
    for start in range(0, len(selected), 512):
        columns = selected[start:start + 512]
        operators = spline(offsets[None, :] + phases[columns, None])
        aligned[:, columns] = np.einsum("xik,kx->ix", operators, strip[:, columns])
        covariance[columns] = ((operators * variance[:, columns].T[:, None, :])
                               @ operators.transpose(0, 2, 1))
    chosen_all, folds = _anchor_fold_columns(nx, complete, int(window), excluded)
    model_mean = np.zeros_like(strip)
    signed_aligned = np.zeros_like(strip)
    calibration_count = np.zeros(nx, int)
    calibration_bvar = np.zeros(nx)
    kernels = [None] * nx
    failure = np.zeros(nx, np.uint8)
    for x, chosen in enumerate(chosen_all):
        if len(chosen) < 3:
            failure[x] = 1
            continue
        candidates = folds[(x + 1) % 3]
        left = np.searchsorted(candidates, x - int(window))
        right = np.searchsorted(candidates, x + int(window), side="right")
        model_columns = candidates[left:right]
        count = int(complete[model_columns].sum())
        if count < 3:
            failure[x] = 2
            continue
        model_mean[:, x] = aligned[:, model_columns].sum(axis=1) / count
        kernel = _anchor_linear_kernel(chosen, x)
        kernels[x] = kernel
        signed_aligned[:, x] = aligned[:, chosen] @ kernel
        calibration_count[x] = len(chosen)
        calibration_bvar[x] = np.sum(background_variance[chosen] * kernel ** 2)
    smooth = (gaussian_filter1d(model_mean, spatial_smooth, axis=0, mode="nearest")
              if spatial_smooth else model_mean)
    positive = np.maximum(smooth, 0.)
    failure[(failure == 0) & (positive.sum(axis=0) <= 0)] = 3
    if failure.any():
        first = int(np.flatnonzero(failure)[0])
        messages = {1: "Insufficient independent complete calibration columns",
                    2: "Insufficient independent training pixels",
                    3: "Insufficient source signal for an empirical profile"}
        raise ValueError(messages[int(failure[first])])
    profiles = np.zeros_like(strip)
    signed = np.zeros_like(strip)
    calibration_covariance = np.empty_like(covariance)
    flat = covariance.reshape(nx, ny * ny)
    for start in range(0, nx, 512):
        stop = min(nx, start + 512)
        pooled = np.empty((stop - start, ny * ny))
        for local, x in enumerate(range(start, stop)):
            chosen, kernel = chosen_all[x], kernels[x]
            first, last = int(chosen[0]), int(chosen[-1])
            span = flat[first:last + 1:3]
            weights = np.zeros(len(span))
            weights[(chosen - first) // 3] = kernel ** 2
            pooled[local] = weights @ span
        backward = spline(offsets[None, :] - phases[start:stop, None])
        calibration_covariance[start:stop] = (backward @ pooled.reshape(stop - start, ny, ny)
                                               @ backward.transpose(0, 2, 1))
        native = np.maximum(np.einsum("xik,kx->ix", backward, positive[:, start:stop]), 0.)
        profiles[:, start:stop] = native / native.sum(axis=0, keepdims=True)
        signed[:, start:stop] = np.einsum("xik,kx->ix", backward, signed_aligned[:, start:stop])
    return profiles, dict(
        signed_calibration=signed,
        calibration_variance=np.diagonal(calibration_covariance, axis1=1, axis2=2).T.copy(),
        calibration_covariance=calibration_covariance,
        calibration_background_variance=calibration_bvar,
        calibration_count=calibration_count, align=True, crossfit=True,
        window=window, pooling_order=1, anchor_exclude=excluded,
        caveat="Cubic training model extrapolates at most half a pixel at aperture edges")


def build_anchor(strip, variance, bad, offsets, phases, window=384,
                  spatial_smooth=.7, crossfit=True, align=False,
                  background_variance=None,pooling_order=0,
                  exclude_features=False,anchor_exclude=None, **kwargs):
    """Return positive empirical profiles plus independent signed calibration.

    Detector arrays have shape ``(spatial pixels, native columns)``. ``offsets``
    gives spatial pixel coordinates and ``phases`` the subpixel trace offset
    per column. ``window`` is the pooling half-width in native columns.
    ``window=None`` pools the full spectrum and is only appropriate for a
    constant trace-relative shape. With ``crossfit=True``, three disjoint
    interleaved column groups separate shape training, normalization
    calibration, and extracted data.
    Optional alignment applies cubic operators to training data only and
    carries their full spatial covariance. It assumes smooth pixel-integrated
    profiles, including up to half a pixel of model extrapolation at each edge.

    Return ``(profiles, diagnostics)`` with normalized nonnegative profiles,
    signed calibration profiles, and their propagated variances/covariances.
    Insufficient independent anchor support raises ``ValueError``.
    """
    strip=np.asarray(strip,float)
    variance=np.asarray(variance,float)
    bad=np.asarray(bad,bool)
    offsets=np.asarray(offsets,float)
    phases=np.asarray(phases,float)
    if strip.ndim != 2 or variance.shape != strip.shape or bad.shape != strip.shape:
        raise ValueError("Native strip, variance, and bad mask must have matching two-dimensional shapes")
    ny,nx=strip.shape
    if (offsets.shape != (ny,) or phases.shape != (nx,)
            or not np.isfinite(offsets).all() or not np.isfinite(phases).all()):
        raise ValueError("Offsets and phases must be finite and match the native strip dimensions")
    if not np.isfinite(pooling_order) or int(pooling_order) != pooling_order or pooling_order < 0:
        raise ValueError("Anchor polynomial order must be a nonnegative integer")
    if window is not None and (not np.isfinite(window) or int(window) != window or window < 0):
        raise ValueError("Anchor window must be a nonnegative integer or None")
    if not np.isfinite(spatial_smooth) or spatial_smooth < 0:
        raise ValueError("Anchor spatial smoothing must be finite and nonnegative")
    valid=~bad&np.isfinite(strip)&np.isfinite(variance)&(variance>0)
    excluded=np.zeros((3,nx),bool) if anchor_exclude is None else np.broadcast_to(anchor_exclude,(3,nx)).copy()
    if exclude_features:
        excluded|=feature_training_exclusions(strip,variance,bad,background_variance)
    if crossfit and align and pooling_order == 1 and window is not None:
        bvar = (np.zeros(nx) if background_variance is None
                else np.broadcast_to(background_variance, (nx,)))
        return _anchor_fast_aligned(strip, variance, valid, offsets, phases, window,
                                    spatial_smooth, bvar, excluded)
    if align:
        values,training_var,usable,covariances,interpolator=_align(
            strip,variance,valid,offsets,phases)
        calibration_cov=np.zeros_like(covariances)
        prefixes={}
        for group in (range(3) if crossfit else [0]):
            indices=np.flatnonzero(np.all(usable,axis=0)&
                       ((np.arange(nx)%3==group) if crossfit else True)&
                       ~excluded[(group+1)%3 if crossfit else 0])
            prefix=np.concatenate([np.zeros((1,ny,ny)),np.cumsum(covariances[indices],axis=0)])
            prefixes[group]=(indices,prefix)
    else:
        values,training_var,usable=strip,variance,valid
    profiles=np.zeros_like(strip)
    signed=np.zeros_like(strip)
    signed_var=np.zeros_like(strip)
    calibration_count=np.zeros(nx,int)
    calibration_bvar=np.zeros(nx,float)
    bvar=np.zeros(nx) if background_variance is None else np.broadcast_to(background_variance,(nx,))
    all_columns=np.arange(nx)
    pooled_cache={}
    for x in range(nx):
        near=np.ones(nx,bool) if window is None else abs(all_columns-x)<=int(window)
        if crossfit:
            model=near&(all_columns%3==(x+1)%3)
            calibration=near&(all_columns%3==(x+2)%3)
        else:
            model=near
            calibration=near
        # Common columns avoid wavelength-dependent flux selection in each row.
        calibration&=np.all(usable,axis=0)
        calibration&=~excluded[x%3 if crossfit else 0]
        if calibration.sum()<max(3,int(pooling_order)+2):
            raise ValueError('Insufficient independent complete calibration columns')
        key=(x%3) if window is None and pooling_order==0 else None
        if key is not None and key in pooled_cache:
            q,count,qcal,vcal,ccount=pooled_cache[key]
        else:
            q,_,count=_pooled(values,training_var,usable,model)
            if pooling_order:
                qcal,vcal,ccount,cindices,ckernel=_local_polynomial_pool(
                    values,training_var,calibration,x,int(pooling_order))
            else:
                qcal,vcal,ccount=_pooled(values,training_var,usable,calibration)
            if key is not None:
                pooled_cache[key]=(q,count,qcal,vcal,ccount)
        if np.min(count)<3 or np.min(ccount)<3:
            raise ValueError('Insufficient independent training pixels')
        smooth=gaussian_filter1d(q,spatial_smooth,mode='nearest') if spatial_smooth else q
        positive=np.maximum(smooth,0)
        if positive.sum()<=0:
            raise ValueError('Insufficient source signal for an empirical profile')
        if align:
            native=offsets-phases[x]
            native_operator=interpolator(native)
            positive=np.maximum(native_operator@positive,0)
            qcal=native_operator@qcal
            if pooling_order:
                covariance=np.einsum('x,xij->ij',ckernel**2,covariances[cindices])
            else:
                indices,prefix=prefixes[(x+2)%3 if crossfit else 0]
                left=0 if window is None else np.searchsorted(indices,x-int(window))
                right=len(indices) if window is None else np.searchsorted(indices,x+int(window),side='right')
                covariance=(prefix[right]-prefix[left])/(right-left)**2
            calibration_cov[x]=native_operator@covariance@native_operator.T
            vcal=np.diag(calibration_cov[x])
        profiles[:,x]=positive/positive.sum()
        signed[:,x]=qcal
        signed_var[:,x]=vcal
        calibration_count[x]=int(ccount.min())
        calibration_bvar[x]=(float(np.sum(bvar[cindices]*ckernel**2)) if pooling_order else
                            float(np.sum(bvar[calibration])/max(calibration.sum(),1)**2))
    diagnostics=dict(signed_calibration=signed,calibration_variance=signed_var,
                         calibration_background_variance=calibration_bvar,
                         calibration_count=calibration_count,align=bool(align),
                         crossfit=bool(crossfit),window=window,pooling_order=pooling_order,
                         anchor_exclude=excluded,
                         caveat='Cubic training model extrapolates at most half a pixel at aperture edges' if align else '')
    if align:
        diagnostics['calibration_covariance']=calibration_cov
    return profiles,diagnostics


def crossfit_profiles(strip, variance, bad, offsets, phases, training_bin=64,
                      n_components=4, x_smooth=20., profile_floor=1e-5, sigma_clip=10.):
    """Fit nonnegative spatial profiles on three independent interleaved folds.

    ``training_bin`` counts columns within each training fold. Return profiles
    on the original spatial-by-column grid and one limited-training flag per
    fold; spectral binning is used only to learn the profile model.
    """
    nx = strip.shape[1]
    index = np.arange(nx)
    result = np.empty_like(strip, dtype=float)
    fallback = []
    for fold in range(3):
        training = index % 3 == (fold + 1) % 3
        testing = index % 3 == fold
        _, _, basis, coefficient, limited = _compact_profile_model(
            strip[:, training], variance[:, training], bad[:, training], offsets,
            phases[training], n_components, int(training_bin), x_smooth / 3, profile_floor, sigma_clip)
        fallback.append(bool(limited))
        interpolator = CubicSpline(offsets, basis, axis=1,
                                  bc_type='natural', extrapolate=False)
        coefficients = np.array([
            np.interp(index, index[training], coefficient[:, k])
            for k in range(len(basis))])
        for x in index[testing]:
            sampled = np.maximum(np.nan_to_num(interpolator(offsets - phases[x])), profile_floor)
            sampled /= sampled.sum(axis=1, keepdims=True)
            result[:, x] = coefficients[:, x] @ sampled
    result /= result.sum(axis=0, keepdims=True)
    return result, fallback


def build_profile(strip, variance, bad, offsets, phases, *, window=128,
                  align=True, pooling_order=1, background_variance=None,
                  training_bin=64, n_components=4, x_smooth=20.,
                  profile_floor=1e-5, sigma_clip=10., **kwargs):
    """Preserve the signed anchor and change only the fixed NMF training bin."""
    strip, variance = np.asarray(strip, float), np.asarray(variance, float)
    bad = np.asarray(bad, bool)
    offsets, phases = np.asarray(offsets, float), np.asarray(phases, float)
    profile, fallback = crossfit_profiles(strip, variance, bad, offsets, phases,
                                         training_bin=training_bin, n_components=n_components,
                                         x_smooth=x_smooth, profile_floor=profile_floor, sigma_clip=sigma_clip)
    _, anchor = build_anchor(
        strip, variance, bad, offsets, phases, window=window,
        crossfit=True, align=align, pooling_order=pooling_order,
        background_variance=background_variance, **kwargs)
    anchor['training_bin_fold_columns'] = int(training_bin)
    anchor['limited_shape_folds'] = fallback
    return profile, anchor


# =============================================================================
# Extraction covariance
# =============================================================================

CONDITIONAL_SCOPE = "FIXED_PROFILE_A_MASK_VARIANCE; SHARED_NATIVE_SKY; CROSSCAL_NOT_MODELED"


ANCHOR_INVALID = 512


PROFILE_UNSTABLE = 1024


STRUCTURAL_FATAL = 1 | 2 | 128 | PROFILE_UNSTABLE


def pooling_operator(n_columns, valid_columns, window=128, order=1,
                     anchor_exclude=None):
    """Build the exact three-fold signed-anchor local polynomial operator."""
    if not np.isfinite(n_columns) or int(n_columns) != n_columns or n_columns < 0:
        raise ValueError("The number of native columns must be a nonnegative integer")
    n_columns = int(n_columns)
    indices = np.arange(n_columns)
    valid = np.asarray(valid_columns, bool)
    if valid.shape != (n_columns,):
        raise ValueError("Anchor validity must contain one value per native column")
    if not np.isfinite(order) or int(order) != order or order < 0:
        raise ValueError("Anchor polynomial order must be a nonnegative integer")
    if window is not None and (not np.isfinite(window) or int(window) != window or window < 0):
        raise ValueError("Anchor window must be a nonnegative integer or None")
    excluded = (np.zeros((3, n_columns), bool) if anchor_exclude is None
                else np.broadcast_to(anchor_exclude, (3, n_columns)))
    if order == 1 and window is not None:
        chosen_all, _ = _anchor_fold_columns(n_columns, valid, int(window), excluded)
        indptr = np.zeros(n_columns + 1, dtype=np.int64)
        chosen_parts, kernel_parts = [], []
        for x, chosen in enumerate(chosen_all):
            if len(chosen) < 3:
                raise ValueError(f"Insufficient independent anchor columns at x={x}")
            chosen_parts.append(chosen)
            kernel_parts.append(_anchor_linear_kernel(chosen, x))
            indptr[x + 1] = indptr[x] + len(chosen)
        if not chosen_parts:
            return sparse.csr_matrix((n_columns, n_columns))
        return sparse.csr_matrix((np.concatenate(kernel_parts), np.concatenate(chosen_parts), indptr),
                                 shape=(n_columns, n_columns))
    rows, columns, coefficients = [], [], []
    for x in indices:
        use = valid & (indices % 3 == (x + 2) % 3) & ~excluded[x % 3]
        if window is not None:
            use &= np.abs(indices - x) <= int(window)
        chosen = indices[use]
        if len(chosen) < max(3, int(order) + 2):
            raise ValueError(f"Insufficient independent anchor columns at x={x}")
        if order:
            coordinate = (chosen - x) / max(np.max(np.abs(chosen - x)), 1)
            design = np.polynomial.polynomial.polyvander(coordinate, int(order))
            kernel = np.linalg.pinv(design)[0]
        else:
            kernel = np.full(len(chosen), 1.0 / len(chosen))
        rows.extend([int(x)] * len(chosen))
        columns.extend(chosen.tolist())
        coefficients.extend(kernel.tolist())
    return sparse.csr_matrix((coefficients, (rows, columns)),
                             shape=(n_columns, n_columns))


def iterated_weight_jacobian(data, profile, variance, usable,
                            noise_floor=None, gain=None, background_level=0.0,
                            dark_model=None):
    """Differentiate the actual finite Horne iteration with its branch fixed.

    The eight-iteration cap and tolerance match the production fit exactly.
    The final weight Jacobian is rank one: ``weight_prime[:, None] *
    previous_gradient[None, :]``.  This avoids assuming that iteration reached
    an implicit fixed point.  The photon clipping branch is differentiated away
    from its zero boundary, including the dark photon contribution.
    """
    data, profile, variance = map(lambda value: np.asarray(value, float),
                                  (data, profile, variance))
    good = np.asarray(usable, bool)
    if not (data.shape == profile.shape == variance.shape == good.shape):
        raise ValueError("Column fit arrays must have identical shapes")
    if (np.any(~np.isfinite(data[good])) or np.any(~np.isfinite(variance[good]))
            or np.any(variance[good] <= 0)):
        raise ValueError("Usable column samples and variances must be finite")
    denominator = np.sum(profile[good] ** 2 / variance[good])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Column fit has no supported profile")
    weights = np.zeros_like(profile)
    weights[good] = profile[good] / variance[good] / denominator
    flux = float(np.sum(weights[good] * data[good]))
    gradient = weights.copy()
    previous_gradient = np.zeros_like(weights)
    weight_prime = np.zeros_like(weights)
    iterations = 0
    if noise_floor is None or gain is None:
        return dict(flux=flux, weights=weights, gradient=gradient,
                    weight_prime=weight_prime, previous_gradient=previous_gradient,
                    iterations=iterations, converged=True)
    floor = np.asarray(noise_floor, float)
    dark = np.zeros_like(data) if dark_model is None else np.asarray(dark_model, float)
    if floor.shape != data.shape or dark.shape != data.shape:
        raise ValueError("Weight detector maps must match the column")
    if (not np.isfinite(gain) or gain <= 0 or np.any(~np.isfinite(floor[good]))
            or np.any(floor[good] < 0) or np.any(~np.isfinite(dark[good]))):
        raise ValueError("Invalid detector terms for photon-weight differentiation")
    floor = np.where(good, floor, 1.0)
    dark = np.where(good, np.maximum(dark, 0.0), 0.0)
    converged = False
    for step in range(8):
        signal = flux * profile + float(background_level) + dark
        unclipped = floor + np.maximum(signal, 0.0) / gain
        working = np.maximum(unclipped, 1e-10)
        vp = np.where((signal > 0) & (unclipped > 1e-10), profile / gain, 0.0)
        denominator = np.sum(profile[good] ** 2 / working[good])
        denominator_prime = -np.sum(profile[good] ** 2 * vp[good] / working[good] ** 2)
        weights = np.zeros_like(profile)
        weights[good] = profile[good] / working[good] / denominator
        weight_prime = np.zeros_like(profile)
        weight_prime[good] = (-profile[good] * vp[good] / working[good] ** 2 / denominator
                              - weights[good] * denominator_prime / denominator)
        previous_gradient = gradient.copy()
        updated = float(np.sum(weights[good] * data[good]))
        gradient = weights + float(weight_prime[good] @ data[good]) * previous_gradient
        iterations = step + 1
        converged = abs(updated - flux) <= 1e-7 * max(abs(flux), 1.0)
        flux = updated
        if converged:
            break
    return dict(flux=flux, weights=weights, gradient=gradient,
                weight_prime=weight_prime, previous_gradient=previous_gradient,
                iterations=iterations, converged=bool(converged))


def calibrated_gradients(fit, signed_anchor, anchor_covariance,
                         anchor_background_variance=0.0, ratio_debias=True,
                         minimum_snr=10.0):
    """Differentiate the implemented second-order signed-ratio correction."""
    q = np.asarray(signed_anchor, float)
    covariance = np.asarray(anchor_covariance, float)
    weights = np.asarray(fit["weights"], float)
    flux = float(fit["flux"])
    bvar = float(anchor_background_variance)
    if covariance.shape != (len(q), len(q)) or weights.shape != q.shape:
        raise ValueError("Signed-anchor covariance dimensions are inconsistent")
    if np.any(~np.isfinite(covariance)) or not np.isfinite(bvar) or bvar < 0:
        raise ValueError("Signed-anchor covariance must be finite with nonnegative sky variance")
    total_covariance = covariance + bvar
    a, b = float(q.sum()), float(weights @ q)
    vb = float(weights @ total_covariance @ weights)
    cab = float(total_covariance.sum(axis=0) @ weights)
    valid = (np.isfinite([a, b, vb, cab, flux]).all() and a > 0 and b > 0
             and vb >= 0 and b >= minimum_snr * np.sqrt(vb))
    if not valid:
        return dict(valid=False, flux=np.nan, correction=np.nan,
                    anchor_snr=b / np.sqrt(vb) if np.isfinite(vb) and vb > 0 else np.nan)
    wp = fit["weight_prime"]
    bp = float(wp @ q)
    vbp = float(2 * wp @ total_covariance @ weights)
    cabp = float(total_covariance.sum(axis=0) @ wp)
    correction = a / b
    correction_prime = -a * bp / b ** 2
    dq = np.full(len(q), 1 / b) - a * weights / b ** 2
    if ratio_debias:
        correction -= a * vb / b ** 3 - cab / b ** 2
        correction_prime += (-a * vbp / b ** 3 + 3 * a * vb * bp / b ** 4
                             + cabp / b ** 2 - 2 * cab * bp / b ** 3)
        dq += -vb / b ** 3 + 3 * a * vb * weights / b ** 4 - 2 * cab * weights / b ** 3
    direct = correction * fit["gradient"] + flux * correction_prime * fit["previous_gradient"]
    anchor = flux * dq
    valid = (np.isfinite(correction) and correction > 0
             and np.isfinite(direct).all() and np.isfinite(anchor).all())
    return dict(valid=bool(valid), flux=flux * correction if valid else np.nan,
                correction=correction if valid else np.nan,
                direct_gradient=direct, anchor_gradient=anchor,
                anchor_snr=b / np.sqrt(vb) if vb > 0 else np.inf)


def spatial_operators(offsets, phases, align=True):
    """Return exact cubic input alignment and output inverse-alignment matrices."""
    offsets, phases = np.asarray(offsets, float), np.asarray(phases, float)
    if (offsets.ndim != 1 or phases.ndim != 1
            or not np.isfinite(offsets).all() or not np.isfinite(phases).all()):
        raise ValueError("Native offsets and phases must be finite vectors")
    identity = np.eye(len(offsets))
    if not align:
        repeated = np.broadcast_to(identity, (len(phases), len(offsets), len(offsets)))
        return repeated, repeated
    interpolator = CubicSpline(offsets, identity, axis=0,
                              bc_type="natural", extrapolate=True)
    forward = interpolator(offsets[None, :] + phases[:, None])
    backward = interpolator(offsets[None, :] - phases[:, None])
    return forward, backward


def physical_anchor_covariance(pixel_variance, pool, forward, backward,
                               background_variance):
    """Propagate the physical detector basis with bounded sparse covariance batches.

    Squared pooling coefficients act on flattened spatial covariance matrices.
    Native-coordinate transforms target eight MiB of dense batch scratch, with
    at most half the columns in each of two buffers and a one-column minimum.
    The output and aligned covariance retain their previous sizes; no dense
    spectral covariance is constructed. Sparse entry squaring preserves the
    original treatment of duplicate CSR entries and signed coefficients.
    """
    pixel_variance = np.asarray(pixel_variance, float)
    ny, nx = pixel_variance.shape
    if np.any(~np.isfinite(pixel_variance)) or np.any(pixel_variance < 0):
        raise ValueError("Physical pixel variance must be finite and nonnegative")
    aligned_covariance = (forward * pixel_variance.T[:, None, :]) @ forward.transpose(0, 2, 1)
    output = np.empty((nx, ny, ny))
    squared_pool = sparse.csr_matrix(
        (pool.data ** 2, pool.indices, pool.indptr), shape=pool.shape)
    matrix_bytes = max(ny * ny * output.dtype.itemsize, 1)
    batch_columns = max(1, min(max(nx // 2, 1), (8 * 1024 ** 2) // (2 * matrix_bytes)))
    flattened = aligned_covariance.reshape(nx, ny * ny)
    for start in range(0, nx, batch_columns):
        stop = min(nx, start + batch_columns)
        pooled = (squared_pool[start:stop] @ flattened).reshape(stop - start, ny, ny)
        transformed = backward[start:stop] @ pooled
        np.matmul(transformed, backward[start:stop].transpose(0, 2, 1), out=output[start:stop])
        del pooled, transformed
    sky_variance = pool.multiply(pool) @ np.asarray(background_variance, float)
    return output, np.asarray(sky_variance)


def _broadcast_map(values, shape, name):
    """Broadcast one supplied detector term without hiding unusable dimensions."""
    array = np.asarray(values, float)
    if array.shape == (shape[0],):
        array = array[:, None]
    try:
        return np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"{name} cannot be broadcast to the native strip") from exc


def correlated_extraction_samples(result, strip, variance, offsets, phases,
                                  weight_floor, weight_dark, physical_floor,
                                  physical_dark, gain, sky, bvar, *, count=256,
                                  seed=0, window=None, align=None, pooling_order=None,
                                  ratio_debias=None, batch_size=16):
    """Finalize A+C means, marginal variances, and correlated absolute draws.

    ``result`` is the preliminary output of :func:`extract_signed`. Native
    detector arrays use ``(spatial pixels, columns)``; fluxes are in ADU and
    variances in ADU squared. Weight-model detector terms remain separate from
    unsmoothed physical detector terms. Reevaluate anchor covariance and the
    second-order ratio gate on one fitted physical variance basis, retaining
    the preliminary estimate as a separate diagnostic in the output writer.

    Differentiate each column's finite weight-iteration path in 512-column
    blocks. Hold the fitted physical variance, learned profiles, and masks
    fixed while propagating noise. ``batch_size`` instead limits the number
    of MC realizations generated together; it does not bin spectral columns.
    Return a dictionary with final flux/variance/quality vectors, correlated
    ``draws`` of shape ``(count, columns)``, sample IDs, and covariance operators.
    """
    strip, variance = np.asarray(strip, float), np.asarray(variance, float)
    if strip.ndim != 2 or variance.shape != strip.shape:
        raise ValueError("Native data and variance must be matching two-dimensional arrays")
    ny, nx = strip.shape
    offsets, phases = np.asarray(offsets, float), np.asarray(phases, float)
    if (offsets.shape != (ny,) or phases.shape != (nx,) or ny < 2 or nx < 1
            or not np.isfinite(offsets).all() or not np.isfinite(phases).all()
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Offsets and phases must be finite and match the native strip dimensions")
    settings = result.get("diagnostics", {})
    window = settings.get("window", 128) if window is None else window
    align = settings.get("align", True) if align is None else bool(align)
    pooling_order = settings.get("pooling_order", 1) if pooling_order is None else pooling_order
    ratio_debias = settings.get("ratio_debias", True) if ratio_debias is None else bool(ratio_debias)
    count = _extraction_control(count, 'realization count', 2, integer=True)
    batch_size = _extraction_control(batch_size, 'draw batch size', 1, integer=True)
    gain = _extraction_control(gain, 'gain')
    if gain <= 0:
        raise ValueError("Gain must be finite and positive")
    bad = np.asarray(result["anchor_mask"], bool)
    rejected = np.asarray(result.get("CR_MASK", np.zeros_like(bad)), bool)
    profile = np.asarray(result["profile"], float)
    if bad.shape != strip.shape or rejected.shape != strip.shape or profile.shape != strip.shape:
        raise ValueError("Final profile and mask must match the native strip")
    bad = bad | rejected
    sky, bvar = np.broadcast_to(sky, (nx,)), np.broadcast_to(bvar, (nx,))
    usable = ~bad & np.isfinite(strip) & np.isfinite(variance) & (variance > 0)
    usable &= np.isfinite(profile) & (profile >= 0)
    source_columns = np.all(usable, axis=0)
    if np.any(~np.isfinite(sky[source_columns])) or np.any(~np.isfinite(bvar[source_columns]) | (bvar[source_columns] < 0)):
        raise ValueError("Anchor source sky estimates and their variances must be valid")
    wf = _broadcast_map(weight_floor, strip.shape, "weight_floor")
    wd = _broadcast_map(weight_dark, strip.shape, "weight_dark")
    pf = _broadcast_map(physical_floor, strip.shape, "physical_floor")
    pd = _broadcast_map(physical_dark, strip.shape, "physical_dark")
    if np.any(~np.isfinite(pf[usable]) | (pf[usable] < 0)) or np.any(~np.isfinite(pd[usable])):
        raise ValueError("Physical detector terms are invalid on usable samples")
    fits_by_column = [None] * nx
    physical = np.zeros(strip.shape)
    profile = np.maximum(np.where(np.isfinite(profile), profile, 0.0), 0.0)
    profile /= np.maximum(profile.sum(axis=0), 1e-300)
    eligible = np.any(usable, axis=0) & np.isfinite(sky) & np.isfinite(bvar) & (bvar >= 0)
    columns = np.flatnonzero(eligible)
    if len(columns):
        batch_fits = chunked_iterated_weight_jacobian(
            strip[:, columns], profile[:, columns], variance[:, columns], usable[:, columns],
            wf[:, columns], gain, sky[columns], wd[:, columns], block_size=512)
    for index, x in enumerate(columns):
        fit = {key: (batch_fits[key][:, index] if np.asarray(batch_fits[key]).ndim == 2
                     else batch_fits[key][index])
               for key in ('flux', 'weights', 'gradient', 'previous_gradient', 'weight_prime', 'iterations', 'converged')}
        fits_by_column[x] = fit
        signal = fit["flux"] * profile[:, x] + sky[x] + np.maximum(pd[:, x], 0)
        physical[:, x] = np.where(usable[:, x],
                                  np.maximum(pf[:, x] + np.maximum(signal, 0) / gain, 1e-10), 0)
    if any(fits_by_column[x] is None for x in np.flatnonzero(source_columns)):
        raise ValueError("An anchor source lacks a valid physical noise prediction")
    previous_anchor = result["anchor"]
    pool = pooling_operator(nx, source_columns, window=window, order=pooling_order,
                             anchor_exclude=previous_anchor.get("anchor_exclude"))
    forward, backward = spatial_operators(offsets, phases, align)
    safe_bvar = np.where(np.isfinite(bvar) & (bvar >= 0), bvar, 0.0)
    qcov, qbvar = physical_anchor_covariance(physical, pool, forward, backward, safe_bvar)
    signed = np.asarray(previous_anchor["signed_calibration"], float)
    quality = np.asarray(result.get("raw_quality", result["quality"]), np.uint16).copy()
    quality |= np.asarray(result["quality"], np.uint16) & np.uint16(65535 ^ (1 | ANCHOR_INVALID))
    if "raw_quality" not in result:
        # Without raw quality, retain every original invalid column conservatively.
        quality |= np.asarray(result["quality"], np.uint16) & np.uint16(1 | 2 | 128 | ANCHOR_INVALID)
    else:
        quality &= np.uint16(65535 ^ ANCHOR_INVALID)
    flux, correction, marginal, snr = (np.full(nx, np.nan) for _ in range(4))
    direct, anchor = np.zeros(strip.shape), np.zeros(strip.shape)
    valid = np.zeros(nx, bool)
    for x, fit in enumerate(fits_by_column):
        if fit is None or quality[x] & (STRUCTURAL_FATAL | ANCHOR_INVALID):
            continue
        calibrated = calibrated_gradients(fit, signed[:, x], qcov[x], qbvar[x], ratio_debias)
        snr[x] = calibrated["anchor_snr"]
        if not calibrated["valid"]:
            quality[x] |= np.uint16(1 | ANCHOR_INVALID)
            continue
        dg, ag = calibrated["direct_gradient"], calibrated["anchor_gradient"]
        value = float(dg ** 2 @ physical[:, x] + dg.sum() ** 2 * safe_bvar[x]
                      + ag @ qcov[x] @ ag + ag.sum() ** 2 * qbvar[x])
        if not np.isfinite(value) or value <= 0:
            quality[x] |= np.uint16(1 | ANCHOR_INVALID)
            continue
        flux[x], correction[x], marginal[x] = calibrated["flux"], calibrated["correction"], value
        direct[:, x], anchor[:, x], valid[x] = dg, ag, True
    aligned_anchor = np.einsum("xji,jx->ix", backward, anchor, optimize=True)
    rng = np.random.default_rng(int(seed))
    draws = np.full((count, nx), np.nan)
    for start in range(0, count, max(1, int(batch_size))):
        end = min(count, start + max(1, int(batch_size)))
        noise = rng.normal(size=(end - start, ny, nx)) * np.sqrt(physical)[None, :, :]
        noise += rng.normal(size=(end - start, nx))[:, None, :] * np.sqrt(safe_bvar)[None, None, :]
        noise[:, ~usable] = 0.0
        aligned = np.einsum("xij,bjx->bix", forward, noise, optimize=True)
        pooled = (pool @ aligned.reshape(-1, nx).T).T.reshape(end - start, ny, nx)
        delta = (np.einsum("ix,bix->bx", direct, noise, optimize=True)
                 + np.einsum("ix,bix->bx", aligned_anchor, pooled, optimize=True))
        draws[start:end, valid] = flux[valid] + delta[:, valid]
    old_flux = np.asarray(result["flux"], float)
    common = valid & np.isfinite(old_flux)
    relative = np.abs(flux[common] - old_flux[common]) / np.maximum(np.abs(old_flux[common]), 1.0)
    updated_anchor = dict(previous_anchor)
    updated_anchor.update(calibration_covariance=qcov,
                           calibration_variance=np.diagonal(qcov, axis1=1, axis2=2).T.copy(),
                           calibration_background_variance=qbvar)
    diagnostics = {
        "scope": CONDITIONAL_SCOPE, "samples": count, "seed": int(seed),
        "draw_batch_size": max(1, int(batch_size)),
        "physical_basis": "UNSMOOTHED_FLOOR_PLUS_FITTED_SOURCE_SKY_DARK_PHOTONS",
        "mean_finalization": "ANCHOR_COVARIANCE_AND_RATIO_GATE_REEVALUATED",
        "pool_nonzero": int(pool.nnz), "original_valid": int(np.isfinite(old_flux).sum()),
        "final_valid": int(valid.sum()), "newly_invalid": int(np.sum(np.isfinite(old_flux) & ~valid)),
        "newly_valid": int(np.sum(~np.isfinite(old_flux) & valid)),
        "max_mean_change_adu": float(np.max(np.abs(flux[common] - old_flux[common]))) if common.any() else None,
        "max_mean_relative_change": float(np.max(relative)) if len(relative) else None,
        "median_mean_relative_change": float(np.median(relative)) if len(relative) else None,
        "finite_iteration_cap_hits": int(sum(fit is not None and not fit["converged"] for fit in fits_by_column)),
        "jacobian_batch_columns": 512,
        "crosscal": "NOT_MODELED", "variance_prediction_fixed_in_derivative": True,
    }
    return dict(flux=flux, correction=correction, variance=marginal, quality=quality,
                draws=draws, sample_ids=np.arange(count, dtype=np.int32),
                anchor_snr=snr, anchor=updated_anchor, diagnostics=diagnostics,
                direct_gradient=direct, anchor_gradient=anchor, physical_variance=physical,
                pool=pool, forward=forward, backward=backward,
                background_variance=safe_bvar)


def band_covariance(result, band_weights):
    """Compute exact conditional covariances for selected bands by adjoints.

    This diagnostic avoids a dense full-spectrum covariance.  Every band's
    direct and anchor gradients are combined before squaring, so all reuse
    cross terms enter on the same native detector and common-sky noise basis.
    """
    h = np.atleast_2d(np.asarray(band_weights, float))
    direct, anchor = result["direct_gradient"], result["anchor_gradient"]
    ny, nx = direct.shape
    if h.shape[1] != nx or np.any(~np.isfinite(h)):
        raise ValueError("Band weights must be finite and match the native spectrum")
    if np.any(np.any(h != 0, axis=0) & ~np.isfinite(result["variance"])):
        raise ValueError("A band includes an invalid extracted column")
    gradients = np.empty((len(h), ny, nx))
    for k, weights in enumerate(h):
        output_gradient = anchor * weights[None, :]
        aligned = np.einsum("xji,jx->ix", result["backward"], output_gradient, optimize=True)
        pooled = (result["pool"].T @ aligned.T).T
        native = np.einsum("xji,jx->ix", result["forward"], pooled, optimize=True)
        gradients[k] = direct * weights[None, :] + native
    detector = (gradients * np.sqrt(result["physical_variance"])[None, :, :]).reshape(len(h), -1)
    sky = gradients.sum(axis=1) * np.sqrt(result["background_variance"])[None, :]
    covariance = detector @ detector.T + sky @ sky.T
    return covariance


# =============================================================================
# Extraction ac
# =============================================================================

class ExtractionStageError(ValueError):
    """Expose a whole-spectrum learning failure without inventing valid flux."""

    def __init__(self, stage, columns, cause):
        """Attach machine-readable stage metadata to the original failure."""
        self.stage = str(stage)
        self.columns = int(columns)
        self.cause_type = type(cause).__name__
        super().__init__(f'{self.stage}: {cause}')


def calibrated_column(data, profile, variance, signed_calibration,
                      calibration_variance, *, mask=None, noise_floor=None,
                      dark_model=None, gain=.087, background_level=20.,
                      ratio_debias=True, background_variance=0.,
                      calibration_background_variance=0., reject_cosmics=False,
                      calibration_covariance=None, profile_sigma=None,
                      physical_noise_floor=None, physical_dark_model=None,
                      sigma_clip=10., max_rejections=4,
                      min_profile_fraction=0.5, min_core_fraction=0.5,
                      min_pixels=5, core_threshold=0.5):
    """Apply signed normalization with separate physical detector variance.

    Dark current remains inside the production photon clipping expression:
    noise_floor + max(F*P + background + max(dark_model, 0), 0)/gain.
    Returned gradients retain the frozen method's first-order conditional
    approximation; they do not differentiate learned profiles or masks.
    """
    original = extract_column_physical(
        data, profile, variance, mask=mask, reject_cosmics=reject_cosmics,
        noise_floor=noise_floor, dark_model=dark_model, gain=gain,
        background_level=background_level, background_variance=background_variance,
        profile_sigma=profile_sigma, physical_noise_floor=physical_noise_floor,
        physical_dark_model=physical_dark_model, sigma_clip=sigma_clip,
        max_rejections=max_rejections, min_profile_fraction=min_profile_fraction,
        min_core_fraction=min_core_fraction, min_pixels=min_pixels, core_threshold=core_threshold)
    return _calibrate_extraction_result(
        original, signed_calibration, calibration_variance, calibration_covariance,
        calibration_background_variance, ratio_debias)


def _calibrate_extraction_result(original, signed_calibration, calibration_variance,
                                 calibration_covariance=None,
                                 calibration_background_variance=0., ratio_debias=True):
    """Apply the original scalar calibration algebra to an already fitted column."""
    weights = original.weights
    q = np.asarray(signed_calibration, float)
    v = np.asarray(calibration_variance, float)
    covariance = None if calibration_covariance is None else np.asarray(calibration_covariance, float)
    a = float(q.sum())
    b = float(weights @ q)
    vb = float((weights**2 @ v if covariance is None else weights @ covariance @ weights)
               + weights.sum()**2 * calibration_background_variance)
    cab = float((weights @ v if covariance is None else covariance.sum(axis=0) @ weights)
                + len(q) * weights.sum() * calibration_background_variance)
    if (not np.isfinite([original.flux, original.variance]).all() or original.variance <= 0
            or a <= 0 or b <= 0 or not np.isfinite([a, b, vb, cab]).all()
            or vb <= 0 or b < 10*np.sqrt(vb)):
        return dict(flux=np.nan, variance=np.nan, uncorrected=original.flux,
                    correction=np.nan, valid=False, quality=int(original.quality) | 512,
                    rejected=original.rejected, raw_quality=int(original.quality), profile_fraction=original.profile_fraction,
                    core_fraction=original.core_fraction, anchor_snr=(b/np.sqrt(vb) if vb > 0 else np.nan))
    correction = a/b
    if ratio_debias:
        correction -= a*vb/b**3 - cab/b**2
    derivative = (b-a*weights)/b**2
    correction_variance = float((derivative**2 @ v if covariance is None
                                else derivative @ covariance @ derivative)
                               + derivative.sum()**2*calibration_background_variance)
    flux = original.flux*correction
    flux_variance = original.variance*correction**2 + original.flux**2*correction_variance
    if not np.isfinite([correction, flux, flux_variance]).all() or correction <= 0 or flux_variance <= 0:
        return dict(flux=np.nan, variance=np.nan, uncorrected=original.flux,
                    correction=np.nan, valid=False, quality=int(original.quality) | 512,
                    rejected=original.rejected, raw_quality=int(original.quality), profile_fraction=original.profile_fraction,
                    core_fraction=original.core_fraction, anchor_snr=b/np.sqrt(vb))
    return dict(flux=flux, variance=flux_variance, uncorrected=original.flux,
                correction=correction, correction_variance=correction_variance,
                valid=True, quality=int(original.quality), raw_quality=int(original.quality), rejected=original.rejected, profile_fraction=original.profile_fraction,
                core_fraction=original.core_fraction, anchor_snr=(b/np.sqrt(vb) if vb > 0 else np.inf),
                calibration_gradient=original.flux*derivative,
                extraction_weights=correction*weights)


def _calibrate_extraction_columns(raw, anchor, ratio_debias=True):
    """Batch signed calibration, reevaluating numerical boundary cases scalarly.

    The raw Horne fit has already frozen each column's stopping state. Columns
    close to the S/N or positivity gates use the original reduction order, so
    the faster quadratic forms do not decide a floating-point boundary.
    """
    q = np.asarray(anchor['signed_calibration'], float)
    v = np.asarray(anchor['calibration_variance'], float)
    weights = np.asarray(raw['weights'], float)
    if q.ndim != 2 or v.shape != q.shape or weights.shape != q.shape:
        raise ValueError('Anchor, variance, and extraction weights must have matching shapes')
    ny, nx = q.shape
    covariance = anchor.get('calibration_covariance')
    if covariance is not None:
        covariance = np.asarray(covariance, float)
        if covariance.shape != (nx, ny, ny):
            raise ValueError('Anchor covariance must match the native column dimensions')
    cbvar = np.asarray(anchor['calibration_background_variance'], float)
    if cbvar.shape != (nx,):
        raise ValueError('Anchor background variance must contain one value per column')
    w, qt, vt = weights.T, q.T, v.T
    a, b, ws = qt.sum(axis=1), np.sum(w * qt, axis=1), w.sum(axis=1)
    if covariance is None:
        vb = np.sum(w * w * vt, axis=1) + ws**2 * cbvar
        cab = np.sum(w * vt, axis=1) + ny * ws * cbvar
    else:
        cw = np.einsum('xij,xj->xi', covariance, w)
        vb = np.sum(w * cw, axis=1) + ws**2 * cbvar
        cab = np.einsum('xij,xj->x', covariance, w) + ny * ws * cbvar
    f0, v0 = np.asarray(raw['flux']), np.asarray(raw['variance'])
    raw_finite = np.isfinite(f0) & np.isfinite(v0) & (v0 > 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        threshold = 10 * np.sqrt(vb)
        valid = (raw_finite & np.isfinite(a) & np.isfinite(b) & np.isfinite(vb)
                 & np.isfinite(cab) & (a > 0) & (b > 0) & (vb > 0) & (b >= threshold))
        correction = a / b
        debias = a * vb / b**3 - cab / b**2 if ratio_debias else np.zeros(nx)
        correction -= debias
        derivative = (b[:, None] - a[:, None] * w) / b[:, None]**2
        if covariance is None:
            cv = np.sum(derivative**2 * vt, axis=1) + derivative.sum(axis=1)**2 * cbvar
        else:
            cv = np.einsum('xi,xij,xj->x', derivative, covariance, derivative)
            cv += derivative.sum(axis=1)**2 * cbvar
        flux, outvar = f0 * correction, v0 * correction**2 + f0**2 * cv
        valid &= (np.isfinite(correction) & np.isfinite(flux) & np.isfinite(outvar)
                  & (correction > 0) & (outvar > 0))
        snr = np.where(vb > 0, b / np.sqrt(vb), np.nan)
        tolerance = 64 * np.finfo(float).eps * max(ny, 1)
        near_snr = np.abs(b - threshold) <= tolerance * np.maximum(np.abs(b) + threshold, 1.)
        near_a = np.abs(a) <= tolerance * np.maximum(np.abs(qt).sum(axis=1), 1.)
        near_b = np.abs(b) <= tolerance * np.maximum(np.abs(w * qt).sum(axis=1), 1.)
        variance_scale = (np.sum(np.abs(w * w * vt), axis=1) if covariance is None
                          else np.sum(np.abs(w * cw), axis=1)) + np.abs(ws**2 * cbvar)
        near_vb = np.abs(vb) <= tolerance * np.maximum(variance_scale, 1.)
        near_corr = np.abs(correction) <= tolerance * np.maximum(np.abs(a / b) + np.abs(debias), 1.)
        near_var = np.abs(outvar) <= tolerance * np.maximum(np.abs(v0 * correction**2) + np.abs(f0**2 * cv), 1.)
        sensitive = raw_finite & (near_snr | near_a | near_b | near_vb | near_corr | near_var)
    quality = np.asarray(raw['quality'], np.uint16).copy()
    quality[~valid] |= np.uint16(512)
    output = dict(flux=np.where(valid, flux, np.nan), variance=np.where(valid, outvar, np.nan),
                  correction=np.where(valid, correction, np.nan), valid=valid, quality=quality,
                  raw_quality=np.asarray(raw['quality'], np.uint16).copy(), anchor_snr=snr,
                  calibration_gradient=np.where(valid[None, :], (f0[:, None] * derivative).T, np.nan),
                  extraction_weights=np.where(valid[None, :], (correction[:, None] * w).T, np.nan))
    for x in np.flatnonzero(sensitive):
        original = ExtractionResult(float(f0[x]), float(v0[x]), raw['rejected'][:, x],
                                    int(raw['quality'][x]), float(raw['profile_fraction'][x]),
                                    float(raw['core_fraction'][x]), weights[:, x])
        scalar = _calibrate_extraction_result(original, q[:, x], v[:, x],
                                              None if covariance is None else covariance[x],
                                              cbvar[x], ratio_debias)
        for key in ('flux', 'variance', 'correction', 'valid', 'quality', 'anchor_snr'):
            output[key][x] = scalar[key]
        for key in ('calibration_gradient', 'extraction_weights'):
            output[key][:, x] = scalar[key] if scalar['valid'] else np.nan
    return output


def _positive_rejection_pass(strip, variance, bad, offsets, phases, floor,
                             dark, gain, sky, bvar, physical_floor, physical_dark,
                             n_components=4, initial_xbin=64, x_smooth=20.,
                             profile_floor=1e-5, sigma_clip=10., max_rejections=4,
                             min_profile_fraction=0.5, min_core_fraction=0.5,
                             min_pixels=5, core_threshold=0.5):
    """Find positive hits using physical variance and the original profile model."""
    profile, profile_sigma, _, _, _ = _compact_profile_model(
        strip, variance, bad, offsets, phases, n_components, initial_xbin, x_smooth, profile_floor, sigma_clip)
    result = chunked_extract_physical(
        strip, profile, variance, mask=bad, profile_sigma=profile_sigma,
        noise_floor=floor, dark_model=dark, gain=gain, background_level=sky,
        background_variance=bvar, reject_cosmics=True,
        physical_noise_floor=physical_floor, physical_dark_model=physical_dark,
        sigma_clip=sigma_clip, max_rejections=max_rejections,
        min_profile_fraction=min_profile_fraction, min_core_fraction=min_core_fraction,
        min_pixels=min_pixels, core_threshold=core_threshold, block_size=512)
    return result['rejected'], result['quality'], profile_sigma


def _pixel_map(values, shape, name, default=None):
    """Broadcast a scalar, spatial vector, or native pixel map explicitly."""
    if values is None:
        if default is None:
            raise ValueError(f'{name} is required')
        values = default
    result = np.asarray(values, float)
    if result.shape == (shape[0],):
        result = result[:, None]
    try:
        return np.broadcast_to(result, shape)
    except ValueError as exc:
        raise ValueError(f'{name} cannot broadcast to the native strip') from exc


def extract_signed(strip, variance, bad, offsets, phases, noise_floor,
                   dark_model, gain, sky, background_variance=None,
                   reject_cosmics=False, *, window=128, align=True,
                   ratio_debias=True, pooling_order=1, max_cr_refinements=3,
                   physical_noise_floor=None, physical_dark_model=None, training_bin=64,
                   n_components=4, initial_xbin=64, x_smooth=20., profile_floor=1e-5,
                   sigma_clip=10., max_rejections=4, min_profile_fraction=0.5,
                   min_core_fraction=0.5, min_pixels=5, core_threshold=0.5):
    """Extract native pixels with separate physical variance and coarser NMF training.

    ``strip``, ``variance``, and ``bad`` have shape ``(spatial pixels, columns)``.
    The strip is sky-subtracted ADU; variance is independent propagated pixel
    variance in ADU squared, excluding the common estimated-sky variance.
    ``noise_floor`` and ``dark_model`` supply detector terms for fitting weights;
    the optional physical maps supply unsmoothed detector terms for variances.
    ``background_variance`` is the variance of the estimated mean sky per pixel.
    ``offsets`` and ``phases`` give spatial coordinates and column trace phases
    in pixels. Input samples remain on their native grid.

    Learn NMF shapes in bins of ``training_bin`` columns within each fold
    (default 64), retaining independent column fits in 512-column blocks.
    Build the signed anchor once after the final CR/profile pass, with pooling
    half-width ``window`` (default 128 native columns). Unsupported profile or
    anchor training raises :class:`ExtractionStageError`. Low-significance
    anchor normalization produces NaN and fatal quality flags for that column;
    an unstable final CR/profile pass invalidates the entire frame.

    Return flux/variance/quality vectors together with native-grid profiles,
    weights, masks, anchor calibration, and diagnostics. Use
    :func:`correlated_extraction_samples` to finalize correlated uncertainties.
    """
    strip = np.asarray(strip, float)
    variance = np.asarray(variance, float)
    bad = np.asarray(bad, bool)
    offsets, phases = np.asarray(offsets, float), np.asarray(phases, float)
    if strip.ndim != 2 or variance.shape != strip.shape or bad.shape != strip.shape:
        raise ValueError('Native strip, variance, and bad mask must have matching two-dimensional shapes')
    ny, nx = strip.shape
    if offsets.shape != (ny,) or phases.shape != (nx,):
        raise ValueError('Offsets and phases must match the native strip dimensions')
    if ny < 2 or nx < 1 or not np.isfinite(offsets).all() or not np.isfinite(phases).all() or np.any(np.diff(offsets) <= 0):
        raise ValueError('Offsets must increase strictly and native coordinates must be finite and nonempty')
    gain = _extraction_control(gain, 'gain')
    if gain <= 0:
        raise ValueError('Gain must be finite and positive')
    max_cr_refinements = _extraction_control(max_cr_refinements, 'max_cr_refinements', 1, integer=True)
    training_bin = _extraction_control(training_bin, 'training_bin', 1, integer=True)
    n_components = _extraction_control(n_components, 'n_components', 1, integer=True)
    initial_xbin = _extraction_control(initial_xbin, 'initial_xbin', 1, integer=True)
    min_pixels = _extraction_control(min_pixels, 'min_pixels', 1, integer=True)
    max_rejections = _extraction_control(max_rejections, 'max_rejections', 0, integer=True)
    sigma_clip = _extraction_control(sigma_clip, 'sigma_clip')
    if sigma_clip <= 0:
        raise ValueError('sigma_clip must be positive')
    x_smooth = _extraction_control(x_smooth, 'x_smooth', 0)
    profile_floor = _extraction_control(profile_floor, 'profile_floor', 0)
    min_profile_fraction = _extraction_control(min_profile_fraction, 'min_profile_fraction', 0, 1)
    min_core_fraction = _extraction_control(min_core_fraction, 'min_core_fraction', 0, 1)
    core_threshold = _extraction_control(core_threshold, 'core_threshold', 0, 1)
    pooling_order = _extraction_control(pooling_order, 'pooling_order', 0, integer=True)
    if window is not None:
        window = _extraction_control(window, 'window', 0, integer=True)
    bvar = np.zeros(nx) if background_variance is None else np.broadcast_to(
        np.asarray(background_variance, float), (nx,))
    sky = np.broadcast_to(np.asarray(sky, float), (nx,))
    floor = _pixel_map(noise_floor, strip.shape, 'noise_floor')
    dark = _pixel_map(dark_model, strip.shape, 'dark_model', default=0.)
    physical_floor = _pixel_map(physical_noise_floor, strip.shape, "physical_noise_floor", default=floor)
    physical_dark = _pixel_map(physical_dark_model, strip.shape, "physical_dark_model", default=dark)
    used_bad = bad.copy()
    cr_mask = np.zeros_like(bad)
    profile_sigma = None
    discovery_passes = 0
    if reject_cosmics:
        for _ in range(max(1, int(max_cr_refinements))):
            try:
                new, _, profile_sigma = _positive_rejection_pass(
                    strip, variance, used_bad, offsets, phases, floor,
                    dark, gain, sky, bvar, physical_floor, physical_dark,
                    n_components, initial_xbin, x_smooth, profile_floor, sigma_clip, max_rejections,
                    min_profile_fraction, min_core_fraction, min_pixels, core_threshold)
            except ValueError as exc:
                raise ExtractionStageError('positive_rejection_profile', nx, exc) from exc
            discovery_passes += 1
            new &= ~used_bad
            if not new.any():
                break
            used_bad |= new
            cr_mask |= new
    unstable = False
    for refinement in range(int(max_cr_refinements)):
        try:
            profile, fallback = crossfit_profiles(
                strip, variance, used_bad, offsets, phases, training_bin=training_bin,
                n_components=n_components, x_smooth=x_smooth,
                profile_floor=profile_floor, sigma_clip=sigma_clip)
        except ValueError as exc:
            raise ExtractionStageError('signed_profile_learning', nx, exc) from exc
        raw = chunked_extract_physical(
            strip, profile, variance, mask=used_bad, noise_floor=floor, dark_model=dark,
            gain=gain, background_level=sky, background_variance=bvar,
            profile_sigma=profile_sigma, reject_cosmics=reject_cosmics,
            physical_noise_floor=physical_floor, physical_dark_model=physical_dark,
            sigma_clip=sigma_clip, max_rejections=max_rejections,
            min_profile_fraction=min_profile_fraction, min_core_fraction=min_core_fraction,
            min_pixels=min_pixels, core_threshold=core_threshold, block_size=512)
        new = raw['rejected'] & ~used_bad
        if not new.any():
            break
        cr_mask |= new
        if refinement == int(max_cr_refinements)-1:
            unstable = True
            break
        used_bad |= new
    try:
        _, anchor = build_anchor(strip, variance, used_bad, offsets, phases, window=window,
                                 crossfit=True, align=align, pooling_order=pooling_order,
                                 background_variance=bvar)
    except ValueError as exc:
        raise ExtractionStageError('signed_profile_learning', nx, exc) from exc
    anchor['training_bin_fold_columns'] = int(training_bin)
    anchor['limited_shape_folds'] = fallback
    calibrated = _calibrate_extraction_columns(raw, anchor, ratio_debias)
    flux, out_variance = calibrated['flux'], calibrated['variance']
    corrections = calibrated['correction']
    quality = calibrated['quality']
    quality[~calibrated['valid']] |= np.uint16(1)
    if unstable:
        flux[:] = np.nan
        out_variance[:] = np.nan
        quality[:] |= np.uint16(1024)
    quality[np.any(cr_mask, axis=0)] |= np.uint16(64)
    return dict(
        flux=flux, variance=out_variance, quality=quality, profile=profile,
        correction=corrections, extraction_weights=calibrated['extraction_weights'],
        profile_fraction=raw['profile_fraction'], core_fraction=raw['core_fraction'],
        anchor_snr=calibrated['anchor_snr'], raw_flux=raw['flux'],
        raw_quality=calibrated['raw_quality'], anchor=anchor, profile_sigma=profile_sigma,
        calibration_gradient=calibrated['calibration_gradient'], anchor_mask=used_bad,
        CR_MASK=cr_mask,
        diagnostics=dict(
            method='A+C_refined_v1', window=window,
            corrections=corrections, align=align, crossfit=True, ratio_debias=ratio_debias,
            pooling_order=pooling_order, independent_folds=3,
            anchor_exclude=anchor.get('anchor_exclude'),
            calibration_count=anchor['calibration_count'],
            initial_cr_discovery_passes=discovery_passes,
            final_cr_refinements=refinement+1,
            anchor_builds=1, horne_batch_columns=512,
            dark_model_forwarded=True, training_bin=training_bin,
            physical_variance_separated=True,
            limited_shape_folds=anchor.get("limited_shape_folds", []),
            limitations=[
                'Cross-column anchor covariance requires the band helper',
                'Covariance is conditional on learned profiles, detector maps, and masks',
                'Data-dependent CR selection requires end-to-end validation']))


def _validate_settings(settings):
    """Reject malformed numerical settings before they can be silently coerced."""
    for name in ('n_components', 'initial_xbin', 'training_bin', 'window', 'max_cr_refinements', 'min_pixels'):
        value = settings[name]
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f'A+C setting {name} must be a positive integer')
    value = settings['niter']
    if not np.isfinite(value) or int(value) != value or value < 0:
        raise ValueError('A+C niter must be a nonnegative integer')
    for name in ('cr_sigma', 'profile_floor'):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f'A+C setting {name} must be finite and positive')
    if not np.isfinite(settings['x_smooth']) or settings['x_smooth'] < 0:
        raise ValueError('A+C x_smooth must be finite and nonnegative')
    for name in ('min_profile_fraction', 'min_core_fraction', 'core_threshold'):
        if not np.isfinite(settings[name]) or not 0 < settings[name] <= 1:
            raise ValueError(f'A+C setting {name} must lie in (0, 1]')
    count = settings['mc_samples']
    if not np.isfinite(count) or int(count) != count or count < 2:
        raise ValueError('A+C covariance requires at least two Monte Carlo samples')


def _negative_pixel_table(result, strip, variance, bad, rows, header, floor, dark,
                          gain, sky, bvar, sigma_clip):
    """Preserve unmasked large negative residuals and their detector coordinates."""
    entries = []
    p_sigma = result['profile_sigma']
    for x in np.flatnonzero(result['quality'] & NEGATIVE_RESIDUAL):
        flux = result['raw_flux'][x]
        if not np.isfinite(flux):
            continue
        physical = pixel_variance(flux, result['profile'][:, x], variance[:, x],
                                  floor[:, x], gain, sky[x], dark[:, x])
        prediction = np.zeros(strip.shape[0]) if p_sigma is None else p_sigma[:, x]
        scale = np.sqrt(physical + bvar[x] + (flux*prediction)**2)
        significance = (strip[:, x] - flux*result['profile'][:, x]) / scale
        usable = ~bad[:, x] & ~result['CR_MASK'][:, x] & np.isfinite(significance)
        for y in np.flatnonzero(usable & (significance < -sigma_clip)):
            entries.append((int(x), int(rows[y, x]), int(rows[y, x] + header.get('TRIM_Y0', 0)),
                            float(significance[y]), True))
    columns = [
        fits.Column(name='X', format='J', array=[row[0] for row in entries]),
        fits.Column(name='TRIM_Y', format='J', array=[row[1] for row in entries]),
        fits.Column(name='RAW_Y', format='J', array=[row[2] for row in entries]),
        fits.Column(name='SIGNIFICANCE', format='D', array=[row[3] for row in entries]),
        fits.Column(name='PIXEL_USED', format='L', array=[row[4] for row in entries]),
    ]
    table = fits.BinTableHDU.from_columns(columns, name='NEGATIVE_PIXELS')
    table.header['COORDSYS'] = 'ZERO_BASED'
    table.header['COMMENT'] = 'Pixel remains in the fit; entire-column validity is given by QUALITY'
    return table


def _sample_provenance(input_fits, header, base_seed=None):
    """Derive independent exposure seeds reproducibly across run directories.

    An explicitly configured seed is a base seed, not a request to reuse the
    same pixel-noise realization in independent exposures. With no base seed,
    the original frame-identity seed remains unchanged.
    """
    identity = hashlib.sha256((Path(input_fits).name + str(header.get('DATE-OBS', '')) +
                               str(header.get('OBJECT', ''))).encode('utf-8')).hexdigest()
    if base_seed is None:
        return identity, int(identity[:8], 16)
    value = int(base_seed)
    if value < 0 or value != base_seed:
        raise ValueError('A+C Monte Carlo base seed must be a nonnegative integer')
    mixed = hashlib.sha256(f'{value}|{identity}'.encode('utf-8')).hexdigest()
    return identity, int(mixed[:16], 16) % (2**63)


def write_ac_extraction(input_fits, header, coeffs, trace, rows, inside, clipped,
                        offsets, phases, strip, variance, bad, noise_floor, dark_model,
                        sky, bvar, background_count, *, aperture_width, sky_outer, niter,
                        cr_sigma, n_components, initial_xbin, x_smooth, profile_floor,
                        min_profile_fraction, min_core_fraction, min_pixels, core_threshold,
                        training_bin, window, max_cr_refinements, mc_samples, mc_seed,
                        variance_fallback=False):
    """Write A+C native flux, conditional correlated samples, and failure diagnostics.

    Finalize the preliminary mean estimate by recomputing anchor covariance
    and its second-order correction on one fitted physical variance basis.
    Save both estimates explicitly, along with the correlated noise draws.
    Invalid anchor normalization remains flagged and is not replaced by an
    alternative extraction estimator.
    """
    
    if noise_floor is None or dark_model is None or variance_fallback:
        raise ExtractionStageError('detector_terms', strip.shape[1], ValueError(
            'A+C requires propagated variance, NOISE_FLOOR and DARK_MODEL; no automatic Horne fallback'))
    settings = dict(niter=niter, cr_sigma=cr_sigma, n_components=n_components,
                    initial_xbin=initial_xbin, x_smooth=x_smooth, profile_floor=profile_floor,
                    min_profile_fraction=min_profile_fraction, min_core_fraction=min_core_fraction,
                    min_pixels=min_pixels, core_threshold=core_threshold, training_bin=training_bin,
                    window=window, max_cr_refinements=max_cr_refinements, mc_samples=mc_samples)
    _validate_settings(settings)
    settings.update(independent_folds=3, fold_smooth=float(x_smooth)/3,
                    ring_rows=3, ring_cols=11, hot_sigma=5., pooling_order=1,
                    anchor_min_snr=10., align=True, ratio_debias=True)
    nx = strip.shape[1]
    xx = np.arange(nx)[None, :]
    row_range = (int(np.floor(trace.min()))-int(np.ceil(sky_outer))-3,
                 int(np.ceil(trace.max()))+int(np.ceil(sky_outer))+4)
    a_floor, a_dark = expected_detector_terms(noise_floor, dark_model, row_range)
    wf, wd = a_floor[clipped, xx], a_dark[clipped, xx]
    pf, pd = noise_floor[clipped, xx], dark_model[clipped, xx]
    gain = float(header['EGAIN'])
    result = extract_signed(
        strip, variance, bad, offsets, phases, wf, wd, gain, sky,
        background_variance=bvar, reject_cosmics=int(niter) > 0,
        window=int(window), align=True, ratio_debias=True, pooling_order=1,
        max_cr_refinements=int(max_cr_refinements), physical_noise_floor=pf,
        physical_dark_model=pd, training_bin=int(training_bin),
        n_components=int(n_components), initial_xbin=int(initial_xbin),
        x_smooth=float(x_smooth), profile_floor=float(profile_floor),
        sigma_clip=float(cr_sigma), max_rejections=int(niter),
        min_profile_fraction=min_profile_fraction, min_core_fraction=min_core_fraction,
        min_pixels=int(min_pixels), core_threshold=core_threshold)
    outside = ~np.all(inside, axis=0)
    invalid_sky = ~np.isfinite(sky) | ~np.isfinite(bvar) | (bvar < 0)
    for name in ('quality', 'raw_quality'):
        result[name][outside] |= APERTURE_OUTSIDE
        result[name][invalid_sky] |= INVALID_BACKGROUND
    identity, seed = _sample_provenance(input_fits, header, mc_seed)
    propagated = correlated_extraction_samples(
        result, strip, variance, offsets, phases, wf, wd, pf, pd, gain, sky, bvar,
        count=int(mc_samples), seed=seed)
    flux = np.asarray(propagated['flux'], float)
    outvar = np.asarray(propagated['variance'], float)
    quality = np.asarray(propagated['quality'], np.uint16)
    invalid = ((quality & FATAL_QUALITY) != 0) | ~np.isfinite(flux) | ~np.isfinite(outvar) | (outvar <= 0)
    quality[invalid & ((quality & FATAL_QUALITY) == 0)] |= INSUFFICIENT_SUPPORT
    flux[invalid], outvar[invalid] = np.nan, np.nan
    draws = np.asarray(propagated['draws'], float)
    if draws.shape != (int(mc_samples), nx):
        raise ValueError('A+C covariance generator returned an inconsistent sample shape')
    draws[:, invalid] = np.nan
    if not np.isfinite(draws[:, ~invalid]).all():
        raise ValueError('A+C covariance samples are nonfinite on valid output columns')
    negative_table = _negative_pixel_table(result, strip, variance, bad, rows, header,
                                           pf, pd, gain, sky, bvar, cr_sigma)
    if invalid.any():
        warnings.warn(f'{Path(input_fits).name}: A+C has {invalid.sum()}/{nx} unusable columns; no fallback applied.', RuntimeWarning)
    if len(negative_table.data):
        warnings.warn(f'{Path(input_fits).name}: {len(negative_table.data)} large negative detector residuals remain in the fit; see NEGATIVE_PIXELS.', RuntimeWarning)
    header = header.copy()
    header['EXTRMETH'] = ('AC', 'Refined independent NMF and signed anchor')
    header['ACVERS'] = 'REFINED_P06_V1'
    header['HFWIDTH'] = float(aperture_width)
    header['EXTRBAD'] = int(invalid.sum())
    header['EXTRCR'] = int(result['CR_MASK'].sum())
    header['EXTRNEG'] = int(np.count_nonzero(quality & NEGATIVE_RESIDUAL))
    header['ACFAIL'] = int(np.count_nonzero(quality & ANCHOR_FAILED))
    header['ACUNST'] = int(np.count_nonzero(quality & CR_PROFILE_UNSTABLE))
    header['ACBIN'] = int(training_bin)
    header['ACWIN'] = int(window)
    header['ACFOLDS'] = 3
    header['ACLIMIT'] = sum(result['diagnostics']['limited_shape_folds'])
    header['ERRMODEL'] = 'AC_CONDITIONAL'
    header['PROFERR'] = ('REJECT', 'Profile-selection uncertainty not included')
    header['CROSSCAL'] = ('NOT_MODELED', 'Cross-frame master covariance omitted')
    header['MCID'] = identity[:24]
    header['MCSEED'] = seed
    header['MCSCOPE'] = 'FIXED_PROFILE_A_MASK'
    header['MCN'] = int(mc_samples)
    header['HISTORY'] = 'A+C native extraction: weights use ring A; physical errors use unsmoothed maps'
    header['HISTORY'] = 'P06 uses one fitted physical variance for direct/anchor covariance and cross terms'
    header['HISTORY'] = 'MC conditions on learned profile, A maps, masks, sky/model variance; no shared-master cross-frame covariance'
    hdus = [fits.PrimaryHDU(header=header)]
    anchor = propagated.get('anchor', result['anchor'])
    values = [
        ('FLUX_ADU', flux), ('VAR_ADU2', outvar), ('BKG_ADU', sky), ('BKG_VAR', bvar),
        ('PROFILE_FRACTION', result['profile_fraction']), ('CORE_FRACTION', result['core_fraction']),
        ('AC_CORRECTION', propagated['correction']), ('ANCHOR_SNR', propagated.get('anchor_snr', result['anchor_snr'])),
        ('FLUX_AC_PORT', result['flux']), ('VAR_AC_PORT', result['variance']),
        ('CORR_AC_PORT', result['correction']), ('PROFILE', result['profile']),
    ]
    for name, value in values:
        hdu = fits.ImageHDU(np.asarray(value, np.float32), name=name)
        hdu.header['BUNIT'] = ('adu2' if name in ('VAR_ADU2', 'VAR_AC_PORT', 'BKG_VAR')
                               else 'adu' if name in ('FLUX_ADU', 'FLUX_AC_PORT', 'BKG_ADU') else '1')
        hdus.append(hdu)
    qhdu = fits.ImageHDU(quality, name='QUALITY')
    qhdu.header['FATALBIT'] = FATAL_QUALITY
    for bit, label in QUALITY_BIT_DESCRIPTIONS.items():
        qhdu.header[f'BIT{bit}'] = label
    mc_hdu = fits.ImageHDU(draws, name='EXTRACT_MC')
    mc_hdu.header['BUNIT'] = 'adu'
    mc_hdu.header['MCTYPE'] = 'ABSOLUTE_FLUX'
    mc_hdu.header['MCID'] = identity[:24]
    hdus.extend([qhdu, fits.ImageHDU(invalid.astype(np.uint8), name='MASK'),
                 fits.ImageHDU(result['CR_MASK'].sum(axis=0).astype(np.int16), name='NREJECT'),
                 fits.ImageHDU(np.asarray(background_count, np.int16), name='BKG_N'),
                 fits.ImageHDU(result['CR_MASK'].astype(np.uint8), name='CR_MASK'),
                 fits.ImageHDU(rows[0].astype(np.int32), name='AP_ROW0'),
                 fits.ImageHDU(np.asarray(anchor['calibration_count'], np.int16), name='ANCHOR_N'),
                 mc_hdu, fits.ImageHDU(np.arange(int(mc_samples), dtype=np.int32), name='MC_IDS'),
                 negative_table])
    settings['mc_seed'] = seed
    settings['mc_base_seed'] = mc_seed
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name='PARAMETER', format='32A', array=list(settings)),
        fits.Column(name='VALUE', format='96A', array=[json.dumps(value) for value in settings.values()]),
    ], name='EXTRACT_CONFIG'))
    output = str(Path(input_fits).with_suffix('')) + '.ec.fits'
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    common = np.isfinite(result['flux']) & ~invalid
    delta = flux[common] - result['flux'][common]
    diagnostics = dict(output=output, coeffs=np.asarray(coeffs).tolist(), method='AC',
                       valid_columns=int((~invalid).sum()), total_columns=nx,
                       anchor_failed_columns=int(np.count_nonzero(quality & ANCHOR_FAILED)),
                       unstable_columns=int(np.count_nonzero(quality & CR_PROFILE_UNSTABLE)),
                       negative_residual_columns=int(np.count_nonzero(quality & NEGATIVE_RESIDUAL)),
                       negative_residual_pixels=len(negative_table.data),
                       profile_mismatch_columns=int(np.count_nonzero(quality & PROFILE_MISMATCH)),
                       cr_pixels=int(result['CR_MASK'].sum()),
                       limited_shape_folds=result['diagnostics']['limited_shape_folds'], settings=settings,
                       p06_changed_valid_columns=int(np.count_nonzero(np.isfinite(result['flux']) != ~invalid)),
                       p06_flux_max_abs_change=float(np.max(np.abs(delta))) if len(delta) else None,
                       p06_flux_rms_change=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                       covariance=propagated['diagnostics'])
    return diagnostics


# =============================================================================
# Response
# =============================================================================

FATAL_EXTRACTION_BITS = FATAL_QUALITY


DEFAULT_LINES = {
    "H2O": {"center": 7240, "width": 100},
    "O2-B": {"center": 6897.5, "width": 50},
    "H-alpha": {"center": 6562.8, "width": 40},
    "H-beta": {"center": 4861.3, "width": 38},
    "H-gamma": {"center": 4340.5, "width": 37},
    "H-delta": {"center": 4101.7, "width": 36},
    "H-epsilon": {"center": 3970.1, "width": 35},
}


class ResponseError(RuntimeError):
    """Indicate an unusable response instead of fabricating a unit response."""


@dataclass
class ArraySpectrum:
    """Carry a one-dimensional spectrum and its statistical provenance."""

    wave: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    valid: np.ndarray
    header: fits.Header = field(default_factory=fits.Header)
    fwhm: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    quality: np.ndarray | None = None


def _quality_header(header=None):
    """Attach shared extraction quality definitions to a product header."""
    result = fits.Header() if header is None else header.copy()
    result["FATALBIT"] = FATAL_EXTRACTION_BITS
    for bit, description in QUALITY_BIT_DESCRIPTIONS.items():
        result[f"BIT{bit}"] = description
    return result


def _quality_array(spectrum):
    """Return all extraction flags, including nonfatal scientific warnings."""
    if spectrum.quality is None:
        return np.zeros(len(spectrum.wave), np.uint16)
    return np.asarray(spectrum.quality, np.uint16)


def _project_quality(operator, quality):
    """OR flags from every nonzero contributor without interpreting their sign."""
    matrix = operator.tocsr()
    quality = np.asarray(quality, np.uint16)
    result = np.zeros(matrix.shape[0], np.uint16)
    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row:row + 2]
        columns = matrix.indices[lo:hi][matrix.data[lo:hi] != 0]
        if len(columns):
            result[row] = np.bitwise_or.reduce(quality[columns])
    return result


def _read_extraction_samples(hdul, n_columns, extension="EXTRACT_MC"):
    """Read correlated absolute-flux realizations and reject malformed provenance."""
    if extension not in hdul:
        return None, None
    if "MC_IDS" not in hdul:
        raise ResponseError(f"{extension} requires MC_IDS; independent redraw is forbidden.")
    samples = np.asarray(hdul[extension].data, float).copy()
    identifiers = np.asarray(hdul["MC_IDS"].data, np.int32).ravel().copy()
    if samples.ndim != 2 or samples.shape != (len(identifiers), n_columns):
        raise ResponseError(f"Invalid {extension} shape or sample identifiers.")
    if len(identifiers) < 8 or len(np.unique(identifiers)) != len(identifiers) or np.any(identifiers < 0):
        raise ResponseError("At least eight unique nonnegative extraction sample IDs are required.")
    return samples, identifiers


def _selected_samples(spectrum, count=None, identifiers=None, native=False):
    """Select saved draws by their IDs instead of regenerating spectral noise."""
    key = "native_samples" if native else "extraction_samples"
    samples = spectrum.metadata.get(key)
    if samples is None:
        if str(spectrum.header.get("EXTRMETH", "")).upper() == "AC":
            raise ResponseError(f"A+C {key} are missing; regenerate extraction/dispersion products.")
        return None, None
    stored = spectrum.metadata["sample_ids"]
    if identifiers is None:
        count = len(stored) if count is None else int(count)
        if count > len(stored):
            raise ResponseError(f"Requested {count} samples but extraction stored only {len(stored)}.")
        selected = np.arange(count)
    else:
        lookup = {int(value): index for index, value in enumerate(stored)}
        if any(int(value) not in lookup for value in identifiers):
            raise ResponseError("Requested uncertainty sample ID is absent from extraction.")
        selected = np.array([lookup[int(value)] for value in identifiers])
    return samples[selected], stored[selected]


def _warning(message, notes=None):
    """Expose a scientific warning to the terminal and the caller's record."""
    if notes is not None and message not in notes:
        notes.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _seed(value):
    """Return a repeatable seed independent of Python's randomized hash."""
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "little")


def _frame_identity(path, header):
    """Identify an exposure independently of its temporary run directory."""
    return "|".join([
        Path(path).name, str(header.get("OBJECT", "")),
        str(header.get("DATE-OBS", header.get("JD", ""))),
        str(header.get("EXPTIME", header.get("EXPOSURE", ""))),
    ])


def bin_edges(wave):
    """Infer increasing wavelength-bin edges from finite increasing centres."""
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1 or len(wave) < 2 or not np.all(np.isfinite(wave)):
        raise ValueError("At least two finite wavelength centres are required.")
    if np.any(np.diff(wave) <= 0):
        raise ValueError("Wavelength centres must be strictly increasing.")
    edges = np.empty(len(wave) + 1, dtype=float)
    edges[1:-1] = (wave[:-1] + wave[1:]) / 2
    edges[0] = wave[0] - (wave[1] - wave[0]) / 2
    edges[-1] = wave[-1] + (wave[-1] - wave[-2]) / 2
    return edges


def density_operator(old_wave, new_wave, source_valid=None):
    """Build a flux-density overlap operator with strict full-bin support.

    The same matrix applies to flux, noise realizations, and covariance.
    Invalid input bins are not bridged or silently renormalized.
    """
    old_edges, new_edges = bin_edges(old_wave), bin_edges(new_wave)
    source_valid = np.ones(len(old_wave), bool) if source_valid is None else np.asarray(source_valid, bool)
    rows, cols, weights = [], [], []
    valid = np.zeros(len(new_wave), dtype=bool)
    for row, (left, right) in enumerate(zip(new_edges[:-1], new_edges[1:])):
        start = max(0, np.searchsorted(old_edges, left, side="right") - 1)
        stop = min(len(old_wave), np.searchsorted(old_edges, right, side="left"))
        total = 0.0
        good = left >= old_edges[0] - 1e-9 and right <= old_edges[-1] + 1e-9
        for col in range(start, stop):
            overlap = min(right, old_edges[col + 1]) - max(left, old_edges[col])
            if overlap <= 0:
                continue
            rows.append(row)
            cols.append(col)
            weights.append(overlap / (right - left))
            total += overlap
            good = good and source_valid[col]
        valid[row] = good and np.isclose(total, right - left, rtol=1e-9, atol=1e-9)
    matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(len(new_wave), len(old_wave)))
    return matrix, valid


def resample_density(wave, flux, variance, new_wave, valid=None):
    """Resample density and propagate variance with squared overlap weights."""
    valid = np.ones(len(wave), bool) if valid is None else np.asarray(valid, bool)
    valid = valid & np.isfinite(flux) & np.isfinite(variance) & (variance >= 0)
    matrix, supported = density_operator(wave, new_wave, valid)
    new_flux = matrix @ np.nan_to_num(flux)
    new_var = matrix.power(2) @ np.nan_to_num(variance)
    new_flux[~supported], new_var[~supported] = np.nan, np.nan
    return new_flux, new_var, supported, matrix


def _linear_operator(wave, grid, valid=None):
    """Interpolate a smooth response without extrapolation or crossing holes."""
    wave, grid = np.asarray(wave, float), np.asarray(grid, float)
    valid = np.ones(len(wave), bool) if valid is None else np.asarray(valid, bool)
    rows, cols, vals = [], [], []
    good = np.zeros(len(grid), bool)
    for row, x in enumerate(grid):
        j = int(np.searchsorted(wave, x))
        if j < len(wave) and np.isclose(x, wave[j], rtol=0, atol=1e-8):
            rows.append(row)
            cols.append(j)
            vals.append(1.0)
            good[row] = valid[j]
        elif 0 < j < len(wave):
            t = (x - wave[j - 1]) / (wave[j] - wave[j - 1])
            rows.extend([row, row])
            cols.extend([j - 1, j])
            vals.extend([1 - t, t])
            good[row] = valid[j - 1] and valid[j]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(grid), len(wave))), good


def smoothing_operator(wave, source_fwhm, target_fwhm, valid=None, truncate=4.0):
    """Construct a local Gaussian LSF operator with explicit edge support.

    Gaussian LSFs are an approximation.  The kernel is normalized for density;
    its finite support and internal invalid pixels remain marked unavailable.
    """
    wave = np.asarray(wave, float)
    source_fwhm = np.broadcast_to(np.asarray(source_fwhm, float), wave.shape)
    target_fwhm = np.broadcast_to(np.asarray(target_fwhm, float), wave.shape)
    if np.any(target_fwhm < source_fwhm - 1e-8):
        raise ValueError("Resolution matching cannot sharpen a spectrum.")
    valid = np.ones(len(wave), bool) if valid is None else np.asarray(valid, bool)
    sigma = np.sqrt(np.maximum(target_fwhm**2 - source_fwhm**2, 0)) / 2.354820045
    edges = bin_edges(wave)
    widths = np.diff(edges)
    rows, cols, vals = [], [], []
    support = np.zeros(len(wave), bool)
    for i, width in enumerate(sigma):
        if not np.isfinite(width):
            continue
        if width < widths[i] * 1e-5:
            rows.append(i)
            cols.append(i)
            vals.append(1.0)
            support[i] = valid[i]
            continue
        radius = truncate * width
        lo = max(0, int(np.searchsorted(wave, wave[i] - radius)))
        hi = min(len(wave), int(np.searchsorted(wave, wave[i] + radius, side="right")))
        jj = np.arange(lo, hi)
        kernel = np.exp(-0.5 * ((wave[jj] - wave[i]) / width)**2) * widths[jj]
        kernel /= kernel.sum()
        rows.extend([i] * len(jj))
        cols.extend(jj.tolist())
        vals.extend(kernel.tolist())
        support[i] = (
            wave[i] - radius >= edges[0] and wave[i] + radius <= edges[-1]
            and np.all(valid[jj])
        )
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(wave), len(wave))), support


def _wavelength_header(header, wave, unit):
    """Replace image-axis metadata and discard checksums belonging to old data."""
    header = header.copy()
    for key in ('CHECKSUM', 'DATASUM'):
        header.remove(key, remove_all=True, ignore_missing=True)
    for key in list(header):
        if re.match(r"^(CRVAL|CRPIX|CDELT|CTYPE|CUNIT|CROTA)[2-9]$", key):
            del header[key]
        elif re.match(r"^(CD|PC)\d+_\d+$", key):
            del header[key]
    header["CRVAL1"] = float(wave[0])
    header["CRPIX1"] = 1.0
    header["CDELT1"] = float(wave[1] - wave[0])
    header["CTYPE1"] = "AWAV"
    header["CUNIT1"] = "Angstrom"
    header["WAVEMED"] = "air"
    header["BUNIT"] = unit
    return header


def _grid_from_header(header, length):
    """Reconstruct a wavelength axis, failing on missing calibration metadata."""
    for name in ("CRVAL1", "CDELT1"):
        if name not in header:
            raise ValueError(f"Missing wavelength metadata {name}.")
    return (
        float(header["CRVAL1"])
        + (np.arange(length) + 1 - float(header.get("CRPIX1", 1))) * float(header["CDELT1"])
    )


def read_array_spectrum(path, require_variance=True):
    """Read full quality and correlated draws without inventing missing errors."""
    metadata = {}
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        flux = np.asarray(hdul[0].data, float).ravel().copy()
        wave = _grid_from_header(header, len(flux))
        if "VARIANCE" not in hdul:
            if require_variance:
                raise ValueError(f"{Path(path).name}: VARIANCE extension is missing.")
            variance = np.full_like(flux, np.nan)
        else:
            variance = np.asarray(hdul["VARIANCE"].data, float).ravel().copy()
        valid = np.isfinite(flux) & np.isfinite(variance) & (variance >= 0)
        if "MASK" in hdul:
            valid &= np.asarray(hdul["MASK"].data).ravel() == 0
        quality = (np.asarray(hdul["QUALITY"].data, np.uint16).ravel().copy()
                   if "QUALITY" in hdul else np.zeros(len(flux), np.uint16))
        valid &= (quality & FATAL_EXTRACTION_BITS) == 0
        samples, identifiers = _read_extraction_samples(hdul, len(flux))
        if samples is not None:
            if np.any(~np.isfinite(samples[:, valid])):
                raise ResponseError("Correlated extraction draws are nonfinite in valid bins.")
            metadata.update(extraction_samples=samples, sample_ids=identifiers)
        if "NATIVE_MC" in hdul:
            if "NATIVE" not in hdul:
                raise ResponseError("NATIVE_MC requires native spectral coordinates.")
            native_samples, native_ids = _read_extraction_samples(hdul, len(hdul["NATIVE"].data), "NATIVE_MC")
            native_data = hdul["NATIVE"].data
            native_valid = ~np.asarray(native_data["MASK"], bool)
            if "QUALITY" in native_data.names:
                native_valid &= (np.asarray(native_data["QUALITY"], np.uint16) & FATAL_EXTRACTION_BITS) == 0
            if np.any(~np.isfinite(native_samples[:, native_valid])):
                raise ResponseError("Native extraction draws are nonfinite in valid native bins.")
            metadata["native_samples"] = native_samples
            metadata["sample_ids"] = native_ids
        if str(header.get("EXTRMETH", "")).upper() == "AC" and "RESPSTAT" not in header and samples is None:
            raise ResponseError("A+C extraction samples are missing; independent redraw is forbidden.")
    bin_edges(wave)
    return ArraySpectrum(wave, flux, variance, valid, header, metadata=metadata, quality=quality)


def read_spectrum(path):
    """Provide the legacy four-array reader while requiring measured variance."""
    spec = read_array_spectrum(path)
    return spec.wave, spec.flux, spec.variance, spec.header


def _resolve_coordinate(name, header=None, coordinates=None, offline=False):
    """Resolve configured coordinates, then explicit FITS coordinates or a name."""
    coordinates = coordinates or {}
    key = str(name).replace(" ", "").lower()
    for candidate, value in coordinates.items():
        if str(candidate).replace(" ", "").lower() == key:
            return SkyCoord(float(value["ra_deg"]) * u.deg, float(value["dec_deg"]) * u.deg)
    header = header or {}
    if "OBJCTRA" in header and "OBJCTDEC" in header:
        return SkyCoord(str(header["OBJCTRA"]), str(header["OBJCTDEC"]), unit=(u.hourangle, u.deg))
    if "RA" in header and "DEC" in header:
        return SkyCoord(float(header["RA"]) * u.deg, float(header["DEC"]) * u.deg)
    if offline:
        raise ValueError(f"No local coordinates for {name}; online resolution is disabled.")
    return SkyCoord.from_name(str(name))


def get_airmass(header, target_coord, location):
    """Compute mid-exposure sec(z), preserving the existing airmass definition."""
    try:
        date = str(header.get("DATE-OBS", header.get("DATE", "")))
        if "T" in date:
            obstime = Time(date, format="isot", scale="utc")
        elif "JD" in header:
            obstime = Time(float(header["JD"]), format="jd")
        else:
            raise ValueError("No exposure time coordinate.")
        exposure = float(header.get("EXPTIME", header.get("EXPOSURE", 0)))
        obstime += TimeDelta(exposure / 2, format="sec")
        result = float(target_coord.transform_to(AltAz(obstime=obstime, location=location)).secz)
        if not np.isfinite(result) or result < 1:
            raise ValueError(f"Invalid airmass {result}.")
        return result
    except Exception as error:
        _warning(f"Airmass calculation failed for {header.get('OBJECT', '?')}: {error}")
        return None


def dispcor(FITS_FILENAME, SITE, WL_ORDER, coordinates=None, offline=False):
    """Write a wavelength/velocity-corrected spectrum on a flux-conserving grid.

    Read the stored wavelength polynomial from the extracted FITS, retaining
    native flux/variance and correlated samples in the output for downstream
    propagation. Resolve coordinates from the supplied mapping or FITS header
    before any permitted online lookup. ``offline=True`` disables online name
    resolution and scopes IERS auto-download suppression to this call.
    """
    context = iers.conf.set_temp("auto_download", False) if offline else nullcontext()
    with context:
        return _dispcor_impl(FITS_FILENAME, SITE, WL_ORDER, coordinates, offline)


def _dispcor_impl(FITS_FILENAME, SITE, WL_ORDER, coordinates=None, offline=False):
    """Apply the existing wavelength/velocity formula and conserve flux bins.

    Native extracted density and variance are retained so subsequent Monte Carlo
    draws pass through the same resampling operator rather than treating output
    bins as independent.  Failed velocity correction is explicitly recorded.
    """
    notes = []
    with fits.open(FITS_FILENAME) as hdul:
        header = hdul[0].header.copy()
        hflux = hdul["FLUX_ADU"].header
        flux = np.asarray(hdul["FLUX_ADU"].data, float).copy()
        variance = np.asarray(hdul["VAR_ADU2"].data, float).copy()
        coefficients = [float(hflux[f"WLCAL{i}"]) for i in range(WL_ORDER + 1)]
        quality = np.asarray(hdul["QUALITY"].data, np.uint16).copy() if "QUALITY" in hdul else np.zeros(len(flux), np.uint16)
        invalid = np.asarray(hdul["MASK"].data, bool).copy() if "MASK" in hdul else (quality & FATAL_EXTRACTION_BITS) != 0
        invalid |= (quality & FATAL_EXTRACTION_BITS) != 0
        samples, identifiers = _read_extraction_samples(hdul, len(flux))
        if str(header.get("EXTRMETH", "")).upper() == "AC" and samples is None:
            raise ResponseError("A+C EXTRACT_MC is missing; dispersion cannot discard its covariance.")
    for index, coefficient in enumerate(coefficients):
        header[f"WLCAL{index}"] = (coefficient, "Native-pixel air wavelength polynomial")
    wave = np.polynomial.polynomial.polyval(np.arange(len(flux)), coefficients)
    bin_edges(wave)
    applied = False
    velocity = 0.0
    try:
        obstime = Time(float(header["JD"]), format="jd") if "JD" in header else Time(header["DATE-OBS"])
        coord = _resolve_coordinate(header.get("OBJECT"), header, coordinates, offline)
        velocity = float(coord.radial_velocity_correction(kind="heliocentric", obstime=obstime, location=SITE).to_value(u.m / u.s))
        applied = True
    except Exception as error:
        _warning(f"{Path(FITS_FILENAME).name}: heliocentric correction FAILED; uncorrected wavelengths retained: {error}", notes)
    wave *= 1 + velocity / constants.c.value
    widths = np.diff(bin_edges(wave))
    density, density_var = flux / widths, variance / widths**2
    valid = ~invalid & np.isfinite(density) & np.isfinite(density_var) & (density_var >= 0)
    step = float(np.max(np.diff(wave)) * 1.5)
    output_wave = np.arange(wave.min() + step / 2, wave.max() - step / 2, step)
    if len(output_wave) < 2:
        raise ValueError("Dispersion correction produced fewer than two bins.")
    values, var, support, matrix = resample_density(wave, density, density_var, output_wave, valid)
    out_quality = _project_quality(matrix, quality)
    out_quality[~support] |= np.uint16(1)
    density_samples, projected_samples = None, None
    if samples is not None:
        if np.any(~np.isfinite(samples[:, valid])):
            raise ResponseError("A+C samples contain nonfinite values in valid native columns.")
        density_samples = samples / widths[None, :]
        projected_samples = (matrix @ np.nan_to_num(density_samples).T).T
        projected_samples[:, ~support] = np.nan
        # The squared diagonal operator alone omits C's inter-column covariance.
        var[support] = np.var(projected_samples[:, support], axis=0, ddof=1)
    density_unit = (u.adu / u.AA).to_string("fits")
    variance_unit = ((u.adu / u.AA)**2).to_string("fits")
    header = _wavelength_header(header, output_wave, density_unit)
    header["HELAPPLY"] = applied
    header["HELSTAT"] = "APPLIED" if applied else "FAILED"
    header["SPECSYS"] = "HELIOCEN" if applied else "TOPOCENT"
    if applied:
        header["VHELIO"] = (velocity, "Applied heliocentric correction in m/s")
    elif "VHELIO" in header:
        del header["VHELIO"]
    header["HISTORY"] = "Flux-conserving dispersion correction; squared weights propagate variance."
    if samples is not None:
        header["EXTCOV"] = True
        header["VARMODEL"] = "CORRELATED_MC"
        header["HISTORY"] = "A+C correlated native draws propagated once; output variance from their scatter."
    header["HISTORY"] = "Heliocentric correction applied." if applied else "Heliocentric correction FAILED; wavelength frame remains topocentric."
    native = fits.BinTableHDU.from_columns([
        fits.Column(name="WAVELENGTH", format="D", unit="Angstrom", array=wave),
        fits.Column(name="FLUX", format="D", unit=density_unit, array=density),
        fits.Column(name="VARIANCE", format="D", unit=variance_unit, array=density_var),
        fits.Column(name="MASK", format="L", array=~valid),
        fits.Column(name="QUALITY", format="J", array=quality.astype(np.int32)),
    ], name="NATIVE")
    output = re.sub(r"\.ec\.fits$", ".ec_dc.fits", str(FITS_FILENAME))
    if output == str(FITS_FILENAME):
        raise ValueError("Expected an .ec.fits filename for dispersion correction.")
    hdus = [
        fits.PrimaryHDU(values.astype(np.float32), header),
        fits.ImageHDU(var.astype(np.float32), fits.Header({"BUNIT": variance_unit}), name="VARIANCE"),
        fits.ImageHDU((~support).astype(np.uint8), name="MASK"),
        fits.ImageHDU(out_quality, _quality_header(), name="QUALITY"),
        native,
    ]
    if samples is not None:
        hdus.extend([
            fits.ImageHDU(density_samples, fits.Header({"BUNIT": density_unit}), name="NATIVE_MC"),
            fits.ImageHDU(projected_samples, fits.Header({"BUNIT": density_unit}), name="EXTRACT_MC"),
            fits.ImageHDU(identifiers, name="MC_IDS"),
        ])
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return {"status": "warning" if notes else "success", "path": output, "warnings": notes}


def _native_source(path, spectrum):
    """Return a pre-resampling source and its operator when provenance exists."""
    with fits.open(path) as hdul:
        if "NATIVE" in hdul:
            data = hdul["NATIVE"].data
            wave = np.asarray(data["WAVELENGTH"], float)
            flux = np.asarray(data["FLUX"], float)
            variance = np.asarray(data["VARIANCE"], float)
            valid = ~np.asarray(data["MASK"], bool)
            if "QUALITY" in data.names:
                valid &= (np.asarray(data["QUALITY"], np.uint16) & FATAL_EXTRACTION_BITS) == 0
            matrix, support = density_operator(wave, spectrum.wave, valid)
            return wave.copy(), flux.copy(), variance.copy(), valid.copy(), matrix, support, True
    return spectrum.wave, spectrum.flux, spectrum.variance, spectrum.valid, sparse.eye(len(spectrum.wave), format="csr"), spectrum.valid, False


def _template_settings(info, config):
    """Merge explicit library/file metadata without inferring wavelength media."""
    templates = (config or {}).get("Templates", {})
    library = info["template"].lower()
    settings = dict(templates.get("libraries", {}).get(library, {}))
    settings.update(templates.get(library, {}))
    settings.update(templates.get("files", {}).get(info["template_file"], {}))
    for key in ("wavelength_medium", "fwhm_angstrom", "resolving_power", "provenance"):
        if key in info:
            settings[key] = info[key]
    return settings


def convert_vacuum_density(wave, flux, variance=None, fwhm=None):
    """Convert optical vacuum bins to air while preserving integrated flux."""
    wave = np.asarray(wave, float)
    old_edges = bin_edges(wave)
    if old_edges[0] <= 2000:
        raise ValueError("Morton2000 conversion is restricted here to wavelengths above 2000 Angstrom.")
    new_wave = vac_to_air(wave * u.AA, method="Morton2000").to_value(u.AA)
    new_edges = vac_to_air(old_edges * u.AA, method="Morton2000").to_value(u.AA)
    jacobian = np.diff(old_edges) / np.diff(new_edges)
    converted_flux = np.asarray(flux, float) * jacobian
    converted_var = None if variance is None else np.asarray(variance, float) * jacobian**2
    converted_fwhm = None if fwhm is None else np.asarray(fwhm, float) / jacobian
    return new_wave, converted_flux, converted_var, converted_fwhm


def load_template(filepath, reflist, refdir, config=None):
    """Load a configured template with explicit statistical and LSF metadata."""
    header = fits.getheader(filepath)
    object_name = str(header.get("OBJECT", "")).replace(" ", "").lower()
    matches = [value for key, value in reflist.items() if str(key).replace(" ", "").lower() == object_name]
    if len(matches) != 1:
        raise ResponseError(f"{Path(filepath).name}: missing or ambiguous template for OBJECT={header.get('OBJECT')!r}.")
    info = matches[0]
    library = str(info["template"]).lower()
    path = Path(refdir) / library / (str(info["template_file"]) + ".fits")
    settings = _template_settings(info, config)
    notes = []
    fwhm = None
    systematic = None
    with fits.open(path) as hdul:
        template_header = hdul[0].header.copy()
        if library == "miles":
            flux = np.asarray(hdul[0].data, float).ravel().copy()
            wave = _grid_from_header(template_header, len(flux))
            wave, flux = wave[40:-20], flux[40:-20]
            variance = np.zeros_like(flux)
            valid = np.isfinite(flux) & (flux > 0)
            stat_known = False
        elif library in ("pickles", "calspec"):
            table = hdul[1].data
            wave = np.asarray(table["WAVELENGTH"], float).copy()
            flux = np.asarray(table["FLUX"], float).copy()
            stat_known = "STATERROR" in table.names
            variance = np.asarray(table["STATERROR"], float)**2 if stat_known else np.zeros_like(flux)
            valid = np.isfinite(flux) & (flux > 0) & np.isfinite(variance) & (variance >= 0)
            if "DATAQUAL" in table.names:
                valid &= np.asarray(table["DATAQUAL"]) == 1
            if "FWHM" in table.names:
                fwhm = np.asarray(table["FWHM"], float).copy()
            if "SYSERROR" in table.names:
                systematic = np.asarray(table["SYSERROR"], float).copy()
        else:
            raise ResponseError(f"Unsupported template library {library!r}.")
    medium = str(settings.get("wavelength_medium", template_header.get("WAVEMED", "unknown"))).lower()
    history = str(template_header.get("HISTORY", "")).lower()
    if medium == "unknown" and "all wavelengths are in vacuum" in history:
        medium = "vacuum"
    if medium not in ("air", "vacuum", "unknown"):
        raise ResponseError(f"Invalid template wavelength_medium={medium!r}.")
    # The pipeline is optical; exclude UV samples outside the conversion domain.
    optical = np.isfinite(wave) & (wave > 2001)
    wave, flux, variance, valid = wave[optical], flux[optical], variance[optical], valid[optical]
    if fwhm is not None:
        fwhm = fwhm[optical]
    if systematic is not None:
        systematic = systematic[optical]
    if "fwhm_angstrom" in settings:
        fwhm = np.full(len(wave), float(settings["fwhm_angstrom"]))
    elif "resolving_power" in settings:
        fwhm = wave / float(settings["resolving_power"])
    if fwhm is not None and np.any(~np.isfinite(fwhm) | (fwhm <= 0)):
        raise ResponseError(f"{path.name}: invalid template LSF widths.")
    if medium == "vacuum":
        original_flux = flux.copy()
        wave, flux, variance, fwhm = convert_vacuum_density(wave, flux, variance, fwhm)
        if systematic is not None:
            systematic *= flux / original_flux
        conversion = "vac_to_air:Morton2000"
    elif medium == "air":
        conversion = "none:air"
    else:
        conversion = "none:unknown"
        _warning(f"{path.name}: template air/vacuum convention is UNKNOWN; axis retained, no guessed conversion. Set Templates metadata after checking provenance.", notes)
    if not stat_known:
        _warning(f"{path.name}: template statistical errors unavailable; response errors are conditional on a fixed template.", notes)
    bin_edges(wave)
    metadata = {
        "path": str(path.resolve()), "library": library, "input_medium": medium,
        "conversion": conversion, "stat_known": stat_known, "warnings": notes,
        "provenance": str(settings.get("provenance", "FITS header" if library == "calspec" else "not supplied")),
        "systematic_fraction": float(np.nanmedian(systematic / np.maximum(flux, np.finfo(float).tiny))) if systematic is not None else None,
        "systematics_included": False,
    }
    return ArraySpectrum(wave, flux, variance, valid, template_header, fwhm, metadata)


def get_ref_template(filepath, reflist, refdir, spec_cont, config=None):
    """Expose a legacy Spectrum template API with full overlap support."""
    template = load_template(filepath, reflist, refdir, config)
    minimum, maximum = np.min(spec_cont.spectral_axis.value), np.max(spec_cont.spectral_axis.value)
    keep = (template.wave >= minimum - 100) & (template.wave <= maximum + 100)
    if np.count_nonzero(keep) < 4:
        raise ResponseError("Observed and template spectra have no usable overlap.")
    result = Spectrum(
        spectral_axis=template.wave[keep] * u.AA,
        flux=template.flux[keep] * u.dimensionless_unscaled,
        uncertainty=VarianceUncertainty(template.variance[keep]),
        mask=~template.valid[keep],
        meta={**template.metadata, "fwhm": None if template.fwhm is None else template.fwhm[keep]},
    )
    return len(result.spectral_axis), result


def _gaussian(x, amplitude, centre, sigma, offset):
    """Evaluate a Gaussian arc-line model for LSF measurement only."""
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma)**2) + offset


def estimate_arc_fwhm(arc_path, wave_grid, config=None):
    """Measure unblended arc widths without changing line IDs or wavelength fit."""
    config = config or {}
    with fits.open(arc_path) as hdul:
        flux = np.asarray(hdul["FLUX_ADU"].data, float)
        header = hdul["FLUX_ADU"].header
        names = sorted([key for key in header if re.fullmatch(r"WLCAL\d+", key)], key=lambda key: int(key[5:]))
        if len(names) < 2:
            raise ResponseError("Arc LSF requires the existing WLCAL coefficients.")
        coefficients = [float(header[key]) for key in names]
    indices = np.arange(len(flux), dtype=float)
    finite = np.isfinite(flux)
    clean = np.where(finite, flux, 0)
    prominence = float(config.get("prominence", max(np.nanmax(clean) * 0.01, 1)))
    peaks, _ = find_peaks(clean, prominence=prominence, distance=5)
    half_window = int(config.get("window", 20))
    maximum_sigma = float(config.get("wmax", 10))
    measurements = []
    for peak in peaks:
        lo, hi = max(0, peak - half_window), min(len(flux), peak + half_window + 1)
        x, y = indices[lo:hi], flux[lo:hi]
        if len(x) < 7 or not np.all(np.isfinite(y)):
            continue
        background = float(np.percentile(y, 20))
        amplitude = float(np.max(y) - background)
        try:
            fitted, covariance = curve_fit(
                _gaussian, x, y, p0=[amplitude, float(peak), min(3, maximum_sigma / 2), background],
                bounds=([0, peak - 2, 0.3, -np.inf], [np.inf, peak + 2, maximum_sigma, np.inf]),
                maxfev=3000,
            )
        except (RuntimeError, ValueError):
            continue
        _, centre, sigma, _ = fitted
        distance = np.abs(peaks - centre)
        if np.count_nonzero(distance < 3 * sigma) > 1:
            continue
        residual = np.sqrt(np.mean((y - _gaussian(x, *fitted))**2))
        if residual > 0.15 * amplitude or sigma >= maximum_sigma * 0.98:
            continue
        wavelength = np.polynomial.polynomial.polyval(centre, coefficients)
        derivative = np.polynomial.polynomial.polyval(centre, np.polynomial.polynomial.polyder(coefficients))
        width = 2.354820045 * sigma * abs(derivative)
        if width > 0:
            measurements.append((wavelength, width))
    if len(measurements) < 4:
        raise ResponseError(f"Only {len(measurements)} usable arc LSF lines.")
    measurements = np.asarray(sorted(measurements))
    # Robust local medians suppress individual residual blends without a new wavelength model.
    bins = np.array_split(measurements, min(12, max(2, len(measurements) // 4)))
    centres = np.array([np.median(group[:, 0]) for group in bins])
    widths = np.array([np.median(group[:, 1]) for group in bins])
    result = np.interp(wave_grid, centres, widths)
    return result, measurements


def _line_mask(wave, lines):
    """Exclude the established Balmer/telluric intervals from response fitting."""
    good = np.ones(len(wave), dtype=bool)
    for name, value in (lines or {}).items():
        distance = np.abs(wave - value["center"])
        good &= distance > value["width"] if name == "H2O" else distance >= value["width"]
    return good


def _spline_fit(wave, ratio, variance, support, lines, order, clip_sigma, n_iter, n_shifts):
    """Fit the existing shifted cubic splines, failing instead of returning ones."""
    support = np.asarray(support, bool)
    good = support & np.isfinite(ratio) & np.isfinite(variance) & (variance >= 0)
    good &= _line_mask(wave, lines)
    order = max(1, int(order))
    if good.sum() < order + 4:
        raise ResponseError(f"Too few usable response points ({good.sum()}) for spline order {order}.")
    lower, upper = float(wave[good][0]), float(wave[good][-1])
    evaluate = support & (wave >= lower) & (wave <= upper)
    if upper <= lower:
        raise ResponseError("Empty response fit domain.")
    positive_var = variance[good & (variance > 0)]
    floor = max(float(np.median(positive_var)) * 1e-12 if positive_var.size else 0, np.finfo(float).tiny)
    errors = np.sqrt(np.maximum(variance, floor))
    models = []
    spacing = (upper - lower) / order
    failures = []
    water = (lines or {}).get("H2O")
    for shift in range(int(n_shifts)):
        knots = np.linspace(lower - spacing, upper + spacing, order + 3) + spacing * shift / n_shifts
        knots = knots[(knots > lower + 1) & (knots < upper - 1)]
        if water is not None:
            # Preserve every original knot outside the excluded water interval.
            gap_low = water["center"] - water["width"]
            gap_high = water["center"] + water["width"]
            knots = knots[(knots < gap_low) | (knots > gap_high)]
        current = good.copy()
        model = None
        for iteration in range(int(n_iter)):
            if current.sum() < len(knots) + 4:
                break
            try:
                spline = LSQUnivariateSpline(
                    wave[current], ratio[current], knots, w=1 / errors[current],
                    bbox=[lower, upper], k=3, check_finite=True,
                )
            except (ValueError, RuntimeError) as error:
                failures.append(str(error).splitlines()[0])
                break
            candidate = np.full(len(wave), np.nan)
            candidate[evaluate] = spline(wave[evaluate])
            residual = (ratio - candidate) / errors
            rms = np.std(residual[current])
            if not np.isfinite(rms):
                break
            model = candidate
            if rms < 1e-10:
                break
            updated = good & (np.abs(residual) < clip_sigma * rms)
            if np.array_equal(updated, current):
                break
            current = updated
        if model is not None:
            models.append(model)
    if not models:
        raise ResponseError("All shifted spline fits failed: " + "; ".join(failures[:3]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(models, axis=0)
    valid = evaluate & np.isfinite(result) & (result > 0)
    if valid.sum() < order + 4:
        raise ResponseError("Spline returned no sufficient positive supported response.")
    normalizer = np.max(result[valid])
    result /= normalizer
    result[~valid] = np.nan
    return result, valid, normalizer, len(models)


def _prepare_response(observed, template, observed_fwhm=None, match_lsf=True):
    """Compose resolution and overlap operators on the observed standard grid."""
    observed_valid = observed.valid.copy()
    template_valid = template.valid.copy()
    observed_operator = sparse.eye(len(observed.wave), format="csr")
    template_operator = sparse.eye(len(template.wave), format="csr")
    matched = False
    if match_lsf and observed_fwhm is not None and template.fwhm is not None:
        observed_fwhm = np.broadcast_to(np.asarray(observed_fwhm, float), observed.wave.shape)
        tpl_on_obs = np.interp(observed.wave, template.wave, template.fwhm)
        obs_on_tpl = np.interp(template.wave, observed.wave, observed_fwhm)
        common_observed = np.maximum(observed_fwhm, tpl_on_obs)
        common_template = np.maximum(template.fwhm, obs_on_tpl)
        observed_operator, observed_valid = smoothing_operator(
            observed.wave, observed_fwhm, common_observed, observed.valid,
        )
        template_operator, template_valid = smoothing_operator(
            template.wave, template.fwhm, common_template, template.valid,
        )
        matched = True
    rebin, rebin_support = density_operator(template.wave, observed.wave, template_valid)
    template_operator = rebin @ template_operator
    support = observed_valid & rebin_support
    source = observed_operator @ np.nan_to_num(observed.flux)
    source_var = observed_operator.power(2) @ np.nan_to_num(observed.variance)
    reference = template_operator @ np.nan_to_num(template.flux)
    reference_var = template_operator.power(2) @ np.nan_to_num(template.variance)
    support &= np.isfinite(source) & np.isfinite(reference) & (reference > 0)
    ratio = np.divide(source, reference, out=np.full_like(source, np.nan), where=reference > 0)
    ratio_var = np.divide(source_var, reference**2, out=np.full_like(source, np.nan), where=reference > 0)
    ratio_var += np.divide(source**2 * reference_var, reference**4, out=np.zeros_like(source), where=reference > 0)
    return observed_operator, template_operator, ratio, ratio_var, support, matched


def _response_ensemble(observed, template, parameters, native=None, observed_fwhm=None, identity="standard"):
    """Repeat the full one-dimensional response fit with shared template draws."""
    count = int(parameters.get("mc_samples", 256))
    if count < 8:
        raise ValueError("At least eight response Monte Carlo samples are required.")
    obs_operator, tpl_operator, ratio, ratio_var, support, matched = _prepare_response(
        observed, template, observed_fwhm, bool(parameters.get("match_resolution", True)),
    )
    if native is None:
        source_flux, source_var = observed.flux, observed.variance
        base_operator = sparse.eye(len(observed.wave), format="csr")
    else:
        _, source_flux, source_var, _, base_operator, source_support, _ = native
        support &= source_support
    total_obs_operator = obs_operator @ base_operator
    # Compose operators before squaring to retain rebin/smoothing covariance.
    source_var_on_grid = total_obs_operator.power(2) @ np.nan_to_num(source_var)
    saved_draws, saved_ids = _selected_samples(observed, count=count, native=native is not None and native[-1])
    if saved_draws is not None:
        transformed_draws = (total_obs_operator @ np.nan_to_num(saved_draws).T).T
        source_var_on_grid = np.var(transformed_draws, axis=0, ddof=1)
    else:
        transformed_draws = None
        saved_ids = np.arange(count, dtype=np.int32)
    reference = tpl_operator @ template.flux
    source = total_obs_operator @ np.nan_to_num(source_flux)
    reference_var = tpl_operator.power(2) @ template.variance
    ratio = np.divide(source, reference, out=np.full_like(source, np.nan), where=reference > 0)
    ratio_var = np.divide(source_var_on_grid, reference**2, out=np.full_like(source, np.nan), where=reference > 0)
    ratio_var += np.divide(source**2 * reference_var, reference**4, out=np.zeros_like(source), where=reference > 0)
    fit_args = (
        parameters.get("lines", DEFAULT_LINES), int(parameters.get("order", 50)),
        float(parameters.get("clip_sigma", 2.5)), int(parameters.get("fit_iterations", 5)),
        int(parameters.get("n_shifts", 3)),
    )
    centre, valid, norm, shifts = _spline_fit(observed.wave, ratio, ratio_var, support, *fit_args)
    obs_rng = np.random.default_rng(_seed(f"{identity}:observation:{parameters.get('seed', 42)}"))
    draws, identifiers, sample_supports = [], [], []
    for index, identifier in enumerate(saved_ids):
        # Key template noise by draw ID so rejected draws never shift correlations.
        tpl_rng = np.random.default_rng(_seed(f"{template.metadata.get('path', 'template')}:template:{parameters.get('seed', 42)}:{int(identifier)}"))
        tpl_draw = template.flux + tpl_rng.normal(size=len(template.flux)) * np.sqrt(np.maximum(template.variance, 0))
        if transformed_draws is None:
            obs_draw = np.nan_to_num(source_flux) + obs_rng.normal(size=len(source_flux)) * np.sqrt(np.maximum(np.nan_to_num(source_var), 0))
            numerator = total_obs_operator @ obs_draw
        else:
            numerator = transformed_draws[index]
        denominator = tpl_operator @ tpl_draw
        sample = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        try:
            curve, sample_valid, _, _ = _spline_fit(observed.wave, sample, ratio_var, support, *fit_args)
        except ResponseError:
            continue
        draws.append(curve)
        identifiers.append(identifier)
        sample_supports.append(sample_valid)
    if len(draws) < max(8, math.ceil(count / 2)):
        raise ResponseError(f"Only {len(draws)}/{count} response uncertainty realizations succeeded.")
    draws = np.asarray(draws)
    nominal_valid = valid.copy()
    valid &= np.all(sample_supports, axis=0)
    trimmed_bins = int(np.count_nonzero(nominal_valid & ~valid))
    if valid.sum() < int(parameters.get("order", 50)) + 4:
        raise ResponseError("Successful response realizations have insufficient common wavelength support.")
    # Select common wavelength support instead of conditioning on whole draws.
    common_normalizer = float(np.max(centre[valid]))
    norm *= common_normalizer
    centre /= common_normalizer
    draws /= np.max(draws[:, valid], axis=1)[:, None]
    centre[~valid] = np.nan
    draws[:, ~valid] = np.nan
    variance = np.full(len(observed.wave), np.nan)
    variance[valid] = np.var(draws[:, valid], axis=0, ddof=1)
    quality = _project_quality(obs_operator, _quality_array(observed))
    quality[~valid] |= np.uint16(1)
    return {
        "wave": observed.wave, "flux": centre, "variance": variance, "valid": valid,
        "samples": draws, "sample_ids": np.asarray(identifiers, np.int32),
        "normalizer": norm, "successful_shifts": shifts, "lsf_matched": matched,
        "raw_ratio": ratio, "mc_requested": count, "mc_trimmed_bins": trimmed_bins,
        "mc_fit_failures": count - len(draws),
        "quality": quality, "extraction_covariance": saved_draws is not None,
    }


def generate_response_spline(obs_spec, ref_spec, balmer_lines=None, order=5,
                             clip_sigma=2.5, n_iter=5, n_shifts=3,
                             RC_PLOT=False, plot_file=None, mc_samples=256,
                             observed_fwhm=None):
    """Provide a compatible response API with variance and explicit failures."""
    if obs_spec.uncertainty is None:
        raise ResponseError("Observed standard variance is required for response calibration.")
    observed = ArraySpectrum(
        obs_spec.spectral_axis.to_value(u.AA), np.asarray(obs_spec.flux.value, float),
        obs_spec.uncertainty.represent_as(VarianceUncertainty).array,
        ~np.asarray(obs_spec.mask, bool) if obs_spec.mask is not None else np.ones(len(obs_spec.flux), bool),
    )
    reference_var = np.zeros(len(ref_spec.flux)) if ref_spec.uncertainty is None else ref_spec.uncertainty.represent_as(VarianceUncertainty).array
    template = ArraySpectrum(
        ref_spec.spectral_axis.to_value(u.AA), np.asarray(ref_spec.flux.value, float),
        np.asarray(reference_var, float),
        ~np.asarray(ref_spec.mask, bool) if ref_spec.mask is not None else np.ones(len(ref_spec.flux), bool),
        fwhm=ref_spec.meta.get("fwhm"), metadata=dict(ref_spec.meta),
    )
    result = _response_ensemble(observed, template, {
        "lines": balmer_lines, "order": order, "clip_sigma": clip_sigma,
        "fit_iterations": n_iter, "n_shifts": n_shifts, "mc_samples": mc_samples,
    }, observed_fwhm=observed_fwhm)
    keep = np.flatnonzero(result["valid"])
    if not len(keep):
        raise ResponseError("No supported response interval.")
    sl = slice(keep[0], keep[-1] + 1)
    if RC_PLOT and plot_file:
        _plot_response(result, plot_file)
    return Spectrum(
        spectral_axis=result["wave"][sl] * u.AA,
        flux=result["flux"][sl] * u.dimensionless_unscaled,
        uncertainty=VarianceUncertainty(result["variance"][sl]),
        mask=~result["valid"][sl],
        meta={"samples": result["samples"][:, sl], "sample_ids": result["sample_ids"]},
    )


def _plot_response(result, path):
    """Render a diagnostic of measured ratio and supported response only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(12, 6))
    valid = result["valid"]
    axis.plot(result["wave"], result["raw_ratio"] / result["normalizer"], ".", color="0.6", markersize=1)
    axis.plot(result["wave"][valid], result["flux"][valid], color="tab:red")
    error = np.sqrt(result["variance"][valid])
    axis.fill_between(result["wave"][valid], result["flux"][valid] - error, result["flux"][valid] + error, alpha=0.2)
    axis.set(xlabel="Air wavelength (Angstrom)", ylabel="Relative response", title=Path(path).name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_response_file(sci_dc_path, references, refpath, config, arc_ec_path=None):
    """Create a supported response product and its reproducible noise ensemble."""
    from scipy.optimize import OptimizeWarning

    notes = []
    observed = read_array_spectrum(sci_dc_path)
    template = load_template(sci_dc_path, references, refpath, config)
    notes.extend(template.metadata["warnings"])
    parameters = dict(config.get("Resgen", {}))
    native = _native_source(sci_dc_path, observed)
    if not native[-1]:
        _warning(f"{Path(sci_dc_path).name}: pre-resampling noise provenance missing; output-bin independence is an approximation.", notes)
    observed_fwhm = None
    arc_measurements = None
    if "observed_fwhm_angstrom" in parameters:
        observed_fwhm = np.full(len(observed.wave), float(parameters["observed_fwhm_angstrom"]))
    elif arc_ec_path is not None and bool(parameters.get("match_resolution", True)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                observed_fwhm, arc_measurements = estimate_arc_fwhm(arc_ec_path, observed.wave, config.get("Wavecal", {}))
        except (ResponseError, ValueError, OSError):
            # Unavailable widths remain reflected by LSFMATCH and LSFMODE.
            pass
    result = _response_ensemble(
        observed, template, parameters, native, observed_fwhm, _frame_identity(sci_dc_path, observed.header),
    )
    if result["successful_shifts"] < int(parameters.get("n_shifts", 3)):
        _warning(f"{Path(sci_dc_path).name}: only {result['successful_shifts']} shifted spline fits succeeded.", notes)
    if len(result["samples"]) < result["mc_requested"]:
        _warning(f"{Path(sci_dc_path).name}: {len(result['samples'])}/{result['mc_requested']} uncertainty realizations succeeded.", notes)
    if result["mc_trimmed_bins"]:
        _warning(f"{Path(sci_dc_path).name}: masked {result['mc_trimmed_bins']} bins outside common Monte Carlo support; retained all {len(result['samples'])} successful fits.", notes)
    keep = np.flatnonzero(result["valid"])
    sl = slice(keep[0], keep[-1] + 1)
    wave = observed.wave[sl]
    header = _wavelength_header(observed.header, wave, "")
    header["RESPSTAT"] = "SUCCESS"
    header["FLUXTYPE"] = "RELATIVE RESPONSE"
    header["TPLFILE"] = Path(template.metadata["path"]).name
    header["TPLMED"] = template.metadata["input_medium"]
    header["TPLCONV"] = template.metadata["conversion"]
    header["TPLSTAT"] = template.metadata["stat_known"]
    header["TPLSYS"] = "NOT_INCLUDED"
    header["CROSSCAL"] = ("NOT_MODELED", "Cross-frame master-calibration covariance")
    header["LSFMATCH"] = result["lsf_matched"]
    header["LSFMODE"] = "GAUSSIAN_ARC_APPROX" if result["lsf_matched"] else "UNMATCHED"
    header["MCSAMPLE"] = len(result["samples"])
    header["MCREQ"] = result["mc_requested"]
    header["MCTRIM"] = (result["mc_trimmed_bins"], "Bins removed for common Monte Carlo support")
    header["MCFITBAD"] = (result["mc_fit_failures"], "Failed Monte Carlo spline fits")
    header["NATIVEMC"] = bool(native[-1])
    header["EXTCOV"] = bool(result["extraction_covariance"])
    header["FATALBIT"] = FATAL_EXTRACTION_BITS
    header["REFQALL"] = int(np.bitwise_or.reduce(result["quality"][result["valid"]])) & ~FATAL_EXTRACTION_BITS
    header["ERRMODEL"] = "MC_STAT_CONDITIONAL"
    header["HISTORY"] = "Only observed/template overlap is calibrated; no response extrapolation."
    header["HISTORY"] = "Input marginal variance includes upstream master-calibration variance."
    header["HISTORY"] = "Cross-frame covariance from shared masters is not modeled."
    header["HISTORY"] = "Template mismatch systematics are not included in VARIANCE."
    header["HISTORY"] = "Template provenance: " + template.metadata["provenance"]
    if result["lsf_matched"]:
        header["HISTORY"] = "Arc LSF approximates stellar LSF; slit-illumination mismatch is not measured."
    output = str(sci_dc_path).replace(".ec_dc.fits", ".res.fits")
    if output == str(sci_dc_path):
        raise ValueError("Expected .ec_dc.fits input for response generation.")
    hdus = [
        fits.PrimaryHDU(result["flux"][sl].astype(np.float32), header),
        fits.ImageHDU(result["variance"][sl].astype(np.float32), name="VARIANCE"),
        fits.ImageHDU((~result["valid"][sl]).astype(np.uint8), name="MASK"),
        fits.ImageHDU(result["samples"][:, sl].astype(np.float32), name="MC_RESPONSE"),
        fits.ImageHDU(result["sample_ids"], name="MC_INDEX"),
        fits.ImageHDU(result["quality"][sl], _quality_header(), name="QUALITY"),
    ]
    if arc_measurements is not None:
        hdus.append(fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", unit="Angstrom", array=arc_measurements[:, 0]),
            fits.Column(name="FWHM", format="D", unit="Angstrom", array=arc_measurements[:, 1]),
        ], name="ARC_LSF"))
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    if parameters.get("plotyn"):
        _plot_response(result, output.replace(".res.fits", "_response.pdf"))
    return {
        "status": "warning" if notes else "success", "path": output, "warnings": notes,
        "coverage": [float(wave[0]), float(wave[-1])],
        "lsf_matched": bool(result["lsf_matched"]),
        "mc_samples": len(result["samples"]),
        "template_path": str(Path(template.metadata["path"]).resolve()),
        "template_sha256": (hashlib.sha256(Path(template.metadata["path"]).read_bytes()).hexdigest()
                            if Path(template.metadata["path"]).is_file() else None),
        "extraction_covariance": bool(result["extraction_covariance"]),
    }


def _load_response(path):
    """Read a response and its aligned Monte Carlo sample identifiers."""
    spectrum = read_array_spectrum(path)
    with fits.open(path) as hdul:
        if "MC_RESPONSE" not in hdul or "MC_INDEX" not in hdul:
            raise ResponseError(f"{Path(path).name}: response uncertainty samples are missing; regenerate this response.")
        samples = np.asarray(hdul["MC_RESPONSE"].data, float).copy()
        identifiers = np.asarray(hdul["MC_INDEX"].data, np.int32).copy()
    if samples.ndim != 2 or samples.shape != (len(identifiers), len(spectrum.wave)):
        raise ResponseError(f"{Path(path).name}: invalid response uncertainty array shape.")
    if len(identifiers) < 8 or len(np.unique(identifiers)) != len(identifiers) or np.any(identifiers < 0):
        raise ResponseError(f"{Path(path).name}: response sample IDs must be unique and nonnegative.")
    spectrum.valid &= (spectrum.flux > 0) & np.all(np.isfinite(samples) & (samples > 0), axis=0)
    return spectrum, samples, identifiers


def _group_response(file_list, star_coord, site):
    """Combine log responses and shared-template noise realizations by median."""
    paths = list(dict.fromkeys(str(Path(path).resolve()) for path in file_list))
    if not paths:
        raise ResponseError("No successful response files in the reference group.")
    loaded = [_load_response(path) for path in paths]
    if any(np.count_nonzero(spec.valid) < 4 for spec, _, _ in loaded):
        raise ResponseError("A reference response has fewer than four valid points.")
    minimum = max(spec.wave[spec.valid][0] for spec, _, _ in loaded if np.any(spec.valid))
    maximum = min(spec.wave[spec.valid][-1] for spec, _, _ in loaded if np.any(spec.valid))
    base_wave = loaded[0][0].wave
    grid = base_wave[(base_wave >= minimum) & (base_wave <= maximum)]
    if len(grid) < 4:
        raise ResponseError("Reference response spectra have no sufficient common support.")
    sample_ids = loaded[0][2]
    for _, _, identifiers in loaded[1:]:
        sample_ids = np.intersect1d(sample_ids, identifiers)
    if len(sample_ids) < 8:
        raise ResponseError("Fewer than eight common response uncertainty realizations.")
    curves, ensembles, airmasses = [], [], []
    combined_quality = np.zeros(len(grid), np.uint16)
    global_warnings = 0
    support = np.ones(len(grid), bool)
    for spec, samples, identifiers in loaded:
        operator, valid = _linear_operator(spec.wave, grid, spec.valid)
        combined_quality |= _project_quality(operator, _quality_array(spec))
        global_warnings |= int(spec.header.get("REFQALL", 0))
        curve = operator @ np.nan_to_num(spec.flux)
        selected = np.array([int(np.flatnonzero(identifiers == identifier)[0]) for identifier in sample_ids])
        ensemble = (operator @ np.nan_to_num(samples[selected]).T).T
        valid &= (curve > 0) & np.all(ensemble > 0, axis=0)
        support &= valid
        curves.append(np.log(np.maximum(curve, np.finfo(float).tiny)))
        ensembles.append(np.log(np.maximum(ensemble, np.finfo(float).tiny)))
        airmass = get_airmass(spec.header, star_coord, site)
        if airmass is None:
            raise ResponseError(f"Reference airmass is unavailable for {spec.header.get('OBJECT', '?')}.")
        airmasses.append(airmass)
    log_curve = np.median(curves, axis=0)
    log_draws = np.median(ensembles, axis=0)
    scatter = np.std(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros(len(grid))
    log_curve[~support] = np.nan
    log_draws[:, ~support] = np.nan
    scatter[~support] = np.nan
    if support.sum() < 4:
        raise ResponseError("Reference group has fewer than four supported response points.")
    return {
        "wave": grid, "log_response": log_curve, "log_samples": log_draws,
        "sample_ids": sample_ids, "valid": support, "airmass": float(np.mean(airmasses)),
        "scatter": scatter, "paths": paths, "nref": len(paths),
        "template_media": sorted(set(str(spec.header.get("TPLMED", "unknown")) for spec, _, _ in loaded)),
        "lsf_matched": all(bool(spec.header.get("LSFMATCH", False)) for spec, _, _ in loaded),
        "template_stat_known": all(bool(spec.header.get("TPLSTAT", False)) for spec, _, _ in loaded),
        "native_mc": all(bool(spec.header.get("NATIVEMC", False)) for spec, _, _ in loaded),
        "extraction_covariance": all(bool(spec.header.get("EXTCOV", False)) for spec, _, _ in loaded),
        "quality": combined_quality, "global_warnings": global_warnings,
    }


def get_group_response(file_list, star_coord, site, grid_points=4500):
    """Provide the legacy group tuple with actual Monte Carlo uncertainties."""
    result = _group_response(file_list, star_coord, site)
    error = np.full(len(result["wave"]), np.nan)
    error[result["valid"]] = np.std(result["log_samples"][:, result["valid"]], axis=0, ddof=1)
    return result["wave"], result["log_response"], error, result["airmass"]


def _project_group(group, grid, sample_ids):
    """Evaluate a smooth group response and its correlated samples on a grid."""
    operator, valid = _linear_operator(group["wave"], grid, group["valid"])
    response = operator @ np.nan_to_num(group["log_response"])
    indices = [int(np.flatnonzero(group["sample_ids"] == identifier)[0]) for identifier in sample_ids]
    samples = (operator @ np.nan_to_num(group["log_samples"][indices]).T).T
    scatter = operator @ np.nan_to_num(group["scatter"])
    response[~valid] = np.nan
    samples[:, ~valid] = np.nan
    scatter[~valid] = np.nan
    return response, samples, scatter, valid


def _noise_realizations(path, spectrum, count, seed, identifiers=None):
    """Reuse correlated extraction draws or explicitly use the legacy noise model."""
    _, flux, variance, _, operator, valid, available = _native_source(path, spectrum)
    samples, _ = _selected_samples(spectrum, count=count, identifiers=identifiers, native=available)
    if samples is not None:
        draws = (operator @ np.nan_to_num(samples).T).T
        draws[:, ~valid] = np.nan
        return draws, available
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(count, len(flux))) * np.sqrt(np.maximum(np.nan_to_num(variance), 0))
    draws = (operator @ (np.nan_to_num(flux)[None, :] + noise).T).T
    draws[:, ~valid] = np.nan
    return draws, available


def _relative_normalization_window(wave, valid, config):
    """Select a fixed science-grid wavelength window with explicit support limits.

    The nominal spectrum and every Monte Carlo realization use this same mask.
    Missing support never changes the configured window or invokes a full-range
    normalization. ``Resgen`` configures the Angstrom interval (default
    5400--5600), minimum coverage (0.9), and minimum valid bin count (20).
    Coverage is the supported bin width inside the window. Return the mask and
    window metadata, raising ``ResponseError`` when support is inadequate.
    """
    settings = config.get("Resgen", {})
    if not isinstance(settings, dict):
        raise ResponseError("Resgen must be a mapping for relative-flux normalization.")
    try:
        window = np.asarray(settings.get("relflux_norm_window", (5400., 5600.)), dtype=float)
        minimum_coverage = float(settings.get("relflux_norm_min_coverage", 0.9))
        count_setting = settings.get("relflux_norm_min_bins", 20)
        minimum_bins = int(count_setting)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResponseError("Invalid fixed relative-flux normalization settings.") from error
    if (window.shape != (2,) or not np.isfinite(window).all()
            or window[0] <= 0 or window[1] <= window[0]):
        raise ResponseError("relflux_norm_window must be two increasing positive Angstrom wavelengths.")
    if (isinstance(count_setting, (bool, np.bool_)) or minimum_bins != count_setting
            or minimum_bins < 2):
        raise ResponseError("relflux_norm_min_bins must be an integer of at least two.")
    if (isinstance(settings.get("relflux_norm_min_coverage", 0.9), (bool, np.bool_))
            or not np.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1):
        raise ResponseError("relflux_norm_min_coverage must be in (0, 1].")
    wave = np.asarray(wave, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if (wave.ndim != 1 or len(wave) < 2 or valid.shape != wave.shape
            or not np.isfinite(wave).all() or np.any(np.diff(wave) <= 0)):
        raise ResponseError("Normalization wavelengths and validity must be matching increasing vectors.")
    lower, upper = map(float, window)
    selected = valid & (wave >= lower) & (wave <= upper)
    edges = bin_edges(wave)
    widths = np.maximum(np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower), 0.)
    coverage = min(float(widths[selected].sum() / (upper - lower)), 1.)
    count = int(selected.sum())
    if count < minimum_bins or coverage + 1e-12 < minimum_coverage:
        raise ResponseError(
            f"Insufficient fixed normalization window support in {lower:g}-{upper:g} Angstrom: "
            f"{count} bins, coverage {coverage:.3f}; require {minimum_bins} bins and {minimum_coverage:.3f}.")
    return selected, dict(lower=lower, upper=upper, count=count, coverage=coverage,
                          minimum_bins=minimum_bins, minimum_coverage=minimum_coverage)


def _correct_one(path, front, back, target_coord, site, config, group_number, target_name):
    """Write response-corrected relative flux and conditional MC uncertainties.

    Project the bracketing standard groups onto the science wavelength grid
    and interpolate their log responses in airmass. Normalize the nominal
    corrected spectrum and each MC realization by its own median over the
    same configured wavelength mask. ``FLUXERR`` is the sample standard
    deviation of these normalized draws, including their spectral correlation;
    template-mismatch systematics and cross-frame master covariance are excluded.

    Write or replace the target's relative-flux FITS product beside ``path``.
    Return its filename, warnings, and supported wavelength bounds. Missing
    response support or invalid normalization raises ``ResponseError``.
    """
    notes = []
    spectrum = read_array_spectrum(path)
    same_group = front is back
    identifiers = np.intersect1d(front["sample_ids"], back["sample_ids"])
    if "sample_ids" in spectrum.metadata:
        identifiers = np.intersect1d(identifiers, spectrum.metadata["sample_ids"])
    if len(identifiers) < 8:
        raise ResponseError("Front/back reference groups have too few common uncertainty samples.")
    log_front, front_draws, scatter_front, supported_front = _project_group(front, spectrum.wave, identifiers)
    if same_group:
        log_back, back_draws, scatter_back, supported_back = log_front, front_draws, scatter_front, supported_front
    else:
        log_back, back_draws, scatter_back, supported_back = _project_group(back, spectrum.wave, identifiers)
    current_airmass = get_airmass(spectrum.header, target_coord, site)
    if current_airmass is None or not np.isfinite(current_airmass) or current_airmass < 1:
        raise ResponseError(f"{Path(path).name}: target airmass unavailable; no invented fallback is permitted.")
    x1, x2 = front["airmass"], back["airmass"]
    if abs(x2 - x1) < 1e-4:
        weight = 0.5
        if not same_group:
            _warning(f"{Path(path).name}: reference airmasses nearly equal ({x1:.6f}, {x2:.6f}); existing weight 0.5 retained.", notes)
    else:
        weight = (current_airmass - x1) / (x2 - x1)
    if not 0 <= weight <= 1:
        _warning(f"{Path(path).name}: airmass EXTRAPOLATION, X1={x1:.5f}, Xt={current_airmass:.5f}, X2={x2:.5f}, w={weight:.5f}.", notes)
    elif not same_group and abs(x2 - x1) < 0.01:
        _warning(f"{Path(path).name}: small airmass span {abs(x2 - x1):.5f}; weight is sensitive to reference differences.", notes)
    with np.errstate(over="ignore", invalid="ignore"):
        response = np.exp((1 - weight) * log_front + weight * log_back)
        response_draws = np.exp((1 - weight) * front_draws + weight * back_draws)
    source_edges = bin_edges(spectrum.wave)
    lower_edge = max(bin_edges(front["wave"])[0], bin_edges(back["wave"])[0])
    upper_edge = min(bin_edges(front["wave"])[-1], bin_edges(back["wave"])[-1])
    valid = (
        spectrum.valid & supported_front & supported_back
        & (source_edges[:-1] >= lower_edge) & (source_edges[1:] <= upper_edge)
        & np.isfinite(response) & (response > 0)
        & np.all(np.isfinite(response_draws) & (response_draws > 0), axis=0)
    )
    if valid.sum() < 4:
        raise ResponseError("No common valid target/standard/template wavelength interval.")
    calibrated = np.full(len(spectrum.wave), np.nan)
    calibrated[valid] = spectrum.flux[valid] / response[valid]
    normalization_mask, normalization_info = _relative_normalization_window(spectrum.wave, valid, config)
    normalization = float(np.median(calibrated[normalization_mask]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ResponseError(f"Invalid positive relative-flux normalization {normalization}.")
    target_draws, native_available = _noise_realizations(
        path, spectrum, len(identifiers), _seed(f"{_frame_identity(path, spectrum.header)}:target:{config.get('Resgen', {}).get('seed', 42)}"),
        identifiers=identifiers,
    )
    if not native_available:
        _warning(f"{Path(path).name}: native target variance provenance unavailable; normalization MC assumes independent saved bins.", notes)
    corrected_draws = target_draws[:, valid] / response_draws[:, valid]
    normalization_draws = np.median(corrected_draws[:, normalization_mask[valid]], axis=1)
    good_draws = np.isfinite(normalization_draws) & (normalization_draws > 0)
    if good_draws.sum() < max(8, math.ceil(len(identifiers) / 2)):
        raise ResponseError("Too few valid relative-flux normalization realizations.")
    if not good_draws.all():
        _warning(f"{Path(path).name}: rejected {len(good_draws) - good_draws.sum()} nonpositive normalization realizations.", notes)
    normalized_draws = corrected_draws[good_draws] / normalization_draws[good_draws, None]
    flux = calibrated / normalization
    error = np.full(len(flux), np.nan)
    error[valid] = np.std(normalized_draws, axis=0, ddof=1)
    first, last = np.flatnonzero(valid)[[0, -1]]
    sl = slice(first, last + 1)
    header = _wavelength_header(spectrum.header, spectrum.wave[sl], "")
    header["FLUXTYPE"] = "RELATIVE"
    header["NORMTYPE"] = "MEDIAN_WINDOW"
    header["NORMVAL"] = normalization
    header["NORMERR"] = float(np.std(normalization_draws[good_draws], ddof=1))
    header["NORMWLO"] = (normalization_info["lower"], "Fixed normalization lower wavelength")
    header["NORMWHI"] = (normalization_info["upper"], "Fixed normalization upper wavelength")
    header["NORMUNIT"] = "Angstrom"
    header["NORMFRAM"] = (spectrum.header.get("SPECSYS", "UNKNOWN"), "Normalization uses this science-grid frame")
    header["NORMNPIX"] = normalization_info["count"]
    header["NORMCOV"] = (normalization_info["coverage"], "Supported fraction of fixed window width")
    header["NORMMIN"] = normalization_info["minimum_bins"]
    header["NORMCVM"] = normalization_info["minimum_coverage"]
    header["ERRMODEL"] = "MC_STAT_CONDITIONAL"
    header["COVAR"] = "SPECTRAL_CORRELATED"
    header["SYSERR"] = ("NOT_INCLUDED", "Template and other unmodeled systematics")
    header["CROSSCAL"] = ("NOT_MODELED", "Cross-frame master-calibration covariance")
    header["TPLMED"] = ",".join(sorted(set(front.get("template_media", ["unknown"]) + back.get("template_media", ["unknown"]))))
    header["LSFMATCH"] = bool(front.get("lsf_matched", False) and back.get("lsf_matched", False))
    header["TPLSTAT"] = bool(front.get("template_stat_known", False) and back.get("template_stat_known", False))
    header["NATIVEMC"] = bool(native_available and front.get("native_mc", False) and back.get("native_mc", False))
    header["EXTCOV"] = bool("extraction_samples" in spectrum.metadata
                              and front.get("extraction_covariance", False)
                              and back.get("extraction_covariance", False))
    header["FATALBIT"] = FATAL_EXTRACTION_BITS
    header["SCATTYPE"] = "LOGR_FRAME_SCATTER"
    header["AIRMASS"] = current_airmass
    header["REFAM1"] = x1
    header["REFAM2"] = x2
    header["RESP_W"] = weight
    header["SAME_REF"] = same_group
    header["NREF1"] = front["nref"]
    header["NREF2"] = back["nref"]
    header["MCSAMPLE"] = int(good_draws.sum())
    header["RESPSTAT"] = "SUCCESS"
    header["HISTORY"] = "Relative flux and error are dimensionless; science wavelength grid retained."
    header["HISTORY"] = "Response/normalization Monte Carlo includes spectral correlation."
    header["HISTORY"] = "Nominal flux and every MC draw use the same fixed normalization window."
    header["HISTORY"] = "Input marginal variance includes upstream master-calibration variance."
    header["HISTORY"] = "Cross-frame covariance from shared masters is not modeled."
    header["HISTORY"] = "Template mismatch systematics are not included."
    match = re.search(r"-(\d+)\.ec_dc\.fits$", Path(path).name)
    suffix = match.group(1) if match else Path(path).stem.replace(".ec_dc", "")
    output = Path(path).parent / f"T{group_number + 1:03d}_{target_name}_relflux_{suffix}.fits"
    scatter = np.abs(1 - weight) * scatter_front + np.abs(weight) * scatter_back
    target_quality = _quality_array(spectrum)
    reference_quality = np.zeros(len(spectrum.wave), np.uint16)
    for group in (front,) if same_group else (front, back):
        operator, _ = _linear_operator(group["wave"], spectrum.wave, group["valid"])
        reference_quality |= _project_quality(operator, group.get("quality", np.zeros(len(group["wave"]), np.uint16)))
        reference_quality |= np.uint16(group.get("global_warnings", 0))
    quality = target_quality | reference_quality
    quality[~valid] |= np.uint16(1)
    header["REFQALL"] = int(np.bitwise_or.reduce(reference_quality[valid])) & ~FATAL_EXTRACTION_BITS
    columns = [
        fits.Column(name="WAVELENGTH", format="D", unit="Angstrom", array=spectrum.wave[sl]),
        fits.Column(name="FLUX", format="E", array=flux[sl].astype(np.float32)),
        fits.Column(name="FLUXERR", format="E", array=error[sl].astype(np.float32)),
        fits.Column(name="MASK", format="L", array=~valid[sl]),
        fits.Column(name="RESP_SCATTER", format="E", array=scatter[sl].astype(np.float32)),
        fits.Column(name="QUALITY", format="J", array=quality[sl].astype(np.int32)),
        fits.Column(name="TARGQUAL", format="J", array=target_quality[sl].astype(np.int32)),
        fits.Column(name="REFQUAL", format="J", array=reference_quality[sl].astype(np.int32)),
    ]
    final_draws = np.full((int(good_draws.sum()), len(flux)), np.nan)
    final_draws[:, valid] = normalized_draws
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(columns, header=_quality_header(header), name="SPECTRUM"),
        fits.ImageHDU(final_draws[:, sl], fits.Header({"BUNIT": ""}), name="MC_RELFLUX"),
        fits.ImageHDU(identifiers[good_draws], name="MC_INDEX"),
    ]).writeto(output, overwrite=True, checksum=True)
    return {"path": str(output), "warnings": notes, "coverage": [float(spectrum.wave[first]), float(spectrum.wave[last])]}


def response_correction(num, obsset, SITE, config=None):
    """Correct targets with scoped IERS offline behavior in every worker."""
    offline = bool((config or {}).get("Runtime", {}).get("offline", False))
    context = iers.conf.set_temp("auto_download", False) if offline else nullcontext()
    with context:
        return _response_correction_impl(num, obsset, SITE, config)


def _response_correction_impl(num, obsset, SITE, config=None):
    """Correct independent targets, returning explicit outputs and failures."""
    config = config or {}
    outputs, failures, notes = [], [], []
    target_name, reference_name = obsset["target_name"], obsset["ref_name"]
    front_files = list(obsset.get("ref_front", []))
    back_files = list(obsset.get("ref_back", []))
    if not front_files or not back_files:
        raise ResponseError("A required reference group has no successful response.")
    reference_header = fits.getheader(front_files[0])
    target_header = fits.getheader(obsset["target"][0]) if obsset.get("target") else fits.Header()
    coordinates = config.get("Coordinates", obsset.get("coordinates", {}))
    offline = bool(config.get("Runtime", {}).get("offline", False))
    target_coord = _resolve_coordinate(target_name, target_header, coordinates, offline)
    reference_coord = _resolve_coordinate(reference_name, reference_header, coordinates, offline)
    front = _group_response(front_files, reference_coord, SITE)
    if set(map(str, map(Path, front_files))) == set(map(str, map(Path, back_files))):
        back = front
    else:
        back = _group_response(back_files, reference_coord, SITE)
    for path in obsset.get("target", []):
        try:
            result = _correct_one(path, front, back, target_coord, SITE, config, num, target_name)
            outputs.append(result["path"])
            notes.extend(result["warnings"])
        except Exception as error:
            message = f"{Path(path).name}: response correction FAILED: {type(error).__name__}: {error}"
            _warning(message, notes)
            failures.append({"file": str(path), "stage": "response_correction", "error": str(error)})
    status = "partial" if outputs and failures else "failed" if failures or not outputs else "warning" if notes else "success"
    return {"status": status, "outputs": outputs, "failed": failures, "warnings": notes}


# =============================================================================
# Runtime
# =============================================================================

def json_default(value):
    """Serialize paths and NumPy scalars without losing numerical information."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def file_sha256(path):
    """Hash one provenance file in bounded chunks without loading its image."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    """Atomically replace one JSON report inside the current run directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    default=json_default), encoding="utf-8")
    # Windows indexers may briefly hold a newly written report open.
    for attempt in range(7):
        try:
            temporary.replace(path)
            break
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == 6:
                raise
            time.sleep(0.02 * 2**attempt)


def new_run(config, observations):
    """Create isolated working storage and reserve a date-based publication path."""
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    work = Path(config["workpath"]).expanduser().resolve() / run_id
    products = Path(config["savepath"]).expanduser().resolve() / Path(observations).name
    observations = Path(observations).resolve()
    if (products.is_relative_to(observations) or observations.is_relative_to(products)
            or products.is_relative_to(work) or work.is_relative_to(products)):
        raise ValueError('Observation, working, and published directories must not overlap')
    work.mkdir(parents=True, exist_ok=False)
    return run_id, work, products


def failure(stage, input_file, error, **extra):
    """Describe one failure so independent observations can continue."""
    return {"status": "failed", "stage": stage, "input": str(input_file),
            "error": f"{type(error).__name__}: {error}", "outputs": [],
            "warnings": [], **extra}


def warning_text(records):
    """Deduplicate captured Python warnings while preserving their order."""
    return list(dict.fromkeys(str(record.message) for record in records))


def worker_count(config, jobs, estimated_bytes):
    """Bound concurrent jobs by CPU count, user preference, and memory budget."""
    if jobs <= 0:
        return 0
    requested = config.get("Runtime", {}).get("max_workers")
    cpu_limit = max(1, (os.cpu_count() or 2) // 2 - 2)
    if requested is not None:
        cpu_limit = max(1, int(requested))
    memory_bytes = float(config.get("memory_limit", 32)) * 1e9
    memory_limit = max(1, int(memory_bytes // max(estimated_bytes, 1)))
    return min(jobs, cpu_limit, memory_limit)


def completed_jobs(worker, tasks, workers):
    """Yield every task outcome even when another process raises an exception."""
    if not tasks:
        return
    if workers <= 1:
        for task in tasks:
            try:
                yield task, worker(task)
            except Exception as exc:
                yield task, failure("worker", task.get("file", task.get("idx", "group")), exc)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                yield task, future.result()
            except Exception as exc:
                yield task, failure("worker", task.get("file", task.get("idx", "group")), exc)


def final_status(expected, completed, failures, warnings_count):
    """Distinguish complete, warned, partial, and unsuccessful reductions."""
    if completed == 0:
        return "failed"
    if completed < expected or failures:
        return "partial_success"
    return "completed_with_warnings" if warnings_count else "completed"


def copy_verified(source, destination):
    """Copy a finished file and reject any destination with different contents."""
    source, destination = Path(source), Path(destination)
    source_hash = file_sha256(source)
    shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size or file_sha256(destination) != source_hash:
        raise OSError(f"Published file verification failed: {destination}")


def publication_files(work):
    """Select the flat set of scientific products and diagnostic records to archive."""
    selected = []
    for path in sorted(Path(work).iterdir()):
        if not path.is_file():
            continue
        if (path.name.endswith((".res.fits", ".ec.fits", ".ec_dc.fits"))
                or "relflux" in path.name
                or path.suffix.lower() in {".png", ".pdf", ".json", ".yaml", ".log"}):
            selected.append(path)
    return selected


def publish_products(work, products):
    """Copy and verify final products, extracted spectra, and run diagnostics."""
    products = Path(products)
    products.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in publication_files(work):
        destination = products / path.name
        copy_verified(path, destination)
        copied.append(str(destination))
    return copied


def relocate_published_paths(value, path_map):
    """Redirect exact artifact paths while preserving historical provenance text."""
    if isinstance(value, (str, Path)):
        return path_map.get(str(value), value)
    if isinstance(value, list):
        return [relocate_published_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: relocate_published_paths(item, path_map) for key, item in value.items()}
    return value


def checked_archive_tree(path, parent):
    """Validate an archive child directory before copying, moving, or removing it."""
    path, parent = Path(path), Path(parent).resolve()
    target = path.resolve(strict=True)
    if target.parent != parent or not target.is_dir():
        raise ValueError(f'Archive directory is outside its designated parent: {path}')
    for entry in [path, *target.rglob('*')]:
        if (entry.is_symlink() or (hasattr(entry, 'is_junction') and entry.is_junction())
                or not entry.resolve().is_relative_to(target)):
            raise ValueError(f'Archive contains a linked or external path: {entry}')
    return target


@contextmanager
def publication_workspace(products, run_id):
    """Hold a date-specific lock through publication and final journal updates."""
    products = Path(products)
    products.parent.mkdir(parents=True, exist_ok=True)
    lock = products.parent / f'.{products.name}.publish.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Publication is locked by another or interrupted run: {lock}') from exc
    try:
        os.write(descriptor, run_id.encode('utf-8'))
        leftovers = [path for path in products.parent.iterdir()
                     if os.path.normcase(path.name).startswith(os.path.normcase(f'.{products.name}.run-'))
                     and os.path.normcase(path.suffix) in {'.staging', '.previous'}]
        if leftovers:
            raise RuntimeError(f'Unfinished publication requires recovery; preserved paths: {leftovers}')
        stage = products.parent / f'.{products.name}.{run_id}.staging'
        backup = products.parent / f'.{products.name}.{run_id}.previous'
        stage.mkdir(exist_ok=False)
        yield stage, backup
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def preserve_unmanaged_products(products, stage):
    """Carry user files and legacy run directories forward, excluding old managed outputs."""
    products, stage = Path(products), Path(stage)
    if not products.exists():
        return
    checked_archive_tree(products, products.parent)
    managed = set()
    manifest = products / 'run_manifest.json'
    if manifest.exists():
        report = json.loads(manifest.read_text(encoding='utf-8'))
        if (not re.fullmatch(r'run-\d{8}T\d{6}Z-[0-9a-f]{8}', report.get('run_id', ''))
                or Path(report.get('output_directory', '')).resolve() != products.resolve()
                or report.get('status') not in {'completed', 'completed_with_warnings'}):
            raise ValueError(f'Existing date manifest is not a completed pipeline archive: {manifest}')
        for value in report.get('published_files', []):
            path = Path(value)
            if path.resolve().parent != products.resolve():
                raise ValueError(f'Existing manifest contains an external product path: {value}')
            managed.add(os.path.normcase(path.name))
        if os.path.normcase('run_manifest.json') not in managed:
            raise ValueError(f'Existing manifest has no managed-file inventory: {manifest}')
    for path in sorted(products.rglob('*')):
        relative = path.relative_to(products)
        if len(relative.parts) == 1 and os.path.normcase(path.name) in managed:
            if not path.is_file():
                raise ValueError(f'Managed product is not a regular file: {path}')
            continue
        destination = stage / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_verified(path, destination)


def install_products(stage, products, backup):
    """Install a verified date archive and restore the previous one if installation fails."""
    stage, products, backup = Path(stage), Path(products), Path(backup)
    checked_archive_tree(stage, products.parent)
    if backup.parent.resolve() != products.parent.resolve() or backup.exists():
        raise ValueError(f'Invalid publication backup path: {backup}')
    had_previous = products.exists()
    if had_previous:
        checked_archive_tree(products, products.parent)
        products.rename(backup)
    try:
        stage.rename(products)
    except BaseException:
        if had_previous:
            try:
                backup.rename(products)
            except OSError as restore_error:
                raise RuntimeError(f'Could not restore the previous archive; recover it from {backup}') from restore_error
        raise


def cleanup_run_work(work, products, work_root, run_id):
    """Remove only the current run under its configured root, without following links."""
    work, products = Path(work), Path(products).resolve()
    root = Path(work_root).expanduser().resolve()
    target = work.resolve(strict=True)
    if (not re.fullmatch(r"run-\d{8}T\d{6}Z-[0-9a-f]{8}", run_id)
            or target != root / run_id or products.is_relative_to(target)):
        raise ValueError(f"Refusing cleanup outside the isolated run directory: {target}")
    for path in [work, *target.rglob('*')]:
        if (path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())
                or not path.resolve().is_relative_to(target)):
            raise ValueError(f"Refusing cleanup of a linked or external path: {path}")
    manifest = json.loads((target / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('run_id') != run_id:
        raise ValueError(f"Run identity does not match cleanup directory: {target}")
    shutil.rmtree(target)


class RunJournal:
    """Persist parent-process diagnostics and the scientific provenance of a run."""

    def __init__(self, run_id, work, products, config, observations, config_path=None):
        """Initialize a report before any calibration files are read or written."""
        self.work = Path(work)
        self.products = Path(products)
        self.path = self.work / "run_manifest.json"
        self.log_path = self.work / "pipeline.log"
        self.report = {"run_id": run_id, "status": "running", "observations": str(observations),
                       "work_directory": str(work), "output_directory": str(products),
                       "config": config, "events": [], "final_outputs": []}
        root = Path(__file__).resolve().parent
        modules = ["spec_reduction_parallel.py", "speclibparallel.py"]
        
        self.report["source_sha256"] = {
            name: file_sha256(root / name)
            for name in modules if (root / name).exists()}
        self.report['input_files'] = {}
        self.report["packages"] = {}
        self.report["python"] = platform.python_version()
        for name in ["numpy", "scipy", "astropy", "ccdproc", "specutils", "scikit-learn", "numba"]:
            try:
                self.report["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.report["packages"][name] = "unavailable"
        (self.work / "effective_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self.report['effective_config_sha256'] = file_sha256(self.work / 'effective_config.yaml')
        if config_path is not None:
            self.report['configuration_source'] = {'path': str(Path(config_path).resolve()),
                                                    'sha256': file_sha256(config_path)}
        self.flush()

    def register_input_files(self, paths, role='observation'):
        """Record each actually used input once, retaining role and file identity."""
        for value in paths:
            path = Path(value).resolve()
            key = str(path)
            if key in self.report['input_files']:
                roles = self.report['input_files'][key]['roles']
                if role not in roles:
                    roles.append(role)
                continue
            if not path.is_file():
                continue
            stat = path.stat()
            self.report['input_files'][key] = {'roles': [role], 'size_bytes': stat.st_size,
                                               'mtime_ns': stat.st_mtime_ns,
                                               'sha256': file_sha256(path)}

    def emit(self, message):
        """Print a diagnostic and append it to this run's durable terminal log."""
        print(message, flush=True)
        stamp = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{stamp} {message}\n")

    def record(self, event):
        """Record a structured result, including all warning and failure reasons."""
        self.report["events"].append(event)
        label = event.get("input", event.get("stage", "run"))
        self.emit(f"[{event.get('status', 'info').upper()}] {label}")
        if event.get("error"):
            self.emit(f"  {event['error']}")
        for message in event.get("warnings", []):
            self.emit(f"  WARNING: {message}")
        calibration = event.get('calibration', {})
        if calibration:
            self.emit(f"  Dark: {calibration.get('DARKWHY', 'UNKNOWN')}, "
                      f"exposure={calibration.get('DARKEXP', 'unknown')} s, "
                      f"temperature={calibration.get('DARKTEMP', 'unknown')} C, "
                      f"scale={calibration.get('DARKSCL', 'unknown')}")
            if calibration.get('DARKFILE'):
                self.register_input_files([calibration['DARKFILE']], role='selected_dark_master')
        extraction = event.get('extraction')
        if extraction:
            self.emit(f"  Extraction {extraction.get('method', 'unknown')}: "
                      f"{extraction.get('valid_columns', 0)}/{extraction.get('total_columns', 0)} valid columns; "
                      f"anchor failures={extraction.get('anchor_failed_columns', 0)}, "
                      f"unstable={extraction.get('unstable_columns', 0)}, "
                      f"negative residual columns={extraction.get('negative_residual_columns', 0)}, "
                      f"profile mismatch={extraction.get('profile_mismatch_columns', 0)}, "
                      f"CR pixels={extraction.get('cr_pixels', 0)}")
        self.register_input_files(event.get('template_files', []), role='standard_template')
        self.flush()

    def flush(self):
        """Save an interruption-readable snapshot of the run's progress."""
        write_json(self.path, self.report)

    def finish(self, expected, outputs, elapsed):
        """Publish verified results and remove intermediates only for a complete run."""
        events = self.report["events"]
        failures = [event for event in events if event.get("status") == "failed"]
        warnings_count = sum(len(event.get("warnings", [])) for event in events)
        outputs = list(dict.fromkeys(str(Path(path).resolve()) for path in outputs if Path(path).is_file()))
        status = final_status(expected, len(outputs), failures, warnings_count)
        self.report.update(status=status, expected_target_frames=expected,
                           completed_target_frames=len(outputs),
                           incomplete_target_frames=max(0, expected-len(outputs)),
                           failed_events=len(failures), warning_count=warnings_count,
                           elapsed_seconds=float(elapsed), final_outputs=outputs)
        self.emit(f"Pipeline {status}: {len(outputs)}/{expected} target frames, "
                  f"{len(failures)} failed events, {warnings_count} warnings, {elapsed:.1f} s")
        complete = (status in {'completed', 'completed_with_warnings'}
                    and expected > 0 and len(outputs) == expected)
        self.report['working_cleanup'] = {'status': 'retained', 'reason': 'run_not_complete'}
        self.report['published_files'] = []
        self.report['publication'] = {'status': 'not_published'}
        self.flush()
        if not complete:
            return self.report

        committed = False
        stage = None
        try:
            with publication_workspace(self.products, self.report['run_id']) as (stage, backup):
                preserve_unmanaged_products(self.products, stage)
                # Reject collisions with preserved files before copying new pipeline outputs.
                preserved = {os.path.normcase(path.name) for path in stage.iterdir()}
                if os.path.normcase('run_manifest.json.tmp') in preserved:
                    raise ValueError('Unmanaged archive entry conflicts with the manifest temporary file')
                for path in publication_files(self.work):
                    if os.path.normcase(path.name) in preserved:
                        raise ValueError(f'New output conflicts with an unmanaged archive entry: {path.name}')
                staged = publish_products(self.work, stage)
                published = [str(self.products / Path(path).name) for path in staged]
                path_map = {str((self.work / Path(path).name).resolve()): path for path in published}
                # Every final target must have a verified destination before removing anything.
                staged_report = {
                    **self.report,
                    'final_outputs': [path_map[path] for path in outputs],
                    'published_files': published,
                    'events': relocate_published_paths(events, path_map),
                    'publication': {'status': 'published'},
                    'working_cleanup': {
                        'status': 'pending',
                        'discarded_intermediate_files': [
                            str(path.resolve()) for path in sorted(self.work.rglob('*'))
                            if path.is_file() and str(path.resolve()) not in path_map],
                    },
                }
                self.report['publication'] = {'status': 'staged', 'staging_directory': str(stage)}
                self.report['working_cleanup'] = {'status': 'retained', 'reason': 'awaiting_publication'}
                self.flush()
                copy_verified(self.log_path, stage / self.log_path.name)
                write_json(stage / self.path.name, staged_report)
                expected_manifest = json.dumps(staged_report, indent=2, ensure_ascii=False, default=json_default)
                if (stage / self.path.name).read_text(encoding='utf-8') != expected_manifest:
                    raise OSError('Staged run manifest verification failed')
                install_products(stage, self.products, backup)
                committed = True
                self.report = staged_report
                self.path = self.products / self.path.name
                self.log_path = self.products / self.log_path.name
                try:
                    if backup.exists():
                        shutil.rmtree(checked_archive_tree(backup, self.products.parent))
                    self.finish_work_cleanup()
                except Exception as exc:
                    self.record_storage_failure('cleanup', exc)
        except Exception as exc:
            if committed:
                self.record_storage_failure('cleanup', exc)
            else:
                self.report['events'] = events
                self.report['final_outputs'] = outputs
                self.report['published_files'] = []
                self.report['publication'] = {'status': 'failed'}
                self.record_storage_failure('publication', exc)
            if stage is not None and stage.exists():
                try:
                    shutil.rmtree(checked_archive_tree(stage, self.products.parent))
                except (OSError, ValueError) as cleanup_error:
                    self.report['publication']['recovery_directory'] = str(stage)
                    self.emit(f'Publication temporary files retained at {stage}: {cleanup_error}')
                    self.flush()
        return self.report

    def finish_work_cleanup(self):
        """Clean a published run while keeping cleanup warnings separate from reduction failures."""
        try:
            cleanup_run_work(self.work, self.products, self.report['config']['workpath'],
                             self.report['run_id'])
        except Exception as exc:
            self.record_storage_failure('cleanup', exc)
        else:
            self.report['working_cleanup']['status'] = 'removed'
            discarded = set(self.report['working_cleanup']['discarded_intermediate_files'])
            for path, metadata in self.report['input_files'].items():
                if path in discarded:
                    metadata['availability'] = 'removed_after_success'
            self.emit(f'Removed completed run working directory: {self.work}')
            self.flush()

    def record_storage_failure(self, stage, error):
        """Record an archive or cleanup failure without losing successful spectra."""
        self.report['working_cleanup'] = {
            'status': 'retained' if stage == 'publication' else 'failed',
            'reason': f'{stage}: {type(error).__name__}: {error}',
        }
        if stage == 'cleanup':
            self.record({'status': 'warning', 'stage': stage, 'input': str(self.work),
                         'warnings': [f'{type(error).__name__}: {error}'], 'outputs': []})
        else:
            self.record(failure(stage, self.products, error))
        events = self.report['events']
        failures = [event for event in events if event.get('status') == 'failed']
        warning_count = sum(len(event.get('warnings', [])) for event in events)
        self.report.update(
            status=final_status(self.report['expected_target_frames'],
                                self.report['completed_target_frames'], failures, warning_count),
            failed_events=len(failures), warning_count=warning_count)
        self.flush()
        if stage == 'cleanup' and self.work.is_dir():
            try:
                write_json(self.work / self.path.name, self.report)
            except OSError as report_error:
                self.report['working_cleanup']['report_update_error'] = str(report_error)
                self.emit(f'Working report could not be updated; use {self.path}: {report_error}')
                self.flush()


# =============================================================================
# Controller
# =============================================================================

TARGETS = ['TCrB', 'HD93521', 'HD37725', 'HD116405', 'HD209458', 'HIP108859',
           'HIP19797', 'HD106252', 'HD111980']


REFERENCES = {
    'PiSer': {'sptype': 'A3V', 'template': 'pickles', 'template_file': 'pickles_11'},
    'HIP20087': {'sptype': 'F0V', 'template': 'pickles', 'template_file': 'pickles_14'},
    'HIP27830': {'sptype': 'A0V', 'template': 'pickles', 'template_file': 'pickles_9'},
    'HIP27973': {'sptype': 'A5IV', 'template': 'pickles', 'template_file': 'pickles_49'},
    'HIP51914': {'sptype': 'F3V', 'template': 'miles', 'template_file': 's0337'},
    'HIP55266': {'sptype': 'A2V', 'template': 'pickles', 'template_file': 'pickles_10'},
    'HIP60804': {'sptype': 'G8III', 'template': 'pickles', 'template_file': 'pickles_76'},
    'HIP66234': {'sptype': 'A4V', 'template': 'calspec', 'template_file': 'hd165459_stisnic_007'},
    'HIP79441': {'sptype': 'F3V', 'template': 'miles', 'template_file': 's0589'},
    'HIP106787': {'sptype': 'F1IV', 'template': 'pickles', 'template_file': 'pickles_50'},
    'HIP107734': {'sptype': 'B2V', 'template': 'pickles', 'template_file': 'pickles_4'},
}


def make_site(config):
    """Construct the observatory location using the existing site configuration."""
    return EarthLocation(lat=config['latitude'] * u.deg,
                         lon=config['longitude'] * u.deg,
                         height=config['elevation'] * u.m)


def frame_memory_estimate(config, image_bytes, columns):
    """Budget the existing frame arrays plus A+C covariance and draw workspaces.

    The worker allocation policy is unchanged. A+C additionally retains several
    native-column spatial covariance matrices, three 16-draw work arrays, and
    the stored ensemble. Legacy Horne extraction keeps its previous estimate.
    """
    baseline = int(image_bytes) * 14
    extract = config.get('Extract', {})
    if str(extract.get('method', 'ac')).lower() != 'ac':
        return baseline
    spatial = 2 * int(np.ceil(float(extract.get('raper', 30)))) + 1
    samples = max(8, int(extract.get('mc_samples', config.get('Resgen', {}).get('mc_samples', 256))))
    batch = min(16, samples)
    covariance = 6 * int(columns) * spatial ** 2 * 8
    draws = (3 * batch * spatial + samples) * int(columns) * 8
    return baseline + covariance + draws


def make_group_masters_worker(task):
    """Combine one unique arc set without flat division and report failures."""
    
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore", message=r"^'datfix' made the change.*", category=FITSFixedWarning)
        try:
            if not task['files']:
                raise ValueError("No associated arc frames")
            bias = ccdp.CCDData.read(task['mbias_path'])
            darks = [ccdp.CCDData.read(path) for path in task['mdark_paths']]
            arc = combine_arc(task['files'], bias, darks, task['darkexptimelist'],
                              task['trim_start'], task['trim_end'], mflat=None,
                              memlim=task['memlim'], saturation=task.get('saturation'),
                              dark_library=task.get('dark_library'),
                              incompatible_local_exposures=task.get('incompatible_local_exposures', ()))
            arc.header['EGAIN'] = task['gain']
            arc.header['RDNOISE'] = task['rdnoise']
            arc.header['FLATCOR'] = (False, 'Flat division intentionally disabled')
            arc.write(task['output'], overwrite=False, checksum=True)
            result = {'status': 'success', 'stage': 'arc_master', 'input': task['key'],
                      'outputs': [task['output']], 'path': task['output']}
            result['calibration'] = {key: arc.meta[key] for key in
                                     ('DARKID', 'DARKSRC', 'DARKFILE', 'DARKWHY', 'DARKEXP',
                                      'DARKTEMP', 'DARKSCL', 'CALVAR') if key in arc.meta}
        except Exception as exc:
            result = failure('arc_master', task['key'], exc)
        result['warnings'] = warning_text(caught)
        result['elapsed_seconds'] = time.perf_counter() - start
        return result


def process_single_frame_worker(task):
    """Reduce one frame with explicit stage boundaries and retained partial outputs."""
    
    
    
    
    start = time.perf_counter()
    stage = 'calibration'
    outputs = []
    diagnostics = []
    config = task['config']
    work = Path(task['workpath'])
    result = {'input': task['file'], 'task_id': task['task_id'], 'is_target': task['is_target']}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        warnings.filterwarnings(
            "ignore", message=r"^'datfix' made the change.*", category=FITSFixedWarning)
        try:
            bias = ccdp.CCDData.read(task['global_data']['mbias_path'])
            darks = [ccdp.CCDData.read(path) for path in task['global_data']['mdark_paths']]
            ccd = config['CCD']
            calibrate_science_frames([task['file']], bias, darks,
                                    task['global_data']['darkexptimelist'],
                                    ccd['datay_start'], ccd['datay_end'], None,
                                    task['prefix'], work, saturation=ccd.get('saturation'),
                                    dark_library=task['global_data'].get('dark_library'),
                                    incompatible_local_exposures=task['global_data'].get('incompatible_local_exposures', ()))
            image_path = work / f"{task['prefix']}_zdf{Path(task['file']).name}"
            outputs.append(str(image_path))
            calibrated_header = fits.getheader(image_path)
            result['calibration'] = {key: calibrated_header[key] for key in
                                     ('DARKID', 'DARKSRC', 'DARKFILE', 'DARKWHY', 'DARKEXP',
                                      'DARKTEMP', 'DARKSCL', 'CALVAR') if key in calibrated_header}
            stage = 'trace'
            trace = config['Trace']
            with fits.open(image_path) as hdul:
                data = np.asarray(hdul[0].data, dtype=float).copy()
                if 'MASK' in hdul:
                    data[np.asarray(hdul['MASK'].data, dtype=bool)] = np.nan
            # Trace thresholds and the single-aperture algorithm are preserved.
            # Isolated invalid pixels are replaced only in this detection copy.
            bad = ~np.isfinite(data)
            if bad.any():
                medians = np.nanmedian(data, axis=1)
                data[bad] = np.broadcast_to(medians[:, None], data.shape)[bad]
            if not np.isfinite(data).all():
                raise ValueError('Trace input contains an entirely invalid spatial row')
            coeffs = run_trace(data, trace['naper'], trace['order'], trace['plotyn'],
                               image_path.name, trace['dW'], trace['dH'], trace['fwhm'],
                               trace['threshold'], work)
            if len(coeffs) != trace['order'] + 1 or not np.isfinite(coeffs).all():
                raise ValueError('Trace did not return finite coefficients')
            with fits.open(image_path, mode='update') as hdul:
                for index, value in enumerate(coeffs):
                    hdul[0].header[f'TRACE{index}'] = float(value)
                hdul[0].header['EGAIN'] = ccd['gain']
                hdul[0].header['RDNOISE'] = ccd['rdnoise']
            del data
            stage = 'extraction'
            extract = config['Extract']
            args = (trace['order'], extract['raper'], extract['bgin'], extract['bgout'],
                    extract['niter'], extract['crreject'], extract['nk'], extract['xbin'],
                    extract['xsmooth'], float(extract['profile_floor']))
            extraction = optimal_extract_fits(
                str(image_path), *args, method=extract.get('method', 'ac'),
                ac_training_bin=extract.get('ac_training_bin', 64),
                ac_window=extract.get('ac_window', 128),
                ac_cr_refinements=extract.get('ac_cr_refinements', 3),
                mc_samples=extract.get('mc_samples', config.get('Resgen', {}).get('mc_samples', 256)),
                mc_seed=extract.get('mc_seed'))
            base = str(image_path.with_suffix(''))
            extracted = base + '.ec.fits'
            outputs.append(extracted)
            result['extracted'] = extracted
            if isinstance(extraction, dict):
                result['extraction'] = {key: value for key, value in extraction.items()
                                        if key not in {'coeffs', 'output'}}
                if extraction.get('valid_columns', 0) == 0:
                    raise ValueError('All extracted columns are invalid; diagnostic extraction file retained')
                if extraction.get('valid_columns', 0) < extraction.get('total_columns', 0):
                    diagnostics.append(
                        f"Extraction retained {extraction['valid_columns']}/{extraction['total_columns']} columns; "
                        'only valid support will be propagated')
                if extraction.get('negative_residual_columns', 0):
                    diagnostics.append(
                        f"Large negative residuals remain in {extraction['negative_residual_columns']} columns; "
                        'see NEGATIVE_PIXELS and QUALITY=8; no additional calibration defect mask applied')
            stage = 'arc_extraction'
            if not task.get('marc') or not Path(task['marc']).exists():
                raise ValueError('No successful arc master; wavelength correction skipped')
            arc_num = int(task['sequence_id'])
            optimal_extract_fits(task['marc'], *args, coeffs=coeffs, num=arc_num)
            arc_extracted = str(Path(task['marc']).with_suffix('')) + f'-{arc_num:03d}.ec.fits'
            outputs.append(arc_extracted)
            stage = 'wavelength'
            wave = config['Wavecal']
            solution = get_wavelength_solution(arc_extracted, wave['refpath'], wave['order'],
                                               wave['lindisp'], wave['window'], wave['wmin'],
                                               wave['wmax'], wave['prominence'], bool(wave['plotyn']))
            if not solution or solution.get('coefficients') is None:
                raise ValueError('Wavelength fit did not produce coefficients')
            for path in [arc_extracted, extracted]:
                with fits.open(path, mode='update') as hdul:
                    for index, value in enumerate(solution['coefficients'][::-1]):
                        hdul['FLUX_ADU'].header[f'WLCAL{index}'] = float(value)
                    hdul[0].header['WLRMS'] = float(solution['rms_error_A'])
                    hdul[0].header['WLNLINES'] = int(solution['points_used'])
            result['wavelength'] = {key: value for key, value in solution.items()
                                    if key != 'coefficients'}
            stage = 'dispersion'
            dispersion = dispcor(extracted, make_site(config), wave['order'],
                                 coordinates=config.get('Coordinates'),
                                 offline=config.get('Runtime', {}).get('offline', False))
            corrected = base + '.ec_dc.fits'
            if not Path(corrected).exists():
                raise ValueError('Dispersion correction produced no output')
            outputs.append(corrected)
            result['dispcor'] = corrected
            if isinstance(dispersion, dict):
                diagnostics.extend(dispersion.get('warnings', []))
            if not task['is_target']:
                stage = 'response_generation'
                response = generate_response_file(corrected, REFERENCES, config['refpath'],
                                                  config, arc_ec_path=arc_extracted)
                diagnostics.extend(response.get('warnings', []))
                if response.get('status') == 'failed' or not response.get('path'):
                    raise ValueError(response.get('error', 'Response generation failed'))
                outputs.append(response['path'])
                result['response'] = response['path']
                if response.get('template_path'):
                    result['template_files'] = [response['template_path']]
                if response.get('template_sha256'):
                    result['template_sha256'] = response['template_sha256']
            result.update(status='success', stage=stage)
        except Exception as exc:
            result.update(status='failed', stage=stage, error=f'{type(exc).__name__}: {exc}')
        result['outputs'] = outputs
        result['warnings'] = list(dict.fromkeys(diagnostics + warning_text(caught)))
        result['elapsed_seconds'] = time.perf_counter() - start
    return result


def find_library_darks(directory, target_file, trim_i=None, trim_f=None):
    """Choose each exposure's nearest compatible master from actual FITS headers."""
    
    header = fits.getheader(target_file)
    if trim_i is not None and trim_f is not None:
        header['TRIM_Y0'], header['TRIM_Y1'] = int(trim_i), int(trim_f)
    height = int(trim_f)-int(trim_i) if trim_i is not None and trim_f is not None else int(header['NAXIS2'])
    candidates, rejected = library_dark_candidates(directory, header, (height, int(header['NAXIS1'])))
    if not candidates:
        raise ValueError('No compatible library dark: ' + '; '.join(rejected[:5]))
    chosen = {}
    for distance, path, exposure, metadata in candidates:
        chosen.setdefault(exposure, Path(path))
    return chosen


def arc_key(files):
    """Identify a calibration set independently of target-specific output prefixes."""
    paths = '\n'.join(sorted(str(Path(path).resolve()) for path in files))
    return hashlib.sha256(paths.encode('utf-8')).hexdigest()[:12]


def main(obspath, configpath):
    """Reduce an observation directory using a YAML configuration.

    Create a dedicated work run, calibrate and extract science/reference
    frames, fit wavelengths, and produce relative-flux products. Persist
    stage outcomes and provenance throughout processing. Return the final
    run-report dictionary; per-stage failures are recorded for review.
    Configured output collisions are handled by the publication policy rather
    than silently overwriting existing products.
    """
    
    
    
    started = time.perf_counter()
    with open(configpath, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    observations = Path(obspath).resolve()
    run_id, work, products = new_run(config, observations)
    journal = RunJournal(run_id, work, products, config, observations, config_path=configpath)
    expected = 0
    final_outputs = []
    try:
        journal.emit(f'KHAO spectral reduction: {observations}; work={work}')
        collection = ccdp.ImageFileCollection(location=str(observations), glob_include='*',
                                             keywords=['EXPTIME', 'IMAGETYP'])
        # Absolute filenames do not change the dedicated installation's root paths.
        biases = list(collection.files_filtered(imagetyp='Bias Frame', include_path=True))
        dark_files = list(collection.files_filtered(imagetyp='Dark Frame', include_path=True))
        groups = file_sort(observations, TARGETS, REFERENCES)
        write_json(work / 'grouping_summary.json', groups)
        expected = sum(len(group['target_files']) for group in groups)
        if not biases:
            raise ValueError('No bias frames found')
        if not groups:
            raise ValueError('No target frames found; no processing jobs were submitted')
        inputs = set(biases + dark_files)
        for group in groups:
            inputs.update(group['target_files'])
            inputs.update(group['target_arc_files'])
            for reference in group.get('references', []):
                inputs.update(reference['reference_files'])
                inputs.update(reference['reference_arc_files'])
        journal.register_input_files(sorted(inputs))
        journal.flush()
        ccd = config['CCD']
        memory = float(config.get('memory_limit', 32)) * 1e9
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warnings.filterwarnings(
                "ignore", message=r"^'datfix' made the change.*", category=FITSFixedWarning)
            bias = make_bias_master(biases, ccd['datay_start'], ccd['datay_end'],
                                    memlim=memory, gain=ccd['gain'], rdnoise=ccd['rdnoise'],
                                    saturation=ccd.get('saturation'))
            bias_path = str(work / 'mbias.fit')
            bias.write(bias_path, overwrite=False, checksum=True)
            dark_paths = []
            dark_times = []
            incompatible_local_exposures = []
            dark_rejections = []
            if dark_files:
                by_exposure = {}
                for path in dark_files:
                    header = fits.getheader(path)
                    exposure = float(header['EXPTIME'])
                    try:
                        validate_calibration(bias.meta, header, f'local dark {Path(path).name}')
                    except (CalibrationMismatchError, ValueError) as exc:
                        incompatible_local_exposures.append(exposure)
                        dark_rejections.append({'file': str(path), 'exposure': exposure, 'reason': str(exc)})
                        warnings.warn(f'Local dark excluded before master creation: {Path(path).name}: {exc}; '
                                      'matching library master will be sought per science/reference/arc frame',
                                      RuntimeWarning, stacklevel=2)
                        continue
                    by_exposure.setdefault(exposure, []).append(path)
                for exposure, files in sorted(by_exposure.items()):
                    dark = make_dark_master(files, bias, ccd['datay_start'], ccd['datay_end'],
                                            memlim=memory, gain=ccd['gain'], rdnoise=ccd['rdnoise'],
                                            saturation=ccd.get('saturation'))
                    path = work / f'mdark{exposure:g}.fit'
                    dark.meta['DARKFILE'] = str(path)
                    dark.write(path, overwrite=False, checksum=True)
                    dark_paths.append(str(path))
                    dark_times.append(exposure)
                    if config.get('Runtime', {}).get('update_dark_library', True):
                        library = Path(config['darkpath'])
                        library.mkdir(parents=True, exist_ok=True)
                        temp = dark.header.get('CCD-TEMP', dark.header.get('SET-TEMP'))
                        date = str(dark.header.get('DATE-OBS', ''))[:10].replace('-', '')
                        # Round half values away from zero for the filename only.
                        filename_temp = int(Decimal(str(temp)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
                        name = f'Mdark_T{filename_temp}_E{exposure:g}_{date}.fit'
                        destination = library / name
                        dark.write(destination, overwrite=True, checksum=True)
            if not dark_paths:
                warnings.warn('No compatible local dark masters; library selection will be performed '
                              'independently for each science/reference/arc frame', RuntimeWarning, stacklevel=2)
            journal.record({'status': 'success', 'stage': 'global_masters', 'input': str(observations),
                            'outputs': [bias_path] + dark_paths, 'warnings': warning_text(caught),
                            'rejected_local_darks': dark_rejections})
        global_data = {'mbias_path': bias_path, 'mdark_paths': dark_paths,
                       'darkexptimelist': dark_times, 'references': REFERENCES,
                       'dark_library': config.get('darkpath'),
                       'incompatible_local_exposures': sorted(set(incompatible_local_exposures))}
        del bias
        arc_tasks = {}
        frame_tasks = {}
        associations = []
        for group_index, group in enumerate(groups, 1):
            association = {'group': group, 'targets': [], 'references': []}
            blocks = [(f'T{group_index:03d}', True, group['target_files'], group['target_arc_files'])]
            for index, reference in enumerate(group.get('references', []), 1):
                blocks.append((f'T{group_index:03d}_R{index:02d}', False,
                               reference['reference_files'], reference['reference_arc_files']))
            for prefix, is_target, files, arcs in blocks:
                key = arc_key(arcs)
                if key not in arc_tasks:
                    arc_tasks[key] = {'key': key, 'files': arcs, 'output': str(work/f'CAL_{key}_marc.fit'),
                                      **global_data, 'trim_start': ccd['datay_start'],
                                      'trim_end': ccd['datay_end'], 'memlim': memory,
                                      'gain': ccd['gain'], 'rdnoise': ccd['rdnoise'],
                                      'saturation': ccd.get('saturation')}
                ids = []
                for path in files:
                    identity = hashlib.sha256(f'{Path(path).resolve()}|{key}|{is_target}'.encode()).hexdigest()[:16]
                    if identity not in frame_tasks:
                        frame_tasks[identity] = {'task_id': identity, 'file': str(path), 'prefix': prefix,
                                                 'arc_key': key, 'is_target': is_target, 'workpath': str(work),
                                                 'config': config, 'global_data': global_data,
                                                 'sequence_id': len(frame_tasks)+1}
                    ids.append(identity)
                if is_target:
                    association['targets'] = ids
                else:
                    association['references'].append(ids)
            associations.append(association)
        sample_header = fits.getheader(groups[0]['target_files'][0])
        image_bytes = (ccd['datay_end']-ccd['datay_start']) * sample_header['NAXIS1'] * 8
        n_arc = worker_count(config, len(arc_tasks), image_bytes*30)
        for task in arc_tasks.values():
            task['memlim'] = memory / max(n_arc, 1)
        arc_results = {}
        for task, result in completed_jobs(make_group_masters_worker, list(arc_tasks.values()), n_arc):
            arc_results[task['key']] = result
            journal.record(result)
        for task in frame_tasks.values():
            task['marc'] = arc_results[task['arc_key']].get('path')
        frame_bytes = frame_memory_estimate(config, image_bytes, sample_header['NAXIS1'])
        n_frames = worker_count(config, len(frame_tasks), frame_bytes)
        journal.report['frame_memory_estimate_bytes'] = frame_bytes
        journal.report['frame_workers'] = n_frames
        journal.emit(f'Processing {len(frame_tasks)} unique frames with {n_frames} workers')
        frame_results = {}
        for task, result in completed_jobs(process_single_frame_worker, list(frame_tasks.values()), n_frames):
            frame_results[task['task_id']] = result
            journal.record(result)
        for index, association in enumerate(associations):
            group = association['group']
            targets = [frame_results[key].get('dispcor') for key in association['targets']]
            targets = [path for path in targets if path]
            if not targets:
                continue
            try:
                if not association['references']:
                    raise ValueError('No reference group associated with target')
                reference_paths = []
                notes = []
                for keys in association['references']:
                    paths = [frame_results[key].get('response') for key in keys]
                    good = [path for path in paths if path]
                    if not good:
                        raise ValueError('A required reference group has no valid response')
                    if len(good) < len(keys):
                        notes.append(f'Reference group used {len(good)}/{len(keys)} valid responses')
                    reference_paths.append(good)
                obsset = {'target_name': group['target_name'], 'target': targets,
                          'ref_name': group['references'][0]['reference_name'],
                          'ref_front': reference_paths[0],
                          'ref_back': reference_paths[-1]}
                outcome = response_correction(index, obsset, make_site(config), config=config)
                if not isinstance(outcome, dict):
                    raise ValueError('Response correction did not return a structured outcome')
                outputs = outcome.get('outputs', [])
                final_outputs.extend(outputs)
                failed = outcome.get('failed', [])
                event = {'status': 'failed' if failed or not outputs else 'success',
                         'stage': 'response_correction', 'input': group['target_name'],
                         'outputs': outputs, 'warnings': notes + outcome.get('warnings', [])}
                if failed:
                    event['error'] = str(failed)
                journal.record(event)
            except Exception as exc:
                journal.record(failure('response_correction', group['target_name'], exc))
    except Exception as exc:
        journal.record(failure('controller', observations, exc))
    return journal.finish(expected, final_outputs, time.perf_counter()-started)
