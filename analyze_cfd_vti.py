"""
nTop LBM CFD VTI File Analyzer
================================
Reads a .vti (VTK ImageData) file exported from an nTop Lattice-Boltzmann
CFD simulation and reports aerodynamic forces directly from the simulation.

Coordinate system  (body-fixed, matches nTop default)
------------------------------------------------------
  +X  →  Streamwise  (body reference axis, freestream at alpha = 0)
  +Z  ↑  Vertical    (lift direction at alpha = 0)
  +Y  ·  Spanwise    (right-hand rule: Y = Z × X)

Sign convention
---------------
  Drag  : positive force in freestream direction  (opposes forward motion)
  Lift  : positive force perpendicular to freestream, upward
  Side  : positive force in +Y direction

Angle of attack  (nTop convention)
------------------------------------
  Positive alpha rotates the FREESTREAM from +X toward +Z in the XZ plane.
  The body geometry stays fixed; only the inflow velocity direction changes.
  This matches the nTop Aircraft Flow Analysis block setting.

  At alpha degrees, the wind axes in body-frame coordinates are:
    drag axis = [cos(alpha),  0,  sin(alpha)]
    lift axis = [-sin(alpha), 0,  cos(alpha)]

Force field — authoritative source
------------------------------------
  'Force time-averaged' (point data, shape (N_pts, 3), float64, [N/point])
  Written by nTop's IB solver kernel.  Non-zero only where CellRegion == 3
  (~35 k points).  Summing over those points gives the total aerodynamic
  force including both pressure (form) and viscous (friction) contributions.

  Preferred over pressure integration because:
    1. No surface reconstruction needed.
    2. Not affected by nTop's p=0 BC inside the solid body.
    3. IB interface layer is compact and free of domain-wall contamination.

nTop LBM field inventory  (all stored as POINT DATA)
------------------------------------------------------
  ImplicitField          (N,)    float32  signed-distance: <0 fluid, >0 solid
  CellRegion             (N,)    uint8    0=outer, 1=fluid, 3=IB, 4/5=BC, 6=near-body
  Pressure time-averaged (N,)    float64  [Pa]
  Velocity time-averaged (N, 3)  float64  [m/s]
  Force time-averaged    (N, 3)  float64  [N per grid point]

  Domain walls are also solid in nTop's IB formulation, so the naive
  ImplicitField=0 iso-surface spans the full domain (~20 m²).  The body
  surface is recovered by clipping to the bounding box of CellRegion==3.

Usage
-----
  Edit VTI_FILE and ALPHA_DEG in the configuration block below, then run:
    python analyze_cfd_vti.py
"""

import sys
import numpy as np
import pyvista as pv
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
#  USER CONFIGURATION — edit these values before running
# ──────────────────────────────────────────────────────────────────────────────

VTI_FILE  = "CFD ATAS 0.1 at 2.5 deg Implicit.vti"  # path to the .vti file
ALPHA_DEG = 2.5                                         # angle of attack [deg]
                                                        # positive = freestream
                                                        # tilts from +X toward +Z

# Reference parameters for nondimensionalisation
V_INF  = 30.48           # freestream speed [m/s]
RHO    = 1.2250123       # air density [kg/m³]
S_REF  = 0.4607990784    # reference wing area [m²]
T_INF  = 288.15          # freestream temperature [K]
MACH   = 0.0897          # freestream Mach number

# Body-axis reference directions (matches nTop default coordinate system)
DRAG_DIRECTION = np.array([1.0, 0.0, 0.0])   # body +X — streamwise reference
LIFT_DIRECTION = np.array([0.0, 0.0, 1.0])   # body +Z — vertical reference

# Field-name candidates (exact match tried first, then case-insensitive substring)
_FORCE_CANDIDATES = [
    "Force time-averaged", "Force Time-Averaged",
    "Force initial",       "Force",
    "bodyForce",
]
_VELOCITY_CANDIDATES = [
    "Velocity time-averaged", "Velocity Time-Averaged",
    "Velocity initial",       "Velocity",
    "U", "u", "UMean",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_field(fields: dict, candidates: list) -> tuple:
    """Return (name, array) for the first matching candidate, else (None, None)."""
    for name in candidates:
        if name in fields:
            return name, fields[name]
    for name in candidates:
        needle = name.lower()
        for fname in fields:
            if needle in fname.lower():
                print(f"  [fuzzy] Matched '{fname}' via candidate '{name}'")
                return fname, fields[fname]
    return None, None


def _unit(v: np.ndarray) -> np.ndarray:
    """Return normalised vector; raise ValueError if magnitude is near zero."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Direction vector has near-zero magnitude: {v}")
    return v / n


def _orthogonalise(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: remove component of v along ref, then normalise."""
    return _unit(v - np.dot(v, ref) * ref)


def _aero_axes(drag_ref: np.ndarray,
               lift_ref: np.ndarray,
               alpha_deg: float) -> tuple:
    """
    Rotate body-axis references into wind (aerodynamic) axes.

    nTop convention: positive alpha tilts the freestream from +X toward +Z.
    Equivalent to applying Ry(-alpha) to the body axes:

      drag_aero = cos(alpha) * drag_ref + sin(alpha) * lift_ref
      lift_aero = -sin(alpha) * drag_ref + cos(alpha) * lift_ref

    Verification at alpha=10 deg (drag_ref=+X, lift_ref=+Z):
      drag_aero = [+0.985, 0, +0.174]   (tilted toward +Z)
      lift_aero = [-0.174, 0, +0.985]   (tilted toward -X)
    """
    d = _unit(drag_ref.copy())
    l = _unit(lift_ref.copy())
    if alpha_deg == 0.0:
        return d, _orthogonalise(l, d)
    a = np.radians(alpha_deg)
    return _unit(np.cos(a) * d + np.sin(a) * l), \
           _unit(-np.sin(a) * d + np.cos(a) * l)


# ──────────────────────────────────────────────────────────────────────────────
#  1.  load_vti
# ──────────────────────────────────────────────────────────────────────────────

def load_vti(filepath: str) -> pv.ImageData:
    """
    Read a .vti file with PyVista and print a mesh summary.

    Returns
    -------
    pv.ImageData
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() not in (".vti", ".vtk"):
        print(f"  [warn] Unexpected extension '{path.suffix}' — expected .vti")

    print(f"\n{'='*64}")
    print(f"  FILE       : {path.resolve()}")
    print(f"{'='*64}")

    mesh = pv.read(str(path))
    b    = np.array(mesh.bounds).reshape(3, 2)

    print(f"  Type       : {type(mesh).__name__}")
    if hasattr(mesh, "dimensions"):
        print(f"  Dimensions : {mesh.dimensions}  (nx, ny, nz)")
    if hasattr(mesh, "spacing"):
        sx, sy, sz = mesh.spacing
        print(f"  Spacing    : ({sx:.5g}, {sy:.5g}, {sz:.5g}) m")
        print(f"  Cell vol   : {sx*sy*sz:.5g} m³")
    print(f"  Domain X   : [{b[0,0]:.4g}, {b[0,1]:.4g}] m  "
          f"span {b[0,1]-b[0,0]:.4g} m")
    print(f"  Domain Y   : [{b[1,0]:.4g}, {b[1,1]:.4g}] m  "
          f"span {b[1,1]-b[1,0]:.4g} m")
    print(f"  Domain Z   : [{b[2,0]:.4g}, {b[2,1]:.4g}] m  "
          f"span {b[2,1]-b[2,0]:.4g} m")
    print(f"  N points   : {mesh.n_points:,}")
    print(f"  N cells    : {mesh.n_cells:,}")
    return mesh


# ──────────────────────────────────────────────────────────────────────────────
#  2.  extract_fields
# ──────────────────────────────────────────────────────────────────────────────

def extract_fields(mesh: pv.ImageData) -> dict:
    """
    Collect every data array from the VTI mesh into a flat dict of NumPy arrays.

    For nTop LBM files all data lives in mesh.point_data.
    Cell data and field data containers are empty.

    Returns
    -------
    dict  { field_name: np.ndarray }
        Scalars shape (N,), vectors shape (N, 3).
    """

    def _inspect(container: pv.DataSetAttributes) -> dict:
        out = {}
        if len(container) == 0:
            print("    (none)")
            return out
        for name in container.keys():
            arr  = np.asarray(container[name])
            flat = arr.ravel()
            fin  = flat[np.isfinite(flat)]
            rng  = f"[{fin.min():.5g}, {fin.max():.5g}]" if fin.size else "all NaN/Inf"
            nnan = int((~np.isfinite(flat)).sum())
            kind = "scalar" if arr.ndim == 1 else f"vec{arr.shape[1]}"
            nan_tag = f"  <- {nnan} NaN/Inf" if nnan else ""
            print(f"    {name:<40s}  {kind:<8s}  shape={str(arr.shape):<20s}"
                  f"  dtype={str(arr.dtype):<10s}  range={rng}{nan_tag}")
            out[name] = arr
        return out

    all_fields: dict = {}

    print(f"\n{'─'*64}")
    print("  POINT DATA")
    print(f"{'─'*64}")
    all_fields.update(_inspect(mesh.point_data))

    print(f"\n{'─'*64}")
    print("  CELL DATA")
    print(f"{'─'*64}")
    all_fields.update(_inspect(mesh.cell_data))

    print(f"\n{'─'*64}")
    print("  FIELD / METADATA")
    print(f"{'─'*64}")
    all_fields.update(_inspect(mesh.field_data))

    # ── Key field summary ─────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  KEY FIELD SUMMARY (nTop LBM)")
    print(f"  {'─'*60}")

    impl = all_fields.get("ImplicitField")
    if impl is not None:
        print(f"  ImplicitField  ->  "
              f"fluid (<0): {int((impl < 0).sum()):,}  "
              f"wall (~0): {int((np.abs(impl) < 0.007).sum()):,}  "
              f"solid (>0): {int((impl > 0).sum()):,}")

    creg = all_fields.get("CellRegion")
    if creg is not None:
        region_label = {0: "outer BC", 1: "fluid", 3: "IB interface",
                        4: "inlet/outlet", 5: "inlet/outlet", 6: "near-body fluid"}
        for v in np.unique(creg.astype(int)):
            print(f"  CellRegion {v}   ->  "
                  f"{int((creg == v).sum()):>12,} pts  "
                  f"({region_label.get(v, 'unknown')})")

    v_name, v_arr = _find_field(all_fields, _VELOCITY_CANDIDATES)
    f_name, f_arr = _find_field(all_fields, _FORCE_CANDIDATES)

    if v_name and v_arr is not None and v_arr.ndim == 2:
        vmag = np.linalg.norm(v_arr, axis=1)
        print(f"  Velocity       : '{v_name}'  "
              f"|v| range [{vmag.min():.4g}, {vmag.max():.4g}] m/s")

    if f_name and f_arr is not None and f_arr.ndim == 2:
        fmag = np.linalg.norm(f_arr, axis=1)
        print(f"  Force          : '{f_name}'  "
              f"non-zero pts: {int((fmag > 1e-8).sum()):,}")

    print(f"\n  All field keys : {list(all_fields.keys())}")
    return all_fields


# ──────────────────────────────────────────────────────────────────────────────
#  3.  compute_surface
# ──────────────────────────────────────────────────────────────────────────────

def compute_surface(mesh: pv.ImageData, fields: dict) -> pv.PolyData:
    """
    Reconstruct the body-only surface from the volume data.

    Strategy
    --------
    1. Locate the body via CellRegion==3 points (IB interface layer).
       These exist only around the body, not at domain walls.
    2. Build a tight bounding box around those points.
    3. Extract ImplicitField=0 iso-surface (marching cubes) over the full domain.
    4. Clip to the bounding box to remove domain-wall triangles.
    5. Compute outward cell normals; validate and flip if inverted.

    Returns
    -------
    pv.PolyData
        Triangulated body surface with cell_data["Normals"] and ["Area"],
        and field_data["total_area_m2"].
    """
    print(f"\n{'─'*64}")
    print("  SURFACE EXTRACTION  (nTop body-only method)")
    print(f"{'─'*64}")

    pts     = mesh.points
    impl    = fields.get("ImplicitField")
    creg    = fields.get("CellRegion")
    _, farr = _find_field(fields, _FORCE_CANDIDATES)

    # ── Locate body ───────────────────────────────────────────────────────
    body_mask = None
    if creg is not None:
        m = np.asarray(creg, dtype=int) == 3
        if m.sum() > 0:
            body_mask = m
            print(f"  Body via CellRegion==3       : {m.sum():,} pts")

    if body_mask is None and farr is not None and farr.ndim == 2:
        m = np.linalg.norm(farr, axis=1) > 1e-8
        body_mask = m
        print(f"  Body via |Force|>0           : {m.sum():,} pts")

    if body_mask is None and impl is not None:
        body_mask = np.asarray(impl) > 0
        print(f"  Body via ImplicitField>0     : {body_mask.sum():,} pts")

    if body_mask is None or body_mask.sum() == 0:
        raise RuntimeError("Cannot locate body. Ensure ImplicitField or CellRegion is present.")

    # ── Bounding box + margin ─────────────────────────────────────────────
    bp     = pts[body_mask]
    margin = 3.0 * max(getattr(mesh, "spacing", (0.01, 0.01, 0.01)))
    bb     = [bp[:,0].min()-margin, bp[:,0].max()+margin,
              bp[:,1].min()-margin, bp[:,1].max()+margin,
              bp[:,2].min()-margin, bp[:,2].max()+margin]
    print(f"  Body bounding box            : "
          f"X=[{bb[0]:.4g},{bb[1]:.4g}]  "
          f"Y=[{bb[2]:.4g},{bb[3]:.4g}]  "
          f"Z=[{bb[4]:.4g},{bb[5]:.4g}] m")

    # ── Iso-surface + clip ────────────────────────────────────────────────
    surface = None
    if impl is not None:
        print("  Marching cubes on ImplicitField=0 ...")
        try:
            full = mesh.contour([0.0], scalars="ImplicitField")
            print(f"  Full iso-surface             : {full.n_cells:,} triangles"
                  f" (includes domain walls)")
            clipped = full.clip_box(bb, invert=False)
            if not isinstance(clipped, pv.PolyData):
                clipped = clipped.extract_surface(algorithm="dataset_surface").triangulate()
            else:
                clipped = clipped.triangulate()
            if clipped.n_cells > 0:
                surface = clipped
                print(f"  Body-only surface (clipped)  : "
                      f"{surface.n_cells:,} triangles  {surface.n_points:,} pts")
            else:
                print("  [warn] Clipped iso-surface empty — falling back.")
        except Exception as e:
            print(f"  [warn] Iso-surface failed: {e}")

    # ── Fallback: voxelised surface ───────────────────────────────────────
    if surface is None or surface.n_cells == 0:
        print("  Fallback: voxelised surface from cell threshold ...")
        try:
            mc   = mesh.point_data_to_cell_data()
            sol  = mc.threshold(0.0, scalars="ImplicitField", preference="cell", invert=False)
            clip = sol.clip_box(bb, invert=False)
            surface = clip.extract_surface(algorithm="dataset_surface").triangulate()
            print(f"  Voxelised surface            : {surface.n_cells:,} faces")
        except Exception as e:
            raise RuntimeError(f"Surface extraction failed: {e}") from e

    # ── Normals and areas ─────────────────────────────────────────────────
    surface = surface.compute_normals(cell_normals=True, point_normals=False,
                                      consistent_normals=True,
                                      auto_orient_normals=True,
                                      flip_normals=False)
    surface = surface.compute_cell_sizes(length=False, area=True, volume=False)

    normals    = np.asarray(surface.cell_data["Normals"], dtype=float)
    areas      = np.asarray(surface.cell_data["Area"],    dtype=float)
    total_area = float(areas.sum())
    mean_n     = normals.mean(axis=0)
    b_s        = np.array(surface.bounds).reshape(3, 2)

    print(f"\n  Surface bounds               : "
          f"X=[{b_s[0,0]:.4g},{b_s[0,1]:.4g}]  "
          f"Y=[{b_s[1,0]:.4g},{b_s[1,1]:.4g}]  "
          f"Z=[{b_s[2,0]:.4g},{b_s[2,1]:.4g}] m")
    print(f"  Total surface area           : {total_area:.5g} m²")
    print(f"  Mean cell normal             : [{mean_n[0]:+.4g}, {mean_n[1]:+.4g},"
          f" {mean_n[2]:+.4g}]  (near zero for closed surface)")

    # ── Validate outward orientation ──────────────────────────────────────
    centroid     = bp.mean(axis=0)
    cc           = np.asarray(surface.cell_centers().points)
    radial       = cc - centroid
    radial      /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    align        = (normals * radial).sum(axis=1)
    frac_out     = float((align > 0).mean())
    print(f"\n  Normal orientation check     : {100*frac_out:.1f}% outward-facing")
    if frac_out < 0.5:
        print("  [warn] Majority inward — flipping all normals.")
        surface.cell_data["Normals"] = -normals
    elif frac_out < 0.8:
        print("  [warn] >20% may be inverted — surface reconstruction may be incomplete.")
    else:
        print("  Normals confirmed outward.")

    surface.field_data["total_area_m2"] = np.array([total_area])
    return surface


# ──────────────────────────────────────────────────────────────────────────────
#  4.  compute_forces
# ──────────────────────────────────────────────────────────────────────────────

def compute_forces(mesh:      pv.ImageData,
                   fields:    dict,
                   alpha_deg: float,
                   drag_dir:  np.ndarray = DRAG_DIRECTION,
                   lift_dir:  np.ndarray = LIFT_DIRECTION,
                   v_inf_ref: float = V_INF,
                   rho:       float = RHO,
                   s_ref:     float = S_REF) -> dict:
    """
    Sum 'Force time-averaged' over the IB interface layer to get aerodynamic forces.

    Forces in the VTI are in body-fixed coordinates.  alpha_deg rotates the
    projection axes into wind axes so that drag is along the freestream and
    lift is perpendicular to it.

    IB detection
    ------------
    Primary  : |Force| > 0.1% of max value.
    Fallback : CellRegion == 3 (cross-checked; warning if >20% disagreement).

    Parameters
    ----------
    mesh      : pv.ImageData
    fields    : dict from extract_fields()
    alpha_deg : float  angle of attack [deg]; positive = freestream toward +Z
    drag_dir  : (3,) body-axis drag reference  (default +X)
    lift_dir  : (3,) body-axis lift reference  (default +Z)

    Returns
    -------
    dict
        total_force_N, drag_N, lift_N, span_N, n_ib_pts,
        force_field_name, v_inf_ms, n_upstream_pts,
        alpha_deg, drag_axis, lift_axis,
        q_inf_pa, CD, CL, LD
    """
    print(f"\n{'─'*64}")
    print("  FORCE COMPUTATION")
    print(f"{'─'*64}")

    # ── Wind axes ─────────────────────────────────────────────────────────
    drag_dir, lift_dir = _aero_axes(drag_dir, lift_dir, alpha_deg)
    span_dir           = np.cross(drag_dir, lift_dir)

    print(f"  Alpha (AoA)    : {alpha_deg:+.4f} deg")
    print(f"  Drag axis      : [{drag_dir[0]:+.6f}, {drag_dir[1]:+.6f}, {drag_dir[2]:+.6f}]")
    print(f"  Lift axis      : [{lift_dir[0]:+.6f}, {lift_dir[1]:+.6f}, {lift_dir[2]:+.6f}]")
    print(f"  Span axis      : [{span_dir[0]:+.6f}, {span_dir[1]:+.6f}, {span_dir[2]:+.6f}]")

    # ── Freestream velocity estimate (context only) ───────────────────────
    print(f"\n  {'─'*56}")
    print("  FREESTREAM VELOCITY")
    print(f"  {'─'*56}")

    v_name, v_vol = _find_field(fields, _VELOCITY_CANDIDATES)
    v_inf      = 0.0
    n_upstream = 0

    if v_name and v_vol is not None and v_vol.ndim == 2:
        impl_arr   = fields.get("ImplicitField")
        pts_vol    = mesh.points
        bounds     = mesh.bounds
        fluid_mask = (np.asarray(impl_arr) < -0.006
                      if impl_arr is not None
                      else np.ones(mesh.n_points, dtype=bool))
        x_cut      = bounds[0] + 0.15 * (bounds[1] - bounds[0])
        upstream   = fluid_mask & (pts_vol[:, 0] <= x_cut)
        if upstream.sum() < 100:
            upstream = fluid_mask
        n_upstream = int(upstream.sum())
        v_inf      = float(np.mean(v_vol[upstream] @ drag_dir))
        print(f"  Field          : '{v_name}'")
        print(f"  Upstream pts   : {n_upstream:,}")
        print(f"  V∞ (drag dir)  : {v_inf:.4f} m/s")
    else:
        print("  [warn] No velocity field found — V∞ not reported.")

    # ── Force field ───────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print("  AERODYNAMIC FORCES")
    print(f"  {'─'*56}")

    f_name, f_arr = _find_field(fields, _FORCE_CANDIDATES)
    zeros = np.zeros(3)

    if f_name is None or f_arr is None or f_arr.ndim != 2:
        print("  [error] No vector Force field found.")
        print(f"  Available keys : {list(fields.keys())}")
        return dict(total_force_N=zeros, drag_N=0., lift_N=0., span_N=0.,
                    n_ib_pts=0, force_field_name="", v_inf_ms=v_inf,
                    n_upstream_pts=n_upstream, alpha_deg=alpha_deg,
                    drag_axis=drag_dir, lift_axis=lift_dir)

    print(f"  Field          : '{f_name}'")

    # Cell volume for unit-check printout
    cell_vol = float(np.prod(mesh.spacing)) if hasattr(mesh, "spacing") else None
    if cell_vol:
        print(f"  Cell volume    : {cell_vol:.5g} m³")

    # ── IB detection: force-magnitude mask (primary) ──────────────────────
    fmag       = np.linalg.norm(f_arr, axis=1)
    thresh     = max(1e-8, 1e-3 * float(fmag.max()))
    force_mask = fmag > thresh
    print(f"  |Force|>{thresh:.3g} pts : {int(force_mask.sum()):,}")

    # ── Cross-check with CellRegion==3 ────────────────────────────────────
    creg_arr   = fields.get("CellRegion")
    iface_mask = force_mask

    if creg_arr is not None:
        mask3        = np.asarray(creg_arr, dtype=int) == 3
        n_creg3      = int(mask3.sum())
        n_overlap    = int((force_mask & mask3).sum())
        overlap_frac = n_overlap / max(n_creg3, 1)
        print(f"  CellRegion==3 pts          : {n_creg3:,}")
        print(f"  Overlap                    : {n_overlap:,}  ({100*overlap_frac:.1f}%)")
        if overlap_frac < 0.80:
            print("  [warn] Masks differ >20% — using force-magnitude mask.")
            print("  [warn] Verify CellRegion labelling for this nTop version.")
        else:
            iface_mask = mask3
            print("  CellRegion==3 consistent — using it.")

    # ── Unit check ────────────────────────────────────────────────────────
    if cell_vol:
        pcmean = float(fmag[iface_mask].mean())
        print(f"\n  Mean |Force| per pt : {pcmean:.5g} N/pt")
        print(f"  Equiv. density     : {pcmean/cell_vol:.5g} N/m³")
        print(f"  (Summing is correct if field is N/pt; "
              f"multiply by {cell_vol:.5g} m³ if N/m³.)")

    # ── Sum ───────────────────────────────────────────────────────────────
    F_pts       = f_arr[iface_mask]
    total_force = F_pts.sum(axis=0)
    n_ib        = int(iface_mask.sum())

    drag = float(np.dot(total_force, drag_dir))
    lift = float(np.dot(total_force, lift_dir))
    span = float(np.dot(total_force, span_dir))

    print(f"\n  IB interface pts    : {n_ib:,}")
    print(f"  Total force (body)  : [{total_force[0]:+.5g}, "
          f"{total_force[1]:+.5g}, {total_force[2]:+.5g}] N")
    print(f"  Drag (freestream)   : {drag:+.5g} N")
    print(f"  Lift (perpendicular): {lift:+.5g} N")
    print(f"  Side (spanwise)     : {span:+.5g} N")

    drag_pc = F_pts @ drag_dir
    lift_pc = F_pts @ lift_dir
    print(f"  Per-pt drag range   : [{drag_pc.min():.4g}, {drag_pc.max():.4g}] N")
    print(f"  Per-pt lift range   : [{lift_pc.min():.4g}, {lift_pc.max():.4g}] N")

    # ── Nondimensionalisation ─────────────────────────────────────────────
    q_inf = 0.5 * rho * v_inf_ref**2       # dynamic pressure [Pa]
    qS    = q_inf * s_ref                  # [N]
    CD    = drag / qS
    CL    = lift / qS
    LD    = lift / drag if abs(drag) > 1e-12 else float("nan")

    return dict(total_force_N=total_force,
                drag_N=drag, lift_N=lift, span_N=span,
                n_ib_pts=n_ib, force_field_name=f_name,
                v_inf_ms=v_inf, n_upstream_pts=n_upstream,
                alpha_deg=alpha_deg, drag_axis=drag_dir, lift_axis=lift_dir,
                q_inf_pa=q_inf, CD=CD, CL=CL, LD=LD)


# ──────────────────────────────────────────────────────────────────────────────
#  5.  print_summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print the final aerodynamic force summary table."""

    print(f"\n{'='*64}")
    print("  AERODYNAMIC FORCE SUMMARY")
    print(f"{'='*64}")

    alpha     = results.get("alpha_deg", 0.0)
    v_inf     = results.get("v_inf_ms",  0.0)
    total     = results.get("total_force_N", np.zeros(3))
    drag      = results.get("drag_N",  0.0)
    lift      = results.get("lift_N",  0.0)
    span      = results.get("span_N",  0.0)
    n_ib      = results.get("n_ib_pts", 0)
    ff        = results.get("force_field_name", "N/A")
    drag_axis = results.get("drag_axis", np.array([1., 0., 0.]))
    lift_axis = results.get("lift_axis", np.array([0., 0., 1.]))
    q_inf     = results.get("q_inf_pa", float("nan"))
    CD        = results.get("CD",       float("nan"))
    CL        = results.get("CL",       float("nan"))
    LD        = results.get("LD",       float("nan"))

    print(f"  Alpha (AoA)         : {alpha:+.4f} deg")
    if v_inf:
        print(f"  Freestream V∞       : {v_inf:.4f} m/s  (along drag axis)")
    print(f"  Dynamic pressure q∞ : {q_inf:.4f} Pa")

    print(f"\n  Body-frame force    : [{total[0]:+.5g}, "
          f"{total[1]:+.5g}, {total[2]:+.5g}] N")

    print(f"\n  {'Component':<26s}  {'Force [N]':>12s}  {'Coefficient':>12s}")
    print(f"  {'─'*26}  {'─'*12}  {'─'*12}")
    print(f"  {'Drag  (freestream)':<26s}  {drag:>+12.4f}  CD = {CD:>+.6f}")
    print(f"  {'Lift  (perpendicular)':<26s}  {lift:>+12.4f}  CL = {CL:>+.6f}")
    print(f"  {'Side  (spanwise)':<26s}  {span:>+12.4f}  (no coeff)")

    print(f"\n  L/D                 : {LD:.4f}  (CL/CD = {CL:.6f} / {CD:.6f})")

    print(f"\n  Source              : '{ff}'  ({n_ib:,} IB pts summed)")

    print(f"\n  WIND AXES USED")
    print(f"  Drag : [{drag_axis[0]:+.6f}, {drag_axis[1]:+.6f}, {drag_axis[2]:+.6f}]")
    print(f"  Lift : [{lift_axis[0]:+.6f}, {lift_axis[1]:+.6f}, {lift_axis[2]:+.6f}]")

    print(f"\n  SIGN CONVENTION")
    print(f"  Positive drag  = force along freestream (opposes motion)")
    print(f"  Positive lift  = force perpendicular to freestream, upward")
    print(f"  Positive side  = force in +Y direction")

    print(f"\n  METHOD")
    print(f"  Sums 'Force time-averaged' (IB solver, N/pt) over CellRegion==3.")
    print(f"  Includes both pressure (form) and viscous (friction) contributions.")
    print(f"  Forces stored in body-fixed frame; projected onto wind axes via alpha.")
    print(f"{'='*64}")


# ──────────────────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Run the full analysis pipeline using VTI_FILE and ALPHA_DEG from config."""

    print(f"\n  VTI file  : {VTI_FILE}")
    print(f"  Alpha     : {ALPHA_DEG:+.4f} deg")

    mesh    = load_vti(VTI_FILE)
    fields  = extract_fields(mesh)

    if not fields:
        print("\n  No data arrays found. Open in ParaView to verify file contents.")
        sys.exit(1)

    surface = compute_surface(mesh, fields)
    results = compute_forces(mesh, fields, ALPHA_DEG,
                             v_inf_ref=V_INF, rho=RHO, s_ref=S_REF)
    print_summary(results)

    return mesh, fields, surface, results


if __name__ == "__main__":
    main()
