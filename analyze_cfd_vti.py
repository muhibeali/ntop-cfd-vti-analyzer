"""
nTop LBM CFD VTI File Analyzer
================================
Reads a .vti (VTK ImageData) file exported from an nTop Lattice-Boltzmann
CFD simulation, extracts every physical field, reconstructs the body surface,
and computes aerodynamic lift and drag forces.

Coordinate system (defaults — edit DRAG_DIRECTION / LIFT_DIRECTION to change):
  +X  →  Streamwise / drag  (freestream flows in +X direction)
  +Z  ↑  Vertical   / lift
  +Y  ·  Spanwise

nTop LBM specifics
------------------
nTop uses an immersed-boundary (IB) method on a Cartesian grid:

  ImplicitField
      Signed-distance field of the geometry.
      < 0  →  fluid domain (outside body)
      = 0  →  body wall
      > 0  →  inside solid body

  CellRegion
      Region label for each grid cell.
      0      : outer boundary
      1      : main fluid domain
      3      : IB fluid-solid interface layer  ← Force field is non-zero here
      4, 5   : inflow / outflow boundary patches
      6      : near-body fluid layer

  Force time-averaged (N per cell)
      Aerodynamic force per grid cell stored by the IB solver.
      Non-zero only in CellRegion == 3 (~36 k cells out of 26 M).
      Summing this field gives the total aerodynamic force (Method A).

  The outer domain walls are also treated as solid boundaries, so a naive
  ImplicitField = 0 contour spans the FULL domain bounding box (≈ 20 m²).
  The correct body surface is obtained by clipping that contour to the
  bounding box of the CellRegion == 3 cells (≈ 0.96 m²).

  Inside the solid, nTop sets pressure = 0.  Pressure sampled at surface
  nodes averages fluid and solid values, underestimating surface pressure.
  Method A (Force field) is immune to this artefact.

Force methods
-------------
  Method A — Force-field integration (PRIMARY, recommended)
      F = Σ  Force_time-averaged_i       over CellRegion == 3 cells
      Captures both pressure (form) and viscous (friction) forces.

  Method B — Pressure integration (SECONDARY, form drag only)
      F = −∫ (p − p∞) · n̂ dA           over the body surface
      Uses the velocity field to derive q∞ → CL, CD.
      Results are lower than Method A because nTop's p=0 BC inside
      the solid reduces the sampled surface pressure.

Usage:
  python analyze_cfd_vti.py [path/to/file.vti]
  Or set VTI_FILE inside main() below.
"""

import sys
import numpy as np
import pyvista as pv
from pathlib import Path

# Encode output as UTF-8 so special characters render on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
#  Global configuration  (edit these before running)
# ──────────────────────────────────────────────────────────────────────────────

# Freestream (drag) direction unit vector
DRAG_DIRECTION = np.array([1.0, 0.0, 0.0])   # +X

# Lift direction unit vector (must be perpendicular to DRAG_DIRECTION)
LIFT_DIRECTION = np.array([0.0, 0.0, 1.0])   # +Z

# Fluid properties
FLUID_DENSITY = 1.225     # kg/m³  (sea-level standard air)

# Field-name candidates searched in order for auto-detection.
# Exact match is tried first, then case-insensitive substring fallback.
_FORCE_CANDIDATES = [
    "Force time-averaged", "Force Time-Averaged",
    "Force initial",       "Force",
    "bodyForce",
]
_PRESSURE_CANDIDATES = [
    "Pressure time-averaged", "Pressure Time-Averaged",
    "Pressure initial",       "Pressure",
    "p", "P", "pMean",        "p_rgh", "StaticPressure",
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
    """
    Return (name, array) for the first candidate found in *fields*.
    Search order: exact match → case-insensitive substring.
    Returns (None, None) if no match is found.
    """
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
    """Return unit vector, raise if near-zero."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Direction vector {v} has near-zero magnitude.")
    return v / n


def _orthogonalise(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Remove the component of *v* along *ref* (Gram-Schmidt) and normalise."""
    v = v - np.dot(v, ref) * ref
    return _unit(v)


# ──────────────────────────────────────────────────────────────────────────────
#  1.  load_vti()
# ──────────────────────────────────────────────────────────────────────────────

def load_vti(filepath: str) -> pv.ImageData:
    """
    Load a VTK ImageData (.vti) file and print a concise mesh summary.

    Parameters
    ----------
    filepath : str
        Path to the .vti file.

    Returns
    -------
    pv.ImageData
        The loaded PyVista dataset.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() not in (".vti", ".vtk"):
        print(f"  [warn] Expected .vti extension, got '{path.suffix}'")

    print(f"\n{'='*64}")
    print(f"  FILE       : {path.resolve()}")
    print(f"{'='*64}")

    mesh = pv.read(str(path))

    b = np.array(mesh.bounds).reshape(3, 2)
    print(f"  Type       : {type(mesh).__name__}")
    if hasattr(mesh, "dimensions"):
        print(f"  Dimensions : {mesh.dimensions}  (nx, ny, nz)")
    if hasattr(mesh, "spacing"):
        sx, sy, sz = mesh.spacing
        print(f"  Spacing    : ({sx:.5g}, {sy:.5g}, {sz:.5g}) m")
        cell_vol = sx * sy * sz
        print(f"  Cell volume: {cell_vol:.5g} m³")
    print(f"  Domain X   : [{b[0,0]:.5g}, {b[0,1]:.5g}] m  "
          f"(span {b[0,1]-b[0,0]:.4g} m)")
    print(f"  Domain Y   : [{b[1,0]:.5g}, {b[1,1]:.5g}] m  "
          f"(span {b[1,1]-b[1,0]:.4g} m)")
    print(f"  Domain Z   : [{b[2,0]:.5g}, {b[2,1]:.5g}] m  "
          f"(span {b[2,1]-b[2,0]:.4g} m)")
    print(f"  N points   : {mesh.n_points:,}")
    print(f"  N cells    : {mesh.n_cells:,}")

    return mesh


# ──────────────────────────────────────────────────────────────────────────────
#  2.  extract_fields()
# ──────────────────────────────────────────────────────────────────────────────

def extract_fields(mesh: pv.ImageData) -> dict:
    """
    Collect every data array from the VTI mesh into NumPy arrays.

    Prints a formatted table for each data container (point data, cell data,
    field/metadata) showing:  name · kind (scalar/vector) · shape · dtype ·
    value range · NaN/Inf count.

    Parameters
    ----------
    mesh : pv.ImageData
        The loaded VTI dataset.

    Returns
    -------
    dict
        Flat dictionary  { field_name: numpy_array }  covering all containers.
        Scalars have shape (N,), vectors shape (N, k).

    Notes for nTop LBM files
    ------------------------
    Typical fields present:
      ImplicitField          – signed-distance function (fluid < 0, solid > 0)
      CellRegion             – region label (0–6, see module docstring)
      Pressure initial       – pressure at t = 0
      Pressure time-averaged – time-averaged pressure (use this for forces)
      Velocity initial       – velocity at t = 0
      Velocity time-averaged – time-averaged velocity
      Force time-averaged    – aerodynamic body force per cell (N) [IB method]
    """

    def _inspect(container: pv.DataSetAttributes, label: str) -> dict:
        """Print stats for one data container and return its numpy arrays."""
        out: dict = {}
        if len(container) == 0:
            print(f"    (none)")
            return out
        for name in container.keys():
            arr  = np.array(container[name])
            flat = arr.ravel()
            fin  = flat[np.isfinite(flat)]
            if fin.size:
                v_rng = f"[{fin.min():.5g}, {fin.max():.5g}]"
            else:
                v_rng = "all NaN/Inf"
            nnan = int(np.sum(~np.isfinite(flat)))
            kind = "scalar" if arr.ndim == 1 else f"vec{arr.shape[1]}"
            nan_tag = f"  ← {nnan} NaN/Inf" if nnan else ""
            print(f"    {name:<40s}  {kind:<8s}  "
                  f"shape={str(arr.shape):<20s}  "
                  f"dtype={str(arr.dtype):<10s}  "
                  f"range={v_rng}{nan_tag}")
            out[name] = arr
        return out

    all_fields: dict = {}

    # ── Point data ─────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  POINT DATA  (one value per mesh node)")
    print(f"{'─'*64}")
    all_fields.update(_inspect(mesh.point_data, "point"))

    # ── Cell data ──────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  CELL DATA   (one value per grid cell)")
    print(f"{'─'*64}")
    all_fields.update(_inspect(mesh.cell_data, "cell"))

    # ── Field metadata ─────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  FIELD / METADATA")
    print(f"{'─'*64}")
    all_fields.update(_inspect(mesh.field_data, "field"))

    # ── Quick summary of key nTop fields ──────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  KEY FIELD SUMMARY (nTop LBM)")
    print(f"  {'─'*60}")

    impl = all_fields.get("ImplicitField")
    if impl is not None:
        n_fluid  = int((impl < 0).sum())
        n_wall   = int((np.abs(impl) < 0.007).sum())
        n_solid  = int((impl > 0).sum())
        print(f"  ImplicitField  →  fluid (< 0): {n_fluid:,}  "
              f"wall (≈ 0): {n_wall:,}  solid (> 0): {n_solid:,}")

    creg = all_fields.get("CellRegion")
    if creg is not None:
        unique_vals = np.unique(creg.astype(int))
        region_desc = {0: "outer BC", 1: "fluid", 3: "IB interface",
                       4: "inlet/outlet", 5: "inlet/outlet", 6: "near-body fluid"}
        for v in unique_vals:
            desc = region_desc.get(v, "unknown")
            print(f"  CellRegion {v}   →  {(creg == v).sum():>12,} pts  ({desc})")

    # Detect time-averaged fields
    p_name, p_arr = _find_field(all_fields, _PRESSURE_CANDIDATES)
    v_name, v_arr = _find_field(all_fields, _VELOCITY_CANDIDATES)
    f_name, f_arr = _find_field(all_fields, _FORCE_CANDIDATES)

    if p_name:
        print(f"  Pressure field : '{p_name}'")
    if v_name and v_arr is not None and v_arr.ndim == 2:
        vmag = np.linalg.norm(v_arr, axis=1)
        print(f"  Velocity field : '{v_name}'  "
              f"speed range [{vmag.min():.4g}, {vmag.max():.4g}] m/s")
    if f_name and f_arr is not None and f_arr.ndim == 2:
        fmag = np.linalg.norm(f_arr, axis=1)
        n_nonzero = int((fmag > 1e-8).sum())
        print(f"  Force field    : '{f_name}'  "
              f"non-zero cells: {n_nonzero:,}")

    print(f"\n  All field keys: {list(all_fields.keys())}")
    return all_fields


# ──────────────────────────────────────────────────────────────────────────────
#  3.  compute_surface()
# ──────────────────────────────────────────────────────────────────────────────

def compute_surface(mesh: pv.ImageData, fields: dict) -> pv.PolyData:
    """
    Reconstruct the aerodynamic body surface from the volume data.

    Why naive contouring fails for nTop LBM
    ----------------------------------------
    nTop treats the outer domain walls as solid boundaries, so the
    ImplicitField = 0 iso-surface spans the ENTIRE domain bounding box
    (~20 m²) instead of just the ATAS body (~0.96 m²).

    Correct extraction procedure
    ----------------------------
    1. Identify the IB interface layer (CellRegion == 3).  These points
       exist only around the actual body, not on domain walls.
    2. Compute the body's tight bounding box from those points.
    3. Extract the smooth ImplicitField = 0 iso-surface (marching cubes).
    4. Clip the iso-surface to the bounding box → body-only surface.
    5. Sample all volume fields onto the surface for downstream use.

    Parameters
    ----------
    mesh   : pv.ImageData  full-volume VTI mesh.
    fields : dict          all extracted numpy arrays from extract_fields().

    Returns
    -------
    pv.PolyData
        Triangulated body surface with:
          • All volume fields sampled as point data.
          • "Normals"  (N, 3) outward cell normals.
          • "Area"     (N,)   cell areas [m²].
          • field_data["total_area_m2"], ["frontal_area_m2"],
            ["planform_area_m2"] stored for downstream use.
    """
    print(f"\n{'─'*64}")
    print("  SURFACE EXTRACTION  (nTop LBM body-only method)")
    print(f"{'─'*64}")

    pts = mesh.points

    # ── Step 1: Locate the actual body via IB interface cells ──────────────
    creg_arr = fields.get("CellRegion")
    f_name, f_arr = _find_field(fields, _FORCE_CANDIDATES)
    impl_arr = fields.get("ImplicitField")

    body_mask = None

    # Priority 1: CellRegion == 3  (IB interface — most reliable nTop marker)
    if creg_arr is not None:
        mask3 = np.array(creg_arr, dtype=int) == 3
        if mask3.sum() > 0:
            body_mask = mask3
            print(f"  Body located via CellRegion == 3  "
                  f"({mask3.sum():,} pts)")

    # Priority 2: Non-zero Force cells
    if body_mask is None and f_arr is not None and f_arr.ndim == 2:
        fmag = np.linalg.norm(f_arr, axis=1)
        body_mask = fmag > 1e-8
        print(f"  Body located via |Force| > 0  ({body_mask.sum():,} pts)")

    # Priority 3: ImplicitField > 0 (inside solid)
    if body_mask is None and impl_arr is not None:
        body_mask = np.array(impl_arr) > 0
        print(f"  Body located via ImplicitField > 0  ({body_mask.sum():,} pts)")

    if body_mask is None or body_mask.sum() == 0:
        raise RuntimeError("Cannot locate body in mesh. "
                           "Ensure ImplicitField or CellRegion is present.")

    # ── Step 2: Bounding box with margin ──────────────────────────────────
    body_pts = pts[body_mask]
    cell_sz   = max(getattr(mesh, "spacing", (0.01, 0.01, 0.01)))
    margin    = 3.0 * cell_sz   # 3 cell spacings safety margin

    bb = [
        float(body_pts[:, 0].min()) - margin,
        float(body_pts[:, 0].max()) + margin,
        float(body_pts[:, 1].min()) - margin,
        float(body_pts[:, 1].max()) + margin,
        float(body_pts[:, 2].min()) - margin,
        float(body_pts[:, 2].max()) + margin,
    ]
    print(f"  Body bounding box : "
          f"X=[{bb[0]:.4g},{bb[1]:.4g}]  "
          f"Y=[{bb[2]:.4g},{bb[3]:.4g}]  "
          f"Z=[{bb[4]:.4g},{bb[5]:.4g}]  m")

    # ── Step 3 & 4: ImplicitField = 0 contour, then clip ──────────────────
    surface = None

    if impl_arr is not None:
        print("  Extracting ImplicitField = 0 iso-surface (marching cubes)...")
        try:
            full_iso = mesh.contour([0.0], scalars="ImplicitField")
            n_full   = full_iso.n_cells
            print(f"  Full iso-surface (incl. domain walls): {n_full:,} triangles")

            # Clip to body bounding box — removes domain-wall triangles
            clipped = full_iso.clip_box(bb, invert=False)

            # clip_box may return UnstructuredGrid; convert to triangulated PolyData
            if not isinstance(clipped, pv.PolyData):
                clipped = clipped.extract_surface(
                    algorithm="dataset_surface"
                ).triangulate()
            else:
                clipped = clipped.triangulate()

            if clipped.n_cells > 0:
                surface = clipped
                print(f"  Body-only iso-surface (clipped)      : "
                      f"{surface.n_cells:,} triangles  {surface.n_points:,} pts")
            else:
                print("  [warn] Clipped iso-surface is empty — trying fallback.")
        except Exception as e:
            print(f"  [warn] Iso-surface extraction failed: {e}")

    # Fallback: voxelised body surface from cell threshold
    if surface is None or surface.n_cells == 0:
        print("  Fallback: cell threshold (ImplicitField > 0) + extract_surface...")
        try:
            mesh_cell   = mesh.point_data_to_cell_data()
            solid       = mesh_cell.threshold(
                value=0.0, scalars="ImplicitField",
                preference="cell", invert=False,
            )
            solid_clip  = solid.clip_box(bb, invert=False)
            surface     = solid_clip.extract_surface(
                algorithm="dataset_surface"
            ).triangulate()
            print(f"  Voxelised body surface : {surface.n_cells:,} faces")
        except Exception as e:
            raise RuntimeError(
                f"Surface extraction failed: {e}. "
                "Verify that ImplicitField is present in the VTI file."
            ) from e

    # ── Step 5: Compute normals and cell areas ─────────────────────────────
    # auto_orient_normals=True ensures normals point outward from the body
    # (into the fluid), which is required for correct pressure integration.
    surface = surface.compute_normals(
        cell_normals=True, point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        flip_normals=False,
    )
    surface = surface.compute_cell_sizes(length=False, area=True, volume=False)

    normals     = np.array(surface.cell_data["Normals"], dtype=float)   # (M, 3)
    areas       = np.array(surface.cell_data["Area"],    dtype=float)   # (M,)
    total_area  = float(areas.sum())

    # Projected areas used as aerodynamic reference areas
    # Frontal area  = projection onto YZ plane → reference for drag (CD)
    # Planform area = projection onto XZ plane → reference for lift (CL)
    frontal_area  = float((areas * np.abs(normals[:, 0])).sum())
    planform_area = float((areas * np.abs(normals[:, 2])).sum())

    mean_n = normals.mean(axis=0)
    b_s    = np.array(surface.bounds).reshape(3, 2)

    print(f"\n  Surface bounds      : "
          f"X=[{b_s[0,0]:.4g},{b_s[0,1]:.4g}]  "
          f"Y=[{b_s[1,0]:.4g},{b_s[1,1]:.4g}]  "
          f"Z=[{b_s[2,0]:.4g},{b_s[2,1]:.4g}]  m")
    print(f"  Total surface area  : {total_area:.5g} m²")
    print(f"  Frontal area  (YZ)  : {frontal_area:.5g} m²   (CD reference)")
    print(f"  Planform area (XZ)  : {planform_area:.5g} m²   (CL reference)")
    print(f"  Mean outward normal : [{mean_n[0]:+.4g}, {mean_n[1]:+.4g}, "
          f"{mean_n[2]:+.4g}]")
    print(f"  (Should be near [0,0,0] for a closed surface)")

    # ── Step 5b: Sample all volume fields onto the surface ─────────────────
    # PyVista's sample() uses probe interpolation from the volume mesh,
    # making every point-data array available on the surface.
    print(f"\n  Sampling volume fields onto surface...")
    try:
        surface = surface.sample(mesh)
        print(f"  Sampled arrays: {[k for k in surface.point_data.keys() if not k.startswith('vtk')]}")
    except Exception as e:
        print(f"  [warn] Sampling failed: {e}")

    # Store geometry for downstream functions
    surface.field_data["total_area_m2"]    = np.array([total_area])
    surface.field_data["frontal_area_m2"]  = np.array([frontal_area])
    surface.field_data["planform_area_m2"] = np.array([planform_area])

    return surface


# ──────────────────────────────────────────────────────────────────────────────
#  4.  compute_forces()
# ──────────────────────────────────────────────────────────────────────────────

def compute_forces(
    mesh:          pv.ImageData,
    surface:       pv.PolyData,
    fields:        dict,
    drag_dir:      np.ndarray = DRAG_DIRECTION,
    lift_dir:      np.ndarray = LIFT_DIRECTION,
    fluid_density: float      = FLUID_DENSITY,
    v_inf_override: float     = None,
    a_frontal_override: float = None,
    a_planform_override: float = None,
) -> dict:
    """
    Compute aerodynamic lift, drag, and span forces via two independent methods.

    Uses the following physical fields:

      Force field  (Method A — primary)
          Aerodynamic force per grid cell stored by nTop's IB solver.
          Summed over CellRegion == 3 (IB interface layer).
          Captures both pressure (form) and viscous (friction) forces.

      Pressure field  (Method B — secondary)
          F = −∫ (p − p∞) · n̂ dA   over the extracted body surface.
          Uses cell normals and cell areas computed by compute_surface().
          Form (pressure) drag only — viscous drag excluded.

      Velocity field  (both methods)
          Used to detect freestream velocity V∞ and dynamic pressure q∞.
          Also used to compute the pressure coefficient Cp at every surface
          cell and to verify the no-slip condition at the body wall.

    Parameters
    ----------
    mesh           : pv.ImageData  full-volume VTI mesh.
    surface        : pv.PolyData   body surface from compute_surface().
    fields         : dict          all numpy arrays from extract_fields().
    drag_dir       : (3,) array    drag (freestream) direction unit vector.
    lift_dir       : (3,) array    lift direction unit vector.
    fluid_density  : float         ρ [kg/m³], default 1.225 (sea-level air).
    v_inf_override : float | None  freestream speed [m/s]; None = auto-detect.
    a_frontal_override  : float | None  frontal reference area [m²]; None = auto.
    a_planform_override : float | None  planform reference area [m²]; None = auto.

    Returns
    -------
    dict  with keys:
        # Geometry
        "surface_area_m2", "frontal_area_m2", "planform_area_m2"
        # Freestream
        "v_inf_ms", "p_inf_Pa", "q_inf_Pa"
        # Method A — Force field (primary)
        "A_total_force_N", "A_drag_N", "A_lift_N", "A_span_N"
        "A_CD", "A_CL", "A_LD"                          (if v_inf > 0)
        "A_n_interface_cells", "A_force_field_name"
        # Method B — Pressure integration (secondary)
        "B_total_force_N", "B_drag_N", "B_lift_N", "B_span_N"
        "B_CD", "B_CL"                                   (if v_inf > 0)
        "B_pressure_field_name", "B_p_ref_Pa"
        # Velocity analysis
        "wall_speed_mean_ms", "wall_speed_max_ms"
        "Cp_mean", "Cp_min", "Cp_max"
        # Per-cell arrays (for post-processing)
        "B_force_per_cell", "B_Cp_per_cell"
    """
    print(f"\n{'─'*64}")
    print("  FORCE COMPUTATION")
    print(f"{'─'*64}")

    # ── Direction vectors ──────────────────────────────────────────────────
    drag_dir = _unit(drag_dir.copy())
    lift_dir = _orthogonalise(lift_dir.copy(), drag_dir)
    span_dir = np.cross(drag_dir, lift_dir)

    print(f"  Drag direction (+X) : {drag_dir}")
    print(f"  Lift direction (+Z) : {lift_dir}")
    print(f"  Span direction (+Y) : {span_dir}")

    # ── Surface geometry from compute_surface() ────────────────────────────
    # Re-compute normals/areas in case the surface was modified downstream
    surf2 = surface.compute_normals(
        cell_normals=True, point_normals=False,
        consistent_normals=True, auto_orient_normals=True,
    )
    surf2 = surf2.compute_cell_sizes(length=False, area=True, volume=False)

    normals_c = np.array(surf2.cell_data["Normals"], dtype=float)   # (M, 3)
    areas_c   = np.array(surf2.cell_data["Area"],    dtype=float)   # (M,)
    total_area  = float(areas_c.sum())
    frontal_area  = (a_frontal_override  or
                     float(surface.field_data["frontal_area_m2"][0]))
    planform_area = (a_planform_override or
                     float(surface.field_data["planform_area_m2"][0]))

    print(f"\n  Surface cells       : {surf2.n_cells:,}")
    print(f"  Total surface area  : {total_area:.5g} m²")
    print(f"  Frontal area  (YZ)  : {frontal_area:.5g} m²   (CD reference)")
    print(f"  Planform area (XZ)  : {planform_area:.5g} m²   (CL reference)")

    # ── VELOCITY FIELD — freestream and wall analysis ──────────────────────
    print(f"\n  {'─'*56}")
    print("  VELOCITY FIELD ANALYSIS")
    print(f"  {'─'*56}")

    v_name, v_vol = _find_field(fields, _VELOCITY_CANDIDATES)
    v_inf  = 0.0
    p_inf  = 0.0
    q_inf  = 0.0

    if v_name and v_vol is not None and v_vol.ndim == 2:
        print(f"  Velocity field : '{v_name}'")

        # Freestream: sample from upstream inlet strip (low X, fluid domain)
        impl_arr = fields.get("ImplicitField")
        pts_vol  = mesh.points
        bounds   = mesh.bounds

        if impl_arr is not None:
            # Fluid mask: ImplicitField < 0 (outside solid body)
            fluid_mask = np.array(impl_arr) < -0.006
        else:
            fluid_mask = np.ones(mesh.n_points, dtype=bool)

        # Upstream strip: lowest 15 % of the X extent
        x_inlet = bounds[0] + 0.15 * (bounds[1] - bounds[0])
        upstream = fluid_mask & (pts_vol[:, 0] <= x_inlet)

        if upstream.sum() < 100:
            upstream = fluid_mask   # fallback to all fluid

        v_up   = v_vol[upstream]
        v_inf_computed = float(np.linalg.norm(v_up, axis=1).mean())
        print(f"  Upstream fluid pts : {upstream.sum():,}")
        print(f"  V∞ (mean speed)    : {v_inf_computed:.4f} m/s")
        print(f"  V∞ x-component     : {v_up[:, 0].mean():.4f} m/s")

        v_inf = v_inf_override if v_inf_override else v_inf_computed

        # Pressure freestream from same upstream region
        p_name, p_vol = _find_field(fields, _PRESSURE_CANDIDATES)
        if p_vol is not None:
            p_inf = float(p_vol[upstream].mean())
            print(f"  p∞ (upstream mean) : {p_inf:.4f} Pa")

        q_inf = 0.5 * fluid_density * v_inf ** 2
        print(f"  q∞ = ½ρV²         : {q_inf:.4f} Pa")
        print(f"  ρ (fluid density)  : {fluid_density} kg/m³")

        # Wall velocity — verify no-slip condition
        # Sample velocity from volume onto surface (already done by compute_surface)
        v_surf_arr = None
        if v_name in surface.point_data:
            v_surf_arr = np.array(surface.point_data[v_name])
        elif v_name in surface.cell_data:
            v_surf_arr = np.array(surface.cell_data[v_name])

        if v_surf_arr is not None and v_surf_arr.ndim == 2:
            surf_speed = np.linalg.norm(v_surf_arr, axis=1)
            # Convert to cell centres for consistency with force integration
            s_tmp = surface.copy()
            s_tmp.point_data["_vsurf"] = v_surf_arr
            s_tmp = s_tmp.point_data_to_cell_data()
            v_cell_arr = np.array(s_tmp.cell_data["_vsurf"])
            wall_speed = np.linalg.norm(v_cell_arr, axis=1)
            print(f"\n  Wall speed (no-slip check):")
            print(f"    Mean  : {wall_speed.mean():.4f} m/s  "
                  f"(should be near 0 for perfect no-slip)")
            print(f"    Max   : {wall_speed.max():.4f} m/s")
            print(f"    % cells < 0.5 m/s : "
                  f"{100*float((wall_speed < 0.5).mean()):.1f}%")
        else:
            wall_speed = np.zeros(surf2.n_cells)
            print(f"  [warn] Velocity not sampled onto surface.")
    else:
        print(f"  [warn] No vector velocity field found.")
        print(f"  CL / CD will not be computed.")
        wall_speed = np.zeros(surf2.n_cells)

    # ─────────────────────────────────────────────────────────────────────
    # METHOD A — Force-field integration  (PRIMARY for nTop LBM)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print("  METHOD A: Force-field integration  (primary, recommended)")
    print(f"  {'─'*56}")
    print("  Sums 'Force time-averaged' over CellRegion == 3 (IB interface).")
    print("  Captures pressure + viscous forces without surface reconstruction.")

    f_name, f_arr = _find_field(fields, _FORCE_CANDIDATES)
    A_drag = A_lift = A_span = 0.0
    A_F_total = np.zeros(3)
    n_iface = 0

    if f_name and f_arr is not None and f_arr.ndim == 2:
        print(f"  Force field : '{f_name}'")

        # Identify IB interface cells ─────────────────────────────────────
        creg_arr = fields.get("CellRegion")
        iface_mask = None

        if creg_arr is not None:
            mask3 = np.array(creg_arr, dtype=int) == 3
            if mask3.sum() > 0:
                iface_mask = mask3
                print(f"  IB cells    : CellRegion == 3  ({mask3.sum():,} pts)")

        if iface_mask is None:
            # Fallback: any cell with non-negligible Force magnitude
            fmag   = np.linalg.norm(f_arr, axis=1)
            thresh = max(1e-8, 0.001 * float(fmag.max()))
            iface_mask = fmag > thresh
            print(f"  IB cells    : |Force| > {thresh:.3g}  ({iface_mask.sum():,} pts)")

        # Sum Force over all IB interface cells
        F_per_cell = f_arr[iface_mask]        # (N_iface, 3)
        A_F_total  = F_per_cell.sum(axis=0)   # total aerodynamic force [N]
        n_iface    = int(iface_mask.sum())

        A_drag = float(np.dot(A_F_total, drag_dir))
        A_lift = float(np.dot(A_F_total, lift_dir))
        A_span = float(np.dot(A_F_total, span_dir))

        # Per-cell contributions
        drag_pc = F_per_cell @ drag_dir
        lift_pc = F_per_cell @ lift_dir
        print(f"\n  IB interface cells  : {n_iface:,}")
        print(f"  Total force vector  : [{A_F_total[0]:+.5g}, "
              f"{A_F_total[1]:+.5g}, {A_F_total[2]:+.5g}] N")
        print(f"  Drag  (+X)          : {A_drag:+.5g} N")
        print(f"  Lift  (+Z)          : {A_lift:+.5g} N")
        print(f"  Span  (+Y)          : {A_span:+.5g} N")
        print(f"  Per-cell drag range : [{drag_pc.min():.4g}, {drag_pc.max():.4g}] N")
        print(f"  Per-cell lift range : [{lift_pc.min():.4g}, {lift_pc.max():.4g}] N")

        if q_inf > 0:
            A_CD = A_drag / (q_inf * frontal_area)
            A_CL = A_lift / (q_inf * planform_area)
            print(f"\n  q∞                  : {q_inf:.4g} Pa")
            print(f"  CD (total)          : {A_CD:+.6f}  "
                  f"(A_frontal = {frontal_area:.4g} m²)")
            print(f"  CL (total)          : {A_CL:+.6f}  "
                  f"(A_planform = {planform_area:.4g} m²)")
            if A_drag != 0:
                print(f"  L/D ratio           : {A_lift/A_drag:.4f}")
        else:
            A_CD = A_CL = 0.0
            print("  [skip] CL/CD not computed (V∞ = 0).")
    else:
        print("  [error] No Force field found — Method A unavailable.")
        print(f"  Available fields: {list(fields.keys())}")
        A_CD = A_CL = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # METHOD B — Pressure integration  (SECONDARY, form drag only)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print("  METHOD B: Pressure integration  (secondary, form drag only)")
    print(f"  {'─'*56}")
    print("  F = -∫(p - p∞) · n̂ dA   over the extracted body surface.")
    print("  [Note] nTop sets p = 0 inside solid → sampled surface pressure")
    print("         averages fluid+solid values, underestimating forces.")

    # Find pressure on surface (sampled by compute_surface via sample())
    p_name, _ = _find_field(fields, _PRESSURE_CANDIDATES)
    B_drag = B_lift = B_span = 0.0
    B_F_total  = np.zeros(3)
    B_force_pc = np.zeros((surf2.n_cells, 3))
    Cp_cells   = np.zeros(surf2.n_cells)

    # Build surface cell-centre data container
    surf_cc = surf2.point_data_to_cell_data()   # convert point→cell

    # Locate pressure at cell centres
    p_cells = None
    p_src   = "none"
    if p_name and p_name in surf_cc.cell_data:
        p_cells = np.array(surf_cc.cell_data[p_name], dtype=float)
        p_src   = "cell data (from point→cell average)"
    elif p_name and p_name in surf2.point_data:
        tmp = surf2.copy()
        tmp = tmp.point_data_to_cell_data()
        if p_name in tmp.cell_data:
            p_cells = np.array(tmp.cell_data[p_name], dtype=float)
            p_src   = "point data (averaged to cell centres)"

    if p_cells is not None:
        print(f"\n  Pressure field : '{p_name}'  ({p_src})")

        # Validity mask: finite pressure, positive area, finite normals
        valid = (
            np.isfinite(p_cells)
            & (areas_c > 0)
            & np.all(np.isfinite(normals_c), axis=1)
        )
        if valid.sum() < surf2.n_cells:
            print(f"  [warn] Discarding {surf2.n_cells - valid.sum()} "
                  f"degenerate/NaN cells")

        p_v = p_cells[valid]
        n_v = normals_c[valid]
        a_v = areas_c[valid]

        print(f"  Valid cells         : {valid.sum():,}")
        print(f"  Pressure range      : [{p_v.min():.4g}, {p_v.max():.4g}] Pa")
        print(f"  p∞ reference        : {p_inf:.4g} Pa")

        # Gauge pressure (remove uniform offset)
        # Using p_inf from far-field avoids cancellation errors.
        p_gauge = p_v - p_inf

        # Pressure force on body:
        #   dF_i = -(p_gauge_i) * n̂_i * dA_i
        # The minus sign: fluid pushes inward on the body; outward normals
        # (pointing from body into fluid) have n̂ pointing away from body,
        # so the force the fluid exerts = -(p-p_ref) * n̂ * dA.
        force_pc_v = -(p_gauge[:, np.newaxis] * n_v * a_v[:, np.newaxis])
        B_F_total_valid = force_pc_v.sum(axis=0)

        # Map back to full arrays
        B_force_pc[valid] = force_pc_v

        B_drag = float(np.dot(B_F_total_valid, drag_dir))
        B_lift = float(np.dot(B_F_total_valid, lift_dir))
        B_span = float(np.dot(B_F_total_valid, span_dir))
        B_F_total = B_F_total_valid

        print(f"\n  Total force vector  : [{B_F_total[0]:+.5g}, "
              f"{B_F_total[1]:+.5g}, {B_F_total[2]:+.5g}] N")
        print(f"  Drag  (+X)          : {B_drag:+.5g} N  (form / pressure only)")
        print(f"  Lift  (+Z)          : {B_lift:+.5g} N  (form / pressure only)")
        print(f"  Span  (+Y)          : {B_span:+.5g} N")

        # ── Pressure coefficient Cp from velocity field ────────────────────
        # Cp_i = (p_i - p∞) / q∞     dimensionless pressure coefficient
        # Requires q∞ > 0 (velocity field present)
        if q_inf > 0:
            Cp_cells[valid] = p_gauge / q_inf
            print(f"\n  Pressure coefficient Cp = (p - p∞)/q∞:")
            print(f"    Mean Cp  : {Cp_cells[valid].mean():+.4f}")
            print(f"    Min Cp   : {Cp_cells[valid].min():+.4f}  "
                  f"(suction peak)")
            print(f"    Max Cp   : {Cp_cells[valid].max():+.4f}  "
                  f"(stagnation)")

            B_CD = B_drag / (q_inf * frontal_area)
            B_CL = B_lift / (q_inf * planform_area)
            print(f"\n  CD (form only)      : {B_CD:+.6f}")
            print(f"  CL (form only)      : {B_CL:+.6f}")
        else:
            B_CD = B_CL = 0.0
            print("  [skip] Cp, CL/CD not computed (V∞ = 0).")
    else:
        print(f"  [error] No pressure field found on surface.")
        print(f"  Surface point arrays: {list(surface.point_data.keys())}")
        B_CD = B_CL = 0.0

    # ── Method comparison ─────────────────────────────────────────────────
    if f_name and p_cells is not None:
        print(f"\n  {'─'*56}")
        print("  COMPARISON: Method A vs Method B")
        print(f"  {'─'*56}")
        print(f"  {'':28s}  {'Drag':>10s}  {'Lift':>10s}  {'Span':>10s}")
        print(f"  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*10}")
        print(f"  {'Method A (Force field) [N]':<28s}  "
              f"{A_drag:>+10.4f}  {A_lift:>+10.4f}  {A_span:>+10.4f}")
        print(f"  {'Method B (Pressure int) [N]':<28s}  "
              f"{B_drag:>+10.4f}  {B_lift:>+10.4f}  {B_span:>+10.4f}")
        print(f"  {'Difference (A - B) [N]':<28s}  "
              f"{A_drag-B_drag:>+10.4f}  {A_lift-B_lift:>+10.4f}  "
              f"{A_span-B_span:>+10.4f}")
        print(f"  Method B is lower: p=0 inside solid reduces sampled pressure.")

    # ── Assemble result dict ──────────────────────────────────────────────
    results = {
        # Geometry
        "surface_area_m2":  total_area,
        "frontal_area_m2":  frontal_area,
        "planform_area_m2": planform_area,
        # Freestream
        "v_inf_ms":  v_inf,
        "p_inf_Pa":  p_inf,
        "q_inf_Pa":  q_inf,
        # Method A
        "A_total_force_N":        A_F_total,
        "A_drag_N":               A_drag,
        "A_lift_N":               A_lift,
        "A_span_N":               A_span,
        "A_CD":                   A_CD,
        "A_CL":                   A_CL,
        "A_LD":                   A_lift / A_drag if A_drag != 0 else 0.0,
        "A_n_interface_cells":    n_iface,
        "A_force_field_name":     f_name or "",
        # Method B
        "B_total_force_N":        B_F_total,
        "B_drag_N":               B_drag,
        "B_lift_N":               B_lift,
        "B_span_N":               B_span,
        "B_CD":                   B_CD,
        "B_CL":                   B_CL,
        "B_pressure_field_name":  p_name or "",
        "B_p_ref_Pa":             p_inf,
        "B_force_per_cell":       B_force_pc,
        "B_Cp_per_cell":          Cp_cells,
        # Velocity analysis
        "wall_speed_mean_ms":     float(wall_speed.mean()) if wall_speed.size else 0.0,
        "wall_speed_max_ms":      float(wall_speed.max())  if wall_speed.size else 0.0,
        "Cp_mean":                float(Cp_cells[Cp_cells != 0].mean()) if (Cp_cells != 0).any() else 0.0,
        "Cp_min":                 float(Cp_cells[Cp_cells != 0].min())  if (Cp_cells != 0).any() else 0.0,
        "Cp_max":                 float(Cp_cells[Cp_cells != 0].max())  if (Cp_cells != 0).any() else 0.0,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  5.  print_summary()
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print a clean final aerodynamic summary table."""

    print(f"\n{'='*64}")
    print("  AERODYNAMIC FORCE SUMMARY")
    print(f"{'='*64}")

    # Freestream conditions
    v_inf = results.get("v_inf_ms", 0.0)
    q_inf = results.get("q_inf_Pa", 0.0)
    p_inf = results.get("p_inf_Pa", 0.0)
    print(f"  Freestream velocity  V∞ : {v_inf:.4g} m/s")
    print(f"  Dynamic pressure     q∞ : {q_inf:.4g} Pa")
    print(f"  Static pressure      p∞ : {p_inf:.4g} Pa")
    print(f"  Fluid density         ρ : {FLUID_DENSITY} kg/m³")

    # Surface geometry
    print(f"\n  Body surface area       : {results.get('surface_area_m2', 0):.5g} m²")
    print(f"  Frontal area (CD ref)   : {results.get('frontal_area_m2', 0):.5g} m²")
    print(f"  Planform area (CL ref)  : {results.get('planform_area_m2', 0):.5g} m²")

    # Wall velocity
    ws_mean = results.get("wall_speed_mean_ms", 0.0)
    ws_max  = results.get("wall_speed_max_ms", 0.0)
    print(f"\n  Wall speed (no-slip)    : mean={ws_mean:.4g} m/s  max={ws_max:.4g} m/s")

    # Cp
    cp_mn, cp_mi, cp_mx = (results.get("Cp_mean", 0),
                            results.get("Cp_min", 0),
                            results.get("Cp_max", 0))
    print(f"  Pressure coeff Cp       : mean={cp_mn:+.4f}  "
          f"min={cp_mi:+.4f}  max={cp_mx:+.4f}")

    # Force table
    A_drag = results.get("A_drag_N", 0.0)
    A_lift = results.get("A_lift_N", 0.0)
    A_span = results.get("A_span_N", 0.0)
    B_drag = results.get("B_drag_N", 0.0)
    B_lift = results.get("B_lift_N", 0.0)
    B_span = results.get("B_span_N", 0.0)

    print(f"\n  {'Component':<32s}  {'Drag':>10s}  {'Lift':>10s}  {'Span':>10s}")
    print(f"  {'─'*32}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(f"  {'Method A — Force field [N]':<32s}  "
          f"{A_drag:>+10.4f}  {A_lift:>+10.4f}  {A_span:>+10.4f}")
    print(f"  {'Method B — Pressure int. [N]':<32s}  "
          f"{B_drag:>+10.4f}  {B_lift:>+10.4f}  {B_span:>+10.4f}")

    if q_inf > 0:
        A_CD = results.get("A_CD", 0.0)
        A_CL = results.get("A_CL", 0.0)
        A_LD = results.get("A_LD", 0.0)
        B_CD = results.get("B_CD", 0.0)
        B_CL = results.get("B_CL", 0.0)
        print(f"\n  {'':32s}  {'CD':>10s}  {'CL':>10s}  {'L/D':>10s}")
        print(f"  {'─'*32}  {'─'*10}  {'─'*10}  {'─'*10}")
        print(f"  {'Method A — Force field':<32s}  "
              f"{A_CD:>+10.6f}  {A_CL:>+10.6f}  {A_LD:>+10.4f}")
        print(f"  {'Method B — Pressure int. (form)':<32s}  "
              f"{B_CD:>+10.6f}  {B_CL:>+10.6f}  {'N/A':>10s}")

    n_if = results.get("A_n_interface_cells", 0)
    ff   = results.get("A_force_field_name", "N/A")
    pf   = results.get("B_pressure_field_name", "N/A")
    print(f"\n  Force field used        : '{ff}'  ({n_if:,} IB interface cells)")
    print(f"  Pressure field used     : '{pf}'")

    print(f"\n  COORDINATE SYSTEM")
    print(f"  Drag (+X) : streamwise / freestream direction")
    print(f"  Lift (+Z) : vertical / normal to freestream")
    print(f"  Span (+Y) : spanwise")

    print(f"\n  METHOD NOTES")
    print(f"  * Method A is the recommended result for nTop LBM data.")
    print(f"    It sums the stored aerodynamic Force field over the IB")
    print(f"    interface layer and captures pressure + viscous forces.")
    print(f"  * Method B integrates (p - p∞) n̂ dA over the body surface.")
    print(f"    It captures form (pressure) drag/lift only.  Values are")
    print(f"    lower because nTop sets p=0 inside the solid body, so")
    print(f"    sampled surface pressure is reduced near the body wall.")
    print(f"  * ImplicitField sign: < 0 = fluid, = 0 = wall, > 0 = solid.")
    print(f"  * CellRegion 3 = IB interface layer (Force field non-zero).")
    print(f"{'='*64}")


# ──────────────────────────────────────────────────────────────────────────────
#  main()
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── File selection ────────────────────────────────────────────────────
    VTI_FILE = "CFD ATAS 0.005 at 0 deg Implicit.vti"   # ← change to your file name

    if len(sys.argv) > 1:
        VTI_FILE = sys.argv[1]

    # ── Optional manual overrides  (set to None for auto-detection) ───────
    FREESTREAM_VEL_MS    = None    # e.g. 30.0  — overrides auto-detected V∞
    REFERENCE_FRONTAL_M2 = None    # e.g. 0.05  — YZ-projected area for CD
    REFERENCE_PLANFORM_M2 = None   # e.g. 0.20  — XZ-projected area for CL

    # ── 1. Load VTI ───────────────────────────────────────────────────────
    mesh = load_vti(VTI_FILE)

    # ── 2. Extract all fields as NumPy arrays ─────────────────────────────
    fields = extract_fields(mesh)

    if not fields:
        print("\n  No data arrays found in the file.")
        print("  Open the file in ParaView to verify its contents.")
        sys.exit(1)

    # ── 3. Reconstruct body surface ───────────────────────────────────────
    surface = compute_surface(mesh, fields)

    # ── 4. Compute forces, CL, CD ─────────────────────────────────────────
    results = compute_forces(
        mesh             = mesh,
        surface          = surface,
        fields           = fields,
        drag_dir         = DRAG_DIRECTION,
        lift_dir         = LIFT_DIRECTION,
        fluid_density    = FLUID_DENSITY,
        v_inf_override   = FREESTREAM_VEL_MS,
        a_frontal_override  = REFERENCE_FRONTAL_M2,
        a_planform_override = REFERENCE_PLANFORM_M2,
    )

    # ── 5. Print final summary ────────────────────────────────────────────
    print_summary(results)

    # Return all objects for interactive / notebook use
    return mesh, fields, surface, results


if __name__ == "__main__":
    main()
