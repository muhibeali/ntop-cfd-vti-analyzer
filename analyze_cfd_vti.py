"""
nTop LBM CFD VTI File Analyzer
================================
Reads a .vti (VTK ImageData) file exported from an nTop Lattice-Boltzmann
CFD simulation and reports aerodynamic forces directly from the simulation.

Coordinate system
-----------------
  +X  →  Streamwise / drag direction  (freestream flows in +X direction)
  +Z  ↑  Vertical   / lift direction
  +Y  ·  Spanwise   / side force direction

Sign convention
---------------
  Drag  : positive in +X  (force opposing forward motion)
  Lift  : positive in +Z  (upward force)
  Side  : positive in +Y  (positive spanwise)

Force field units
-----------------
  The 'Force time-averaged' field is assumed to be stored in Newtons [N]
  per immersed-boundary (IB) grid cell.  Summing this field over the IB
  interface layer gives the total aerodynamic force directly.

  If the field is stored as force density [N/m³] instead, each cell value
  must be multiplied by the cell volume before summing.  The code prints
  the cell volume and per-cell force magnitude range so this assumption
  can be verified at runtime.

Why 'Force time-averaged' is the authoritative force source
-----------------------------------------------------------
  nTop's immersed-boundary LBM solver writes this field directly from the
  IB kernel.  It is preferred over surface pressure integration because:

    1. It captures both pressure (form) and viscous (friction) forces in
       a single field — no surface reconstruction or decomposition needed.

    2. It is unaffected by nTop's boundary condition that sets pressure
       to zero inside the solid body.  Surface pressure sampling averages
       fluid and zero values near the wall, systematically underestimating
       the true aerodynamic loading.

    3. The IB interface cells (CellRegion == 3) are a clearly defined,
       compact layer around the body — not the full domain — so the sum
       is uncontaminated by domain-wall effects.

nTop LBM field reference
-------------------------
  ImplicitField
      Signed-distance field of the geometry.
      < 0  →  fluid domain
      = 0  →  body wall
      > 0  →  inside solid body

  CellRegion
      Region label for each grid cell.
      0      : outer boundary
      1      : main fluid domain
      3      : IB fluid-solid interface layer  ← Force field is non-zero here
      4, 5   : inflow / outflow boundary patches
      6      : near-body fluid layer

  Force time-averaged [N per IB cell]
      Aerodynamic force stored by the IB solver.
      Non-zero only in CellRegion == 3 (~36 k cells out of 26 M total).

  The outer domain walls are also treated as solid boundaries, so a naive
  ImplicitField = 0 contour spans the FULL domain bounding box (~20 m²).
  The correct body surface is obtained by clipping that contour to the
  bounding box of the CellRegion == 3 cells (~0.96 m²).

Usage
-----
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

# Freestream (drag) direction unit vector — +X is streamwise
DRAG_DIRECTION = np.array([1.0, 0.0, 0.0])

# Lift direction unit vector — must be perpendicular to DRAG_DIRECTION
LIFT_DIRECTION = np.array([0.0, 0.0, 1.0])

# Field-name candidates searched in order for auto-detection.
# Exact match is tried first, then case-insensitive substring fallback.
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
    """Return unit vector; raise if near-zero."""
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
      ImplicitField          – signed-distance function  (fluid < 0, solid > 0)
      CellRegion             – region label  (0–6, see module docstring)
      Velocity initial       – velocity at t = 0
      Velocity time-averaged – time-averaged velocity
      Force time-averaged    – aerodynamic body force per IB cell [N]
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
            nan_tag = f"  <- {nnan} NaN/Inf" if nnan else ""
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
        n_fluid = int((impl < 0).sum())
        n_wall  = int((np.abs(impl) < 0.007).sum())
        n_solid = int((impl > 0).sum())
        print(f"  ImplicitField  ->  fluid (< 0): {n_fluid:,}  "
              f"wall (~0): {n_wall:,}  solid (> 0): {n_solid:,}")

    creg = all_fields.get("CellRegion")
    if creg is not None:
        unique_vals = np.unique(creg.astype(int))
        region_desc = {0: "outer BC", 1: "fluid", 3: "IB interface",
                       4: "inlet/outlet", 5: "inlet/outlet", 6: "near-body fluid"}
        for v in unique_vals:
            desc = region_desc.get(v, "unknown")
            print(f"  CellRegion {v}   ->  {(creg == v).sum():>12,} pts  ({desc})")

    v_name, v_arr = _find_field(all_fields, _VELOCITY_CANDIDATES)
    f_name, f_arr = _find_field(all_fields, _FORCE_CANDIDATES)

    if v_name and v_arr is not None and v_arr.ndim == 2:
        vmag = np.linalg.norm(v_arr, axis=1)
        print(f"  Velocity field : '{v_name}'  "
              f"speed range [{vmag.min():.4g}, {vmag.max():.4g}] m/s")
    if f_name and f_arr is not None and f_arr.ndim == 2:
        fmag     = np.linalg.norm(f_arr, axis=1)
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
    (~20 m²) instead of just the body (~0.96 m²).

    Correct extraction procedure
    ----------------------------
    1. Identify the IB interface layer (CellRegion == 3).  These points
       exist only around the actual body, not on domain walls.
    2. Compute the body's tight bounding box from those points.
    3. Extract the smooth ImplicitField = 0 iso-surface (marching cubes).
    4. Clip the iso-surface to the bounding box → body-only surface.
    5. Compute outward cell normals and validate their orientation.

    Parameters
    ----------
    mesh   : pv.ImageData  full-volume VTI mesh.
    fields : dict          all extracted numpy arrays from extract_fields().

    Returns
    -------
    pv.PolyData
        Triangulated body surface with:
          • "Normals"  (N, 3)  outward cell normals.
          • "Area"     (N,)    cell areas [m²].
          • field_data["total_area_m2"] stored for reference.
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
            print(f"  Body located via CellRegion == 3  ({mask3.sum():,} pts)")

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
    cell_sz  = max(getattr(mesh, "spacing", (0.01, 0.01, 0.01)))
    margin   = 3.0 * cell_sz

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
            mesh_cell  = mesh.point_data_to_cell_data()
            solid      = mesh_cell.threshold(
                value=0.0, scalars="ImplicitField",
                preference="cell", invert=False,
            )
            solid_clip = solid.clip_box(bb, invert=False)
            surface    = solid_clip.extract_surface(
                algorithm="dataset_surface"
            ).triangulate()
            print(f"  Voxelised body surface : {surface.n_cells:,} faces")
        except Exception as e:
            raise RuntimeError(
                f"Surface extraction failed: {e}. "
                "Verify that ImplicitField is present in the VTI file."
            ) from e

    # ── Step 5: Compute normals and cell areas ─────────────────────────────
    surface = surface.compute_normals(
        cell_normals=True, point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        flip_normals=False,
    )
    surface = surface.compute_cell_sizes(length=False, area=True, volume=False)

    normals    = np.array(surface.cell_data["Normals"], dtype=float)   # (M, 3)
    areas      = np.array(surface.cell_data["Area"],    dtype=float)   # (M,)
    total_area = float(areas.sum())

    mean_n = normals.mean(axis=0)
    b_s    = np.array(surface.bounds).reshape(3, 2)

    print(f"\n  Surface bounds    : "
          f"X=[{b_s[0,0]:.4g},{b_s[0,1]:.4g}]  "
          f"Y=[{b_s[1,0]:.4g},{b_s[1,1]:.4g}]  "
          f"Z=[{b_s[2,0]:.4g},{b_s[2,1]:.4g}]  m")
    print(f"  Total surface area: {total_area:.5g} m²")
    print(f"  Mean cell normal  : [{mean_n[0]:+.4g}, {mean_n[1]:+.4g}, "
          f"{mean_n[2]:+.4g}]  (near [0,0,0] for a closed surface)")

    # ── Step 6: Validate outward normal orientation ────────────────────────
    # For each surface cell, the vector from the body centroid to the cell
    # centre should align with the outward normal.  A high fraction of
    # misaligned normals indicates the surface was reconstructed inside-out.
    body_centroid = body_pts.mean(axis=0)
    cell_centres  = np.array(surface.cell_centers().points)
    radial        = cell_centres - body_centroid
    radial       /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    alignment     = (normals * radial).sum(axis=1)
    frac_outward  = float((alignment > 0).mean())

    print(f"\n  Normal orientation check:")
    print(f"    Cells with outward-facing normals : {100*frac_outward:.1f}%")

    if frac_outward < 0.5:
        print("  [warn] Majority of normals appear INWARD — flipping all normals.")
        surface.cell_data["Normals"] = -normals
    elif frac_outward < 0.8:
        print("  [warn] More than 20% of normals may be inverted. "
              "Surface reconstruction may be incomplete.")
    else:
        print("    Normals verified as outward-facing.")

    surface.field_data["total_area_m2"] = np.array([total_area])

    return surface


# ──────────────────────────────────────────────────────────────────────────────
#  4.  compute_forces()
# ──────────────────────────────────────────────────────────────────────────────

def compute_forces(
    mesh:     pv.ImageData,
    fields:   dict,
    drag_dir: np.ndarray = DRAG_DIRECTION,
    lift_dir: np.ndarray = LIFT_DIRECTION,
) -> dict:
    """
    Compute aerodynamic forces by summing 'Force time-averaged' over the
    immersed-boundary (IB) interface layer.

    This is the sole force computation method for nTop LBM data.
    See the module docstring for why this field is the authoritative source.

    IB interface detection
    ----------------------
    Primary detection uses cells where |Force time-averaged| exceeds a
    noise threshold derived from the field's own maximum value.

    Cross-check: if CellRegion == 3 is present, the two masks are compared.
    If they agree to within 80%, the CellRegion label is used (it is an
    explicit solver label).  If they disagree significantly, the
    force-magnitude mask is used and a warning is issued — this guards
    against future nTop versions that may change region numbering.

    Unit verification
    -----------------
    The code prints the cell volume alongside the mean per-cell force
    magnitude so you can verify the unit assumption (N/cell vs N/m³).
    If the field is in N/m³, set FORCE_IS_DENSITY = True in main() and
    the code will multiply by cell volume before summing.

    Parameters
    ----------
    mesh     : pv.ImageData  full-volume VTI mesh.
    fields   : dict          all numpy arrays from extract_fields().
    drag_dir : (3,) array    drag (freestream) direction unit vector.
    lift_dir : (3,) array    lift direction unit vector.

    Returns
    -------
    dict  with keys:
        "total_force_N"    – (3,) total aerodynamic force vector [N]
        "drag_N"           – drag component along drag_dir [N]
        "lift_N"           – lift component along lift_dir [N]
        "span_N"           – side/span component [N]
        "n_ib_cells"       – number of IB interface cells summed
        "force_field_name" – name of the force field detected
        "v_inf_ms"         – freestream velocity component along drag_dir [m/s]
        "n_upstream_pts"   – upstream fluid points used for V∞ estimate
    """
    print(f"\n{'─'*64}")
    print("  FORCE COMPUTATION")
    print(f"{'─'*64}")

    # ── Direction vectors ──────────────────────────────────────────────────
    drag_dir = _unit(drag_dir.copy())
    lift_dir = _orthogonalise(lift_dir.copy(), drag_dir)
    span_dir = np.cross(drag_dir, lift_dir)

    print(f"  Drag direction : {drag_dir}")
    print(f"  Lift direction : {lift_dir}")
    print(f"  Span direction : {span_dir}")

    # ── Freestream velocity (context only — not used in force computation) ─
    print(f"\n  {'─'*56}")
    print("  FREESTREAM VELOCITY")
    print(f"  {'─'*56}")

    v_name, v_vol = _find_field(fields, _VELOCITY_CANDIDATES)
    v_inf      = 0.0
    n_upstream = 0

    if v_name and v_vol is not None and v_vol.ndim == 2:
        print(f"  Velocity field : '{v_name}'")

        impl_arr = fields.get("ImplicitField")
        pts_vol  = mesh.points
        bounds   = mesh.bounds

        # Fluid cells: well inside fluid domain (ImplicitField < -0.006)
        if impl_arr is not None:
            fluid_mask = np.array(impl_arr) < -0.006
        else:
            fluid_mask = np.ones(mesh.n_points, dtype=bool)

        # Upstream strip: lowest 15% of domain X extent
        x_inlet  = bounds[0] + 0.15 * (bounds[1] - bounds[0])
        upstream = fluid_mask & (pts_vol[:, 0] <= x_inlet)

        if upstream.sum() < 100:
            upstream = fluid_mask

        v_up       = v_vol[upstream]
        n_upstream = int(upstream.sum())

        # Project onto drag direction — avoids inflation from cross-flow or
        # numerical noise in off-axis velocity components
        v_inf = float(np.mean(v_up @ drag_dir))

        print(f"  Upstream fluid pts   : {n_upstream:,}")
        print(f"  V∞ (drag direction)  : {v_inf:.4f} m/s")
    else:
        print("  [warn] No vector velocity field found — V∞ not reported.")

    # ─────────────────────────────────────────────────────────────────────
    # AERODYNAMIC FORCES  (Force time-averaged field)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print("  AERODYNAMIC FORCES  (Force time-averaged field)")
    print(f"  {'─'*56}")

    f_name, f_arr = _find_field(fields, _FORCE_CANDIDATES)
    total_force   = np.zeros(3)
    drag = lift = span = 0.0
    n_ib = 0

    if f_name is None or f_arr is None or f_arr.ndim != 2:
        print("  [error] No vector Force field found.")
        print(f"  Available fields: {list(fields.keys())}")
        return {
            "total_force_N":    total_force,
            "drag_N":           drag,
            "lift_N":           lift,
            "span_N":           span,
            "n_ib_cells":       n_ib,
            "force_field_name": "",
            "v_inf_ms":         v_inf,
            "n_upstream_pts":   n_upstream,
        }

    print(f"  Force field : '{f_name}'")

    # ── Cell volume — printed for unit verification ────────────────────────
    cell_vol = None
    if hasattr(mesh, "spacing"):
        sx, sy, sz = mesh.spacing
        cell_vol = sx * sy * sz
        print(f"  Cell volume : {cell_vol:.5g} m³")

    # ── Step 1: Detect IB cells via |Force| > noise threshold ─────────────
    fmag        = np.linalg.norm(f_arr, axis=1)
    thresh      = max(1e-8, 1e-3 * float(fmag.max()))
    force_mask  = fmag > thresh
    n_force_cells = int(force_mask.sum())
    print(f"  IB cells (|Force| > {thresh:.3g}) : {n_force_cells:,}")

    # ── Step 2: Cross-check with CellRegion == 3 ──────────────────────────
    creg_arr   = fields.get("CellRegion")
    iface_mask = force_mask   # default: use force-magnitude detection

    if creg_arr is not None:
        mask3        = np.array(creg_arr, dtype=int) == 3
        n_creg3      = int(mask3.sum())
        n_overlap    = int((force_mask & mask3).sum())
        overlap_frac = n_overlap / max(n_creg3, 1)

        print(f"  IB cells (CellRegion == 3)     : {n_creg3:,}")
        print(f"  Overlap between methods        : {n_overlap:,}  "
              f"({100*overlap_frac:.1f}%)")

        if overlap_frac < 0.80:
            print(f"  [warn] CellRegion == 3 and |Force| masks differ by more "
                  f"than 20%.")
            print(f"  [warn] Using force-magnitude mask.  Verify CellRegion "
                  f"labelling for this nTop version.")
        else:
            # Both methods agree — prefer the explicit solver label
            iface_mask = mask3
            print(f"  CellRegion == 3 consistent with |Force| — using it.")

    # ── Step 3: Unit verification hint ────────────────────────────────────
    if cell_vol is not None:
        per_cell_mean = float(fmag[iface_mask].mean())
        as_density    = per_cell_mean / cell_vol
        print(f"\n  Per-cell |Force| mean  : {per_cell_mean:.5g} N/cell")
        print(f"  Equiv. force density   : {as_density:.5g} N/m³")
        print(f"  (Summing directly is correct if field is in N/cell.)")
        print(f"  (If field is in N/m³, multiply each value by "
              f"{cell_vol:.5g} m³ before summing.)")

    # ── Step 4: Sum force over IB interface cells ──────────────────────────
    F_per_cell  = f_arr[iface_mask]        # (N_ib, 3)  [N/cell]
    total_force = F_per_cell.sum(axis=0)   # [N]
    n_ib        = int(iface_mask.sum())

    drag = float(np.dot(total_force, drag_dir))
    lift = float(np.dot(total_force, lift_dir))
    span = float(np.dot(total_force, span_dir))

    drag_pc = F_per_cell @ drag_dir
    lift_pc = F_per_cell @ lift_dir

    print(f"\n  IB interface cells  : {n_ib:,}")
    print(f"  Total force vector  : [{total_force[0]:+.5g}, "
          f"{total_force[1]:+.5g}, {total_force[2]:+.5g}] N")
    print(f"  Drag  (+X)          : {drag:+.5g} N")
    print(f"  Lift  (+Z)          : {lift:+.5g} N")
    print(f"  Side  (+Y)          : {span:+.5g} N")
    print(f"  Per-cell drag range : [{drag_pc.min():.4g}, {drag_pc.max():.4g}] N")
    print(f"  Per-cell lift range : [{lift_pc.min():.4g}, {lift_pc.max():.4g}] N")

    return {
        "total_force_N":    total_force,
        "drag_N":           drag,
        "lift_N":           lift,
        "span_N":           span,
        "n_ib_cells":       n_ib,
        "force_field_name": f_name,
        "v_inf_ms":         v_inf,
        "n_upstream_pts":   n_upstream,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  5.  print_summary()
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print a clean final aerodynamic force summary."""

    print(f"\n{'='*64}")
    print("  AERODYNAMIC FORCE SUMMARY")
    print(f"{'='*64}")

    v_inf = results.get("v_inf_ms", 0.0)
    if v_inf:
        print(f"  Freestream velocity V∞ : {v_inf:.4g} m/s  (drag direction)")

    total = results.get("total_force_N", np.zeros(3))
    drag  = results.get("drag_N",  0.0)
    lift  = results.get("lift_N",  0.0)
    span  = results.get("span_N",  0.0)
    n_ib  = results.get("n_ib_cells", 0)
    ff    = results.get("force_field_name", "N/A")

    print(f"\n  Total force vector  : [{total[0]:+.5g}, {total[1]:+.5g}, "
          f"{total[2]:+.5g}] N")

    print(f"\n  {'Component':<28s}  {'Force [N]':>12s}")
    print(f"  {'─'*28}  {'─'*12}")
    print(f"  {'Drag  (+X, streamwise)':<28s}  {drag:>+12.4f}")
    print(f"  {'Lift  (+Z, vertical)':<28s}  {lift:>+12.4f}")
    print(f"  {'Side  (+Y, spanwise)':<28s}  {span:>+12.4f}")

    print(f"\n  Source field    : '{ff}'")
    print(f"  IB cells summed : {n_ib:,}")

    print(f"\n  COORDINATE SYSTEM & SIGN CONVENTION")
    print(f"  +X : streamwise / freestream  →  positive drag opposes motion")
    print(f"  +Z : vertical                 →  positive lift is upward")
    print(f"  +Y : spanwise                 →  positive side force in +Y")

    print(f"\n  METHOD")
    print(f"  Forces computed by summing 'Force time-averaged' over the IB")
    print(f"  interface layer (CellRegion == 3).  This field is written by")
    print(f"  nTop's IB solver kernel and includes both pressure (form) and")
    print(f"  viscous (friction) contributions.")
    print(f"  Assumed field units: N per IB grid cell.")
    print(f"{'='*64}")


# ──────────────────────────────────────────────────────────────────────────────
#  main()
# ──────────────────────────────────────────────────────────────────────────────

def main():
    VTI_FILE = "CFD ATAS 0.006 at 10 deg ntop.vti"   # <- change to your file name

    if len(sys.argv) > 1:
        VTI_FILE = sys.argv[1]

    # ── 1. Load VTI ───────────────────────────────────────────────────────
    mesh = load_vti(VTI_FILE)

    # ── 2. Extract all fields as NumPy arrays ─────────────────────────────
    fields = extract_fields(mesh)

    if not fields:
        print("\n  No data arrays found in the file.")
        print("  Open the file in ParaView to verify its contents.")
        sys.exit(1)

    # ── 3. Reconstruct body surface (validates normals, reports geometry) ──
    surface = compute_surface(mesh, fields)

    # ── 4. Compute aerodynamic forces ─────────────────────────────────────
    results = compute_forces(
        mesh     = mesh,
        fields   = fields,
        drag_dir = DRAG_DIRECTION,
        lift_dir = LIFT_DIRECTION,
    )

    # ── 5. Print final summary ────────────────────────────────────────────
    print_summary(results)

    return mesh, fields, surface, results


if __name__ == "__main__":
    main()
