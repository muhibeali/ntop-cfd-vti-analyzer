"""
nTop CFD VTI File Analyzer
==========================
Loads a .vti (VTK ImageData) file from an nTop CFD simulation, inspects all
physical fields, reconstructs the body surface, and computes aerodynamic
lift and drag forces via pressure integration.

Coordinate System (default — change DRAG_DIRECTION / LIFT_DIRECTION below):
  +X  →  Streamwise / drag direction
  +Z  ↑  Vertical   / lift direction
  +Y  ·  Spanwise

Force Method:
  F_pressure = -∫ (p - p_ref) · n̂ dA   (pressure / form force)
  Viscous (friction) drag is NOT computed — it requires wall shear stress,
  which is rarely stored in VTI volume exports.

Usage:
  python analyze_cfd_vti.py [path/to/file.vti]
  — or set VTI_FILE in the main() block below.
"""

import sys
import numpy as np
import pyvista as pv
from pathlib import Path

# Force UTF-8 output so box-drawing and math symbols print correctly on Windows
# (Windows terminals default to cp1252 which cannot encode U+2500, U+221E, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
#  Global configuration  (edit these before running)
# ──────────────────────────────────────────────────────────────────────────────

# Flow / drag direction unit vector
DRAG_DIRECTION = np.array([1.0, 0.0, 0.0])

# Lift direction unit vector (must be perpendicular to drag)
LIFT_DIRECTION = np.array([0.0, 0.0, 1.0])

# Candidate field names searched in order for auto-detection.
# Exact names come first; a fallback substring search is also applied
# (see _find_field_fuzzy) so nTop names like "Pressure time-averaged" match.
PRESSURE_CANDIDATES = [
    # nTop-style names (space-separated with time stamp or qualifier)
    "Pressure time-averaged", "Pressure Time-Averaged",
    "Pressure initial", "Pressure Initial",
    # Generic CFD names
    "p", "P", "pressure", "Pressure", "PRESSURE",
    "p_rgh", "static_pressure", "StaticPressure",
    "pMean", "p_static", "p_total",
]

VELOCITY_CANDIDATES = [
    # nTop-style names
    "Velocity time-averaged", "Velocity Time-Averaged",
    "Velocity initial", "Velocity Initial",
    # Generic CFD names
    "U", "u", "velocity", "Velocity", "VELOCITY",
    "V", "vel", "UMean", "Umean", "v_velocity",
]

# Fields whose iso-surface at the listed value represents the body wall.
# nTop exports the implicit/signed-distance field as "ImplicitField";
# values < 0 = inside the body, 0 = body wall, > 0 = fluid domain.
BODY_LEVEL_SET_CANDIDATES = {
    # field name : iso value
    "ImplicitField": 0.0,  # nTop implicit geometry field
    "sdf":           0.0,
    "phi":           0.0,
    "levelset":      0.0,
    "LevelSet":      0.0,
    "alpha":         0.5,  # volume-of-fluid fraction
    "Alpha":         0.5,
    "alpha.water":   0.5,
    "body":          0.5,
    "Body":          0.5,
    "solid":         0.5,
    "Solid":         0.5,
    "mask":          0.5,
    "Mask":          0.5,
    "BodyFlag":      0.5,
    "bodyFlag":      0.5,
}

# ──────────────────────────────────────────────────────────────────────────────
#  1. Load
# ──────────────────────────────────────────────────────────────────────────────

def load_vti(filepath: str) -> pv.DataSet:
    """Read a .vti file and return the PyVista dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() not in (".vti", ".vtk"):
        print(f"  [warn] Expected .vti extension, got '{path.suffix}'")

    print(f"\n{'='*64}")
    print(f"  FILE  : {path.resolve()}")
    print(f"{'='*64}")

    mesh = pv.read(str(path))

    # Print basic mesh info
    print(f"  Type       : {type(mesh).__name__}")
    if hasattr(mesh, "dimensions"):
        print(f"  Dimensions : {mesh.dimensions}")
    if hasattr(mesh, "spacing"):
        print(f"  Spacing    : {mesh.spacing}")
    bounds = np.array(mesh.bounds).reshape(3, 2)
    print(f"  Bounds X   : [{bounds[0,0]:.4g},  {bounds[0,1]:.4g}]")
    print(f"  Bounds Y   : [{bounds[1,0]:.4g},  {bounds[1,1]:.4g}]")
    print(f"  Bounds Z   : [{bounds[2,0]:.4g},  {bounds[2,1]:.4g}]")
    print(f"  N points   : {mesh.n_points:,}")
    print(f"  N cells    : {mesh.n_cells:,}")

    return mesh


# ──────────────────────────────────────────────────────────────────────────────
#  2. Extract fields
# ──────────────────────────────────────────────────────────────────────────────

def extract_fields(mesh: pv.DataSet) -> dict[str, np.ndarray]:
    """
    Collect every data array (point, cell, field) into one flat dict.
    Prints a formatted table showing name, kind, shape, dtype, and value range.
    Returns:  { array_name: numpy_array, ... }
    """

    def _print_arrays(data_container, label: str) -> dict:
        out = {}
        if len(data_container) == 0:
            print("    (empty)")
            return out
        for name in data_container.keys():
            arr = np.array(data_container[name])          # ensure numpy
            kind = "scalar" if arr.ndim == 1 else f"vector-{arr.shape[1]}"
            # Flatten for min/max even on vectors
            flat = arr.ravel()
            finite = flat[np.isfinite(flat)]
            if finite.size == 0:
                vmin, vmax = float("nan"), float("nan")
            else:
                vmin, vmax = float(finite.min()), float(finite.max())
            n_nan = int(np.sum(~np.isfinite(flat)))
            nan_note = f"  [{n_nan} NaN/Inf]" if n_nan else ""
            print(f"    {name:<32s} {kind:<14s} "
                  f"shape={str(arr.shape):<22s} "
                  f"dtype={str(arr.dtype):<10s} "
                  f"range=[{vmin:.4g}, {vmax:.4g}]{nan_note}")
            out[name] = arr
        return out

    all_fields = {}

    print(f"\n{'─'*64}")
    print("  POINT DATA")
    print(f"{'─'*64}")
    all_fields.update(_print_arrays(mesh.point_data, "point"))

    print(f"\n{'─'*64}")
    print("  CELL DATA")
    print(f"{'─'*64}")
    all_fields.update(_print_arrays(mesh.cell_data, "cell"))

    print(f"\n{'─'*64}")
    print("  FIELD / METADATA")
    print(f"{'─'*64}")
    all_fields.update(_print_arrays(mesh.field_data, "field"))

    print(f"\n  All field names: {list(all_fields.keys())}")
    return all_fields


def _find_field(fields: dict, candidates: list[str]) -> tuple:
    """
    Return (name, array) for the first candidate found, else (None, None).
    Falls back to case-insensitive substring matching so names like
    "Pressure time-averaged" are matched by the candidate "pressure".
    """
    # 1. Exact match
    for name in candidates:
        if name in fields:
            return name, fields[name]
    # 2. Case-insensitive substring: field name contains a candidate token
    for name in candidates:
        needle = name.lower()
        for fname in fields:
            if needle in fname.lower():
                return fname, fields[fname]
    return None, None


# ──────────────────────────────────────────────────────────────────────────────
#  3. Surface extraction
# ──────────────────────────────────────────────────────────────────────────────

def compute_surface(
    mesh: pv.DataSet,
    fields: dict,
    iso_field: str = None,
    iso_value: float = None,
) -> pv.PolyData:
    """
    Extract the body surface from volume data using one of three strategies
    (tried in order):

      1. Explicit: caller supplies iso_field + iso_value.
      2. Auto:     scan BODY_LEVEL_SET_CANDIDATES for a matching field and
                   contour at its natural iso value.
      3. Fallback: extract the outer boundary of the volume mesh.

    For force computation, strategy 1 or 2 (body surface) gives accurate
    body forces.  Strategy 3 gives domain-boundary forces, which is correct
    only if the body completely fills one face of the domain.

    After extraction, all volume point arrays are sampled onto the surface
    so downstream force computation can access pressure etc.
    """
    print(f"\n{'─'*64}")
    print("  SURFACE EXTRACTION")
    print(f"{'─'*64}")

    surface = None

    # ── Strategy 1: explicit iso-surface ─────────────────────────────────────
    if iso_field is not None and iso_value is not None:
        print(f"  Strategy 1: iso-surface  '{iso_field}' = {iso_value}")
        if iso_field not in mesh.point_data and \
           iso_field not in mesh.cell_data:
            # Put the field on the mesh from our fields dict
            if iso_field in fields and fields[iso_field].shape[0] == mesh.n_points:
                mesh.point_data[iso_field] = fields[iso_field]
        try:
            surface = mesh.contour([iso_value], scalars=iso_field)
        except Exception as e:
            print(f"  [warn] Contour failed: {e}")

    # ── Strategy 2: auto-detect body / level-set field ────────────────────────
    if surface is None or surface.n_cells == 0:
        for fname, fval in BODY_LEVEL_SET_CANDIDATES.items():
            if fname in fields:
                arr = fields[fname]
                if arr.ndim != 1:
                    continue                       # need a scalar field
                print(f"  Strategy 2: iso-surface  '{fname}' = {fval}")
                # Attach to mesh if not already there
                if fname not in mesh.point_data:
                    if arr.shape[0] == mesh.n_points:
                        mesh.point_data[fname] = arr
                    elif arr.shape[0] == mesh.n_cells:
                        mesh.cell_data[fname] = arr
                        mesh = mesh.cell_data_to_point_data()
                    else:
                        continue
                try:
                    surf_candidate = mesh.contour([fval], scalars=fname)
                    if surf_candidate.n_cells > 0:
                        surface = surf_candidate
                        print(f"  Found body surface via '{fname}'")
                        break
                except Exception as e:
                    print(f"  [warn] Contour on '{fname}' failed: {e}")

    # ── Strategy 3: domain boundary ──────────────────────────────────────────
    if surface is None or surface.n_cells == 0:
        print(f"  Strategy 3: mesh boundary surface (domain walls)")
        print(f"  [warn] This includes ALL domain boundaries, not just the body.")
        print(f"         Set iso_field + iso_value in main() for body-only forces.")
        surface = mesh.extract_surface(algorithm="dataset_surface")

    # ── Report ────────────────────────────────────────────────────────────────
    if surface.n_cells == 0:
        raise RuntimeError("Surface extraction produced an empty mesh. "
                           "Check iso_field / iso_value settings.")

    print(f"\n  Surface: {surface.n_points:,} points, {surface.n_cells:,} cells")
    bounds = np.array(surface.bounds).reshape(3, 2)
    print(f"  Bounds X: [{bounds[0,0]:.4g}, {bounds[0,1]:.4g}]")
    print(f"  Bounds Y: [{bounds[1,0]:.4g}, {bounds[1,1]:.4g}]")
    print(f"  Bounds Z: [{bounds[2,0]:.4g}, {bounds[2,1]:.4g}]")

    # ── Sample all volume fields onto the surface ─────────────────────────────
    # pv.PolyData.sample() performs probe interpolation from the source volume.
    print(f"\n  Sampling volume data onto surface...")
    try:
        surface = surface.sample(mesh)
        print(f"  Sampled point arrays : {list(surface.point_data.keys())}")
    except Exception as e:
        print(f"  [warn] Sampling failed: {e}")
        print(f"         Forces will fall back to volume array lookup.")

    return surface


# ──────────────────────────────────────────────────────────────────────────────
#  4. Force computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_forces(
    surface: pv.PolyData,
    fields: dict,
    pressure_name: str = None,
    velocity_name: str = None,
    drag_dir: np.ndarray = DRAG_DIRECTION,
    lift_dir: np.ndarray = LIFT_DIRECTION,
    freestream_velocity: float = None,
    reference_area: float = None,
    fluid_density: float = 1.225,
) -> dict:
    """
    Integrate pressure over the surface to get aerodynamic forces.

    Pressure force on body:
        F = -∑_i (p_i - p_ref) * n̂_i * A_i

    The minus sign: outward normals point away from the body into the fluid,
    so the force the fluid exerts on the body is -p * n̂.

    Parameters
    ----------
    surface            PyVista PolyData surface (with sampled pressure).
    fields             All extracted volume arrays (fallback).
    pressure_name      Override auto-detection with a specific field name.
    velocity_name      Optional velocity field (used only for reporting).
    drag_dir           Unit vector in the drag (freestream) direction.
    lift_dir           Unit vector in the lift direction.
    freestream_velocity  m/s — if given, computes CL / CD.
    reference_area     m²  — reference area for CL/CD (defaults to total surface area).
    fluid_density      kg/m³ (default: 1.225 = sea-level air).

    Returns
    -------
    dict  with keys:
        total_force_N, drag_N, lift_N, span_N,
        CL, CD (if freestream_velocity given),
        and metadata.

    LIMITATIONS
    -----------
    - Only pressure (form) drag is computed.  Viscous friction drag requires
      wall shear stress τ_w = μ(∂u/∂n), not typically exported in VTI files.
    - p_ref defaults to the spatial mean of p over the surface (gauge approach).
      Set p_ref manually if your simulation uses absolute pressure with a known
      far-field value (e.g. 101325 Pa for sea-level standard atmosphere).
    - Results are most accurate when the surface is the true body wall, not the
      outer domain boundary.
    """
    print(f"\n{'─'*64}")
    print("  FORCE COMPUTATION")
    print(f"{'─'*64}")

    # ── Direction vectors ─────────────────────────────────────────────────────
    drag_dir = drag_dir / np.linalg.norm(drag_dir)
    lift_dir = lift_dir / np.linalg.norm(lift_dir)

    # Orthogonalise lift w.r.t. drag (Gram–Schmidt) so they're truly perpendicular
    lift_dir = lift_dir - np.dot(lift_dir, drag_dir) * drag_dir
    if np.linalg.norm(lift_dir) < 1e-10:
        raise ValueError("LIFT_DIRECTION is parallel to DRAG_DIRECTION — choose perpendicular axes.")
    lift_dir = lift_dir / np.linalg.norm(lift_dir)
    span_dir = np.cross(drag_dir, lift_dir)

    print(f"  Drag direction (+X) : {drag_dir}")
    print(f"  Lift direction (+Z) : {lift_dir}")
    print(f"  Span direction (+Y) : {span_dir}")

    # ── Locate pressure array ─────────────────────────────────────────────────
    p_candidates = [pressure_name] if pressure_name else PRESSURE_CANDIDATES
    p_name = None
    p_at_cells = None

    def _search_surface_arrays(data_dict, candidates, label):
        """Exact then case-insensitive substring match against a surface data dict."""
        # Exact match first
        for name in candidates:
            if name in data_dict:
                return name
        # Fuzzy: any candidate token is a substring of a field name
        for cand in candidates:
            needle = cand.lower()
            for fname in data_dict.keys():
                if needle in fname.lower():
                    print(f"  [fuzzy] Matched '{fname}' via candidate '{cand}' in {label}")
                    return fname
        return None

    # Search surface point data → convert to cells
    matched = _search_surface_arrays(surface.point_data, p_candidates, "surface point data")
    if matched:
        p_name = matched
        surf_cc = surface.point_data_to_cell_data()
        p_at_cells = np.array(surf_cc.cell_data[p_name], dtype=float)
        print(f"\n  Pressure '{p_name}' found in surface point data → averaged to cell centers")

    # Search surface cell data directly
    if p_name is None:
        matched = _search_surface_arrays(surface.cell_data, p_candidates, "surface cell data")
        if matched:
            p_name = matched
            p_at_cells = np.array(surface.cell_data[p_name], dtype=float)
            print(f"\n  Pressure '{p_name}' found in surface cell data")

    # Fallback: search the volume fields dict
    if p_name is None:
        p_name_v, p_arr_v = _find_field(fields, p_candidates)
        if p_name_v is not None:
            print(f"\n  [warn] Pressure '{p_name_v}' not on surface — "
                  f"using volume point array (less accurate).")
            p_name = p_name_v
            # If it's a point field and matches the number of surface points,
            # we can use it directly (this happens when the surface IS the mesh boundary)
            if p_arr_v.shape[0] == surface.n_points:
                surf_cc = surface.copy()
                surf_cc.point_data[p_name] = p_arr_v
                surf_cc = surf_cc.point_data_to_cell_data()
                p_at_cells = np.array(surf_cc.cell_data[p_name], dtype=float)
            else:
                print(f"  [error] Shape mismatch: pressure has {p_arr_v.shape[0]} entries, "
                      f"surface has {surface.n_points} points.")
                print(f"  Cannot reliably map volume pressure to surface cells.")
                print(f"  Available fields: {list(fields.keys())}")
                return {}

    if p_name is None:
        print(f"\n  ERROR: No pressure field found.")
        print(f"  Searched for: {p_candidates}")
        print(f"  Surface arrays (point): {list(surface.point_data.keys())}")
        print(f"  Surface arrays (cell) : {list(surface.cell_data.keys())}")
        print(f"  Volume fields         : {list(fields.keys())}")
        print(f"\n  FIX: Set PRESSURE_FIELD = '<name>' in main() to override.")
        return {}

    # ── Report pressure stats ─────────────────────────────────────────────────
    p_finite = p_at_cells[np.isfinite(p_at_cells)]
    if p_finite.size == 0:
        print("  ERROR: All pressure values are NaN/Inf — cannot compute forces.")
        return {}

    print(f"  N cells with pressure : {p_finite.size:,}")
    print(f"  Pressure range        : [{p_finite.min():.4g},  {p_finite.max():.4g}]")

    # ── Velocity field (for reporting only) ───────────────────────────────────
    v_candidates = [velocity_name] if velocity_name else VELOCITY_CANDIDATES
    v_name, v_arr = _find_field(fields, v_candidates)
    if v_name:
        v_mag = np.linalg.norm(v_arr, axis=1) if v_arr.ndim == 2 else np.abs(v_arr)
        print(f"  Velocity '{v_name}'        : "
              f"magnitude range [{v_mag.min():.4g}, {v_mag.max():.4g}]")
    else:
        print(f"  Velocity field not found (set VELOCITY_FIELD= to specify)")

    # ── Cell normals and areas ────────────────────────────────────────────────
    # auto_orient_normals=True: VTK orients all normals to point away from
    # the mesh centroid (outward from body into fluid). This is essential —
    # with False, the seed direction is arbitrary and forces can be negated.
    surface = surface.compute_normals(
        cell_normals=True, point_normals=False,
        consistent_normals=True, auto_orient_normals=True,
        flip_normals=False,
    )
    surface = surface.compute_cell_sizes(length=False, area=True, volume=False)

    normals = np.array(surface.cell_data["Normals"], dtype=float)   # (N,3)
    areas   = np.array(surface.cell_data["Area"],    dtype=float)   # (N,)

    # Diagnostic: mean normal should be near zero for a closed surface.
    # A large mean indicates an unclosed surface or orientation problem.
    mean_normal = normals.mean(axis=0)
    print(f"  Mean normal vector    : [{mean_normal[0]:+.4g}, {mean_normal[1]:+.4g}, {mean_normal[2]:+.4g}]")
    print(f"  (Should be ~[0,0,0] for a closed surface; large values → unclosed or flipped normals)")

    total_area = float(areas.sum())
    print(f"\n  Surface cells         : {surface.n_cells:,}")
    print(f"  Total surface area    : {total_area:.6g} m²")

    # Safety: discard any cells with zero/nan area or degenerate normals
    valid = (areas > 0) & np.all(np.isfinite(normals), axis=1) & np.isfinite(p_at_cells)
    if valid.sum() < surface.n_cells:
        print(f"  [warn] Discarding {surface.n_cells - valid.sum()} degenerate cells")
    p_valid  = p_at_cells[valid]
    n_valid  = normals[valid]
    a_valid  = areas[valid]

    # ── Reference (gauge) pressure ────────────────────────────────────────────
    # Subtract mean to convert absolute → gauge, which eliminates the net
    # pressure force on a closed surface in a uniform field.
    p_ref   = float(p_valid.mean())
    p_gauge = p_valid - p_ref
    print(f"  Reference pressure    : {p_ref:.6g} (mean, gauge approach)")
    print(f"  Gauge pressure range  : [{p_gauge.min():.4g}, {p_gauge.max():.4g}]")

    # ── Integrate ─────────────────────────────────────────────────────────────
    # dF_i = -p_gauge_i * n̂_i * dA_i   (fluid pushes inward on body)
    force_per_cell = -p_gauge[:, np.newaxis] * n_valid * a_valid[:, np.newaxis]  # (N,3)
    F_total = force_per_cell.sum(axis=0)   # (3,)

    drag = float(np.dot(F_total, drag_dir))
    lift = float(np.dot(F_total, lift_dir))
    span = float(np.dot(F_total, span_dir))

    print(f"\n  ── Force Results {'─'*44}")
    print(f"  Total force vector    : [{F_total[0]:+.6g}, {F_total[1]:+.6g}, {F_total[2]:+.6g}] N")
    print(f"  Drag  (along {drag_dir}) : {drag:+.6g} N")
    print(f"  Lift  (along {lift_dir}) : {lift:+.6g} N")
    print(f"  Span  (along {span_dir}) : {span:+.6g} N")

    results = {
        "total_force_N":       F_total,
        "drag_N":              drag,
        "lift_N":              lift,
        "span_N":              span,
        "pressure_name":       p_name,
        "velocity_name":       v_name,
        "surface_area_m2":     total_area,
        "p_ref_Pa":            p_ref,
        "force_per_cell":      force_per_cell,   # (N,3) for post-processing
        "valid_cell_mask":     valid,
    }

    # ── Aerodynamic coefficients ──────────────────────────────────────────────
    if freestream_velocity is not None and freestream_velocity > 0.0:
        q_inf = 0.5 * fluid_density * freestream_velocity**2
        A_ref = reference_area if reference_area else total_area
        CL = lift / (q_inf * A_ref)
        CD = drag / (q_inf * A_ref)
        print(f"\n  Dynamic pressure q∞   : {q_inf:.6g} Pa")
        print(f"  Reference area A_ref  : {A_ref:.6g} m²")
        print(f"  CL                    : {CL:.6f}")
        print(f"  CD                    : {CD:.6f}")
        if CD != 0:
            print(f"  L/D ratio             : {CL/CD:.4f}")
        results.update({"CL": CL, "CD": CD, "q_inf_Pa": q_inf, "A_ref_m2": A_ref})

    # ── Per-cell contribution stats ───────────────────────────────────────────
    drag_contrib = force_per_cell @ drag_dir        # (N,)
    lift_contrib = force_per_cell @ lift_dir        # (N,)
    print(f"\n  Per-cell drag range   : [{drag_contrib.min():.4g}, {drag_contrib.max():.4g}] N")
    print(f"  Per-cell lift range   : [{lift_contrib.min():.4g}, {lift_contrib.max():.4g}] N")
    results["drag_per_cell"] = drag_contrib
    results["lift_per_cell"] = lift_contrib

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  5. Viscous (friction) force computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_viscous_forces(
    mesh: pv.DataSet,
    surface: pv.PolyData,
    fields: dict,
    velocity_name: str = None,
    drag_dir: np.ndarray = DRAG_DIRECTION,
    lift_dir: np.ndarray = LIFT_DIRECTION,
    dynamic_viscosity: float = 1.81e-5,
) -> dict:
    """
    Compute viscous (friction / skin-friction) forces from the velocity gradient.

    For a Newtonian incompressible fluid the viscous stress tensor is:
        tau_ij = mu * (du_i/dx_j + du_j/dx_i)

    The traction on the wall (force per unit area) is:
        t_i = sum_j  tau_ij * n_j

    Integrating over the surface gives the viscous aerodynamic force.

    Parameters
    ----------
    mesh              Full-volume PyVista mesh (VTI ImageData).
    surface           Body surface PolyData (must have been sampled from mesh).
    fields            Dict of all extracted volume arrays.
    velocity_name     Override auto-detected velocity field name.
    drag_dir          Drag (streamwise) direction unit vector.
    lift_dir          Lift direction unit vector.
    dynamic_viscosity mu in Pa*s (default: 1.81e-5 = air at 20 degC, 1 atm).

    Returns
    -------
    dict with viscous_force_N, viscous_drag_N, viscous_lift_N, etc.

    LIMITATIONS
    -----------
    - The velocity gradient is computed by finite differences on the Cartesian
      grid (PyVista compute_derivative).  At the body surface the gradient
      stencil straddles both solid and fluid cells, which introduces smearing
      over roughly one grid cell width (~0.02 m here).  This is acceptable as
      a first-order approximation but will underestimate wall shear stress
      compared to a body-fitted mesh with resolved boundary layer cells.
    - For high-Re flows the boundary layer is very thin; viscous drag is often
      <5% of pressure drag.  These results are most meaningful for low-Re or
      transitional flows.
    - Only Newtonian fluid model is implemented; turbulent effective viscosity
      is not included.
    """
    print(f"\n{'─'*64}")
    print("  VISCOUS FORCE COMPUTATION")
    print(f"{'─'*64}")
    print(f"  Dynamic viscosity mu  : {dynamic_viscosity:.4g} Pa*s")

    drag_dir = drag_dir / np.linalg.norm(drag_dir)
    lift_dir = lift_dir - np.dot(lift_dir, drag_dir) * drag_dir
    lift_dir = lift_dir / np.linalg.norm(lift_dir)
    span_dir = np.cross(drag_dir, lift_dir)

    # ── Find velocity field ───────────────────────────────────────────────────
    v_candidates = [velocity_name] if velocity_name else VELOCITY_CANDIDATES
    v_name, _ = _find_field(fields, v_candidates)
    if v_name is None:
        print("  [error] No velocity field found — cannot compute viscous forces.")
        print(f"  Available fields: {list(fields.keys())}")
        return {}
    print(f"  Velocity field        : '{v_name}'")

    # ── Ensure velocity is in point_data (compute_derivative requires it) ────
    if v_name not in mesh.point_data:
        if v_name in mesh.cell_data:
            print(f"  '{v_name}' is in cell_data — converting to point_data for gradient...")
            mesh = mesh.cell_data_to_point_data()
        else:
            print(f"  [error] '{v_name}' not in mesh point_data or cell_data.")
            return {}

    # ── Compute velocity gradient on full volume (finite differences) ─────────
    print("  Computing velocity gradient tensor on volume mesh...")
    try:
        grad_mesh = mesh.compute_derivative(
            scalars=v_name,
            gradient=True, divergence=False, vorticity=False, qcriterion=False,
        )
    except Exception as e:
        print(f"  [error] compute_derivative failed: {e}")
        return {}

    # 'gradient' is (N, 9) = flattened 3x3 Jacobian: J[i,j] = du_i/dx_j
    grad_arr = np.array(grad_mesh.point_data["gradient"])
    print(f"  Gradient tensor shape : {grad_arr.shape}  "
          f"range=[{grad_arr.min():.4g}, {grad_arr.max():.4g}]")

    # ── Sample gradient onto body surface ─────────────────────────────────────
    print("  Sampling gradient onto body surface...")
    try:
        surf_with_grad = surface.sample(grad_mesh)
    except Exception as e:
        print(f"  [error] Sampling failed: {e}")
        return {}

    if "gradient" not in surf_with_grad.point_data.keys():
        print("  [error] 'gradient' key missing after surface sample.")
        return {}

    # Interpolate point data to cell centers for integration
    surf_cc = surf_with_grad.point_data_to_cell_data()

    # ── Recompute surface normals and areas ───────────────────────────────────
    surf_cc = surf_cc.compute_normals(
        cell_normals=True, point_normals=False,
        consistent_normals=True, auto_orient_normals=True, flip_normals=False,
    )
    surf_cc = surf_cc.compute_cell_sizes(length=False, area=True, volume=False)

    normals = np.array(surf_cc.cell_data["Normals"], dtype=float)   # (N, 3)
    areas   = np.array(surf_cc.cell_data["Area"],    dtype=float)   # (N,)

    if "gradient" not in surf_cc.cell_data.keys():
        print("  [error] 'gradient' not in cell data after point_data_to_cell_data.")
        return {}

    grad_cells = np.array(surf_cc.cell_data["gradient"], dtype=float)  # (N, 9)

    # ── Filter valid cells ────────────────────────────────────────────────────
    valid = (
        (areas > 0)
        & np.all(np.isfinite(normals), axis=1)
        & np.all(np.isfinite(grad_cells), axis=1)
    )
    n_invalid = int((~valid).sum())
    if n_invalid:
        print(f"  [warn] Discarding {n_invalid} cells with NaN/zero data")

    J = grad_cells[valid].reshape(-1, 3, 3)   # (M, 3, 3)  J[m, i, j] = du_i/dx_j
    n_v = normals[valid]                        # (M, 3)
    a_v = areas[valid]                          # (M,)

    # ── Viscous stress tensor and wall traction ───────────────────────────────
    # tau_ij = mu * (du_i/dx_j + du_j/dx_i) = mu * (J_ij + J_ji)
    tau = dynamic_viscosity * (J + J.transpose(0, 2, 1))          # (M, 3, 3)

    # Wall traction: t_i = sum_j tau_ij * n_j  (force/area the fluid exerts on body)
    t = np.einsum("mij,mj->mi", tau, n_v)                          # (M, 3)

    # ── Integrate ─────────────────────────────────────────────────────────────
    visc_force_per_cell = t * a_v[:, np.newaxis]                    # (M, 3)
    F_viscous = visc_force_per_cell.sum(axis=0)                     # (3,)

    v_drag = float(np.dot(F_viscous, drag_dir))
    v_lift = float(np.dot(F_viscous, lift_dir))
    v_span = float(np.dot(F_viscous, span_dir))

    # Wall shear stress magnitude per cell
    tau_w_mag = np.linalg.norm(t, axis=1)
    print(f"\n  Wall shear stress     : mean={tau_w_mag.mean():.4g}  "
          f"max={tau_w_mag.max():.4g}  Pa")
    print(f"\n  ── Viscous Force Results {'─'*38}")
    print(f"  Viscous force vector  : [{F_viscous[0]:+.6g}, "
          f"{F_viscous[1]:+.6g}, {F_viscous[2]:+.6g}] N")
    print(f"  Viscous drag          : {v_drag:+.6g} N")
    print(f"  Viscous lift          : {v_lift:+.6g} N")
    print(f"  Viscous span          : {v_span:+.6g} N")

    return {
        "viscous_force_N":    F_viscous,
        "viscous_drag_N":     v_drag,
        "viscous_lift_N":     v_lift,
        "viscous_span_N":     v_span,
        "velocity_name":      v_name,
        "dynamic_viscosity":  dynamic_viscosity,
        "tau_w_mean_Pa":      float(tau_w_mag.mean()),
        "tau_w_max_Pa":       float(tau_w_mag.max()),
        "visc_force_per_cell": visc_force_per_cell,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  6. Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(pressure_results: dict, viscous_results: dict = None) -> None:
    """Print a clean final summary combining pressure and viscous forces."""
    if not pressure_results:
        print("\n  No force results to display.")
        return

    F_p = pressure_results["total_force_N"]
    p_drag = pressure_results["drag_N"]
    p_lift = pressure_results["lift_N"]
    p_span = pressure_results["span_N"]

    have_visc = bool(viscous_results)
    if have_visc:
        F_v    = viscous_results["viscous_force_N"]
        v_drag = viscous_results["viscous_drag_N"]
        v_lift = viscous_results["viscous_lift_N"]
        v_span = viscous_results["viscous_span_N"]
        F_tot  = F_p + F_v
        t_drag = p_drag + v_drag
        t_lift = p_lift + v_lift
        t_span = p_span + v_span
    else:
        F_tot  = F_p
        t_drag, t_lift, t_span = p_drag, p_lift, p_span

    print(f"\n{'='*64}")
    print("  AERODYNAMIC FORCE SUMMARY")
    print(f"{'='*64}")

    if have_visc:
        print(f"  {'Component':<28s}  {'Drag':>12s}  {'Lift':>12s}  {'Span':>12s}")
        print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*12}")
        print(f"  {'Pressure (form)  [N]':<28s}  {p_drag:>+12.4f}  {p_lift:>+12.4f}  {p_span:>+12.4f}")
        print(f"  {'Viscous (friction) [N]':<28s}  {v_drag:>+12.4f}  {v_lift:>+12.4f}  {v_span:>+12.4f}")
        print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*12}")
        print(f"  {'TOTAL  [N]':<28s}  {t_drag:>+12.4f}  {t_lift:>+12.4f}  {t_span:>+12.4f}")
        if p_drag != 0:
            print(f"\n  Viscous / Pressure drag ratio : {abs(v_drag/p_drag)*100:.2f}%")
        print(f"  Wall shear stress   : "
              f"mean={viscous_results['tau_w_mean_Pa']:.4g} Pa  "
              f"max={viscous_results['tau_w_max_Pa']:.4g} Pa")
    else:
        print(f"  Total force vector  : [{F_tot[0]:+.4f},  {F_tot[1]:+.4f},  {F_tot[2]:+.4f}]  N")
        print(f"  Drag                : {t_drag:+.4f} N")
        print(f"  Lift                : {t_lift:+.4f} N")
        print(f"  Span (side force)   : {t_span:+.4f} N")

    if "CL" in pressure_results:
        print(f"\n  CL (pressure)       : {pressure_results['CL']:.6f}")
        print(f"  CD (pressure)       : {pressure_results['CD']:.6f}")

    print(f"\n  Pressure field used : '{pressure_results['pressure_name']}'")
    print(f"  Surface area        : {pressure_results['surface_area_m2']:.4g} m^2")
    print(f"  Reference pressure  : {pressure_results['p_ref_Pa']:.4g} Pa")
    if have_visc:
        print(f"  Velocity field used : '{viscous_results['velocity_name']}'")
        print(f"  Dynamic viscosity   : {viscous_results['dynamic_viscosity']:.4g} Pa*s")

    print(f"\n  NOTES")
    print(f"  * Viscous forces are a first-order approximation from finite-difference")
    print(f"    velocity gradients on the Cartesian grid (~1 cell width smearing).")
    print(f"  * p_ref = mean surface pressure (gauge).  For absolute p,")
    print(f"    set p_ref = far-field value (e.g. 101325 Pa).")
    print(f"  * ImplicitField iso-surface used as body wall (nTop SDF at 0).")
    print(f"{'='*64}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── File path ─────────────────────────────────────────────────────────────
    # Either pass the path as a command-line argument or set it here:
    VTI_FILE = "CFD Analysis of ATAS implicit.vti"   # <- change to your actual file name/path

    if len(sys.argv) > 1:
        VTI_FILE = sys.argv[1]

    # ── Override auto-detection (set to None to use auto) ────────────────────
    PRESSURE_FIELD      = None        # e.g. "Pressure time-averaged"
    VELOCITY_FIELD      = None        # e.g. "Velocity time-averaged"

    # Body surface extraction:
    #   ImplicitField is the nTop SDF; iso-surface at 0 = body wall.
    #   Set to None for auto-detection (will find ImplicitField automatically).
    ISO_FIELD           = None        # e.g. "ImplicitField"
    ISO_VALUE           = None        # e.g. 0.0

    # Freestream conditions (set for CL / CD output)
    FREESTREAM_VEL_MS   = None        # m/s  — e.g. 50.0
    REFERENCE_AREA_M2   = None        # m^2  — e.g. 0.04  (frontal area)
    FLUID_DENSITY_KGM3  = 1.225       # kg/m^3

    # Fluid dynamic viscosity for viscous drag computation
    # Air at 20 degC, 1 atm: 1.81e-5 Pa*s
    DYNAMIC_VISCOSITY   = 1.81e-5     # Pa*s

    # ─────────────────────────────────────────────────────────────────────────

    # 1. Load the VTI file
    mesh = load_vti(VTI_FILE)

    # 2. Inspect and extract all available field arrays
    fields = extract_fields(mesh)

    if not fields:
        print("\n  No data arrays found in the file.")
        print("  Verify the VTI contains field data (open in ParaView to inspect).")
        sys.exit(1)

    print(f"\n  >>> To override pressure / velocity field names, set:")
    print(f"      PRESSURE_FIELD = '<name from list above>'")
    print(f"      VELOCITY_FIELD = '<name from list above>'")

    # 3. Extract the body surface from the volume mesh
    surface = compute_surface(
        mesh, fields,
        iso_field=ISO_FIELD,
        iso_value=ISO_VALUE,
    )

    # 4. Compute pressure (form) forces
    pressure_results = compute_forces(
        surface,
        fields,
        pressure_name       = PRESSURE_FIELD,
        velocity_name       = VELOCITY_FIELD,
        drag_dir            = DRAG_DIRECTION,
        lift_dir            = LIFT_DIRECTION,
        freestream_velocity = FREESTREAM_VEL_MS,
        reference_area      = REFERENCE_AREA_M2,
        fluid_density       = FLUID_DENSITY_KGM3,
    )

    # 5. Compute viscous (friction) forces from velocity gradient
    viscous_results = compute_viscous_forces(
        mesh,
        surface,
        fields,
        velocity_name       = VELOCITY_FIELD,
        drag_dir            = DRAG_DIRECTION,
        lift_dir            = LIFT_DIRECTION,
        dynamic_viscosity   = DYNAMIC_VISCOSITY,
    )

    # 6. Print combined summary
    print_summary(pressure_results, viscous_results)

    # Return all objects for interactive / notebook use
    return mesh, fields, surface, pressure_results, viscous_results


if __name__ == "__main__":
    main()
