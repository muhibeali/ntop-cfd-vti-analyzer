"""
Smoke test — VTI field discovery
=================================
Run this FIRST before touching analyze_cfd_vti.py.
Confirms exactly what PyVista can see in the file, with exact field names
printed via repr() so hidden characters or encoding differences are visible.

Usage:
    python smoke_test_fields.py
    python smoke_test_fields.py "path/to/file.vti"
"""

import sys
import numpy as np
import pyvista as pv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VTI_FILE = "CFD ATAS 0.006 at 2.5 deg Implicit.vti"
if len(sys.argv) > 1:
    VTI_FILE = sys.argv[1]

# ── Reference parameters for nondimensionalisation ────────────────────────────
ALPHA_DEG  = 2.5             # angle of attack [deg] — positive: freestream from +X toward +Z
V_INF      = 30.48           # freestream speed [m/s]
RHO        = 1.2250123       # air density [kg/m³]
S_REF      = 0.4607990784    # reference wing area [m²]
T_INF      = 288.15          # freestream temperature [K]
MACH       = 0.0897          # freestream Mach number

print(f"File : {VTI_FILE}\n")

# ── Step 1: Read ──────────────────────────────────────────────────────────────
mesh = pv.read(VTI_FILE)

print(f"PyVista type : {type(mesh).__name__}")
print(f"Dimensions   : {getattr(mesh, 'dimensions', 'N/A')}")
print(f"N points     : {mesh.n_points:,}")
print(f"N cells      : {mesh.n_cells:,}")
if hasattr(mesh, "spacing"):
    print(f"Spacing      : {mesh.spacing}")

# ── Step 2: mesh.array_names (quickest complete inventory) ───────────────────
print(f"\n{'='*60}")
print("  ALL ARRAY NAMES  (mesh.array_names)")
print(f"{'='*60}")
for name in mesh.array_names:
    # Use repr() — shows spaces, dashes, encoding anomalies
    print(f"  {repr(name)}")

# ── Step 3: Point data ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  POINT DATA  (mesh.point_data)")
print(f"{'='*60}")
if len(mesh.point_data) == 0:
    print("  (empty)")
for name in mesh.point_data.keys():
    arr = np.asarray(mesh.point_data[name])
    fin = arr.ravel()[np.isfinite(arr.ravel())]
    rng = f"[{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "all NaN"
    print(f"  {repr(name):<45s}  shape={arr.shape}  dtype={arr.dtype}  range={rng}")

# ── Step 4: Cell data ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  CELL DATA  (mesh.cell_data)")
print(f"{'='*60}")
if len(mesh.cell_data) == 0:
    print("  (empty)")
for name in mesh.cell_data.keys():
    arr = np.asarray(mesh.cell_data[name])
    fin = arr.ravel()[np.isfinite(arr.ravel())]
    rng = f"[{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "all NaN"
    print(f"  {repr(name):<45s}  shape={arr.shape}  dtype={arr.dtype}  range={rng}")

# ── Step 5: Field / metadata ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FIELD DATA  (mesh.field_data)")
print(f"{'='*60}")
if len(mesh.field_data) == 0:
    print("  (empty)")
for name in mesh.field_data.keys():
    arr = np.asarray(mesh.field_data[name])
    print(f"  {repr(name):<45s}  shape={arr.shape}  dtype={arr.dtype}")

# ── Step 6: Force field — targeted extraction ─────────────────────────────────
print(f"\n{'='*60}")
print("  FORCE FIELD — targeted extraction")
print(f"{'='*60}")

all_names = (
    list(mesh.point_data.keys()) +
    list(mesh.cell_data.keys()) +
    list(mesh.field_data.keys())
)

force_hits = [n for n in all_names if "force" in n.lower()]
if not force_hits:
    print("  [!] No field with 'force' in its name found.")
else:
    for name in force_hits:
        # Find which container holds it
        for label, container in [
            ("point_data", mesh.point_data),
            ("cell_data",  mesh.cell_data),
            ("field_data", mesh.field_data),
        ]:
            if name in container:
                arr = np.asarray(container[name])
                print(f"  Found in {label}: {repr(name)}")
                print(f"    shape : {arr.shape}")
                print(f"    dtype : {arr.dtype}")
                if arr.ndim == 2:
                    mag = np.linalg.norm(arr, axis=1)
                    n_nonzero = int((mag > 1e-12).sum())
                    fin = mag[np.isfinite(mag)]
                    print(f"    |F| range     : [{fin.min():.5g}, {fin.max():.5g}]")
                    print(f"    Non-zero cells: {n_nonzero:,}")
                    print(f"    Sum of vectors: {arr.sum(axis=0)}")
                else:
                    fin = arr.ravel()[np.isfinite(arr.ravel())]
                    print(f"    range : [{fin.min():.5g}, {fin.max():.5g}]")
                break

# ── Step 7: Exact byte representation of field names ─────────────────────────
print(f"\n{'='*60}")
print("  FIELD NAME BYTES  (catches invisible character differences)")
print(f"{'='*60}")
for name in all_names:
    encoded = name.encode("utf-8")
    print(f"  {repr(name):<45s}  bytes={encoded}")

# ── Step 8: Aerodynamic forces + coefficients ────────────────────────────────
print(f"\n{'='*60}")
print("  AERODYNAMIC FORCES & COEFFICIENTS")
print(f"{'='*60}")
print(f"  Reference parameters:")
print(f"    Alpha  = {ALPHA_DEG} deg")
print(f"    V∞     = {V_INF} m/s")
print(f"    rho    = {RHO} kg/m³")
print(f"    S_ref  = {S_REF} m²")
print(f"    T∞     = {T_INF} K")
print(f"    Mach   = {MACH}")

# ── 8a. Extract CellRegion and Force time-averaged ────────────────────────────
if "CellRegion" not in mesh.point_data:
    print("  [!] CellRegion not found — cannot isolate IB interface.")
elif "Force time-averaged" not in mesh.point_data:
    print("  [!] 'Force time-averaged' not found in point_data.")
else:
    cell_region = np.asarray(mesh.point_data["CellRegion"])
    force_raw   = np.asarray(mesh.point_data["Force time-averaged"])  # (N,3) [N/pt]

    ib_mask   = cell_region == 3
    n_ib      = int(ib_mask.sum())
    F_ib      = force_raw[ib_mask]                    # (n_ib, 3)  body-frame [N]
    F_total   = F_ib.sum(axis=0)                      # (3,)       body-frame total [N]

    # ── 8b. Wind-axis rotation (nTop convention) ──────────────────────────────
    # Positive alpha: freestream tilts from +X toward +Z (about +Y axis)
    #   drag_axis = [cos(a), 0, sin(a)]
    #   lift_axis = [-sin(a), 0, cos(a)]
    alpha_rad = np.radians(ALPHA_DEG)
    drag_axis = np.array([ np.cos(alpha_rad), 0.0,  np.sin(alpha_rad)])
    lift_axis = np.array([-np.sin(alpha_rad), 0.0,  np.cos(alpha_rad)])
    span_axis = np.array([ 0.0,              -1.0,  0.0              ])

    drag_N = float(F_total @ drag_axis)
    lift_N = float(F_total @ lift_axis)
    side_N = float(F_total @ span_axis)

    # ── 8c. Dynamic pressure and coefficients ─────────────────────────────────
    q_inf = 0.5 * RHO * V_INF**2          # [Pa]
    qS    = q_inf * S_REF                  # [N]
    CD    = drag_N / qS
    CL    = lift_N / qS
    LD    = lift_N / drag_N if abs(drag_N) > 1e-12 else float("nan")

    print(f"\n  Dynamic pressure  q∞ = 0.5 × {RHO} × {V_INF}² = {q_inf:.4f} Pa")
    print(f"  q∞ × S_ref            = {qS:.4f} N")

    print(f"\n  IB interface points   : {n_ib:,}")
    print(f"  Body-frame total force: [{F_total[0]:+.5g}, {F_total[1]:+.5g}, {F_total[2]:+.5g}] N")

    print(f"\n  Wind axes (body frame):")
    print(f"    Drag axis : [{drag_axis[0]:+.6f}, {drag_axis[1]:+.6f}, {drag_axis[2]:+.6f}]")
    print(f"    Lift axis : [{lift_axis[0]:+.6f}, {lift_axis[1]:+.6f}, {lift_axis[2]:+.6f}]")

    print(f"\n  {'Component':<26s}  {'Force [N]':>12s}  {'Coefficient':>12s}")
    print(f"  {'─'*26}  {'─'*12}  {'─'*12}")
    print(f"  {'Drag  (freestream)':<26s}  {drag_N:>+12.4f}  CD = {CD:>+.6f}")
    print(f"  {'Lift  (perpendicular)':<26s}  {lift_N:>+12.4f}  CL = {CL:>+.6f}")
    print(f"  {'Side  (spanwise)':<26s}  {side_N:>+12.4f}  (no coeff)")

    print(f"\n  L/D = CL / CD = {CL:.6f} / {CD:.6f} = {LD:.4f}")

print(f"\n{'='*60}")
