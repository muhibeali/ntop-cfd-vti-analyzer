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

VTI_FILE = "CFD ATAS 0.006 at 10 deg ntop.vti"
if len(sys.argv) > 1:
    VTI_FILE = sys.argv[1]

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
