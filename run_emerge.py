import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import emerge as em
from emerge.plot import smith
from emerge.plot import plot_ff_polar

# NEW: exporters
from emerge.write import FarFieldExporter  # :contentReference[oaicite:2]{index=2}


# ----------------------------- Helpers ---------------------------------

def unit_scale(units: str) -> float:
    u = units.strip().lower()
    if u in ("mm", "millimeter", "millimeters"):
        return 0.001
    if u in ("m", "meter", "meters"):
        return 1.0
    raise ValueError(f"Unsupported units '{units}'. Use 'mm' or 'm'.")


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_exists(path: Path, what: str = "file") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def resolve_cfg_path(cfg_path: Path, maybe_rel: str) -> Path:
    """
    Resolve a file path from sim.json:
      - absolute -> as-is
      - relative -> relative to the folder containing sim.json
    """
    p = Path(maybe_rel).expanduser()
    if not p.is_absolute():
        p = cfg_path.parent / p
    return p.resolve()


def prompt_save_basename(initial_dir: Path, default_name: str) -> str:
    """
    Prompts user for a "filename" (Save As dialog when available).
    Returns the base-name without extension.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filename = filedialog.asksaveasfilename(
            title="Choose a name for storing EMerge simulation data",
            initialdir=str(initial_dir),
            initialfile=default_name,
            defaultextension=".emerge",
            filetypes=[("Any", "*.*")],
        )

        root.destroy()

        if not filename:
            raise RuntimeError("User cancelled file selection.")

        base = Path(filename).stem.strip()
        if not base:
            raise ValueError("Empty name selected.")
        return base

    except Exception as e:
        print(f"[INFO] File dialog not available ({e}). Falling back to console input.")
        base = input(f"Enter simulation base name [{default_name}]: ").strip() or default_name
        return base


def prompt_multiline_comment() -> str:
    """
    Multiline comment input. End by entering an empty line.
    """
    print("\nEnter comment (multi-line). Finish with an empty line:\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def make_port_plate_z(p1_m: np.ndarray, p2_m: np.ndarray, width_m: float) -> tuple[em.geo.Plate, float]:
    """
    Z-directed lumped port plate in XZ plane at y = yc, spanning X by width and Z by |p1.z - p2.z|.
    Returns (plate, port_height).
    """
    zmin = float(min(p1_m[2], p2_m[2]))
    zmax = float(max(p1_m[2], p2_m[2]))
    port_h = zmax - zmin
    if port_h <= 0:
        raise ValueError("Port height computed <= 0. Check port p1/p2 z coordinates.")

    xc = 0.5 * (float(p1_m[0]) + float(p2_m[0]))
    yc = 0.5 * (float(p1_m[1]) + float(p2_m[1]))

    corner = np.array([xc - width_m / 2.0, yc, zmin], dtype=float)
    wvec = np.array([width_m, 0.0, 0.0], dtype=float)
    hvec = np.array([0.0, 0.0, port_h], dtype=float)
    return em.geo.Plate(corner, wvec, hvec), port_h


def _safe_bounds_from_stepitems(stepitems):
    """
    Best-effort bounding box read (min,max) in meters.
    Returns (None, None) if not available.
    """
    candidates = ["bounds", "bbox", "bounding_box", "aabb"]
    for attr in candidates:
        if hasattr(stepitems, attr):
            try:
                b = getattr(stepitems, attr)
                b = b() if callable(b) else b
                if isinstance(b, (tuple, list)) and len(b) == 2:
                    bmin = np.array(b[0], dtype=float)
                    bmax = np.array(b[1], dtype=float)
                    return bmin, bmax
            except Exception:
                pass

    # Try per-volume bounds if available
    if hasattr(stepitems, "volumes") and stepitems.volumes:
        v0 = stepitems.volumes[0]
        for attr in candidates:
            if hasattr(v0, attr):
                try:
                    b = getattr(v0, attr)
                    b = b() if callable(b) else b
                    if isinstance(b, (tuple, list)) and len(b) == 2:
                        bmin = np.array(b[0], dtype=float)
                        bmax = np.array(b[1], dtype=float)
                        return bmin, bmax
                except Exception:
                    pass

    # If STEP only has surfaces (no volumes), try those as seed geometry
    if hasattr(stepitems, "surfaces") and stepitems.surfaces:
        s0 = stepitems.surfaces[0]
        for attr in candidates:
            if hasattr(s0, attr):
                try:
                    b = getattr(s0, attr)
                    b = b() if callable(b) else b
                    if isinstance(b, (tuple, list)) and len(b) == 2:
                        bmin = np.array(b[0], dtype=float)
                        bmax = np.array(b[1], dtype=float)
                        return bmin, bmax
                except Exception:
                    pass

    return None, None


def make_air_sphere(seed_stepitems, margin_m: float):
    """
    Create spherical air background that encloses seed geometry with margin.
    Auto-sizes from bounds if available; otherwise uses radius=max(margin, 0.05m).
    Returns: (air_sphere_volume, radius_m, center_xyz)
    """
    bmin, bmax = _safe_bounds_from_stepitems(seed_stepitems)
    if bmin is not None and bmax is not None:
        diag = float(np.linalg.norm(bmax - bmin))
        radius = 0.5 * diag + float(margin_m)
        center = 0.5 * (bmin + bmax)
    else:
        radius = max(float(margin_m), 0.05)  # fallback
        center = np.array([0.0, 0.0, 0.0], dtype=float)

    # Some Emerge versions accept position=, some don't -> try both.
    try:
        air = em.geo.Sphere(radius, position=tuple(center)).background()
    except TypeError:
        air = em.geo.Sphere(radius).background()

    return air, radius, center


def selection_face_tags(sel):
    """
    Best-effort extraction of face tags from an EMerge face selection.
    Returns list[int] or raises RuntimeError if not possible.
    """
    for attr in ("tags", "face_tags", "facetag", "facetags"):
        if hasattr(sel, attr):
            v = getattr(sel, attr)
            v = v() if callable(v) else v
            try:
                return [int(x) for x in list(v)]
            except Exception:
                pass

    try:
        return [int(x) for x in sel]
    except Exception:
        pass

    raise RuntimeError("Could not extract face tags from selection (unknown selection type/API).")


def _split_comment_lines(txt: str) -> list[str]:
    if not txt:
        return []
    lines = [ln.strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln]

def export_touchstone(grid, out_path: Path, z0_ref: float, comment: str):
    comments = _split_comment_lines(comment)
    try:
        grid.export_touchstone(
            str(out_path),
            Z0ref=float(z0_ref),
            format="RI",
            custom_comments=comments,
        )
        print(f"[OK] Touchstone exported: {out_path.name}")
    except Exception as e:
        print(f"[WARN] Touchstone export failed: {e}")

def export_farfield_3d_emff(data, boundary_selection, out_base: Path, theta: np.ndarray, phi: np.ndarray):
    # data.field is iterable BaseDataset in EMerge 2.3; convert to list to get len()
    fields = list(data.field)

    print(f"[INFO] Computing 3D farfield for all {len(fields)} frequencies...")
    exportdata = []

    for i, field in enumerate(fields):
        ff3d = field.farfield_3d(boundary_selection, theta, phi)
        exportdata.append(ff3d)

        if len(fields) <= 10 or (i + 1) % max(1, len(fields) // 10) == 0:
            print(f"  ... {i+1}/{len(fields)}")

    ffexp = FarFieldExporter(str(out_base), exportdata, precision=6)
    ffexp.addcol().Ex
    ffexp.addcol().Ey
    ffexp.addcol().Ez
    ffexp.write()

    out_emff = out_base.with_suffix(".emff")
    if out_emff.exists():
        print(f"[OK] Farfield exported: {out_emff.name}")
    else:
        print(f"[OK] Farfield export done (base={out_base.name})")


def resolve_simdata_dir(run_dir: Path, base_name: str) -> Path:
    """
    Resolve the EMerge simulation data folder for a run.
    If no known folder exists yet, create the default <base_name>.EMResults.
    """
    exact_candidates = [
        run_dir / f"{base_name}.EMResults",
        run_dir / f"{base_name}.EMResult",
        run_dir / f"{base_name}.emresults",
        run_dir / f"{base_name}.emresult",
    ]
    for c in exact_candidates:
        if c.exists() and c.is_dir():
            return c

    pattern_hits = sorted(
        [
            p for p in run_dir.iterdir()
            if p.is_dir()
            and p.name.lower().startswith(base_name.lower())
            and "emresult" in p.name.lower()
        ],
        key=lambda p: p.name.lower(),
    )
    if pattern_hits:
        return pattern_hits[0]

    out = run_dir / f"{base_name}.EMResults"
    out.mkdir(parents=True, exist_ok=True)
    return out


def move_emerge_files_to_simdata_dir(
    run_dir: Path,
    workspace_dir: Path,
    simdata_dir: Path,
    base_name: str,
    run_started_ts: float,
) -> list[Path]:
    """
    Move .emerge dataset files produced by this run into simdata_dir.

    EMerge may write simdata.emerge either in run_dir, or into a sibling
    <base_name>.EMResults folder under the workspace directory. We scan both
    locations and only move files updated during this run.
    """
    moved: list[Path] = []
    simdata_dir.mkdir(parents=True, exist_ok=True)

    candidate_parents = [run_dir, workspace_dir]
    candidate_dirs: list[Path] = []

    for parent in candidate_parents:
        candidate_dirs.append(parent)
        for d in parent.iterdir():
            if (
                d.is_dir()
                and d.name.lower().startswith(base_name.lower())
                and "emresult" in d.name.lower()
            ):
                candidate_dirs.append(d)

    seen: set[Path] = set()
    for cdir in candidate_dirs:
        for src in cdir.glob("*.emerge"):
            src_resolved = src.resolve()
            if src_resolved in seen:
                continue
            seen.add(src_resolved)

            # Avoid moving stale files from older runs.
            if src.stat().st_mtime < (run_started_ts - 2.0):
                continue

            dst = simdata_dir / src.name
            if src_resolved == dst.resolve():
                continue

            if dst.exists():
                dst.unlink()

            shutil.move(str(src), str(dst))
            moved.append(dst)

    # Remove empty, now-unused .EMResults dirs under workspace root.
    for d in workspace_dir.iterdir():
        if d.is_dir() and d.name.lower().startswith(base_name.lower()) and "emresult" in d.name.lower():
            try:
                next(d.iterdir())
            except StopIteration:
                d.rmdir()

    return moved

# ----------------------------- Main ------------------------------------

def main():
    cfg_path = Path("sim.json").resolve()
    ensure_exists(cfg_path, "config")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Results root
    outputs = cfg.get("outputs", {})
    results_dir = Path(outputs.get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ask user for dataset name + comment
    default_base = outputs.get("run_name", "").strip() or "emerge_run"
    base_name = prompt_save_basename(results_dir, default_base)
    comment = prompt_multiline_comment()

    run_dir = results_dir / f"{timestamp_tag()}_{base_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started_ts = time.time()

    # Units, sweep, mesh, port
    u = unit_scale(cfg.get("units", "mm"))
    STEP_UNIT = u

    sweep = cfg["sweep"]
    fstart = float(sweep["fstart_hz"])
    fstop = float(sweep["fstop_hz"])
    npoints = int(sweep["npoints"])

    mesh_cfg = cfg["mesh"]
    resolution = float(mesh_cfg["resolution"])
    air_margin_user_units = float(mesh_cfg["air_margin"])
    air_margin_m = air_margin_user_units * u

    # Optional manual air-sphere override from config (radius/center in config units, e.g. mm)
    air_sphere_cfg = cfg.get("air_sphere")

    port_cfg = cfg["port"]
    if port_cfg.get("type", "lumped_z") != "lumped_z":
        raise ValueError("This script supports only port.type='lumped_z'")

    p1 = np.array(port_cfg["p1"], dtype=float) * u
    p2 = np.array(port_cfg["p2"], dtype=float) * u
    port_width_m = float(port_cfg["width"]) * u
    port_face_size_m = float(port_cfg["face_size"]) * u
    z0 = float(port_cfg.get("z0", 50.0))

    preview = cfg.get("preview", {})
    show_geometry = bool(preview.get("show_geometry", False))
    show_mesh = bool(preview.get("show_mesh", False))

    imports = cfg["imports"]

    # NEW: farfield export configuration (optional)
    ff_cfg = cfg.get("farfield_export", {})
    ff_enable = bool(ff_cfg.get("enable", True))
    theta_points = int(ff_cfg.get("theta_points", 181))  # 0..pi inclusive => 1 degree
    phi_points = int(ff_cfg.get("phi_points", 361))      # -pi..pi inclusive => 1 degree
    ff_basename = str(ff_cfg.get("basename", "farfield_3d"))
    ff_precision = int(ff_cfg.get("precision", 6))  # currently fixed in exporter call, but keep as metadata
    # (If you want precision wired, we can do it—keeping now simple.)

    # Resolve ALL input paths BEFORE changing cwd
    dielectrics_resolved = []
    for item in imports.get("dielectrics", []):
        p = resolve_cfg_path(cfg_path, item["file"])
        ensure_exists(p, "dielectric STEP")
        dielectrics_resolved.append((item, p))

    metals_resolved = []
    for item in imports.get("metals", []):
        p = resolve_cfg_path(cfg_path, item["file"])
        ensure_exists(p, "metal STEP")
        metals_resolved.append((item, p))

    surfaces_resolved = []
    for item in imports.get("surfaces", []):
        p = resolve_cfg_path(cfg_path, item["file"])
        ensure_exists(p, "surface STEP")
        surfaces_resolved.append((item, p))

    # Optional STEP for sizing air sphere bounds
    full_model_path = resolve_cfg_path(cfg_path, cfg.get("full_model_step", "geometry/full_model.step"))
    if not full_model_path.exists():
        full_model_path = None

    print(f"\nUnits: {cfg.get('units','mm')} (scale={u} m/unit)")
    print(f"Run dir: {run_dir}")
    print(f"Simulation name: {base_name}")

    old_cwd = Path.cwd()

    try:
        # Change cwd so EMerge writes its *.EMResults directory under run_dir
        os.chdir(run_dir)

        model = em.Simulation(base_name, save_file=True)

        # Store helpful stuff inside the EMerge dataset too (pickle-able)
        model.data.globals["comment"] = comment
        model.data.globals["config_path"] = str(cfg_path)
        model.data.globals["config"] = cfg

        model.mw.set_resolution(resolution)
        model.mw.set_frequency_range(fstart, fstop, npoints)

        # --- Import dielectric volumes
        dielectric_items: list[tuple[em.geo.STEPItems, dict]] = []
        for item, p in dielectrics_resolved:
            di = em.geo.STEPItems(item.get("name", "dielectric"), str(p), unit=STEP_UNIT)
            if len(di.volumes) < 1:
                raise RuntimeError(f"{p} produced zero dielectric volumes.")
            er = float(item["material"]["er"])
            tand = float(item["material"].get("tand", 0.0))
            mat = em.Material(er=er, tand=tand)
            for v in di.volumes:
                v.set_material(mat)
            dielectric_items.append((di, item))
            print(f"Dielectric import: {p.name} volumes={len(di.volumes)} er={er} tand={tand}")

        # --- Import metal volumes
        metal_items: list[tuple[em.geo.STEPItems, dict]] = []
        for item, p in metals_resolved:
            mi = em.geo.STEPItems(item.get("name", "metal"), str(p), unit=STEP_UNIT)
            if len(mi.volumes) < 1:
                raise RuntimeError(f"{p} produced zero metal volumes.")
            kind = item["material"]["kind"].strip().lower()
            if kind == "pec_volume":
                mat = em.lib.PEC
            elif kind == "conductor_volume":
                sigma = float(item["material"]["sigma"])
                mat = em.Material(cond=sigma)
            else:
                raise ValueError(f"Unknown metal material.kind '{kind}' (use 'pec_volume' or 'conductor_volume').")
            for v in mi.volumes:
                v.set_material(mat)
            metal_items.append((mi, item))
            print(f"Metal import: {p.name} volumes={len(mi.volumes)} kind={kind}")

        # --- Import surfaces
        surface_items: list[tuple[em.geo.STEPItems, dict]] = []
        for item, p in surfaces_resolved:
            si = em.geo.STEPItems(item.get("name", "surface"), str(p), unit=STEP_UNIT)
            if len(si.surfaces) < 1:
                raise RuntimeError(f"{p} produced zero surfaces. Export as SURFACE bodies from CAD.")
            surface_items.append((si, item))
            print(f"Surface import: {p.name} surfaces={len(si.surfaces)} bc={item['bc']['kind']}")

        # --- Create spherical air domain
        # Prefer manual settings from config if provided, otherwise auto-size from geometry.
        if air_sphere_cfg is not None:
            # radius and center are specified in same units as the rest of the config (e.g. mm)
            radius_units = float(air_sphere_cfg["radius"])
            radius_m = radius_units * u

            center_units = air_sphere_cfg.get("center", [0.0, 0.0, 0.0])
            if len(center_units) != 3:
                raise ValueError("air_sphere.center must be a 3-element list [x,y,z].")
            center_m = np.array(center_units, dtype=float) * u

            try:
                air = em.geo.Sphere(radius_m, position=tuple(center_m)).background()
            except TypeError:
                air = em.geo.Sphere(radius_m).background()

            air_radius_m = radius_m
            air_center = center_m
            seed_name = "manual_cfg"

            print(
                f"Air domain: SPHERE (manual) seed={seed_name} "
                f"R={air_radius_m:.6g} m center={tuple(np.round(air_center, 6))}"
            )
        else:
            # Auto-sized sphere, seeded by full_model.step if available
            seed = None
            seed_name = None

            if full_model_path is not None:
                fm = em.geo.STEPItems("full_model_bounds", str(full_model_path), unit=STEP_UNIT)
                n_vol = len(getattr(fm, "volumes", []))
                n_surf = len(getattr(fm, "surfaces", []))

                if n_vol > 0 or n_surf > 0:
                    seed = fm
                    seed_name = str(full_model_path)
                    print(
                        f"Air-sphere seed: {full_model_path.name} "
                        f"(volumes={n_vol}, surfaces={n_surf})"
                    )
                else:
                    print(
                        f"Air-sphere seed: {full_model_path.name} had zero volumes/surfaces; "
                        f"falling back to imports."
                    )

            if seed is None:
                if dielectric_items:
                    seed = dielectric_items[0][0]
                    seed_name = "first dielectric import"
                elif metal_items:
                    seed = metal_items[0][0]
                    seed_name = "first metal import"
                else:
                    raise RuntimeError("No volumetric geometry found to size air sphere.")

            air, air_radius_m, air_center = make_air_sphere(seed, air_margin_m)
            print(
                f"Air domain: SPHERE seed={seed_name} "
                f"margin={air_margin_user_units} {cfg.get('units','mm')} "
                f"R={air_radius_m:.6g} m center={tuple(np.round(air_center, 6))}"
            )

        # --- Port plate
        port_plate, port_h = make_port_plate_z(p1, p2, port_width_m)

        # --- Commit geometry
        model.commit_geometry()

        if show_geometry:
            model.view(labels=True)

            # Optional pause after geometry preview
            while True:
                ans = input("Continue with meshing and simulation? [Y/N]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    break
                if ans in ("n", "no"):
                    print("Stopping at geometry preview by user request.")
                    return
                print("Please answer Y or N.")

        # --- Mesh controls
        model.mesher.set_face_size(port_plate, port_face_size_m)

        for di, item in dielectric_items:
            bs = float(item.get("mesh", {}).get("boundary_size", 0.0))
            if bs > 0:
                bs_m = bs * u
                for v in di.volumes:
                    model.mesher.set_boundary_size(v, bs_m)

        for mi, item in metal_items:
            bs = float(item.get("mesh", {}).get("boundary_size", 0.0))
            if bs > 0:
                bs_m = bs * u
                for v in mi.volumes:
                    model.mesher.set_boundary_size(v, bs_m)

        for si, item in surface_items:
            fs = float(item.get("mesh", {}).get("face_size", 0.0))
            if fs > 0:
                fs_m = fs * u
                for s in si.surfaces:
                    model.mesher.set_face_size(s, fs_m)

        # --- Mesh
        model.generate_mesh()

        # Mesh statistics (Mesh3D)
        try:
            mesh3d = model.mesh
            n_nodes = getattr(mesh3d, "n_nodes", None)
            n_tets = getattr(mesh3d, "n_tets", None)
            n_tris = getattr(mesh3d, "n_tris", None)
            n_edges = getattr(mesh3d, "n_edges", None)

            print("\n[Mesh statistics]")
            if n_nodes is not None:
                print(f"  Nodes         : {int(n_nodes):,}")
            if n_tets is not None:
                print(f"  Tetrahedra    : {int(n_tets):,}")
            if n_tris is not None:
                print(f"  Triangles     : {int(n_tris):,}")
            if n_edges is not None:
                print(f"  Edges         : {int(n_edges):,}")
        except Exception as e:
            print(f"[WARN] Could not print mesh statistics: {e}")

        # IMPORTANT: create selections AFTER mesh so facetags match mesh
        boundary_selection = air.boundary()

        # Store ABC / farfield integration face tags for post-processing scripts
        try:
            abc_tags = selection_face_tags(boundary_selection)
            model.data.globals["abc_face_tags"] = abc_tags  # list[int]
            print(f"Stored abc_face_tags: {len(abc_tags)} faces")
        except Exception as e:
            print(f"[WARN] Could not store abc_face_tags: {e}")

        if show_mesh:
            mesh_sel = []
            for si, _ in surface_items:
                mesh_sel.extend(si.surfaces)
            model.view(selections=mesh_sel, plot_mesh=True, volume_mesh=False)

            # Optional pause after mesh preview
            while True:
                ans = input("Continue with boundary conditions and simulation? [Y/N]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    break
                if ans in ("n", "no"):
                    print("Stopping at mesh preview by user request.")
                    return
                print("Please answer Y or N.")

        # --- Boundary conditions
        model.mw.bc.AbsorbingBoundary(boundary_selection)

        for si, item in surface_items:
            bc = item["bc"]
            kind = bc["kind"].strip().lower()
            if kind == "pec":
                for s in si.surfaces:
                    model.mw.bc.PEC(s)
            elif kind == "surface_impedance":
                sigma = float(bc["sigma"])
                mat = em.Material(cond=sigma)
                print(f"SurfaceImpedance on '{item.get('name','surface')}': sigma={sigma:g} S/m")
                for s in si.surfaces:
                    model.mw.bc.SurfaceImpedance(s, mat)
            else:
                raise ValueError(f"Unknown surface bc.kind '{kind}' (use 'pec' or 'surface_impedance').")

        model.mw.bc.LumpedPort(
            port_plate, 1,
            width=port_width_m,
            height=port_h,
            direction=em.ZAX,
            Z0=z0
        )

        # --- Solve
        data = model.mw.run_sweep()
        print("boundary_selection tags count:", len(selection_face_tags(boundary_selection)))

        # --- Quick sanity plot: Smith chart (keep as-is)
        grid = data.scalar.grid
        freqs = np.array(grid.freq, dtype=float)
        s11 = np.array(grid.S(1, 1), dtype=np.complex128)
        smith(s11, f=freqs, labels="S11")

        # ---- Quick far-field debug plots (single frequency) (keep as-is)
        f_mid = 0.5 * (fstart + fstop)
        field_mid = data.field.find(freq=float(f_mid))

        ff_xy = field_mid.farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
        ff_xz = field_mid.farfield_2d((0, 1, 0), (1, 0, 0), boundary_selection)
        ff_yz = field_mid.farfield_2d((1, 0, 0), (0, 1, 0), boundary_selection)

        plot_ff_polar(ff_xy.ang, [ff_xy.normE / em.lib.EISO], dB=True, dBfloor=-30)
        plot_ff_polar(ff_xz.ang, [ff_xz.normE / em.lib.EISO], dB=True, dBfloor=-30)
        plot_ff_polar(ff_yz.ang, [ff_yz.normE / em.lib.EISO], dB=True, dBfloor=-30)

        # Put all output artifacts into the same simulation data folder.
        simdata_dir = resolve_simdata_dir(run_dir, base_name)

        # Save comment as plain text
        (simdata_dir / "comment.txt").write_text((comment + "\n") if comment else "", encoding="utf-8")

        # ---------------- Export S-parameters (Touchstone) ----------------
        # You have one port -> s1p
        ts_path = simdata_dir / f"{base_name}.s1p"
        export_touchstone(grid, ts_path, z0_ref=z0, comment=comment)

        # ---------------- Export 3D farfield for all freqs ----------------
        if ff_enable:
            theta = np.linspace(0.0, np.pi, theta_points)
            phi = np.linspace(-np.pi, np.pi, phi_points)

            # Store grid meta for later
            model.data.globals["farfield_export"] = {
                "theta_points": int(theta_points),
                "phi_points": int(phi_points),
                "theta_min": 0.0,
                "theta_max": float(np.pi),
                "phi_min": float(-np.pi),
                "phi_max": float(np.pi),
                "basename": ff_basename,
                "precision": int(ff_precision),
            }

            ff_base = simdata_dir / ff_basename
            # FarFieldExporter takes a filename base (it writes <base>.emff typically)
            export_farfield_3d_emff(
                data=data,
                boundary_selection=boundary_selection,
                out_base=ff_base,
                theta=theta,
                phi=phi
            )
        else:
            print("[INFO] farfield_export.enable is false -> skipping 3D farfield export")

        # Human-readable meta
        meta = {
            "base_name": base_name,
            "run_dir": str(run_dir),
            "simdata_dir": str(simdata_dir),
            "config_file": str(cfg_path),
            "air_sphere": {
                "radius_m": float(air_radius_m),
                "center_m": [float(x) for x in air_center],
                "margin_user_units": float(air_margin_user_units),
                "units": cfg.get("units", "mm"),
            },
            "exports": {
                "touchstone": str(ts_path),
                "farfield_emff": str((simdata_dir / ff_basename).with_suffix(".emff")) if ff_enable else None,
            }
        }
        (simdata_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Save after all globals/meta updates are ready.
        model.save()

        moved_emerge_files = move_emerge_files_to_simdata_dir(
            run_dir=run_dir,
            workspace_dir=old_cwd,
            simdata_dir=simdata_dir,
            base_name=base_name,
            run_started_ts=run_started_ts,
        )
        if moved_emerge_files:
            print("Moved EMerge dataset file(s) into simulation results folder:")
            for p in moved_emerge_files:
                print(f"  {p}")

        print("\nSaved EMerge dataset into this run folder:")
        print(f"  {run_dir}")
        print(f"All exports were saved into: {simdata_dir}")
        print("Saved: comment.txt, meta.json, Touchstone, farfield .emff\n")

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()