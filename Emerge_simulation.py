"""
Unified EMerge simulation script for N-port electromagnetic simulations.
Supports any number of ports (1, 4, or more) configured via JSON.
Automatically consolidates all outputs into a single run folder.
"""
import json
import os
import shutil
import argparse
import time
from pathlib import Path

import numpy as np
import emerge as em
from emerge.plot import smith, plot_ff_polar
from emerge.write import FarFieldExporter


# ----------------------------- Helpers ---------------------------------

def unit_scale_to_m(units: str) -> float:
    """Convert unit string to meters."""
    u = (units or "mm").strip().lower()
    if u == "m":
        return 1.0
    if u == "cm":
        return 1e-2
    if u == "mm":
        return 1e-3
    if u in ("um", "µm"):
        return 1e-6
    raise ValueError(f"Unsupported units '{units}'")


def timestamp_tag() -> str:
    """Generate timestamp string for folder naming."""
    return time.strftime("%Y%m%d_%H%M%S")


def safe_name(s: str) -> str:
    """Sanitize string for folder/file names."""
    s = (s or "").strip()
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)
    return s or "run"


def ensure_exists(path: Path, what: str = "file") -> None:
    """Check file/folder exists, raise if not."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def resolve_cfg_path(cfg_path: Path, maybe_rel: str) -> Path:
    """Resolve file path from config: absolute as-is, relative to config folder."""
    p = Path(maybe_rel).expanduser()
    if not p.is_absolute():
        p = cfg_path.parent / p
    return p.resolve()


def prompt_multiline_comment() -> str:
    """Prompt user for optional multiline comment. End with empty line."""
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


def selection_face_tags(sel):
    """Best-effort extraction of face tags from EMerge selection."""
    for attr in ("tags", "face_tags", "facetags"):
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
    raise RuntimeError("Could not extract face tags from selection")


def make_port_plate_z(p1_m: np.ndarray, p2_m: np.ndarray, width_m: float) -> tuple[em.geo.Plate, float]:
    """
    Create Z-directed lumped port plate in XZ plane.
    Returns (plate, port_height).
    """
    zmin = float(min(p1_m[2], p2_m[2]))
    zmax = float(max(p1_m[2], p2_m[2]))
    port_h = zmax - zmin
    if port_h <= 0:
        raise ValueError("Port height <= 0. Check port p1/p2 z coordinates.")

    xc = 0.5 * (float(p1_m[0]) + float(p2_m[0]))
    yc = 0.5 * (float(p1_m[1]) + float(p2_m[1]))

    corner = np.array([xc - width_m / 2.0, yc, zmin], dtype=float)
    wvec = np.array([width_m, 0.0, 0.0], dtype=float)
    hvec = np.array([0.0, 0.0, port_h], dtype=float)
    return em.geo.Plate(corner, wvec, hvec), port_h


def parse_port_direction(direction: str):
    """Map config direction string to EMerge axis constant."""
    d = (direction or "z").strip().lower()
    mapping = {
        "x": em.XAX,
        "y": em.YAX,
        "z": em.ZAX,
    }
    if d not in mapping:
        raise ValueError(f"Unsupported port direction '{direction}'. Use x, y, or z.")
    return mapping[d]


def _safe_bounds_from_stepitems(stepitems):
    """Best-effort bounding box read. Returns (None, None) if not available."""
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

    # Try per-volume bounds
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

    # Try surfaces if available
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
    """Create spherical air background. Auto-sizes from bounds if available."""
    bmin, bmax = _safe_bounds_from_stepitems(seed_stepitems)
    if bmin is not None and bmax is not None:
        diag = float(np.linalg.norm(bmax - bmin))
        radius = 0.5 * diag + float(margin_m)
        center = 0.5 * (bmin + bmax)
    else:
        radius = max(float(margin_m), 0.05)
        center = np.array([0.0, 0.0, 0.0], dtype=float)

    try:
        air = em.geo.Sphere(radius, position=tuple(center)).background()
    except TypeError:
        air = em.geo.Sphere(radius).background()

    return air, radius, center


def _split_comment_lines(txt: str) -> list[str]:
    """Split comment into lines, strip whitespace."""
    if not txt:
        return []
    lines = [ln.strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln]


def export_touchstone(grid, out_path: Path, nports: int, z0_ref: float, comment: str = ""):
    """Export S-parameters to Touchstone format."""
    comments = _split_comment_lines(comment)
    try:
        grid.export_touchstone(
            str(out_path),
            Z0ref=float(z0_ref),
            format="RI",
            custom_comments=comments,
        )
        ext = f".s{nports}p"
        print(f"[OK] Touchstone exported: {out_path.name}")
    except Exception as e:
        print(f"[WARN] Touchstone export failed: {e}")


def export_farfield_3d_per_port_emff(
    data,
    boundary_selection,
    out_base: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    precision: int,
    port_index: int,
):
    """
    Export embedded-element 3D farfield for a driven port.
    Uses MWField.excite_port(port_index) to set active port.
    """
    fields = list(data.field)
    ff_list = []

    for field in fields:
        field.excite_port(int(port_index))
        ff3d = field.farfield_3d(boundary_selection, theta, phi)
        ff_list.append(ff3d)

    exp = FarFieldExporter(str(out_base), ff_list, precision=int(precision))
    exp.addcol().Ex
    exp.addcol().Ey
    exp.addcol().Ez
    exp.write()


def resolve_simdata_dir(run_dir: Path, base_name: str) -> Path:
    """Resolve the EMerge simulation data folder for a run."""
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
    Move .emerge dataset files into simdata_dir.
    Scans run_dir and any .EMResults folders under workspace_dir.
    """
    moved: list[Path] = []
    simdata_dir.mkdir(parents=True, exist_ok=True)

    candidate_parents = [run_dir, workspace_dir]
    candidate_dirs: list[Path] = []

    for parent in candidate_parents:
        candidate_dirs.append(parent)
        try:
            for d in parent.iterdir():
                if (
                    d.is_dir()
                    and d.name.lower().startswith(base_name.lower())
                    and "emresult" in d.name.lower()
                ):
                    candidate_dirs.append(d)
        except (OSError, PermissionError):
            pass

    seen: set[Path] = set()
    for cdir in candidate_dirs:
        try:
            for src in cdir.glob("*.emerge"):
                src_resolved = src.resolve()
                if src_resolved in seen:
                    continue
                seen.add(src_resolved)

                if src.stat().st_mtime < (run_started_ts - 2.0):
                    continue

                dst = simdata_dir / src.name
                if src_resolved == dst.resolve():
                    continue

                if dst.exists():
                    dst.unlink()

                shutil.move(str(src), str(dst))
                moved.append(dst)
        except (OSError, PermissionError):
            pass

    # Remove empty .EMResults dirs under workspace root
    try:
        for d in workspace_dir.iterdir():
            if (
                d.is_dir()
                and d.name.lower().startswith(base_name.lower())
                and "emresult" in d.name.lower()
            ):
                try:
                    next(d.iterdir())
                except StopIteration:
                    d.rmdir()
    except (OSError, PermissionError):
        pass

    return moved


# ----------------------------- Main ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run unified EMerge simulation from JSON config.")
    parser.add_argument(
        "config",
        nargs="?",
        default="define_simulation.json",
        help="Path to simulation JSON file (default: define_simulation.json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    ensure_exists(cfg_path, "config file")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Units
    units = cfg.get("units", "mm")
    u = unit_scale_to_m(units)
    print(f"Units: {units} (scale {u} m/unit)")

    # Results directory
    outputs = cfg.get("outputs", {})
    results_dir = Path(outputs.get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Base name + run folder
    default_base = outputs.get("run_name", "").strip() or "emerge_run"
    base_name = safe_name(default_base)
    comment = str(cfg.get("comment", "") or "").strip()

    run_dir = results_dir / f"{timestamp_tag()}_{base_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started_ts = time.time()

    # Ports (now flexible: 1, 4, or any number)
    ports_cfg = cfg.get("ports", [])
    if not ports_cfg:
        raise ValueError("No ports defined in config['ports']")
    nports = len(ports_cfg)
    print(f"Simulation: {nports}-port")

    # Sweep
    sweep = cfg["sweep"]
    fstart = float(sweep["fstart_hz"])
    fstop = float(sweep["fstop_hz"])
    npoints = int(sweep["npoints"])
    print(f"Sweep: {fstart/1e9:.3f}..{fstop/1e9:.3f} GHz, N={npoints}")

    # Mesh
    mesh_cfg = cfg["mesh"]
    resolution = float(mesh_cfg["resolution"])
    air_margin_user_units = float(mesh_cfg["air_margin"])
    air_margin_m = air_margin_user_units * u

    # Optional air sphere override
    air_sphere_cfg = cfg.get("air_sphere")

    # Farfield export config
    ff_cfg = cfg.get("farfield_export", {})
    ff_enable = bool(ff_cfg.get("enable", True))
    theta_points = int(ff_cfg.get("theta_points", 181))
    phi_points = int(ff_cfg.get("phi_points", 361))
    ff_basename = ff_cfg.get("basename", "farfield_3d")
    ff_precision = int(ff_cfg.get("precision", 6))

    theta = np.linspace(0.0, np.pi, int(theta_points))
    phi = np.linspace(-np.pi, np.pi, int(phi_points))

    # Preview flags
    preview = cfg.get("preview", {})
    show_geometry = bool(preview.get("show_geometry", False))
    show_mesh = bool(preview.get("show_mesh", False))

    # Resolve ALL input paths BEFORE changing cwd
    imports = cfg["imports"]

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

    # Optional STEP for air sphere sizing
    full_model_path = resolve_cfg_path(cfg_path, cfg.get("full_model_step", "geometry/full_model.step"))
    if not full_model_path.exists():
        full_model_path = None

    print(f"Run dir: {run_dir}")
    print(f"Simulation name: {base_name}\n")

    old_cwd = Path.cwd()

    try:
        os.chdir(run_dir)

        model = em.Simulation(base_name, save_file=True)

        # Store metadata
        model.data.globals["comment"] = comment
        model.data.globals["config_path"] = str(cfg_path)
        model.data.globals["config"] = cfg

        model.mw.set_resolution(resolution)
        model.mw.set_frequency_range(fstart, fstop, npoints)

        # --- Import dielectrics
        dielectric_items: list[tuple[em.geo.STEPItems, dict]] = []
        for item, p in dielectrics_resolved:
            di = em.geo.STEPItems(item.get("name", "dielectric"), str(p), unit=u)
            if len(di.volumes) < 1:
                raise RuntimeError(f"{p} produced zero dielectric volumes.")
            er = float(item["material"]["er"])
            tand = float(item["material"].get("tand", 0.0))
            mat = em.Material(er=er, tand=tand)
            for v in di.volumes:
                v.set_material(mat)
            dielectric_items.append((di, item))
            print(f"Dielectric: {p.name} volumes={len(di.volumes)} er={er} tand={tand}")

        # --- Import metals
        metal_items: list[tuple[em.geo.STEPItems, dict]] = []
        for item, p in metals_resolved:
            mi = em.geo.STEPItems(item.get("name", "metal"), str(p), unit=u)
            if len(mi.volumes) < 1:
                raise RuntimeError(f"{p} produced zero metal volumes.")
            kind = item["material"]["kind"].strip().lower()
            if kind == "pec_volume":
                mat = em.lib.PEC
            elif kind == "conductor_volume":
                sigma = float(item["material"]["sigma"])
                mat = em.Material(cond=sigma)
            else:
                raise ValueError(f"Unknown metal material.kind '{kind}'.")
            for v in mi.volumes:
                v.set_material(mat)
            metal_items.append((mi, item))
            print(f"Metal: {p.name} volumes={len(mi.volumes)} kind={kind}")

        # --- Import surfaces
        surface_items: list[tuple[em.geo.STEPItems, dict]] = []
        for item, p in surfaces_resolved:
            si = em.geo.STEPItems(item.get("name", "surface"), str(p), unit=u)
            if len(si.surfaces) < 1:
                raise RuntimeError(f"{p} produced zero surfaces.")
            surface_items.append((si, item))
            print(f"Surface: {p.name} surfaces={len(si.surfaces)} bc={item['bc']['kind']}")

        # --- Air sphere
        if air_sphere_cfg is not None:
            radius_units = float(air_sphere_cfg["radius"])
            radius_m = radius_units * u
            center_units = air_sphere_cfg.get("center", [0.0, 0.0, 0.0])
            if len(center_units) != 3:
                raise ValueError("air_sphere.center must be [x, y, z].")
            center_m = np.array(center_units, dtype=float) * u

            try:
                air = em.geo.Sphere(radius_m, position=tuple(center_m)).background()
            except TypeError:
                air = em.geo.Sphere(radius_m).background()

            air_radius_m = radius_m
            air_center = center_m
            print(f"Air sphere (manual): R={air_radius_m:.6g} m center={tuple(np.round(air_center, 6))}")
        else:
            seed = None
            if full_model_path is not None:
                fm = em.geo.STEPItems("full_model_bounds", str(full_model_path), unit=u)
                n_vol = len(getattr(fm, "volumes", []))
                n_surf = len(getattr(fm, "surfaces", []))
                if n_vol > 0 or n_surf > 0:
                    seed = fm
                    print(f"Air-sphere seed: {full_model_path.name} (volumes={n_vol}, surfaces={n_surf})")

            if seed is None:
                if dielectric_items:
                    seed = dielectric_items[0][0]
                elif metal_items:
                    seed = metal_items[0][0]
                else:
                    raise RuntimeError("No volumetric geometry found to size air sphere.")

            air, air_radius_m, air_center = make_air_sphere(seed, air_margin_m)
            print(f"Air sphere (auto): R={air_radius_m:.6g} m margin={air_margin_user_units} {units}")

        # --- Port plates
        port_plates = []
        port_heights = []
        port_widths = []
        port_face_sizes = []
        port_z0s = []

        for i, pc in enumerate(ports_cfg, start=1):
            if pc.get("type", "lumped_z") != "lumped_z":
                raise ValueError("Only port.type='lumped_z' is supported.")

            p1 = np.array(pc["p1"], dtype=float) * u
            p2 = np.array(pc["p2"], dtype=float) * u
            w = float(pc["width"]) * u
            fs = float(pc["face_size"]) * u
            z0 = float(pc.get("z0", 50.0))
            direction = parse_port_direction(pc.get("direction", "z"))

            plate, h = make_port_plate_z(p1, p2, w)
            port_plates.append(plate)
            port_heights.append(h)
            port_widths.append(w)
            port_face_sizes.append(fs)
            port_z0s.append(z0)
            # Keep direction aligned with port index
            if "port_dirs" not in locals():
                port_dirs = []
            port_dirs.append(direction)

            port_name = pc.get("name", f"P{i}")
            print(f"Port {i} ({port_name}): z0={z0:.1f} ohm width={w*1e3:.3f}mm height={h*1e3:.3f}mm")

        # --- Commit geometry
        model.commit_geometry()

        if show_geometry:
            model.view(labels=True)
            while True:
                ans = input("Continue with meshing and simulation? [Y/N]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    break
                if ans in ("n", "no"):
                    print("Stopping at geometry preview.")
                    return
                print("Please answer Y or N.")

        # --- Mesh controls
        for plate, fs in zip(port_plates, port_face_sizes):
            model.mesher.set_face_size(plate, fs)

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

        # --- Generate mesh
        model.generate_mesh()

        # Mesh statistics
        try:
            mesh3d = model.mesh
            n_nodes = getattr(mesh3d, "n_nodes", None)
            n_tets = getattr(mesh3d, "n_tets", None)
            n_tris = getattr(mesh3d, "n_tris", None)
            n_edges = getattr(mesh3d, "n_edges", None)

            print("\n[Mesh statistics]")
            if n_nodes is not None:
                print(f"  Nodes      : {int(n_nodes):,}")
            if n_tets is not None:
                print(f"  Tetrahedra : {int(n_tets):,}")
            if n_tris is not None:
                print(f"  Triangles  : {int(n_tris):,}")
            if n_edges is not None:
                print(f"  Edges      : {int(n_edges):,}")
        except Exception as e:
            print(f"[WARN] Mesh statistics unavailable: {e}")

        # Create boundary selection AFTER mesh
        boundary_selection = air.boundary()

        # Store ABC face tags
        try:
            abc_tags = selection_face_tags(boundary_selection)
            model.data.globals["abc_face_tags"] = abc_tags
            print(f"[INFO] Stored abc_face_tags: {len(abc_tags)} faces")
        except Exception as e:
            print(f"[WARN] Could not store abc_face_tags: {e}")

        if show_mesh:
            mesh_sel = []
            for si, _ in surface_items:
                mesh_sel.extend(si.surfaces)
            model.view(selections=mesh_sel, plot_mesh=True, volume_mesh=False)

            while True:
                ans = input("Continue with boundary conditions and simulation? [Y/N]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    break
                if ans in ("n", "no"):
                    print("Stopping at mesh preview.")
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
                for s in si.surfaces:
                    model.mw.bc.SurfaceImpedance(s, mat)
            else:
                raise ValueError(f"Unknown bc.kind '{kind}'.")

        # Add ports (works for any number)
        for i in range(nports):
            model.mw.bc.LumpedPort(
                port_plates[i], i + 1,
                width=port_widths[i],
                height=port_heights[i],
                direction=port_dirs[i],
                Z0=port_z0s[i],
            )

        # --- Solve once (for all ports)
        print("\n[INFO] Running sweep...")
        data = model.mw.run_sweep()

        # --- Quick debug plots
        grid = data.scalar.grid
        freqs = np.array(grid.freq, dtype=float)
        s11 = np.array(grid.S(1, 1), dtype=np.complex128)
        smith(s11, f=freqs, labels=f"S11 (Port 1)")

        f_mid = 0.5 * (fstart + fstop)
        field_mid = data.field.find(freq=float(f_mid))

        ff_xy = field_mid.farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
        ff_xz = field_mid.farfield_2d((0, 1, 0), (1, 0, 0), boundary_selection)
        ff_yz = field_mid.farfield_2d((1, 0, 0), (0, 1, 0), boundary_selection)

        plot_ff_polar(ff_xy.ang, [ff_xy.normE / em.lib.EISO], dB=True, dBfloor=-30)
        plot_ff_polar(ff_xz.ang, [ff_xz.normE / em.lib.EISO], dB=True, dBfloor=-30)
        plot_ff_polar(ff_yz.ang, [ff_yz.normE / em.lib.EISO], dB=True, dBfloor=-30)

        # Resolve output folders
        simdata_dir = resolve_simdata_dir(run_dir, base_name)

        # Save comment
        (simdata_dir / "comment.txt").write_text((comment + "\n") if comment else "", encoding="utf-8")

        # --- Export S-parameters
        ts_path = simdata_dir / f"{base_name}.s{nports}p"
        export_touchstone(grid, ts_path, nports, z0_ref=port_z0s[0], comment=comment)

        # --- Export farfield per port
        if ff_enable:
            try:
                nports_reported = int(field_mid.Nports)
                print(f"[INFO] MWField reports Nports = {nports_reported}")
            except Exception:
                nports_reported = nports

            for port_idx in range(1, nports + 1):
                out_base = simdata_dir / f"{ff_basename}_P{port_idx}"
                port_name = ports_cfg[port_idx - 1].get("name", f"P{port_idx}")
                print(f"[INFO] Exporting farfield for {port_name} -> {out_base.with_suffix('.emff').name}")
                export_farfield_3d_per_port_emff(
                    data=data,
                    boundary_selection=boundary_selection,
                    out_base=out_base,
                    theta=theta,
                    phi=phi,
                    precision=ff_precision,
                    port_index=port_idx,
                )
        else:
            print("[INFO] Farfield export disabled")

        # Store metadata
        meta = {
            "base_name": base_name,
            "run_dir": str(run_dir),
            "simdata_dir": str(simdata_dir),
            "config_file": str(cfg_path),
            "nports": nports,
            "air_sphere": {
                "radius_m": float(air_radius_m),
                "center_m": [float(x) for x in air_center],
            },
        }
        (simdata_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Save model
        model.save()

        # Move any workspace-level .emerge files into simdata_dir
        moved_emerge_files = move_emerge_files_to_simdata_dir(
            run_dir=run_dir,
            workspace_dir=old_cwd,
            simdata_dir=simdata_dir,
            base_name=base_name,
            run_started_ts=run_started_ts,
        )
        if moved_emerge_files:
            print("\nMoved EMerge dataset file(s) into simulation results:")
            for p in moved_emerge_files:
                print(f"  {p}")

        print(f"\n[OK] Simulation complete!")
        print(f"  Touchstone: {ts_path}")
        print(f"  Results:    {simdata_dir}")

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
