import os
import json
import time
from pathlib import Path

import numpy as np
import emerge as em
from emerge.plot import smith, plot_ff_polar
from emerge.write import FarFieldExporter


# ----------------------------- Helpers ---------------------------------

def unit_scale_to_m(units: str) -> float:
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


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)
    return s or "run"


def ensure_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def resolve_cfg_path(cfg_path: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel).expanduser()
    if not p.is_absolute():
        p = cfg_path.parent / p
    return p.resolve()


def selection_face_tags(sel):
    # Best-effort extraction of face tags from EMerge selection
    for attr in ("tags", "face_tags", "facetags"):
        if hasattr(sel, attr):
            v = getattr(sel, attr)
            v = v() if callable(v) else v
            try:
                return [int(x) for x in list(v)]
            except Exception:
                pass
    raise RuntimeError("Could not extract face tags from selection")


def make_port_plate_z(p1_m: np.ndarray, p2_m: np.ndarray, width_m: float):
    """
    Z-directed lumped port plate in XZ plane at y = yc.
    """
    zmin = float(min(p1_m[2], p2_m[2]))
    zmax = float(max(p1_m[2], p2_m[2]))
    h = zmax - zmin
    if h <= 0:
        raise ValueError("Port height <= 0, check p1/p2 z")

    xc = 0.5 * (float(p1_m[0]) + float(p2_m[0]))
    yc = 0.5 * (float(p1_m[1]) + float(p2_m[1]))

    corner = np.array([xc - width_m / 2.0, yc, zmin], dtype=float)
    wvec = np.array([width_m, 0.0, 0.0], dtype=float)
    hvec = np.array([0.0, 0.0, h], dtype=float)
    return em.geo.Plate(corner, wvec, hvec), h


def make_air_sphere(seed_stepitems, margin_m: float):
    """
    Create spherical air background from seed bounds if available.
    """
    bmin = None
    bmax = None

    for attr in ("bounds", "bbox", "bounding_box"):
        if hasattr(seed_stepitems, attr):
            try:
                b = getattr(seed_stepitems, attr)
                b = b() if callable(b) else b
                if isinstance(b, (tuple, list)) and len(b) == 2:
                    bmin = np.array(b[0], dtype=float)
                    bmax = np.array(b[1], dtype=float)
                    break
            except Exception:
                pass

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


def export_touchstone(grid, out_path: Path, comment: str = ""):
    comments = [ln.strip() for ln in (comment or "").splitlines() if ln.strip()]
    # IMPORTANT: don't force funit="GHz" (you got "'GHz'" earlier)
    grid.export_touchstone(
        str(out_path),
        Z0ref=50.0,
        format="RI",
        custom_comments=comments
    )


def export_farfield_3d_per_port_emff(
    data,
    boundary_selection,
    out_base_no_ext: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    precision: int,
    port_index: int,
):
    """
    Export embedded-element 3D farfield for a given driven port index.
    The crucial step is: MWField.excite_port(port_index) before farfield_3d().

    data.field is a BaseDataset of MWField entries (typically one per frequency point).
    """
    fields = list(data.field)  # BaseDataset is iterable
    ff_list = []

    for field in fields:
        # 🔑 Select excitation (this is what was missing before)
        field.excite_port(int(port_index))
        ff3d = field.farfield_3d(boundary_selection, theta, phi)
        ff_list.append(ff3d)

    exp = FarFieldExporter(str(out_base_no_ext), ff_list, precision=int(precision))
    exp.addcol().Ex
    exp.addcol().Ey
    exp.addcol().Ez
    exp.write()


# ----------------------------- Main ------------------------------------

def main():
    here = Path(__file__).resolve().parent

    # Config file
    cfg_path = (here / "sim-4port.json").resolve()
    if not cfg_path.exists():
        # fallback if you kept old name
        cfg_path = (here / "sim.json").resolve()
    ensure_exists(cfg_path, "config json")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    units = cfg.get("units", "mm")
    u = unit_scale_to_m(units)

    # Outputs
    outputs = cfg.get("outputs", {})
    results_dir = (here / outputs.get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_name(outputs.get("run_name", "")) or "run"
    run_dir = results_dir / f"{now_stamp()}_{base_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional user comment file (keep simple; you can extend later)
    comment = cfg.get("comment", "") or ""
    (run_dir / "comment.txt").write_text((comment + "\n") if comment else "", encoding="utf-8")

    # Preview flags (show only once)
    preview = cfg.get("preview", {})
    show_geometry = bool(preview.get("show_geometry", False))
    show_mesh = bool(preview.get("show_mesh", False))

    # Sweep
    sweep = cfg["sweep"]
    fstart = float(sweep["fstart_hz"])
    fstop = float(sweep["fstop_hz"])
    npoints = int(sweep["npoints"])

    # Mesh
    mesh_cfg = cfg["mesh"]
    resolution = float(mesh_cfg["resolution"])
    air_margin_user_units = float(mesh_cfg["air_margin"])
    air_margin_m = air_margin_user_units * u

    # Optional manual air-sphere override from config (radius/center in config units, e.g. mm)
    air_sphere_cfg = cfg.get("air_sphere")

    # Ports (must be 4)
    ports_cfg = cfg.get("ports", [])
    if len(ports_cfg) != 4:
        raise ValueError(f"Expected 4 ports in cfg['ports'], got {len(ports_cfg)}")

    # Farfield export config
    ff_cfg = cfg.get("farfield_export", {})
    ff_enable = bool(ff_cfg.get("enable", True))
    theta_pts = int(ff_cfg.get("theta_points", 181))
    phi_pts = int(ff_cfg.get("phi_points", 361))
    ff_basename = safe_name(ff_cfg.get("basename", "farfield_3d"))
    ff_prec = int(ff_cfg.get("precision", 6))

    theta = np.linspace(0.0, np.pi, int(theta_pts))
    phi = np.linspace(-np.pi, np.pi, int(phi_pts))

    # Resolve imports
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

    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Units: {units} (scale {u} m/unit)")
    print(f"[INFO] Sweep: {fstart/1e9:.3f}..{fstop/1e9:.3f} GHz, N={npoints}")
    print("[INFO] 4-port run: ONE run_sweep(), then farfield export per port using MWField.excite_port().")

    old_cwd = Path.cwd()
    try:
        # Put EMerge save directories under run_dir
        os.chdir(run_dir)

        # --- Create simulation
        model = em.Simulation(base_name, save_file=True)

        # Traceability
        model.data.globals["config_path"] = str(cfg_path)
        model.data.globals["config"] = cfg

        model.mw.set_resolution(resolution)
        model.mw.set_frequency_range(fstart, fstop, npoints)

        # --- Import dielectrics
        dielectric_items = []
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
            print(f"[INFO] Dielectric: {p.name} volumes={len(di.volumes)} er={er} tand={tand}")

        # --- Import metals (volumes)
        metal_items = []
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
            print(f"[INFO] Metal: {p.name} volumes={len(mi.volumes)} kind={kind}")

        # --- Import surfaces
        surface_items = []
        for item, p in surfaces_resolved:
            si = em.geo.STEPItems(item.get("name", "surface"), str(p), unit=u)
            if len(si.surfaces) < 1:
                raise RuntimeError(f"{p} produced zero surfaces (export as SURFACE bodies).")
            surface_items.append((si, item))
            print(f"[INFO] Surface: {p.name} surfaces={len(si.surfaces)} bc={item['bc']['kind']}")

        # --- Air sphere
        # Prefer manual settings from config if provided, otherwise auto-size from geometry.
        if air_sphere_cfg is not None:
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
            print(
                f"[INFO] Air sphere (manual): R={air_radius_m:.6g} m "
                f"center={tuple(np.round(air_center, 6))}"
            )
        else:
            seed = dielectric_items[0][0] if dielectric_items else metal_items[0][0]
            air, air_radius_m, air_center = make_air_sphere(seed, air_margin_m)
            print(
                f"[INFO] Air sphere (auto): R={air_radius_m:.6g} m "
                f"center={tuple(np.round(air_center, 6))} "
                f"margin={air_margin_user_units} {units}"
            )

        # --- Ports geometry (plates)
        port_plates = []
        port_heights = []
        port_widths = []
        port_face_sizes = []
        port_z0s = []

        for pc in ports_cfg:
            if pc.get("type", "lumped_z") != "lumped_z":
                raise ValueError("This script supports only port.type='lumped_z'")

            p1 = np.array(pc["p1"], dtype=float) * u
            p2 = np.array(pc["p2"], dtype=float) * u
            w = float(pc["width"]) * u
            fs = float(pc["face_size"]) * u
            z0 = float(pc.get("z0", 50.0))

            plate, h = make_port_plate_z(p1, p2, w)
            port_plates.append(plate)
            port_heights.append(h)
            port_widths.append(w)
            port_face_sizes.append(fs)
            port_z0s.append(z0)

        # --- Commit geometry
        model.commit_geometry()

        # ✅ show geometry once
        if show_geometry:
            try:
                model.view(labels=True)
            except Exception as e:
                print(f"[WARN] model.view(labels=True) failed: {e}")

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

        boundary_selection = air.boundary()

        # store ABC face tags for post tools
        try:
            model.data.globals["abc_face_tags"] = selection_face_tags(boundary_selection)
            print(f"[INFO] Stored abc_face_tags: {len(model.data.globals['abc_face_tags'])}")
        except Exception as e:
            print(f"[WARN] Could not store abc_face_tags: {e}")

        # ✅ show mesh once
        if show_mesh:
            try:
                mesh_sel = []
                for si, _ in surface_items:
                    mesh_sel.extend(si.surfaces)
                model.view(selections=mesh_sel, plot_mesh=True, volume_mesh=False)
            except Exception as e:
                print(f"[WARN] model.view(plot_mesh=...) failed: {e}")

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

        # surfaces BC
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
                raise ValueError(f"Unknown surface bc.kind '{kind}'.")

        # 4 lumped ports
        for i in range(4):
            model.mw.bc.LumpedPort(
                port_plates[i], i + 1,
                width=port_widths[i],
                height=port_heights[i],
                direction=em.ZAX,
                Z0=port_z0s[i],
            )

        # --- Solve ONCE
        data = model.mw.run_sweep()

        # --- Export S4P once
        grid = data.scalar.grid
        ts_path = run_dir / f"{base_name}.s4p"
        print(f"[INFO] Exporting S-data to {ts_path}")
        export_touchstone(grid, ts_path, comment=comment)

        # --- Debug plots (keep as-is)
        freqs = np.array(grid.freq, dtype=float)
        s11 = np.array(grid.S(1, 1), dtype=np.complex128)
        smith(s11, f=freqs, labels="S11")

        f_mid = 0.5 * (fstart + fstop)
        field_mid = data.field.find(freq=float(f_mid))

        ff_xy = field_mid.farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
        ff_xz = field_mid.farfield_2d((0, 1, 0), (1, 0, 0), boundary_selection)
        ff_yz = field_mid.farfield_2d((1, 0, 0), (0, 1, 0), boundary_selection)

        plot_ff_polar(ff_xy.ang, [ff_xy.normE / em.lib.EISO], dB=True, dBfloor=-30)
        plot_ff_polar(ff_xz.ang, [ff_xz.normE / em.lib.EISO], dB=True, dBfloor=-30)
        plot_ff_polar(ff_yz.ang, [ff_yz.normE / em.lib.EISO], dB=True, dBfloor=-30)

        # --- Export embedded-element farfields per port
        if ff_enable:
            # sanity: verify EMerge sees 4 ports in field object
            try:
                nports = int(field_mid.Nports)
                print(f"[INFO] MWField reports Nports = {nports}")
            except Exception:
                nports = 4

            for p in range(1, 5):
                out_base = run_dir / f"{ff_basename}_P{p}"
                print(f"[INFO] Exporting 3D farfield for port P{p} -> {out_base.with_suffix('.emff').name}")
                export_farfield_3d_per_port_emff(
                    data=data,
                    boundary_selection=boundary_selection,
                    out_base_no_ext=out_base,
                    theta=theta,
                    phi=phi,
                    precision=ff_prec,
                    port_index=p,
                )
        else:
            print("[INFO] farfield_export.enable is false -> skipping farfield export")

        # --- Save model (as before)
        model.save()

        print("\n[OK] Done. Outputs:")
        print(f"  Touchstone: {ts_path}")
        if ff_enable:
            for p in range(1, 5):
                print(f"  Farfield:   {run_dir / (ff_basename + '_P' + str(p) + '.emff')}")
        print(f"  Model save: {run_dir / (base_name + '.EMResults')} (or similar)")

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()