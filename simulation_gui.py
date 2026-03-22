import json
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


def default_config() -> dict:
    return {
        "comment": "",
        "units": "mm",
        "sweep": {
            "fstart_hz": 6.5e9,
            "fstop_hz": 9.0e9,
            "npoints": 28,
        },
        "mesh": {
            "resolution": 0.2,
            "air_margin": 20.0,
        },
        "air_sphere": {
            "radius": 20.0,
            "center": [0.0, 0.0, 0.0],
        },
        "ports": [],
        "imports": {
            "dielectrics": [],
            "metals": [],
            "surfaces": [],
        },
        "preview": {
            "show_geometry": False,
            "show_mesh": False,
        },
        "outputs": {
            "results_dir": "results",
            "run_name": "emerge_run",
        },
        "farfield_export": {
            "enable": True,
            "theta_points": 181,
            "phi_points": 361,
            "basename": "farfield_3d",
            "precision": 6,
        },
        "full_model_step": "geometry/full_model.step",
    }


def parse_vec3(text_value: str):
    parts = [p.strip() for p in text_value.split(",")]
    if len(parts) != 3:
        raise ValueError("Need exactly 3 comma-separated values")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


class SimulationGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EMerge Simulation Builder")
        self.root.geometry("1100x760")

        self.workspace = Path(__file__).resolve().parent
        self.sim_script = self.workspace / "Emerge_simulation.py"

        self.cfg = default_config()
        self.current_json_path: Path | None = None
        self.run_process = None
        self.solid_index_map: list[tuple[str, int]] = []
        self.surface_index_map: list[int] = []

        self._build_ui()
        self.refresh_lists()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top_actions = ttk.Frame(outer)
        top_actions.pack(fill=tk.X)

        ttk.Button(top_actions, text="LOAD SOLID", command=self.load_solid).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="REMOVE SOLID", command=self.remove_selected_solid).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="LOAD SURFACE", command=self.load_surface).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="REMOVE SURFACE", command=self.remove_selected_surface).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="ADD PORT", command=self.add_port).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="REMOVE PORT", command=self.remove_selected_port).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="DEFINE FREQUENCIES", command=self.define_frequencies).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_actions, text="AIRBOX", command=self.define_airbox).pack(side=tk.LEFT, padx=4)

        lists_frame = ttk.Frame(outer)
        lists_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 6))

        self.solids_list = self._make_labeled_list(lists_frame, "Solids", 0)
        self.surfaces_list = self._make_labeled_list(lists_frame, "Surfaces", 1)
        self.ports_list = self._make_labeled_list(lists_frame, "Ports", 2)

        settings = ttk.LabelFrame(outer, text="Run Settings", padding=8)
        settings.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(settings, text="Run Name").grid(row=0, column=0, sticky="w")
        self.run_name_var = tk.StringVar(value=self.cfg["outputs"]["run_name"])
        ttk.Entry(settings, textvariable=self.run_name_var, width=28).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Label(settings, text="Results Dir").grid(row=0, column=2, sticky="w")
        self.results_dir_var = tk.StringVar(value=self.cfg["outputs"]["results_dir"])
        ttk.Entry(settings, textvariable=self.results_dir_var, width=24).grid(row=0, column=3, padx=6, sticky="w")

        ttk.Label(settings, text="Global Mesh Resolution").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.mesh_res_var = tk.DoubleVar(value=float(self.cfg["mesh"]["resolution"]))
        ttk.Entry(settings, textvariable=self.mesh_res_var, width=12).grid(row=1, column=1, padx=6, pady=(6, 0), sticky="w")

        self.ff_enable_var = tk.BooleanVar(value=bool(self.cfg["farfield_export"]["enable"]))
        ttk.Checkbutton(
            settings,
            text="Save Farfield Data",
            variable=self.ff_enable_var,
            command=self.on_toggle_farfield,
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))

        self.show_geom_var = tk.BooleanVar(value=bool(self.cfg["preview"]["show_geometry"]))
        ttk.Checkbutton(
            settings,
            text="Show Geometry",
            variable=self.show_geom_var,
            command=self.on_toggle_show_geometry,
        ).grid(row=2, column=2, sticky="w", pady=(6, 0))

        self.show_mesh_var = tk.BooleanVar(value=bool(self.cfg["preview"]["show_mesh"]))
        ttk.Checkbutton(
            settings,
            text="Show Mesh",
            variable=self.show_mesh_var,
            command=self.on_toggle_show_mesh,
        ).grid(row=2, column=3, sticky="w", pady=(6, 0))

        self.freq_info_var = tk.StringVar(value="")
        ttk.Label(settings, text="Frequencies").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Label(settings, textvariable=self.freq_info_var).grid(
            row=3, column=1, columnspan=3, sticky="w", padx=6, pady=(6, 0)
        )

        self._update_frequency_info()

        file_actions = ttk.Frame(outer)
        file_actions.pack(fill=tk.X, pady=(2, 8))

        ttk.Button(file_actions, text="SAVE", command=self.save_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(file_actions, text="LOAD", command=self.load_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(file_actions, text="RUN", command=self.run_simulation).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(file_actions, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        log_frame = ttk.LabelFrame(outer, text="Run Output", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=18, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=yscroll.set)

    def _make_labeled_list(self, parent: ttk.Frame, title: str, col: int):
        section = ttk.LabelFrame(parent, text=title, padding=6)
        section.grid(row=0, column=col, padx=5, sticky="nsew")
        parent.columnconfigure(col, weight=1)

        lb = tk.Listbox(section, width=44, height=10)
        lb.pack(fill=tk.BOTH, expand=True)
        return lb

    def log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def refresh_lists(self):
        self.solids_list.delete(0, tk.END)
        self.surfaces_list.delete(0, tk.END)
        self.ports_list.delete(0, tk.END)
        self.solid_index_map = []
        self.surface_index_map = []

        for d_idx, d in enumerate(self.cfg["imports"]["dielectrics"]):
            mat = d.get("material", {})
            txt = f"DIELECTRIC | {d['name']} | er={mat.get('er')} tand={mat.get('tand')} | {d['file']}"
            self.solids_list.insert(tk.END, txt)
            self.solid_index_map.append(("dielectrics", d_idx))

        for m_idx, m in enumerate(self.cfg["imports"]["metals"]):
            mat = m.get("material", {})
            kind = mat.get("kind", "")
            if kind == "pec_volume":
                spec = "PEC"
            else:
                spec = f"sigma={mat.get('sigma')}"
            txt = f"METAL | {m['name']} | {spec} | {m['file']}"
            self.solids_list.insert(tk.END, txt)
            self.solid_index_map.append(("metals", m_idx))

        for s_idx, s in enumerate(self.cfg["imports"]["surfaces"]):
            bc = s.get("bc", {})
            kind = bc.get("kind", "")
            if kind == "pec":
                spec = "PEC"
            else:
                spec = f"surface_impedance sigma={bc.get('sigma')}"
            txt = f"SURFACE | {s['name']} | {spec} | {s['file']}"
            self.surfaces_list.insert(tk.END, txt)
            self.surface_index_map.append(s_idx)

        for idx, p in enumerate(self.cfg["ports"], start=1):
            txt = (
                f"P{idx} | dir={p.get('direction', 'z')} | "
                f"p1={p.get('p1')} p2={p.get('p2')} | "
                f"w={p.get('width')} fs={p.get('face_size')} z0={p.get('z0')}"
            )
            self.ports_list.insert(tk.END, txt)

    def remove_selected_solid(self):
        selection = self.solids_list.curselection()
        if not selection:
            messagebox.showinfo("Remove solid", "Select a solid entry to remove")
            return

        list_idx = int(selection[0])
        if list_idx < 0 or list_idx >= len(self.solid_index_map):
            messagebox.showerror("Remove solid", "Invalid solid selection")
            return

        group, item_idx = self.solid_index_map[list_idx]
        item = self.cfg["imports"][group][item_idx]
        item_name = item.get("name", "unnamed")

        self.cfg["imports"][group].pop(item_idx)
        self.refresh_lists()
        self.status_var.set(f"Removed solid: {item_name}")

    def remove_selected_surface(self):
        selection = self.surfaces_list.curselection()
        if not selection:
            messagebox.showinfo("Remove surface", "Select a surface entry to remove")
            return

        list_idx = int(selection[0])
        if list_idx < 0 or list_idx >= len(self.surface_index_map):
            messagebox.showerror("Remove surface", "Invalid surface selection")
            return

        item_idx = self.surface_index_map[list_idx]
        item = self.cfg["imports"]["surfaces"][item_idx]
        item_name = item.get("name", "unnamed")

        self.cfg["imports"]["surfaces"].pop(item_idx)
        self.refresh_lists()
        self.status_var.set(f"Removed surface: {item_name}")

    def remove_selected_port(self):
        selection = self.ports_list.curselection()
        if not selection:
            messagebox.showinfo("Remove port", "Select a port entry to remove")
            return

        item_idx = int(selection[0])
        if item_idx < 0 or item_idx >= len(self.cfg["ports"]):
            messagebox.showerror("Remove port", "Invalid port selection")
            return

        item = self.cfg["ports"][item_idx]
        item_name = item.get("name", f"P{item_idx + 1}")

        self.cfg["ports"].pop(item_idx)

        # Renumber default-style port names to keep ordering clean.
        for i, p in enumerate(self.cfg["ports"], start=1):
            if p.get("name", "").startswith("P"):
                p["name"] = f"P{i}"

        self.refresh_lists()
        self.status_var.set(f"Removed port: {item_name}")

    def load_solid(self):
        step_file = filedialog.askopenfilename(
            title="Select solid STEP file",
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
        )
        if not step_file:
            return

        rel_path = self._to_rel_path(step_file)
        name = Path(step_file).stem

        mat_type = simpledialog.askstring(
            "Material type",
            "Enter solid material type: dielectric, metal, or PEC",
            parent=self.root,
        )
        if not mat_type:
            return
        mat_type = mat_type.strip().lower()

        mesh_res = simpledialog.askfloat(
            "Mesh boundary size",
            "Enter mesh boundary_size for this solid",
            parent=self.root,
            minvalue=0.0,
        )
        if mesh_res is None:
            return

        if mat_type == "dielectric":
            er = simpledialog.askfloat("Dielectric constant", "Enter er", parent=self.root, minvalue=1e-6)
            if er is None:
                return
            tand = simpledialog.askfloat("Loss tangent", "Enter tand", parent=self.root, minvalue=0.0)
            if tand is None:
                return

            self.cfg["imports"]["dielectrics"].append(
                {
                    "name": name,
                    "file": rel_path,
                    "material": {"er": float(er), "tand": float(tand)},
                    "mesh": {"boundary_size": float(mesh_res)},
                }
            )

        elif mat_type == "metal":
            sigma = simpledialog.askfloat(
                "Conductivity",
                "Enter conductivity sigma (S/m)",
                parent=self.root,
                minvalue=0.0,
            )
            if sigma is None:
                return

            self.cfg["imports"]["metals"].append(
                {
                    "name": name,
                    "file": rel_path,
                    "material": {"kind": "conductor_volume", "sigma": float(sigma)},
                    "mesh": {"boundary_size": float(mesh_res)},
                }
            )

        elif mat_type == "pec":
            self.cfg["imports"]["metals"].append(
                {
                    "name": name,
                    "file": rel_path,
                    "material": {"kind": "pec_volume"},
                    "mesh": {"boundary_size": float(mesh_res)},
                }
            )
        else:
            messagebox.showerror("Invalid type", "Use dielectric, metal, or PEC")
            return

        self.refresh_lists()
        self.status_var.set(f"Loaded solid: {name}")

    def load_surface(self):
        step_file = filedialog.askopenfilename(
            title="Select surface STEP file",
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
        )
        if not step_file:
            return

        rel_path = self._to_rel_path(step_file)
        name = Path(step_file).stem

        surf_type = simpledialog.askstring(
            "Surface type",
            "Enter surface property: PEC or conductive",
            parent=self.root,
        )
        if not surf_type:
            return
        surf_type = surf_type.strip().lower()

        mesh_res = simpledialog.askfloat(
            "Mesh face size",
            "Enter mesh face_size for this surface",
            parent=self.root,
            minvalue=0.0,
        )
        if mesh_res is None:
            return

        if surf_type == "pec":
            bc = {"kind": "pec"}
        elif surf_type == "conductive":
            sigma = simpledialog.askfloat(
                "Surface conductivity",
                "Enter conductivity sigma for surface impedance (S/m)",
                parent=self.root,
                minvalue=0.0,
            )
            if sigma is None:
                return
            bc = {"kind": "surface_impedance", "sigma": float(sigma)}
        else:
            messagebox.showerror("Invalid type", "Use PEC or conductive")
            return

        self.cfg["imports"]["surfaces"].append(
            {
                "name": name,
                "file": rel_path,
                "bc": bc,
                "mesh": {"face_size": float(mesh_res)},
            }
        )

        self.refresh_lists()
        self.status_var.set(f"Loaded surface: {name}")

    def add_port(self):
        pidx = len(self.cfg["ports"]) + 1

        p1_text = simpledialog.askstring(
            "Port p1",
            "Enter p1 as x,y,z (config units)",
            parent=self.root,
            initialvalue="0,0,0",
        )
        if not p1_text:
            return

        p2_text = simpledialog.askstring(
            "Port p2",
            "Enter p2 as x,y,z (config units)",
            parent=self.root,
            initialvalue="0,0,0.3",
        )
        if not p2_text:
            return

        try:
            p1 = parse_vec3(p1_text)
            p2 = parse_vec3(p2_text)
        except ValueError as exc:
            messagebox.showerror("Invalid coordinates", str(exc))
            return

        direction = simpledialog.askstring(
            "Feed direction",
            "Enter feed direction: x, y, or z",
            parent=self.root,
            initialvalue="z",
        )
        if not direction:
            return
        direction = direction.strip().lower()
        if direction not in ("x", "y", "z"):
            messagebox.showerror("Invalid direction", "Use x, y, or z")
            return

        width = simpledialog.askfloat("Port width", "Enter port width", parent=self.root, minvalue=0.0)
        if width is None:
            return
        face_size = simpledialog.askfloat("Port face size", "Enter port face_size", parent=self.root, minvalue=0.0)
        if face_size is None:
            return
        z0 = simpledialog.askfloat("Port impedance", "Enter Z0 (ohms)", parent=self.root, minvalue=1e-6)
        if z0 is None:
            return

        self.cfg["ports"].append(
            {
                "name": f"P{pidx}",
                "type": "lumped_z",
                "p1": p1,
                "p2": p2,
                "direction": direction,
                "width": float(width),
                "face_size": float(face_size),
                "z0": float(z0),
            }
        )

        self.refresh_lists()
        self.status_var.set(f"Added port P{pidx}")

    def define_frequencies(self):
        fstart = simpledialog.askfloat("Start frequency", "fstart_hz", parent=self.root, minvalue=0.0)
        if fstart is None:
            return
        fstop = simpledialog.askfloat("Stop frequency", "fstop_hz", parent=self.root, minvalue=0.0)
        if fstop is None:
            return
        npoints = simpledialog.askinteger("Points", "npoints", parent=self.root, minvalue=2)
        if npoints is None:
            return

        if fstop <= fstart:
            messagebox.showerror("Invalid sweep", "Stop frequency must be larger than start frequency")
            return

        self.cfg["sweep"]["fstart_hz"] = float(fstart)
        self.cfg["sweep"]["fstop_hz"] = float(fstop)
        self.cfg["sweep"]["npoints"] = int(npoints)
        self._update_frequency_info()
        self.status_var.set("Frequencies updated")

    def define_airbox(self):
        radius = simpledialog.askfloat("Air sphere radius", "Enter radius (config units)", parent=self.root, minvalue=0.0)
        if radius is None:
            return

        center_text = simpledialog.askstring(
            "Air sphere center",
            "Enter center as x,y,z (config units)",
            parent=self.root,
            initialvalue="0,0,0",
        )
        if not center_text:
            return

        try:
            center = parse_vec3(center_text)
        except ValueError as exc:
            messagebox.showerror("Invalid center", str(exc))
            return

        self.cfg["air_sphere"] = {
            "radius": float(radius),
            "center": center,
        }
        self.status_var.set("Air sphere updated")

    def on_toggle_farfield(self):
        self.cfg["farfield_export"]["enable"] = bool(self.ff_enable_var.get())

    def on_toggle_show_geometry(self):
        self.cfg["preview"]["show_geometry"] = bool(self.show_geom_var.get())

    def on_toggle_show_mesh(self):
        self.cfg["preview"]["show_mesh"] = bool(self.show_mesh_var.get())

    def _update_frequency_info(self):
        sweep = self.cfg.get("sweep", {})
        fstart = float(sweep.get("fstart_hz", 0.0))
        fstop = float(sweep.get("fstop_hz", 0.0))
        npoints = int(sweep.get("npoints", 0))
        self.freq_info_var.set(f"start={fstart:.3e} Hz   stop={fstop:.3e} Hz   npoints={npoints}")

    def _sync_form_to_config(self):
        self.cfg["outputs"]["run_name"] = self.run_name_var.get().strip() or "emerge_run"
        self.cfg["outputs"]["results_dir"] = self.results_dir_var.get().strip() or "results"
        self.cfg["mesh"]["resolution"] = float(self.mesh_res_var.get())
        self.cfg["farfield_export"]["enable"] = bool(self.ff_enable_var.get())
        self.cfg["preview"]["show_geometry"] = bool(self.show_geom_var.get())
        self.cfg["preview"]["show_mesh"] = bool(self.show_mesh_var.get())

    def _to_rel_path(self, selected_file: str) -> str:
        p = Path(selected_file).resolve()
        try:
            return p.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(p)

    def save_json(self):
        self._sync_form_to_config()

        target = filedialog.asksaveasfilename(
            title="Save simulation JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialdir=str(self.workspace),
            initialfile=(self.current_json_path.name if self.current_json_path else "define_simulation.json"),
        )
        if not target:
            return

        out_path = Path(target).resolve()
        out_path.write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")
        self.current_json_path = out_path
        self.status_var.set(f"Saved: {out_path.name}")
        self.log(f"Saved JSON: {out_path}")

    def load_json(self):
        selected = filedialog.askopenfilename(
            title="Load simulation JSON",
            filetypes=[("JSON files", "*.json")],
            initialdir=str(self.workspace),
        )
        if not selected:
            return

        path = Path(selected).resolve()
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not parse JSON:\n{exc}")
            return

        self.cfg = loaded
        self.current_json_path = path

        self.cfg.setdefault("imports", {})
        self.cfg["imports"].setdefault("dielectrics", [])
        self.cfg["imports"].setdefault("metals", [])
        self.cfg["imports"].setdefault("surfaces", [])
        self.cfg.setdefault("ports", [])
        self.cfg.setdefault("outputs", {})
        self.cfg["outputs"].setdefault("run_name", "emerge_run")
        self.cfg["outputs"].setdefault("results_dir", "results")
        self.cfg.setdefault("mesh", {})
        self.cfg["mesh"].setdefault("resolution", 0.2)
        self.cfg.setdefault("farfield_export", {})
        self.cfg["farfield_export"].setdefault("enable", True)
        self.cfg.setdefault("preview", {})
        self.cfg["preview"].setdefault("show_geometry", False)
        self.cfg["preview"].setdefault("show_mesh", False)
        self.cfg.setdefault("sweep", {})
        self.cfg["sweep"].setdefault("fstart_hz", 6.5e9)
        self.cfg["sweep"].setdefault("fstop_hz", 9.0e9)
        self.cfg["sweep"].setdefault("npoints", 28)

        self.run_name_var.set(self.cfg["outputs"]["run_name"])
        self.results_dir_var.set(self.cfg["outputs"]["results_dir"])
        self.mesh_res_var.set(float(self.cfg["mesh"]["resolution"]))
        self.ff_enable_var.set(bool(self.cfg["farfield_export"]["enable"]))
        self.show_geom_var.set(bool(self.cfg["preview"]["show_geometry"]))
        self.show_mesh_var.set(bool(self.cfg["preview"]["show_mesh"]))
        self._update_frequency_info()

        self.refresh_lists()
        self.status_var.set(f"Loaded: {path.name}")
        self.log(f"Loaded JSON: {path}")

    def run_simulation(self):
        self._sync_form_to_config()

        if not self.sim_script.exists():
            messagebox.showerror("Missing script", f"Cannot find simulator script:\n{self.sim_script}")
            return

        if not self.current_json_path:
            ask = messagebox.askyesno("Save first", "Config not saved yet. Save now before RUN?")
            if not ask:
                return
            self.save_json()
            if not self.current_json_path:
                return

        cmd = [sys.executable, str(self.sim_script), str(self.current_json_path)]
        self.log("Running: " + " ".join(cmd))
        self.status_var.set("Simulation running...")

        if self.run_process and self.run_process.poll() is None:
            messagebox.showwarning("Already running", "A simulation process is already running")
            return

        def worker():
            try:
                self.run_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.workspace),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                assert self.run_process.stdout is not None
                for line in self.run_process.stdout:
                    self.log(line.rstrip("\n"))

                code = self.run_process.wait()
                if code == 0:
                    self.status_var.set("Simulation completed")
                    self.log("Simulation finished successfully")
                else:
                    self.status_var.set(f"Simulation failed (exit {code})")
                    self.log(f"Simulation failed with exit code {code}")
            except Exception as exc:
                self.status_var.set("Simulation launch failed")
                self.log(f"Run failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    app = SimulationGui(root)
    app.log("EMerge Simulation Builder ready")
    root.mainloop()


if __name__ == "__main__":
    main()
