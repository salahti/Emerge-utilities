# array_plotter.py
#
# GUI tool to combine N embedded-element farfield .emff files with per-element
# amplitude and phase excitation, then plot array pattern cuts in XY/XZ/YZ.
# Supports 2 to 16 ports. Amplitude and phase controls are created dynamically
# after loading the folder.
#
# Usage:
#   python array_plotter.py
#
# Expected folder contents:
#   ..._P1.emff, ..._P2.emff, ...  (recommended naming)
# or any *.emff files found in the selected folder (2-16 files)

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import skrf as rf
except Exception:
    rf = None

import emerge as em


# ---------------------- Frequency band overlays ----------------------
# Each entry: (f_start_GHz, f_stop_GHz, label, color)
# Edit freely – bands are drawn as light shaded regions on the gain vs. freq plot.
FREQ_BANDS = [
    (7.25, 7.75, "Downlink (Space-to-Earth)", "#4da6ff"),   # blue
    (7.90, 8.40, "Uplink (Earth-to-Space)",   "#ff884d"),   # orange
]


# ---------------------- EMFF parsing ----------------------

def _read_number_list_lines(f, stop_prefixes=("%", "#", "$")):
    nums = []
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            return nums, ""
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(p) for p in stop_prefixes):
            f.seek(pos)
            return nums, s
        nums.extend([float(x) for x in s.split()])


def parse_emff(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        theta_deg = None
        phi_deg = None
        freqs = None

        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("% Theta"):
                theta_list, _ = _read_number_list_lines(f)
                theta_deg = np.array(theta_list, dtype=float)
            elif s.startswith("% Phi"):
                phi_list, _ = _read_number_list_lines(f)
                phi_deg = np.array(phi_list, dtype=float)
            elif s.startswith("% Frequencies"):
                freq_list, _ = _read_number_list_lines(f)
                freqs = np.array(freq_list, dtype=float)
                break

        if theta_deg is None or phi_deg is None or freqs is None:
            raise ValueError(
                f"Failed to parse EMFF header in {path.name}. "
                f"Got theta={theta_deg is not None}, phi={phi_deg is not None}, freqs={freqs is not None}"
            )

        Nt = theta_deg.size
        Np = phi_deg.size
        Nf = freqs.size

        theta_to_i = {float(v): i for i, v in enumerate(theta_deg)}
        phi_to_j = {float(v): j for j, v in enumerate(phi_deg)}

        Ex = np.zeros((Nf, Nt, Np), dtype=np.complex128)
        Ey = np.zeros((Nf, Nt, Np), dtype=np.complex128)
        Ez = np.zeros((Nf, Nt, Np), dtype=np.complex128)

        cur_fi = -1
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                parts = s[1:].strip().split()
                if not parts:
                    continue
                f_hz = float(parts[0])
                cur_fi = int(np.argmin(np.abs(freqs - f_hz)))
                continue

            if s.startswith("$"):
                continue

            if cur_fi < 0:
                continue

            cols = s.split()
            if len(cols) < 8:
                continue

            th = float(cols[0])
            ph = float(cols[1])

            i = theta_to_i.get(th, None)
            j = phi_to_j.get(ph, None)
            if i is None or j is None:
                th2 = float(round(th))
                ph2 = float(round(ph))
                i = theta_to_i.get(th2, None)
                j = phi_to_j.get(ph2, None)
                if i is None or j is None:
                    continue

            ex = float(cols[2]) + 1j * float(cols[3])
            ey = float(cols[4]) + 1j * float(cols[5])
            ez = float(cols[6]) + 1j * float(cols[7])

            Ex[cur_fi, i, j] = ex
            Ey[cur_fi, i, j] = ey
            Ez[cur_fi, i, j] = ez

    return freqs, theta_deg, phi_deg, Ex, Ey, Ez


# ---------------------- Pattern math ----------------------

def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("Zero-length vector")
    return v / n


def sph_unit_vectors(theta: np.ndarray, phi: np.ndarray):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    th_hat = np.stack([-ct * cp, -ct * sp, st], axis=-1)
    ph_hat = np.stack([-sp, cp, np.zeros_like(theta)], axis=-1)
    return th_hat, ph_hat


def cart_to_sph_E(Ex, Ey, Ez, theta, phi):
    th_hat, ph_hat = sph_unit_vectors(theta, phi)
    E = np.stack([Ex, Ey, Ez], axis=-1)
    Etheta = np.sum(E * th_hat, axis=-1)
    Ephi = np.sum(E * ph_hat, axis=-1)
    return Etheta, Ephi


def dir_unit_from_theta_phi(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def inplane_angle(theta, phi, plane_normal, ref_dir):
    n = unit(plane_normal)
    e1 = unit(ref_dir)
    e2 = unit(np.cross(n, e1))
    u = dir_unit_from_theta_phi(theta, phi)
    x = np.sum(u * e1, axis=-1)
    y = np.sum(u * e2, axis=-1)
    return np.arctan2(y, x)


def to_db_20(x_lin, floor_db):
    x_lin = np.maximum(x_lin, 1e-300)
    db = 20.0 * np.log10(x_lin)
    return np.maximum(db, floor_db)


def nearest_index(arr, target):
    arr = np.array(arr, dtype=float)
    return int(np.argmin(np.abs(arr - target)))


def interp_complex_1d(x, y, x_new):
    """Linear interpolation for complex-valued 1-D data (extrapolates at edges)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=np.complex128)
    re = np.interp(x_new, x, np.real(y))
    im = np.interp(x_new, x, np.imag(y))
    return re + 1j * im


def read_s2p(path: Path):
    """
    Read a 2-port Touchstone file.  Returns (freqs_hz ndarray (N,), sparams ndarray (N,2,2)).
    Requires scikit-rf (pip install scikit-rf).
    """
    if rf is None:
        raise RuntimeError(
            "scikit-rf is required to read .s2p files.\nInstall with:  pip install scikit-rf"
        )
    ntwk = rf.Network(str(path))
    if ntwk.number_of_ports != 2:
        raise ValueError(f"Expected a 2-port network, got {ntwk.number_of_ports} ports.")
    return np.array(ntwk.f, dtype=float), np.array(ntwk.s, dtype=np.complex128)


def read_snp_antenna_s11(path: Path, num_ports: int):
    """
    Read an N-port Touchstone file and extract diagonal S11 parameters (port i → port i).
    Returns dict with key "port_k" (0-indexed): {"freqs": ndarray (N,), "s11": ndarray (N,)}.
    Ignores off-diagonal mutual couplings.
    """
    if rf is None:
        raise RuntimeError("scikit-rf required for .snp files. Install with: pip install scikit-rf")
    
    ntwk = rf.Network(str(path))
    if ntwk.number_of_ports != num_ports:
        raise ValueError(
            f"Expected {num_ports}-port network, got {ntwk.number_of_ports} ports in {path.name}"
        )
    
    freqs = np.array(ntwk.f, dtype=float)
    sparams = np.array(ntwk.s, dtype=np.complex128)  # (Nfreq, num_ports, num_ports)
    
    result = {}
    for k in range(num_ports):
        result[f"port_{k}"] = {
            "freqs": freqs,
            "s11": sparams[:, k, k].copy(),
        }
    
    return result


def mag_over_eiso(Etheta, Ephi, pol: str):
    pol = pol.upper()

    if pol == "THETA":
        E = Etheta
    elif pol == "PHI":
        E = Ephi
    elif pol == "ABS":
        E = np.sqrt(np.abs(Etheta) ** 2 + np.abs(Ephi) ** 2)
    elif pol == "RHCP":
        E = (Etheta - 1j * Ephi) / np.sqrt(2.0)
    elif pol == "LHCP":
        E = (Etheta + 1j * Ephi) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unknown polarization '{pol}'")

    return np.abs(E) / float(em.lib.EISO)


# ---------------------- Array combiner app ----------------------

MIN_PORTS = 2
MAX_PORTS = 16


class ArrayPatternApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EMerge Array Pattern Viewer (EMFF)")

        self.run_dir: Path | None = None
        self.emff_paths: list[Path] = []

        self.freqs = None            # (Nf,)
        self.theta = None            # radians (Nt,)
        self.phi = None              # radians (Np,)

        self.elements: list[dict] = []

        # Per-element excitation vars; populated in _rebuild_element_controls()
        self.amp_vars: list[tk.StringVar] = []
        self.phase_vars: list[tk.StringVar] = []

        # Persistent plot figures: plane -> (fig, ax)
        self._polar_figs: dict[str, tuple] = {}
        self._cart_figs: dict[str, tuple] = {}

        # Persistent gain-vs-frequency figure
        self._max_gain_fig = None
        self._max_gain_ax = None

        # Per-port matching circuits: list[dict|None], one entry per loaded element.
        # Each dict: {"path": Path, "freqs": ndarray (N,), "sparams": ndarray (N,2,2)}
        self.match_circuits: list = []

        # Per-port antenna S11 data: dict with keys "port_0", "port_1", ...
        # Each key maps to {"freqs": ndarray (N,), "s11": ndarray (N,)} (diagonal S11 only).
        self.antenna_s11_data: dict | None = None

        self._build_ui()

    def _build_ui(self):
        outer = tk.Frame(self.root, padx=10, pady=10)
        outer.pack(fill="both", expand=True)

        top = tk.Frame(outer)
        top.pack(fill="x", pady=(0, 8))

        tk.Button(top, text="LOAD FOLDER", width=18, command=self.load_folder).pack(side="left", padx=4)
        tk.Button(top, text="PLOT", width=18, command=self.plot_selected).pack(side="left", padx=4)
        tk.Button(top, text="MAX GAIN vs FREQ", width=18, command=self.plot_max_gain_vs_freq).pack(side="left", padx=4)
        tk.Button(top, text="LOAD MATCHING CIRCUIT", width=22, command=self.load_matching_circuit).pack(side="left", padx=4)
        tk.Button(top, text="CLEAR PLOTS", width=18, command=self.clear_plots).pack(side="left", padx=4)

        self.status = tk.StringVar(value=f"Load a folder with {MIN_PORTS}-{MAX_PORTS} embedded-element *.emff files.")
        tk.Label(outer, textvariable=self.status, anchor="w").pack(fill="x", pady=(0, 8))

        mid = tk.Frame(outer)
        mid.pack(fill="both", expand=True)

        # LEFT: frequency list
        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="Frequencies (select one):").pack(anchor="w")
        self.freq_list = tk.Listbox(left, height=18, width=34)
        self.freq_list.pack(fill="both", expand=True)

        # CENTER: options panel
        right = tk.Frame(mid)
        right.pack(side="left", fill="y")

        # Cut planes (multi-select)
        tk.Label(right, text="Cut plane(s):").pack(anchor="w")
        self.plane_list = tk.Listbox(right, height=3, exportselection=False, selectmode=tk.MULTIPLE, width=12)
        for p in ("XY", "XZ", "YZ"):
            self.plane_list.insert(tk.END, p)
        self.plane_list.selection_set(0)
        self.plane_list.pack(anchor="w", pady=(0, 10))

        # Plot type (single-select)
        tk.Label(right, text="Plot type:").pack(anchor="w")
        self.plot_type_list = tk.Listbox(right, height=2, exportselection=False, selectmode=tk.SINGLE, width=12)
        for t in ("Polar", "Cartesian"):
            self.plot_type_list.insert(tk.END, t)
        self.plot_type_list.selection_set(0)
        self.plot_type_list.pack(anchor="w", pady=(0, 10))

        # Polarization
        tk.Label(right, text="Polarization:").pack(anchor="w")
        self.pol_var = tk.StringVar(value="ABS")
        tk.OptionMenu(right, self.pol_var, "ABS", "THETA", "PHI", "RHCP", "LHCP").pack(anchor="w", pady=(0, 8))

        # dB limits
        tk.Label(right, text="dB floor:").pack(anchor="w")
        self.floor_var = tk.StringVar(value="-30")
        tk.Entry(right, textvariable=self.floor_var, width=8).pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="dB max:").pack(anchor="w")
        self.max_var = tk.StringVar(value="10")
        tk.Entry(right, textvariable=self.max_var, width=8).pack(anchor="w", pady=(0, 8))

        self.norm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Normalize (0 dB max)", variable=self.norm_var).pack(anchor="w", pady=(0, 10))

        self.info = tk.StringVar(value="")
        tk.Label(right, textvariable=self.info, justify="left", anchor="w").pack(anchor="w", pady=(4, 0))

        # RIGHT: scrollable per-element excitation controls
        exc_outer = tk.Frame(mid)
        exc_outer.pack(side="left", fill="y", padx=(14, 0))

        tk.Label(exc_outer, text="Element excitation:").pack(anchor="w")

        hdrs = tk.Frame(exc_outer)
        hdrs.pack(anchor="w")
        tk.Label(hdrs, text="Port", width=6, anchor="w").pack(side="left")
        tk.Label(hdrs, text="Ampl.", width=7, anchor="w").pack(side="left")
        tk.Label(hdrs, text="Phase (deg)", width=10, anchor="w").pack(side="left")
        tk.Label(hdrs, text="Matching", width=16, anchor="w").pack(side="left")

        # scrollable canvas for the rows
        canvas = tk.Canvas(exc_outer, width=340, height=340, highlightthickness=0)
        vsb = tk.Scrollbar(exc_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.exc_frame = tk.Frame(canvas)
        self._exc_canvas_win = canvas.create_window((0, 0), window=self.exc_frame, anchor="nw")

        def _on_frame_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.exc_frame.bind("<Configure>", _on_frame_resize)
        self._exc_canvas = canvas

    # ------------------------------------------------------------------ #
    # Dynamic element-control builder                                      #
    # ------------------------------------------------------------------ #

    def _rebuild_element_controls(self, n: int):
        """Destroy existing rows and create n amplitude + phase entry rows."""
        for widget in self.exc_frame.winfo_children():
            widget.destroy()

        self.amp_vars = []
        self.phase_vars = []
        self.match_label_vars = []
        # Preserve existing matching circuits if port count unchanged, else reset.
        if len(self.match_circuits) != n:
            self.match_circuits = [None] * n

        for k in range(n):
            row = tk.Frame(self.exc_frame)
            row.pack(anchor="w", pady=2)

            tk.Label(row, text=f"P{k+1}", width=6, anchor="w").pack(side="left")

            amp_v = tk.StringVar(value="1.0")
            tk.Entry(row, textvariable=amp_v, width=7).pack(side="left", padx=(0, 4))

            ph_v = tk.StringVar(value="0")
            tk.Entry(row, textvariable=ph_v, width=10).pack(side="left", padx=(0, 4))

            mc = self.match_circuits[k]
            match_text = mc["path"].name if mc else "\u2014"
            m_v = tk.StringVar(value=match_text)
            tk.Label(row, textvariable=m_v, width=16, anchor="w",
                     fg="#006600" if mc else "gray").pack(side="left")

            self.amp_vars.append(amp_v)
            self.phase_vars.append(ph_v)
            self.match_label_vars.append(m_v)

        # refresh canvas scroll region
        self.exc_frame.update_idletasks()
        self._exc_canvas.configure(scrollregion=self._exc_canvas.bbox("all"))

    # ------------------------------------------------------------------ #
    # Folder loading                                                       #
    # ------------------------------------------------------------------ #

    def load_folder(self):
        d = filedialog.askdirectory(
            title=f"Select folder containing {MIN_PORTS}-{MAX_PORTS} *.emff files"
        )
        if not d:
            return
        run_dir = Path(d)

        emffs = sorted(run_dir.glob("*.emff"))
        nf = len(emffs)

        if nf < MIN_PORTS:
            messagebox.showerror(
                "Not enough files",
                f"Found {nf} *.emff file(s) in:\n{run_dir}\n"
                f"Need at least {MIN_PORTS}.",
            )
            return

        if nf > MAX_PORTS:
            messagebox.showwarning(
                "Too many files",
                f"Found {nf} *.emff files. Using first {MAX_PORTS}.",
            )
            emffs = emffs[:MAX_PORTS]
            nf = MAX_PORTS

        # Sort by port number if names contain _P<n>
        def port_sort_key(p: Path):
            for part in reversed(p.stem.split("_")):
                if part.upper().startswith("P"):
                    try:
                        return int(part[1:])
                    except ValueError:
                        pass
            return 9999

        emffs = sorted(emffs, key=port_sort_key)

        elements = []
        ref_freqs = ref_theta_deg = ref_phi_deg = None

        try:
            for p in emffs:
                freqs, theta_deg, phi_deg, Ex, Ey, Ez = parse_emff(p)
                elements.append({
                    "path": p,
                    "freqs": freqs,
                    "theta_deg": theta_deg,
                    "phi_deg": phi_deg,
                    "Ex": Ex,
                    "Ey": Ey,
                    "Ez": Ez,
                })

                if ref_freqs is None:
                    ref_freqs, ref_theta_deg, ref_phi_deg = freqs, theta_deg, phi_deg
                else:
                    if len(freqs) != len(ref_freqs) or np.max(np.abs(freqs - ref_freqs)) > 0:
                        raise ValueError(f"Frequency grid mismatch in {p.name}")
                    if len(theta_deg) != len(ref_theta_deg) or np.max(np.abs(theta_deg - ref_theta_deg)) > 0:
                        raise ValueError(f"Theta grid mismatch in {p.name}")
                    if len(phi_deg) != len(ref_phi_deg) or np.max(np.abs(phi_deg - ref_phi_deg)) > 0:
                        raise ValueError(f"Phi grid mismatch in {p.name}")

        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.run_dir = run_dir
        self.emff_paths = [e["path"] for e in elements]
        self.elements = elements

        self.freqs = ref_freqs
        self.theta = np.deg2rad(ref_theta_deg)
        self.phi = np.deg2rad(ref_phi_deg)

        self.freq_list.delete(0, tk.END)
        for i, f in enumerate(self.freqs):
            self.freq_list.insert(tk.END, f"{i:3d}: {f/1e9:.6f} GHz")
        self.freq_list.selection_clear(0, tk.END)
        self.freq_list.selection_set(0)

        # Auto-load antenna S11 if available
        self.antenna_s11_data = None
        snp_file = self._find_antenna_snp(run_dir, nf)
        if snp_file:
            try:
                self.antenna_s11_data = read_snp_antenna_s11(snp_file, nf)
            except Exception as e:
                print(f"Warning: Could not load antenna S11 from {snp_file.name}: {e}")

        self._rebuild_element_controls(nf)

        self.status.set(f"Loaded {nf} EMFF file(s) from: {run_dir.name}")
        self.info.set(
            f"{nf} elements loaded\n"
            f"Freq points: {len(self.freqs)}\n"
            f"Theta: {len(self.theta)} pts\n"
            f"Phi: {len(self.phi)} pts"
        )

    # ------------------------------------------------------------------ #
    # Matching circuit                                                     #
    # ------------------------------------------------------------------ #

    def load_matching_circuit(self):
        if not self.elements:
            messagebox.showinfo("No data", "Load a folder with EMFF files first.")
            return

        p = filedialog.askopenfilename(
            title="Select matching network Touchstone file (*.s2p)",
            filetypes=[("Touchstone 2-port", "*.s2p *.S2P"), ("All files", "*.*")],
        )
        if not p:
            return

        s2p_path = Path(p)
        try:
            match_freqs, match_sparams = read_s2p(s2p_path)
        except Exception as e:
            messagebox.showerror("S2P load error", f"Failed to parse:\n{s2p_path}\n\n{e}")
            return

        n = len(self.elements)
        port_num = simpledialog.askinteger(
            "Select port",
            f"Apply matching circuit to which port? (1\u2013{n})",
            minvalue=1,
            maxvalue=n,
            parent=self.root,
        )
        if port_num is None:
            return

        k = port_num - 1
        self.match_circuits[k] = {
            "path": s2p_path,
            "freqs": match_freqs,
            "sparams": match_sparams,
        }
        self.match_label_vars[k].set(s2p_path.name)
        self.status.set(f"Matching circuit for P{port_num}: {s2p_path.name}")

    def _matching_s21_for_port(self, k: int, f_hz: float) -> float:
        """
        Return power correction factor |S21|² / |1 - S22·Γ_ant|² for matching circuit on port k.
        If no matching circuit or antenna S11 loaded, returns 1.0 (no correction).
        Formula: Pout/Pin = |S21|² / |1 - S22·Γ_ant|² accounts for mismatch loss.
        """
        mc = self.match_circuits[k] if k < len(self.match_circuits) else None
        if mc is None:
            return 1.0

        # Get antenna S11 (reflection coefficient)
        ant_data = None
        if self.antenna_s11_data:
            ant_data = self.antenna_s11_data.get(f"port_{k}", None)
        
        if ant_data is None:
            # No antenna S11, use just |S21|² (ignores mismatch loss)
            s21 = interp_complex_1d(mc["freqs"], mc["sparams"][:, 1, 0], float(f_hz))
            return float(np.abs(s21) ** 2)
        
        # Get matching network parameters
        s21_m = interp_complex_1d(mc["freqs"], mc["sparams"][:, 1, 0], float(f_hz))
        s22_m = interp_complex_1d(mc["freqs"], mc["sparams"][:, 1, 1], float(f_hz))
        
        # Get antenna S11 (reflection coefficient)
        gamma_ant = interp_complex_1d(ant_data["freqs"], ant_data["s11"], float(f_hz))
        
        # Power ratio: |S21|² / |1 - S22·Γ_ant|²
        den = np.abs(1.0 - s22_m * gamma_ant) ** 2
        den = max(float(den), 1e-300)
        ratio = float(np.abs(s21_m) ** 2) / den
        
        return max(float(ratio), 0.0) if np.isfinite(ratio) else 1.0

    def _find_antenna_snp(self, search_dir: Path, num_ports: int) -> Path | None:
        """
        Search for an N-port Touchstone file (*.snp) in the given directory.
        Looks for common patterns like *.s2p, *.s4p, etc. matching num_ports.
        Returns the first match found, or None.
        """
        snp_pattern = f"*.s{num_ports}p"
        matches = sorted(search_dir.glob(snp_pattern))
        if matches:
            return matches[0]
        
        # Fallback: try parent directory
        if search_dir.parent != search_dir:
            matches = sorted(search_dir.parent.glob(snp_pattern))
            if matches:
                return matches[0]
        
        return None


    def _selected_freq_index(self):
        sel = self.freq_list.curselection()
        return int(sel[0]) if sel else None

    def _selected_planes(self):
        idxs = self.plane_list.curselection()
        return [self.plane_list.get(i).upper() for i in idxs]

    def _selected_plot_type(self):
        sel = self.plot_type_list.curselection()
        return self.plot_type_list.get(sel[0]).upper() if sel else "POLAR"

    def _plane_spec(self, plane: str):
        plane = plane.upper()
        if plane == "XY":
            return (0, 0, 1), (1, 0, 0)
        if plane == "XZ":
            return (0, 1, 0), (1, 0, 0)
        if plane == "YZ":
            return (1, 0, 0), (0, 1, 0)
        raise ValueError(f"Unknown plane '{plane}'")

    # ------------------------------------------------------------------ #
    # Plane-cut extraction (unchanged logic)                               #
    # ------------------------------------------------------------------ #

    def _extract_plane_cut_from_element(self, el, idx_f: int, plane: str):
        plane_normal, ref_dir = self._plane_spec(plane)

        theta = self.theta
        phi = self.phi

        Ex = el["Ex"][idx_f, :, :]
        Ey = el["Ey"][idx_f, :, :]
        Ez = el["Ez"][idx_f, :, :]

        if plane.upper() == "XY":
            it = nearest_index(theta, np.pi / 2)
            th = theta[it] * np.ones_like(phi)
            ph = phi.copy()
            Exs, Eys, Ezs = Ex[it, :], Ey[it, :], Ez[it, :]

        elif plane.upper() == "XZ":
            ip0 = nearest_index(phi, 0.0)
            ip1 = nearest_index(phi, np.pi)

            th_a, ph_a = theta.copy(), phi[ip0] * np.ones_like(theta)
            Ex_a, Ey_a, Ez_a = Ex[:, ip0], Ey[:, ip0], Ez[:, ip0]

            th_b, ph_b = theta[::-1].copy(), phi[ip1] * np.ones_like(theta)
            Ex_b, Ey_b, Ez_b = Ex[::-1, ip1], Ey[::-1, ip1], Ez[::-1, ip1]

            th = np.concatenate([th_a, th_b])
            ph = np.concatenate([ph_a, ph_b])
            Exs = np.concatenate([Ex_a, Ex_b])
            Eys = np.concatenate([Ey_a, Ey_b])
            Ezs = np.concatenate([Ez_a, Ez_b])

        elif plane.upper() == "YZ":
            ip0 = nearest_index(phi, np.pi / 2)
            ip1 = nearest_index(phi, -np.pi / 2)

            th_a, ph_a = theta.copy(), phi[ip0] * np.ones_like(theta)
            Ex_a, Ey_a, Ez_a = Ex[:, ip0], Ey[:, ip0], Ez[:, ip0]

            th_b, ph_b = theta[::-1].copy(), phi[ip1] * np.ones_like(theta)
            Ex_b, Ey_b, Ez_b = Ex[::-1, ip1], Ey[::-1, ip1], Ez[::-1, ip1]

            th = np.concatenate([th_a, th_b])
            ph = np.concatenate([ph_a, ph_b])
            Exs = np.concatenate([Ex_a, Ex_b])
            Eys = np.concatenate([Ey_a, Ey_b])
            Ezs = np.concatenate([Ez_a, Ez_b])

        else:
            raise ValueError(f"Unknown plane '{plane}'")

        Etheta, Ephi = cart_to_sph_E(Exs, Eys, Ezs, th, ph)
        ang = inplane_angle(th, ph, plane_normal, ref_dir)
        order = np.argsort(ang)
        return ang[order], Etheta[order], Ephi[order]

    # ------------------------------------------------------------------ #
    # Complex excitation weights (amplitude × exp(jφ))                   #
    # ------------------------------------------------------------------ #

    def _element_weights(self):
        n = len(self.elements)
        weights = np.zeros(n, dtype=np.complex128)
        for k in range(n):
            try:
                amp = float(self.amp_vars[k].get().strip())
            except Exception:
                raise ValueError(f"Invalid amplitude for P{k+1}: '{self.amp_vars[k].get()}'")
            try:
                ph_deg = float(self.phase_vars[k].get().strip())
            except Exception:
                raise ValueError(f"Invalid phase for P{k+1}: '{self.phase_vars[k].get()}'")
            weights[k] = amp * np.exp(1j * np.deg2rad(ph_deg))
        return weights

    # ------------------------------------------------------------------ #
    # Combined pattern computation                                         #
    # ------------------------------------------------------------------ #

    def _compute_plane_db(self, idx_f: int, plane: str, pol: str, floor_db: float, normalize: bool):
        w = self._element_weights()
        f_hz = float(self.freqs[idx_f])

        ang_ref = None
        Etheta_sum = None
        Ephi_sum = None

        for k, el in enumerate(self.elements):
            ang, Etheta, Ephi = self._extract_plane_cut_from_element(el, idx_f, plane)

            if ang_ref is None:
                ang_ref = ang
                Etheta_sum = np.zeros_like(Etheta, dtype=np.complex128)
                Ephi_sum = np.zeros_like(Ephi, dtype=np.complex128)
            else:
                if len(ang) != len(ang_ref) or np.max(np.abs(ang - ang_ref)) > 1e-12:
                    raise ValueError(f"Angle grid mismatch for plane {plane} in {el['path'].name}")

            # Apply per-port excitation weight and matching power correction
            power_factor = self._matching_s21_for_port(k, f_hz)
            correction = np.sqrt(power_factor)  # Convert power factor to amplitude correction
            Etheta_sum += w[k] * correction * Etheta
            Ephi_sum += w[k] * correction * Ephi

        mag = mag_over_eiso(Etheta_sum, Ephi_sum, pol)
        db = to_db_20(mag, floor_db=floor_db)
        if normalize:
            db = db - np.max(db)

        return ang_ref, db

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #

    def _figure_exists(self, fig) -> bool:
        return fig is not None and plt.fignum_exists(fig.number)

    def clear_plots(self):
        for fig, _ax in self._polar_figs.values():
            if self._figure_exists(fig):
                plt.close(fig)
        self._polar_figs.clear()
        for fig, _ax in self._cart_figs.values():
            if self._figure_exists(fig):
                plt.close(fig)
        self._cart_figs.clear()
        if self._figure_exists(self._max_gain_fig):
            plt.close(self._max_gain_fig)
        self._max_gain_fig = None
        self._max_gain_ax = None

    def _plot_polar(self, ang, db, plane, label, floor_db, db_max):
        if plane not in self._polar_figs or not self._figure_exists(self._polar_figs[plane][0]):
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            self._polar_figs[plane] = (fig, ax)
        else:
            fig, ax = self._polar_figs[plane]
        ax.plot(ang, db, label=label)
        ax.set_title(f"Array Pattern — {plane} (Polar)")
        ax.set_rlim(floor_db, db_max)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.show()
        plt.show(block=False)

    def _plot_cartesian(self, ang, db, plane, label, floor_db, db_max):
        x_deg = np.rad2deg(ang)
        if plane not in self._cart_figs or not self._figure_exists(self._cart_figs[plane][0]):
            fig, ax = plt.subplots()
            self._cart_figs[plane] = (fig, ax)
        else:
            fig, ax = self._cart_figs[plane]
        ax.plot(x_deg, db, label=label)
        ax.set_title(f"Array Pattern — {plane} (Cartesian)")
        ax.set_xlabel(f"{plane} in-plane angle [deg]")
        ax.set_ylabel("Gain [dBi] (relative to EISO)")
        ax.set_ylim(floor_db, db_max)
        ax.grid(True)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.show()
        plt.show(block=False)

    def plot_selected(self):
        if self.freqs is None or not self.elements:
            messagebox.showinfo("Nothing loaded", "Load a folder with *.emff files first.")
            return

        idx = self._selected_freq_index()
        if idx is None:
            messagebox.showinfo("Select frequency", "Select one frequency from the list.")
            return

        planes = self._selected_planes()
        if not planes:
            messagebox.showinfo("Select plane", "Select at least one cut plane (XY/XZ/YZ).")
            return

        pol = self.pol_var.get().upper()
        plot_type = self._selected_plot_type()

        try:
            floor_db = float(self.floor_var.get().strip())
            db_max = float(self.max_var.get().strip())
        except Exception:
            messagebox.showerror("Invalid limits", "Enter valid numbers for dB floor and dB max.")
            return

        try:
            w = self._element_weights()
        except Exception as e:
            messagebox.showerror("Invalid excitation", str(e))
            return

        fsel = float(self.freqs[idx])
        n = len(self.elements)

        try:
            for plane in planes:
                ang, db = self._compute_plane_db(
                    idx_f=idx,
                    plane=plane,
                    pol=pol,
                    floor_db=floor_db,
                    normalize=self.norm_var.get(),
                )

                label = f"{pol} {fsel/1e9:.4f} GHz"

                if plot_type == "POLAR":
                    self._plot_polar(ang, db, plane, label, floor_db, db_max)
                else:
                    self._plot_cartesian(ang, db, plane, label, floor_db, db_max)

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            return


    def plot_max_gain_vs_freq(self):
        if self.freqs is None or not self.elements:
            messagebox.showinfo("Nothing loaded", "Load a folder with *.emff files first.")
            return

        pol = self.pol_var.get().upper()

        try:
            w = self._element_weights()
        except Exception as e:
            messagebox.showerror("Invalid excitation", str(e))
            return

        try:
            th_grid, ph_grid = np.meshgrid(self.theta, self.phi, indexing="ij")  # (Nt, Np)
            max_gain_db = np.zeros(len(self.freqs))

            for idx_f in range(len(self.freqs)):
                f_hz = float(self.freqs[idx_f])
                Etheta_sum = np.zeros((len(self.theta), len(self.phi)), dtype=np.complex128)
                Ephi_sum   = np.zeros((len(self.theta), len(self.phi)), dtype=np.complex128)

                for k, el in enumerate(self.elements):
                    Ex = el["Ex"][idx_f, :, :]
                    Ey = el["Ey"][idx_f, :, :]
                    Ez = el["Ez"][idx_f, :, :]
                    Etheta, Ephi = cart_to_sph_E(Ex, Ey, Ez, th_grid, ph_grid)
                    # Apply per-port excitation weight and matching power correction
                    power_factor = self._matching_s21_for_port(k, f_hz)
                    correction = np.sqrt(power_factor)  # Convert power factor to amplitude correction
                    Etheta_sum += w[k] * correction * Etheta
                    Ephi_sum   += w[k] * correction * Ephi

                mag = mag_over_eiso(Etheta_sum, Ephi_sum, pol)
                max_gain_db[idx_f] = 20.0 * np.log10(max(float(np.max(mag)), 1e-300))

        except Exception as e:
            messagebox.showerror("Computation error", str(e))
            return

        n = len(self.elements)
        exc_short = "  ".join(
            f"P{k+1}:{abs(w[k]):.2f}\u2220{np.rad2deg(np.angle(w[k])):.0f}\u00b0"
            for k in range(n)
        )
        trace_label = f"{pol} | {exc_short}"

        if not self._figure_exists(self._max_gain_fig):
            self._max_gain_fig, self._max_gain_ax = plt.subplots()
            for f_start, f_stop, band_label, color in FREQ_BANDS:
                self._max_gain_ax.axvspan(f_start, f_stop, alpha=0.15, color=color, label=band_label)
            self._max_gain_ax.set_xlabel("Frequency (GHz)")
            self._max_gain_ax.set_ylabel("Max array gain (dBi)")
            self._max_gain_ax.grid(True)

        self._max_gain_ax.plot(
            self.freqs / 1e9, max_gain_db, marker="o", markersize=4, zorder=3, label=trace_label
        )
        self._max_gain_ax.set_title("Maximum Array Gain vs Frequency")
        self._max_gain_ax.legend(loc="best", fontsize="small")
        self._max_gain_fig.tight_layout()
        self._max_gain_fig.show()
        plt.show(block=False)


def main():
    root = tk.Tk()
    ArrayPatternApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()