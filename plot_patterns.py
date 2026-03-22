import tkinter as tk
from tkinter import filedialog, messagebox
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
    """
    Read space-separated numbers from subsequent lines until next line starts
    with one of stop_prefixes or EOF. Returns (numbers, last_line).
    last_line is the first non-data line we stopped at (or "" on EOF).
    """
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
            # rewind so caller can re-read this line if desired
            f.seek(pos)
            return nums, s
        nums.extend([float(x) for x in s.split()])
    # unreachable


def parse_emff(path: Path):
    """
    Parse EMerge FarFieldExporter .emff format as produced by your run_emerge.py:
      - Explicit Theta(deg) list
      - Explicit Phi(deg) list
      - Frequencies(Hz) list
      - Blocks per frequency with Theta, Phi, Ex/Ey/Ez re/im

    Returns:
      freqs_hz: (Nf,)
      theta_deg: (Nt,)
      phi_deg: (Np,)
      Ex/Ey/Ez: complex arrays (Nf, Nt, Np)
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        theta_deg = None
        phi_deg = None
        freqs = None

        # Scan header sections
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
                # After frequencies, header is done; next comes blocks
                break

        if theta_deg is None or phi_deg is None or freqs is None:
            raise ValueError(
                f"Failed to parse EMFF header in {path.name}. "
                f"Got theta={theta_deg is not None}, phi={phi_deg is not None}, freqs={freqs is not None}"
            )

        Nt = theta_deg.size
        Np = phi_deg.size
        Nf = freqs.size

        # Maps for fast indexing (theta/phi are integers in your file)
        theta_to_i = {float(v): i for i, v in enumerate(theta_deg)}
        phi_to_j = {float(v): j for j, v in enumerate(phi_deg)}

        Ex = np.zeros((Nf, Nt, Np), dtype=np.complex128)
        Ey = np.zeros((Nf, Nt, Np), dtype=np.complex128)
        Ez = np.zeros((Nf, Nt, Np), dtype=np.complex128)

        # Now parse per-frequency blocks
        cur_fi = -1
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                # Example: "# 6.8e+09 Hz"
                parts = s[1:].strip().split()
                if not parts:
                    continue
                f_hz = float(parts[0])
                # Find which index this is (should match exactly)
                # Use nearest in case of formatting differences
                cur_fi = int(np.argmin(np.abs(freqs - f_hz)))
                continue

            if s.startswith("$"):
                # Column header line, ignore
                continue

            if cur_fi < 0:
                continue

            # Data row: theta phi Exre Exim Eyre Eyim Ezre Ezim
            cols = s.split()
            if len(cols) < 8:
                continue

            th = float(cols[0])
            ph = float(cols[1])

            i = theta_to_i.get(th, None)
            j = phi_to_j.get(ph, None)
            if i is None or j is None:
                # If float formatting differs, try rounding to nearest int
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


# ---------------------- Touchstone (simple S1P parser) ----------------------

def read_s1p(path: Path):
    """
    Minimal Touchstone S1P reader (RI/MA/DB). Returns freqs_hz, s11_complex.
    """
    freqs = []
    s11 = []

    fmt = None
    f_unit = "hz"
    unit_scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line.lower().split()
                if len(parts) >= 2:
                    f_unit = parts[1]
                if "ri" in parts:
                    fmt = "ri"
                elif "ma" in parts:
                    fmt = "ma"
                elif "db" in parts:
                    fmt = "db"
                continue

            cols = line.split()
            if len(cols) < 3:
                continue

            f0 = float(cols[0]) * unit_scale.get(f_unit, 1.0)
            a = float(cols[1])
            b = float(cols[2])

            if fmt == "ma":
                c = a * np.exp(1j * np.deg2rad(b))
            elif fmt == "db":
                c = (10.0 ** (a / 20.0)) * np.exp(1j * np.deg2rad(b))
            else:  # RI or unknown -> assume RI
                c = a + 1j * b

            freqs.append(f0)
            s11.append(c)

    if not freqs:
        raise ValueError("No S11 points found in file.")
    return np.array(freqs, dtype=float), np.array(s11, dtype=np.complex128)


def read_s2p(path: Path):
    """
    Read a 2-port Touchstone file and return frequency in Hz and S-parameter matrix.
    Returns: freqs_hz (N,), sparams (N,2,2)
    """
    if rf is None:
        raise RuntimeError("scikit-rf is required to read .s2p files. Install with: pip install scikit-rf")

    ntwk = rf.Network(str(path))
    if ntwk.number_of_ports != 2:
        raise ValueError(f"Expected a 2-port network, got {ntwk.number_of_ports} ports.")
    return np.array(ntwk.f, dtype=float), np.array(ntwk.s, dtype=np.complex128)


def interp_complex_1d(x, y, x_new):
    """
    Linear interpolation for complex-valued 1D data.
    Extrapolates with edge values outside the source range.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=np.complex128)

    re = np.interp(x_new, x, np.real(y))
    im = np.interp(x_new, x, np.imag(y))
    return re + 1j * im


# ---------------------- Pattern math ----------------------

def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("Zero-length vector")
    return v / n


def sph_unit_vectors(theta: np.ndarray, phi: np.ndarray):
    """
    EMerge convention: theta from +Z, phi around Z, phi=0 at +X.
    Returns th_hat, ph_hat (..,3)
    """
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


# ---------------------- App ----------------------

class PatternViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EMerge Pattern Viewer (EMFF + Touchstone)")

        self.run_dir: Path | None = None
        self.emff_path: Path | None = None
        self.s1p_path: Path | None = None
        self.match_path: Path | None = None

        self.freqs = None
        self.theta = None   # radians 1D
        self.phi = None     # radians 1D
        self.Ex = None      # (Nf,Nt,Np) complex
        self.Ey = None
        self.Ez = None

        self.s11_freqs = None
        self.s11 = None
        self.match_freqs = None
        self.match_sparams = None

        self.pattern_fig = None
        self.pattern_ax = None
        self.max_gain_fig = None
        self.max_gain_ax = None
        self.eff_fig = None
        self.eff_ax = None

        outer = tk.Frame(root, padx=10, pady=10)
        outer.pack(fill="both", expand=True)

        top = tk.Frame(outer)
        top.pack(fill="x", pady=(0, 8))

        tk.Button(top, text="LOAD RUN / EMFF", width=18, command=self.load).pack(side="left", padx=4)
        tk.Button(top, text="LOAD MATCHING (.S2P)", width=18, command=self.load_matching_s2p).pack(side="left", padx=4)
        tk.Button(top, text="PLOT PATTERN", width=18, command=self.plot_pattern).pack(side="left", padx=4)
        tk.Button(top, text="PLOT S11", width=18, command=self.plot_s11).pack(side="left", padx=4)
        tk.Button(top, text="MAX GAIN vs FREQ", width=18, command=self.plot_max_gain_vs_freq).pack(side="left", padx=4)
        tk.Button(top, text="TOTAL EFF vs FREQ", width=18, command=self.plot_total_efficiency_vs_freq).pack(side="left", padx=4)
        tk.Button(top, text="CLEAR PLOTS", width=18, command=self.clear_plots).pack(side="left", padx=4)

        self.status = tk.StringVar(value="Load a run folder or a .emff file.")
        tk.Label(outer, textvariable=self.status, anchor="w").pack(fill="x", pady=(0, 8))

        mid = tk.Frame(outer)
        mid.pack(fill="both", expand=True)

        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="Frequencies (select one):").pack(anchor="w")
        self.freq_list = tk.Listbox(left, height=18, width=32)
        self.freq_list.pack(fill="both", expand=True)

        right = tk.Frame(mid)
        right.pack(side="left", fill="y")

        tk.Label(right, text="Plane:").pack(anchor="w")
        self.plane_var = tk.StringVar(value="XY")
        tk.OptionMenu(right, self.plane_var, "XY", "XZ", "YZ").pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="Polarization:").pack(anchor="w")
        self.pol_var = tk.StringVar(value="ABS")
        tk.OptionMenu(right, self.pol_var, "ABS", "THETA", "PHI", "RHCP", "LHCP").pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="dB floor:").pack(anchor="w")
        self.floor_var = tk.StringVar(value="-30")
        tk.Entry(right, textvariable=self.floor_var, width=8).pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="dB max:").pack(anchor="w")
        self.max_var = tk.StringVar(value="10")
        tk.Entry(right, textvariable=self.max_var, width=8).pack(anchor="w", pady=(0, 8))

        self.norm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Normalize (0 dB max)", variable=self.norm_var).pack(anchor="w", pady=(0, 8))

        self.info = tk.StringVar(value="")
        tk.Label(right, textvariable=self.info, justify="left", anchor="w").pack(anchor="w", pady=(8, 0))

    def load(self):
        p = filedialog.askopenfilename(
            title="Select Farfield file (*.emff) or Cancel to choose folder",
            filetypes=[("EMerge Farfield", "*.emff"), ("All files", "*.*")]
        )

        if p:
            emff = Path(p)
            run_dir = emff.parent
        else:
            d = filedialog.askdirectory(title="Select run folder (contains *.emff and *.s1p)")
            if not d:
                return
            run_dir = Path(d)
            emffs = sorted(run_dir.glob("*.emff"))
            if not emffs:
                messagebox.showerror("Not found", f"No *.emff files found in:\n{run_dir}")
                return
            emff = emffs[0]

        s1ps = sorted(run_dir.glob("*.s1p"))
        s2ps = sorted(run_dir.glob("*.s2p"))
        s1p = s1ps[0] if s1ps else None
        s2p = s2ps[0] if s2ps else None

        try:
            freqs, theta_deg, phi_deg, Ex, Ey, Ez = parse_emff(emff)
        except Exception as e:
            messagebox.showerror("EMFF load error", f"Failed to parse:\n{emff}\n\n{e}")
            return

        # Convert degrees -> radians
        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        # Load S11 if present
        s11_freqs = None
        s11 = None
        if s1p is not None:
            try:
                s11_freqs, s11 = read_s1p(s1p)
            except Exception as e:
                messagebox.showwarning("S1P load warning", f"Failed to read:\n{s1p}\n\n{e}")

        # Load matching network if present
        match_freqs = None
        match_sparams = None
        if s2p is not None:
            try:
                match_freqs, match_sparams = read_s2p(s2p)
            except Exception as e:
                messagebox.showwarning("S2P load warning", f"Failed to read:\n{s2p}\n\n{e}")

        self.run_dir = run_dir
        self.emff_path = emff
        self.s1p_path = s1p
        self.match_path = s2p

        self.freqs = freqs
        self.theta = theta
        self.phi = phi
        self.Ex, self.Ey, self.Ez = Ex, Ey, Ez

        self.s11_freqs, self.s11 = s11_freqs, s11
        self.match_freqs, self.match_sparams = match_freqs, match_sparams

        self.freq_list.delete(0, tk.END)
        for i, f in enumerate(self.freqs):
            self.freq_list.insert(tk.END, f"{i:3d}: {f/1e9:.6f} GHz")
        self.freq_list.selection_clear(0, tk.END)
        self.freq_list.selection_set(0)

        status = f"Loaded: {emff.name}"
        status += f"  +  {s1p.name}" if s1p else "  (no .s1p found)"
        status += f"  +  {s2p.name}" if s2p else "  (no .s2p found)"
        self.status.set(status)
        self._update_info()

        # Useful debug in console:
        print("Parsed shapes:",
              "freqs", self.freqs.shape,
              "theta", self.theta.shape,
              "phi", self.phi.shape,
              "Ex", self.Ex.shape)

    def load_matching_s2p(self):
        p = filedialog.askopenfilename(
            title="Select matching network Touchstone file (*.s2p)",
            filetypes=[("Touchstone 2-port", "*.s2p *.S2P"), ("All files", "*.*")]
        )
        if not p:
            return

        s2p_path = Path(p)
        try:
            match_freqs, match_sparams = read_s2p(s2p_path)
        except Exception as e:
            messagebox.showerror("S2P load error", f"Failed to parse:\n{s2p_path}\n\n{e}")
            return

        self.match_path = s2p_path
        self.match_freqs = match_freqs
        self.match_sparams = match_sparams

        self.status.set(f"Loaded matching network: {s2p_path.name}")
        self._update_info()

    def _update_info(self):
        if self.emff_path is None:
            self.info.set("")
            return

        s1p_txt = self.s1p_path.name if self.s1p_path else "—"
        s2p_txt = self.match_path.name if self.match_path else "—"
        self.info.set(
            f"EMFF: {self.emff_path.name}\n"
            f"S1P: {s1p_txt}\n"
            f"Matching S2P: {s2p_txt}\n"
            f"Freq points: {len(self.freqs)}\n"
            f"Theta: {len(self.theta)} pts\n"
            f"Phi: {len(self.phi)} pts"
        )

    def _antenna_s11_at_freq(self, f_hz: float):
        if self.s11_freqs is None or self.s11 is None:
            return 0.0 + 0.0j
        return interp_complex_1d(self.s11_freqs, self.s11, float(f_hz))

    def _matching_power_ratio(self, f_hz: float):
        """
        Return power ratio at antenna port with matching network relative
        to direct feed (same source/reference impedance).
        """
        if self.match_freqs is None or self.match_sparams is None:
            return 1.0

        s11_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 0, 0], f_hz)
        s21_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 1, 0], f_hz)
        s22_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 1, 1], f_hz)
        _ = s11_m  # kept for readability and future extensions

        gamma_ant = self._antenna_s11_at_freq(f_hz)
        den = np.abs(1.0 - s22_m * gamma_ant) ** 2
        den = max(float(den), 1e-300)

        ratio = (np.abs(s21_m) ** 2) / den
        if not np.isfinite(ratio):
            return 1.0
        return max(float(ratio), 0.0)

    def _s11_with_matching(self, freqs_hz: np.ndarray, gamma_ant: np.ndarray):
        """
        Compute input S11 seen at matching-network port 1 when antenna S11 is
        connected to matching-network port 2.
        """
        if self.match_freqs is None or self.match_sparams is None:
            return gamma_ant

        freqs_hz = np.array(freqs_hz, dtype=float)
        gamma_ant = np.array(gamma_ant, dtype=np.complex128)

        s11_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 0, 0], freqs_hz)
        s12_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 0, 1], freqs_hz)
        s21_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 1, 0], freqs_hz)
        s22_m = interp_complex_1d(self.match_freqs, self.match_sparams[:, 1, 1], freqs_hz)

        den = 1.0 - s22_m * gamma_ant
        den = np.where(np.abs(den) < 1e-300, 1e-300 + 0j, den)

        gamma_in = s11_m + (s12_m * s21_m * gamma_ant) / den
        return gamma_in

    def _selected_freq_index(self):
        sel = self.freq_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def _plane_spec(self, plane: str):
        plane = plane.upper()
        if plane == "XY":
            return (0, 0, 1), (1, 0, 0)
        if plane == "XZ":
            return (0, 1, 0), (1, 0, 0)
        if plane == "YZ":
            return (1, 0, 0), (0, 1, 0)
        raise ValueError(f"Unknown plane '{plane}'")

    def _extract_plane_cut(self, idx_f: int, plane: str):
        plane_normal, ref_dir = self._plane_spec(plane)

        theta = self.theta
        phi = self.phi

        Ex = self.Ex[idx_f, :, :]  # (Nt,Np)
        Ey = self.Ey[idx_f, :, :]
        Ez = self.Ez[idx_f, :, :]

        if plane.upper() == "XY":
            it = nearest_index(theta, np.pi / 2)
            th = theta[it] * np.ones_like(phi)
            ph = phi.copy()
            Exs, Eys, Ezs = Ex[it, :], Ey[it, :], Ez[it, :]

        elif plane.upper() == "XZ":
            ip0 = nearest_index(phi, 0.0)
            ip1 = nearest_index(phi, np.pi)

            th_a = theta.copy()
            ph_a = phi[ip0] * np.ones_like(theta)
            Ex_a, Ey_a, Ez_a = Ex[:, ip0], Ey[:, ip0], Ez[:, ip0]

            th_b = theta[::-1].copy()
            ph_b = phi[ip1] * np.ones_like(theta)
            Ex_b, Ey_b, Ez_b = Ex[::-1, ip1], Ey[::-1, ip1], Ez[::-1, ip1]

            th = np.concatenate([th_a, th_b])
            ph = np.concatenate([ph_a, ph_b])
            Exs = np.concatenate([Ex_a, Ex_b])
            Eys = np.concatenate([Ey_a, Ey_b])
            Ezs = np.concatenate([Ez_a, Ez_b])

        elif plane.upper() == "YZ":
            ip0 = nearest_index(phi, np.pi / 2)
            ip1 = nearest_index(phi, -np.pi / 2)

            th_a = theta.copy()
            ph_a = phi[ip0] * np.ones_like(theta)
            Ex_a, Ey_a, Ez_a = Ex[:, ip0], Ey[:, ip0], Ez[:, ip0]

            th_b = theta[::-1].copy()
            ph_b = phi[ip1] * np.ones_like(theta)
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

    def _mag_over_eiso(self, Etheta, Ephi, pol: str):
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

    def _figure_exists(self, fig):
        return fig is not None and plt.fignum_exists(fig.number)

    def clear_plots(self):
        if self._figure_exists(self.pattern_fig):
            plt.close(self.pattern_fig)
        if self._figure_exists(self.max_gain_fig):
            plt.close(self.max_gain_fig)
        if self._figure_exists(self.eff_fig):
            plt.close(self.eff_fig)

        self.pattern_fig = None
        self.pattern_ax = None
        self.max_gain_fig = None
        self.max_gain_ax = None
        self.eff_fig = None
        self.eff_ax = None

    def plot_pattern(self):
        if self.freqs is None:
            messagebox.showinfo("Nothing loaded", "Load a run folder / .emff first.")
            return

        idx = self._selected_freq_index()
        if idx is None:
            messagebox.showinfo("Select frequency", "Select one frequency from the list.")
            return

        plane = self.plane_var.get().upper()
        pol = self.pol_var.get().upper()

        try:
            floor_db = float(self.floor_var.get().strip())
            db_max = float(self.max_var.get().strip())
        except Exception:
            messagebox.showerror("Invalid limits", "Enter valid numbers for dB floor and dB max.")
            return

        try:
            ang, Etheta, Ephi = self._extract_plane_cut(idx, plane)
            mag = self._mag_over_eiso(Etheta, Ephi, pol)

            # Scale field magnitude by sqrt(power ratio) when matching network is loaded.
            fsel = float(self.freqs[idx])
            p_ratio = self._matching_power_ratio(fsel)
            mag = mag * np.sqrt(p_ratio)

            db = to_db_20(mag, floor_db=floor_db)

            if self.norm_var.get():
                db = db - np.max(db)

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            return

        if not self._figure_exists(self.pattern_fig):
            self.pattern_fig, self.pattern_ax = plt.subplots(subplot_kw={"projection": "polar"})

        label = f"{plane} {pol} {fsel/1e9:.3f} GHz"
        if self.match_path is not None and self.match_sparams is not None:
            label += " + matching"
        self.pattern_ax.plot(ang, db, label=label)
        self.pattern_ax.set_title("Radiation Pattern Overlays")
        self.pattern_ax.set_rlim(floor_db, db_max)
        self.pattern_ax.legend(loc="best")
        self.pattern_fig.tight_layout()
        self.pattern_fig.show()
        plt.show(block=False)

    # --- Replace ONLY the plot_s11() method in your current plot_patterns.py with this one ---

    def plot_s11(self):
        if self.s11 is None or self.s11_freqs is None:
            messagebox.showinfo("No S11", "No .s1p loaded (or it failed to parse).")
            return

        f = np.array(self.s11_freqs, dtype=float)
        s11_ant = np.array(self.s11, dtype=np.complex128)

        if self.match_freqs is not None and self.match_sparams is not None:
            s11 = self._s11_with_matching(f, s11_ant)
            title_suffix = "with matching network"
        else:
            s11 = s11_ant
            title_suffix = "antenna only"

        # 1) |S11| in dB vs frequency
        s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-300))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(f / 1e9, s11_db)
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("|S11| (dB)")
        ax1.grid(True)
        ax1.set_title(f"S11 magnitude (dB), {title_suffix}")
        plt.show()

        # 2) Smith chart with frequency markers
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        # Smith chart background (unit circle + axes)
        t = np.linspace(0, 2*np.pi, 800)
        ax2.plot(np.cos(t), np.sin(t))        # unit circle
        ax2.axhline(0, linewidth=0.8)
        ax2.axvline(0, linewidth=0.8)
        ax2.set_aspect("equal", "box")
        ax2.set_xlabel("Re{S11}")
        ax2.set_ylabel("Im{S11}")
        ax2.set_title(f"Smith chart (S11), {title_suffix}")

        # Plot trace
        ax2.plot(np.real(s11), np.imag(s11))

        # Frequency markers (start, mid, end + a few evenly spaced)
        n = len(f)
        if n > 1:
            marker_indices = sorted(set([
                0,
                n // 2,
                n - 1,
                n // 4,
                (3 * n) // 4
            ]))
        else:
            marker_indices = [0]

        ax2.scatter(np.real(s11[marker_indices]), np.imag(s11[marker_indices]))

        # Annotate each marker with frequency in GHz
        for i in marker_indices:
            ax2.annotate(
                f"{f[i]/1e9:.3f}G",
                (np.real(s11[i]), np.imag(s11[i])),
                textcoords="offset points",
                xytext=(6, 6)
            )

        ax2.grid(True)
        plt.show()


    def plot_max_gain_vs_freq(self):
        if self.freqs is None:
            messagebox.showinfo("Nothing loaded", "Load a run folder / .emff first.")
            return

        pol = self.pol_var.get().upper()

        try:
            th_grid, ph_grid = np.meshgrid(self.theta, self.phi, indexing="ij")  # (Nt, Np)
            max_gain_db = np.zeros(len(self.freqs))

            for idx_f in range(len(self.freqs)):
                Ex = self.Ex[idx_f, :, :]
                Ey = self.Ey[idx_f, :, :]
                Ez = self.Ez[idx_f, :, :]

                Etheta, Ephi = cart_to_sph_E(Ex, Ey, Ez, th_grid, ph_grid)
                mag = self._mag_over_eiso(Etheta, Ephi, pol)

                f_hz = float(self.freqs[idx_f])
                p_ratio = self._matching_power_ratio(f_hz)
                mag = mag * np.sqrt(p_ratio)

                max_gain_db[idx_f] = 20.0 * np.log10(max(float(np.max(mag)), 1e-300))

        except Exception as e:
            messagebox.showerror("Computation error", str(e))
            return

        if not self._figure_exists(self.max_gain_fig):
            self.max_gain_fig, self.max_gain_ax = plt.subplots()

            # Draw shaded frequency bands (defined in FREQ_BANDS at the top of this file)
            for f_start, f_stop, label, color in FREQ_BANDS:
                self.max_gain_ax.axvspan(f_start, f_stop, alpha=0.15, color=color, label=label)

            self.max_gain_ax.set_xlabel("Frequency (GHz)")
            self.max_gain_ax.set_ylabel("Max gain (dBi)")
            self.max_gain_ax.grid(True)

        trace_label = f"{pol}"
        self.max_gain_ax.plot(self.freqs / 1e9, max_gain_db, marker="o", markersize=4, zorder=3, label=trace_label)
        self.max_gain_ax.set_title("Maximum Antenna Gain vs Frequency")
        self.max_gain_ax.legend(loc="best")
        self.max_gain_fig.tight_layout()
        self.max_gain_fig.show()
        plt.show(block=False)

    def plot_total_efficiency_vs_freq(self):
        if self.freqs is None:
            messagebox.showinfo("Nothing loaded", "Load a run folder / .emff first.")
            return

        try:
            th_grid, ph_grid = np.meshgrid(self.theta, self.phi, indexing="ij")  # (Nt, Np)
            sin_th = np.sin(th_grid)
            eff_lin = np.zeros(len(self.freqs), dtype=float)

            for idx_f in range(len(self.freqs)):
                Ex = self.Ex[idx_f, :, :]
                Ey = self.Ey[idx_f, :, :]
                Ez = self.Ez[idx_f, :, :]

                Etheta, Ephi = cart_to_sph_E(Ex, Ey, Ez, th_grid, ph_grid)
                mag_abs = self._mag_over_eiso(Etheta, Ephi, "ABS")
                gain_abs = np.abs(mag_abs) ** 2

                # Include optional matching network by scaling gain with power ratio.
                f_hz = float(self.freqs[idx_f])
                p_ratio = self._matching_power_ratio(f_hz)
                gain_abs = gain_abs * p_ratio

                # eta_tot = (1 / 4pi) * integral(G(theta,phi) dOmega)
                integrand = gain_abs * sin_th
                int_theta = np.trapz(integrand, self.theta, axis=0)
                integral = float(np.trapz(int_theta, self.phi))
                eff_lin[idx_f] = max(integral / (4.0 * np.pi), 1e-300)

            eff_db = 10.0 * np.log10(eff_lin)

        except Exception as e:
            messagebox.showerror("Computation error", str(e))
            return

        if not self._figure_exists(self.eff_fig):
            self.eff_fig, self.eff_ax = plt.subplots()
            self.eff_ax.set_xlabel("Frequency (GHz)")
            self.eff_ax.set_ylabel("Total efficiency (dB)")
            self.eff_ax.grid(True)

        label = "ABS integrated"
        if self.match_path is not None and self.match_sparams is not None:
            label += " + matching"

        self.eff_ax.plot(self.freqs / 1e9, eff_db, marker="o", markersize=4, zorder=3, label=label)
        self.eff_ax.set_title("Total Antenna Efficiency vs Frequency")
        self.eff_ax.legend(loc="best")
        self.eff_fig.tight_layout()
        self.eff_fig.show()
        plt.show(block=False)


def main():
    root = tk.Tk()
    PatternViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()