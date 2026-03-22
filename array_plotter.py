# array_plot_patterns_4port.py
#
# GUI tool to combine 4 embedded-element farfield .emff files with per-element phase shifts
# and plot array pattern cuts in XY/XZ/YZ, with selectable plane(s) and plot style.
#
# Usage:
#   python array_plot_patterns_4port.py
#
# Expected folder contents:
#   ..._P1.emff, ..._P2.emff, ..._P3.emff, ..._P4.emff   (recommended naming)
# or at least 4 *.emff files (will take first 4 sorted)

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import emerge as em


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

class ArrayPattern4PortApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EMerge 4-Element Array Pattern Viewer (EMFF)")

        self.run_dir: Path | None = None
        self.emff_paths: list[Path] = []

        self.freqs = None            # (Nf,)
        self.theta = None            # radians (Nt,)
        self.phi = None              # radians (Np,)

        self.elements = []

        outer = tk.Frame(root, padx=10, pady=10)
        outer.pack(fill="both", expand=True)

        top = tk.Frame(outer)
        top.pack(fill="x", pady=(0, 8))

        tk.Button(top, text="LOAD FOLDER", width=18, command=self.load_folder).pack(side="left", padx=4)
        tk.Button(top, text="PLOT", width=18, command=self.plot_selected).pack(side="left", padx=4)

        self.status = tk.StringVar(value="Load a folder with 4 embedded-element *.emff files.")
        tk.Label(outer, textvariable=self.status, anchor="w").pack(fill="x", pady=(0, 8))

        mid = tk.Frame(outer)
        mid.pack(fill="both", expand=True)

        # LEFT: frequency list
        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="Frequencies (select one):").pack(anchor="w")
        self.freq_list = tk.Listbox(left, height=18, width=34)
        self.freq_list.pack(fill="both", expand=True)

        # RIGHT: options
        right = tk.Frame(mid)
        right.pack(side="left", fill="y")

        # Cut planes (multi-select)
        tk.Label(right, text="Cut plane(s):").pack(anchor="w")
        self.plane_list = tk.Listbox(right, height=3, exportselection=False, selectmode=tk.MULTIPLE, width=12)
        for p in ("XY", "XZ", "YZ"):
            self.plane_list.insert(tk.END, p)
        self.plane_list.selection_set(0)  # default XY
        self.plane_list.pack(anchor="w", pady=(0, 10))

        # Plot type (single-select)
        tk.Label(right, text="Plot type:").pack(anchor="w")
        self.plot_type_list = tk.Listbox(right, height=2, exportselection=False, selectmode=tk.SINGLE, width=12)
        for t in ("Polar", "Cartesian"):
            self.plot_type_list.insert(tk.END, t)
        self.plot_type_list.selection_set(0)  # default Polar
        self.plot_type_list.pack(anchor="w", pady=(0, 10))

        # Polarization
        tk.Label(right, text="Polarization:").pack(anchor="w")
        self.pol_var = tk.StringVar(value="ABS")
        tk.OptionMenu(right, self.pol_var, "ABS", "THETA", "PHI", "RHCP", "LHCP").pack(anchor="w", pady=(0, 8))

        # Limits
        tk.Label(right, text="dB floor:").pack(anchor="w")
        self.floor_var = tk.StringVar(value="-30")
        tk.Entry(right, textvariable=self.floor_var, width=8).pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="dB max:").pack(anchor="w")
        self.max_var = tk.StringVar(value="10")
        tk.Entry(right, textvariable=self.max_var, width=8).pack(anchor="w", pady=(0, 8))

        self.norm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Normalize (0 dB max)", variable=self.norm_var).pack(anchor="w", pady=(0, 10))

        # Phase entries
        tk.Label(right, text="Phase shifts (deg):").pack(anchor="w")
        self.phase_vars = [tk.StringVar(value="0") for _ in range(4)]
        for k in range(4):
            row = tk.Frame(right)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=f"P{k+1}:", width=4, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=self.phase_vars[k], width=8).pack(side="left")

        self.info = tk.StringVar(value="")
        tk.Label(right, textvariable=self.info, justify="left", anchor="w").pack(anchor="w", pady=(10, 0))

    def load_folder(self):
        d = filedialog.askdirectory(title="Select folder containing 4x *.emff (P1..P4)")
        if not d:
            return
        run_dir = Path(d)

        emffs = sorted(run_dir.glob("*.emff"))
        if len(emffs) < 4:
            messagebox.showerror("Not found", f"Need at least 4 *.emff files in:\n{run_dir}")
            return

        # Prefer *_P1..P4 naming if present
        def find_port(pn: int):
            hits = [p for p in emffs if f"_P{pn}" in p.stem or p.stem.endswith(f"P{pn}")]
            return hits[0] if hits else None

        ports = [find_port(i) for i in range(1, 5)]
        selected = ports if all(p is not None for p in ports) else emffs[:4]

        elements = []
        ref_freqs = ref_theta_deg = ref_phi_deg = None

        try:
            for p in selected:
                freqs, theta_deg, phi_deg, Ex, Ey, Ez = parse_emff(p)
                elements.append({"path": p, "freqs": freqs, "theta_deg": theta_deg, "phi_deg": phi_deg,
                                 "Ex": Ex, "Ey": Ey, "Ez": Ez})

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

        self.status.set(f"Loaded 4 EMFF files from: {run_dir.name}")
        self.info.set(
            f"Files:\n- " + "\n- ".join([p.name for p in self.emff_paths]) + "\n"
            f"Freq points: {len(self.freqs)}\n"
            f"Theta: {len(self.theta)} pts\n"
            f"Phi: {len(self.phi)} pts"
        )

    def _selected_freq_index(self):
        sel = self.freq_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def _selected_planes(self):
        idxs = self.plane_list.curselection()
        planes = [self.plane_list.get(i).upper() for i in idxs]
        return planes

    def _selected_plot_type(self):
        sel = self.plot_type_list.curselection()
        if not sel:
            return "POLAR"
        return self.plot_type_list.get(sel[0]).upper()

    def _plane_spec(self, plane: str):
        plane = plane.upper()
        if plane == "XY":
            return (0, 0, 1), (1, 0, 0)
        if plane == "XZ":
            return (0, 1, 0), (1, 0, 0)
        if plane == "YZ":
            return (1, 0, 0), (0, 1, 0)
        raise ValueError(f"Unknown plane '{plane}'")

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

    def _phase_weights(self):
        ph_deg = []
        for k in range(4):
            try:
                ph_deg.append(float(self.phase_vars[k].get().strip()))
            except Exception:
                raise ValueError(f"Invalid phase for P{k+1}: '{self.phase_vars[k].get()}'")
        ph_rad = np.deg2rad(np.array(ph_deg, dtype=float))
        return np.exp(1j * ph_rad)

    def _compute_plane_db(self, idx_f: int, plane: str, pol: str, floor_db: float, normalize: bool):
        w = self._phase_weights()

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

            Etheta_sum += w[k] * Etheta
            Ephi_sum += w[k] * Ephi

        mag = mag_over_eiso(Etheta_sum, Ephi_sum, pol)
        db = to_db_20(mag, floor_db=floor_db)
        if normalize:
            db = db - np.max(db)

        return ang_ref, db

    def _plot_polar(self, ang, db, plane, title, floor_db, db_max):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        ax.set_title(title)
        ax.plot(ang, db)
        ax.set_rlim(floor_db, db_max)
        plt.show()

    def _plot_cartesian(self, ang, db, plane, title, floor_db, db_max):
        # Map in-plane angle to degrees for a clean cut-style plot
        x_deg = np.rad2deg(ang)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.plot(x_deg, db)

        ax.set_xlabel(f"{plane} in-plane angle [deg]")
        ax.set_ylabel("Gain [dBi] (relative to EISO)")

        ax.set_ylim(floor_db, db_max)
        ax.grid(True)
        plt.show()

    def plot_selected(self):
        if self.freqs is None or not self.elements:
            messagebox.showinfo("Nothing loaded", "Load a folder with 4 .emff files first.")
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
            _ = self._phase_weights()
        except Exception as e:
            messagebox.showerror("Invalid phase", str(e))
            return

        fsel = float(self.freqs[idx])
        phases_txt = ", ".join([v.get().strip() for v in self.phase_vars])

        try:
            for plane in planes:
                ang, db = self._compute_plane_db(
                    idx_f=idx,
                    plane=plane,
                    pol=pol,
                    floor_db=floor_db,
                    normalize=self.norm_var.get()
                )

                title = (
                    f"ARRAY {plane} — {pol} @ {fsel/1e9:.6f} GHz\n"
                    f"Phases: {phases_txt} deg"
                )

                if plot_type == "POLAR":
                    self._plot_polar(ang, db, plane, title, floor_db, db_max)
                else:
                    self._plot_cartesian(ang, db, plane, title, floor_db, db_max)

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            return


def main():
    root = tk.Tk()
    ArrayPattern4PortApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()