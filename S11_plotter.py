import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf


def load_s11_file():
    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Select S11 Touchstone file",
        filetypes=[
            ("Touchstone files", "*.s1p *.S1P *.s2p *.S2P *.snp *.SNP"),
            ("All files", "*.*"),
        ],
    )
    return filename


def main():
    filename = load_s11_file()

    if not filename:
        print("No file selected.")
        return

    try:
        ntwk = rf.Network(filename)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file:\n{e}")
        return

    freq_ghz = ntwk.f / 1e9
    s11_db = 20 * np.log10(np.abs(ntwk.s[:, 0, 0]))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axvspan(7.25, 7.75, color="lightgray", alpha=0.35)
    ax.axvspan(7.90, 8.40, color="lightgray", alpha=0.35)

    ax.plot(freq_ghz, s11_db, "k-", linewidth=1.5)

    ax.set_title("S11 Plot")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)

    y_top = ax.get_ylim()[1]
    ax.text(7.50, y_top - 1, "Downlink", ha="center", va="top", color="black")
    ax.text(8.15, y_top - 1, "Uplink", ha="center", va="top", color="black")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()