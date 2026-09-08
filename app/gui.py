from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import matplotlib as mpl
from matplotlib import pyplot as plt

from .core import (
    DEFAULT_AIRMASS,
    DEFAULT_FIBER_COUPLING_EFFICIENCY,
    DEFAULT_SKY_BACKGROUND,
    ETCCalculator,
    get_default_spectrum_file,
)

# Pick the first available preferred style (fallback to default)
for _style in ("seaborn-whitegrid", "seaborn", "ggplot", "classic", "default"):
    if _style in plt.style.available:
        plt.style.use(_style)
        break
else:
    plt.style.use("default")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.frameon": False,
})

GUI_BG = "#000020"
ACCENT_COLORS = ["#f7f702"]
PREFERRED_FONTS = [
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "Nimbus Sans",
    "fixed",
]


class PlaceholderEntry(ttk.Entry):
    def __init__(self, master, placeholder: str, **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.default_fg = "#000000"
        self.placeholder_fg = "#808080"
        self._show_placeholder()
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._restore_if_empty)

    def _show_placeholder(self):
        self.configure(foreground=self.placeholder_fg)
        self.delete(0, tk.END)
        self.insert(0, self.placeholder)

    def _clear_placeholder(self, _event=None):
        if self.get() == self.placeholder and self.cget("foreground") == self.placeholder_fg:
            self.delete(0, tk.END)
            self.configure(foreground=self.default_fg)

    def _restore_if_empty(self, _event=None):
        if not self.get().strip():
            self._show_placeholder()

    def value(self) -> str:
        value = self.get().strip()
        if value == self.placeholder and self.cget("foreground") == self.placeholder_fg:
            return ""
        return value


class SquareToggle(ttk.Frame):
    def __init__(self, master, text: str, var: tk.BooleanVar, color: str = "#f7f702"):
        super().__init__(master, style="TFrame")
        self.var = var
        self.color = color
        self.canvas = tk.Canvas(self, width=18, height=18, highlightthickness=0, bg=GUI_BG, bd=0)
        self.rect = self.canvas.create_rectangle(2, 2, 16, 16, fill=self.color if self.var.get() else "#ffffff", outline=self.color)
        self.check = self.canvas.create_line(5, 10, 8, 13, 14, 5, width=2, fill="black", capstyle=tk.ROUND, joinstyle=tk.ROUND)
        if not self.var.get():
            self.canvas.itemconfig(self.check, state="hidden")
        self.canvas.pack(side=tk.LEFT)
        self.label = ttk.Label(self, text=text)
        self.label.pack(side=tk.LEFT, padx=(6, 0))
        for widget in (self.canvas, self.label, self):
            widget.bind("<Button-1>", self._toggle)
            widget.bind("<Enter>", lambda _event: "break")
            widget.bind("<Leave>", lambda _event: "break")
        self.var.trace_add("write", lambda *_args: self._update())
        self._update()

    def _update(self):
        fill = self.color if self.var.get() else "#ffffff"
        self.canvas.itemconfig(self.rect, fill=fill)
        self.canvas.itemconfig(self.check, state="normal" if self.var.get() else "hidden")

    def _toggle(self, _event=None):
        self.var.set(not self.var.get())


class ETCGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLO CLASSI Spectrograph ETC")
        self.geometry("1040x800")

        style = ttk.Style(self)
        chosen_font = PREFERRED_FONTS[-1]
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            available_fonts = set(tkfont.families())
            for font in PREFERRED_FONTS:
                if font in available_fonts:
                    default_font.configure(family=font)
                    chosen_font = font
                    break
        except Exception:
            pass
        style.configure(".", font=(chosen_font, 11), foreground="white")
        style.configure("TFrame", background=GUI_BG)
        style.configure("TLabel", background=GUI_BG, foreground="white")
        style.configure("TLabelframe", background=GUI_BG)
        style.configure("TLabelframe.Label", background=GUI_BG, foreground="white")
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#000000")
        style.configure("TCombobox", fieldbackground="#ffffff", foreground="#000000")
        style.configure(
            "NoOutline.TCombobox",
            fieldbackground=GUI_BG,
            foreground="white",
            relief="flat",
            borderwidth=0,
        )
        self.configure(bg=GUI_BG)

        self._resource_stack = ExitStack()
        self._default_spectrum_path = self._resource_stack.enter_context(
            as_file(get_default_spectrum_file())
        )

        self.calc = ETCCalculator()
        self.spectrum_path = tk.StringVar(value="")
        self.camera_model = tk.StringVar(value=self.calc.available_camera_models[0])
        self.grating = tk.StringVar(value="1294")
        self.airmass = tk.StringVar(value=str(DEFAULT_AIRMASS))
        self.magnitude_band = tk.StringVar(value="g")
        self.limiting_magnitude_band = tk.StringVar(value="g")
        self.limiting_snr = tk.StringVar(value="5")
        self.sky_background = tk.StringVar(value=DEFAULT_SKY_BACKGROUND)

        self.toggle_vars = {
            name: tk.BooleanVar(value=True)
            for name in self.calc.THROUGHPUT_COMPONENTS
        }

        self._build_ui()
        self.after(100, self._prompt_for_spectrum)

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        path_frame = ttk.LabelFrame(root, text="Spectrum:")
        path_frame.pack(fill=tk.X, pady=6)
        picker_frame = ttk.Frame(path_frame)
        picker_frame.pack(fill=tk.X, pady=(6, 0))
        self.path_entry = tk.Entry(picker_frame, textvariable=self.spectrum_path, fg="#000000", bg="#ffffff", insertbackground="#000000", relief="solid", bd=1, width=60)
        self.path_entry.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.X, expand=True)
        tk.Button(picker_frame, text="Browse", command=lambda: self._browse(initial_dir=None), bg=GUI_BG, fg="white", activebackground=GUI_BG, activeforeground="white", bd=0).pack(side=tk.LEFT, padx=6)

        scale_frame = ttk.Frame(path_frame)
        scale_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(scale_frame, text="Flux scale magnitude (AB):").pack(side=tk.LEFT)
        self.magnitude_entry = PlaceholderEntry(scale_frame, "optional", width=12, foreground="black")
        self.magnitude_entry.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(scale_frame, text="Band:").pack(side=tk.LEFT)
        ttk.Combobox(
            scale_frame,
            textvariable=self.magnitude_band,
            values=self.calc.available_magnitude_bands,
            state="readonly",
            width=5,
        ).pack(side=tk.LEFT, padx=(6, 12))

        mode_frame = ttk.LabelFrame(root, text="Instrument options")
        mode_frame.pack(fill=tk.X, pady=6)

        ttk.Label(mode_frame, text="Grating:").grid(row=0, column=0, padx=6, pady=6, sticky=tk.W)
        ttk.Combobox(mode_frame, textvariable=self.grating, values=[str(grating) for grating in self.calc.available_gratings], state="readonly", width=12).grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(mode_frame, text="Airmass:").grid(row=0, column=2, padx=6, pady=6, sticky=tk.W)
        ttk.Entry(mode_frame, textvariable=self.airmass, width=10).grid(row=0, column=3, padx=6, pady=6)

        ttk.Label(mode_frame, text="Camera model:").grid(row=0, column=4, padx=6, pady=6, sticky=tk.W)
        ttk.Combobox(mode_frame, textvariable=self.camera_model, values=self.calc.available_camera_models, state="readonly", width=12,
        ).grid(row=0, column=5, padx=6, pady=6)

        toggles_frame = ttk.LabelFrame(root, text="Included Throughput Factors:")
        toggles_frame.pack(fill=tk.X, pady=6)
        for index, (name, var) in enumerate(self.toggle_vars.items()):
            row, column = divmod(index, 4)
            toggle = SquareToggle(toggles_frame, name.capitalize(), var, color=ACCENT_COLORS[0])
            toggle.grid(row=row, column=column, padx=8, pady=6, sticky=tk.W)

        fields_frame = ttk.LabelFrame(root, text="SNR inputs")
        fields_frame.pack(fill=tk.X, pady=6)

        self.entries = {}
        specs = [
            ("exp_time", "", "s", "Exposure Time:"),
            ("wave_centers_nm", "450,550,650,750", "nm", "Wave Centers (comma-separated):"),
            ("binsize_nm", "", "nm", "Bin Size:"),
            ("fiber_length_m", "10", "m", "Fiber Length:"),
            ("fiber_coupling_efficiency", f"{DEFAULT_FIBER_COUPLING_EFFICIENCY:g}", "(0-1)", "Fiber Coupling Efficiency:"),
        ]

        for index, (key, placeholder, unit, display) in enumerate(specs):
            row, column = divmod(index, 3)
            ttk.Label(fields_frame, text=display).grid(row=row * 2, column=column, padx=6, pady=(6, 0), sticky=tk.W)
            holder = ttk.Frame(fields_frame)
            holder.grid(row=row * 2 + 1, column=column, padx=6, pady=(0, 6), sticky=tk.W)
            entry = PlaceholderEntry(holder, placeholder, width=18, foreground="black")
            entry.pack(side=tk.LEFT)
            if unit:
                ttk.Label(holder, text=unit).pack(side=tk.LEFT, padx=(6, 0))
            self.entries[key] = entry

        ttk.Label(fields_frame, text="Sky Background:").grid(row=2, column=2, padx=6, pady=(6, 0), sticky=tk.W)
        ttk.Combobox(fields_frame, textvariable=self.sky_background, values=self.calc.available_sky_backgrounds, state="readonly", width=16).grid(row=3, column=2, padx=6, pady=(0, 6), sticky=tk.W)

        ttk.Label(fields_frame, text="Limiting-magnitude SNR:").grid(row=4, column=0, padx=6, pady=(6, 0), sticky=tk.W)
        ttk.Entry(fields_frame, textvariable=self.limiting_snr, width=18).grid(row=5, column=0, padx=6, pady=(0, 6), sticky=tk.W)

        ttk.Label(fields_frame, text="Limiting-magnitude band:").grid(row=4, column=1, padx=6, pady=(6, 0), sticky=tk.W)
        ttk.Combobox(
            fields_frame,
            textvariable=self.limiting_magnitude_band,
            values=self.calc.available_magnitude_bands,
            state="readonly",
            width=16,
        ).grid(row=5, column=1, padx=6, pady=(0, 6), sticky=tk.W)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Button(btn_frame, text="Compute SNR", command=self._compute, bg=GUI_BG, fg="white", activebackground=GUI_BG, activeforeground="white", bd=0).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Compute limiting magnitude(s)", command=self._compute_limiting_magnitudes, bg=GUI_BG, fg="white", activebackground=GUI_BG, activeforeground="white", bd=0).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Plot throughput", command=self._plot_throughput, bg=GUI_BG, fg="white", activebackground=GUI_BG, activeforeground="white", bd=0).pack(side=tk.LEFT, padx=6)

        out_frame = ttk.LabelFrame(root, text="Results")
        out_frame.pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(out_frame, wrap=tk.WORD, height=16, bg=GUI_BG, fg="white", insertbackground="white")
        self.output.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _browse(self, initial_dir: str | None = None):
        dialog = tk.Toplevel(self)
        dialog.title("Select spectrum file")
        dialog.configure(bg=GUI_BG)
        dialog.transient(self)
        dialog.grab_set()

        if initial_dir:
            current_dir = Path(initial_dir)
        elif self.spectrum_path.get():
            current_dir = Path(self.spectrum_path.get()).parent
        else:
            current_dir = Path.cwd()

        dir_frame = ttk.Frame(dialog)
        dir_frame.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(
            dir_frame,
            text="Directory:",
            fg="white",
            bg=GUI_BG,
        ).pack(side=tk.LEFT)
        dir_var = tk.StringVar(value=str(current_dir))
        dir_entry = tk.Entry(dir_frame, textvariable=dir_var, fg="#000000", bg="#ffffff")
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        type_frame = ttk.Frame(dialog)
        type_frame.pack(fill=tk.X, padx=8)
        tk.Label(type_frame, text="Files of type:", fg="white", bg=GUI_BG, relief="flat").pack(side=tk.LEFT)
        filetypes = ["All files (*.*)", "FITS (*.fits)", "Text (*.txt)"]
        type_var = tk.StringVar(value=filetypes[0])
        type_combo = ttk.Combobox(type_frame, textvariable=type_var, values=filetypes, state="readonly", width=20, style="NoOutline.TCombobox")
        type_combo.pack(side=tk.LEFT, padx=6)

        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE, bg="white", fg="black", highlightthickness=0)
        listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        def update_file_list(_event=None):
            listbox.delete(0, tk.END)
            directory = Path(dir_var.get())
            if not directory.exists():
                return
            pattern = "*"
            selected_type = type_var.get()
            if "FITS" in selected_type:
                pattern = "*.fits"
            elif "Text" in selected_type:
                pattern = "*.txt"
            for file_path in sorted(directory.glob(pattern)):
                if file_path.is_file():
                    listbox.insert(tk.END, file_path.name)

        dir_entry.bind("<Return>", update_file_list)
        type_combo.bind("<<ComboboxSelected>>", update_file_list)
        update_file_list()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        def do_open():
            selection = listbox.curselection()
            if not selection:
                return
            path = Path(dir_var.get()) / listbox.get(selection[0])
            self.spectrum_path.set(str(path))
            self.path_entry.config(fg="#000000")
            dialog.destroy()

        tk.Button(btn_frame, text="Open", command=do_open, bg=GUI_BG, fg="white", bd=0, activebackground=GUI_BG, activeforeground="white").pack(side=tk.RIGHT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg=GUI_BG, fg="white", bd=0, activebackground=GUI_BG, activeforeground="white").pack(side=tk.RIGHT)

        dialog.wait_window()

    def _prompt_for_spectrum(self):
        data_dir = self._default_spectrum_path.parent
        while not self.spectrum_path.get():
            self._browse(initial_dir=str(data_dir))
            if not self.spectrum_path.get():
                messagebox.showwarning("Reference spectrum required", "Please select a reference spectrum file to continue.")

    def destroy(self):
        self._resource_stack.close()
        super().destroy()

    def _read_inputs(self):
        try:
            wave_centers = [
                float(value.strip())
                for value in self.entries["wave_centers_nm"].value().split(",")
                if value.strip()
            ]
            magnitude_value = self.magnitude_entry.value()
            target_magnitude = (
                float(magnitude_value)
                if magnitude_value and magnitude_value != self.magnitude_entry.placeholder
                else None
            )
            grating = self.grating.get()
            return {
                "exp_time": float(self.entries["exp_time"].value()),
                "spectrum_file": self.spectrum_path.get(),
                "wave_centers": wave_centers,
                "binsize": float(self.entries["binsize_nm"].value()),
                "sky_background": self.sky_background.get(),
                "camera_model": self.camera_model.get(),
                "grating_id": int(grating) if grating.isdigit() else grating,
                "airmass": float(self.airmass.get()),
                "fiber_length_m": float(self.entries["fiber_length_m"].value()),
                "fiber_coupling_efficiency": float(self.entries["fiber_coupling_efficiency"].value()),
                "target_magnitude": target_magnitude,
                "magnitude_band": self.magnitude_band.get(),
                "throughput_toggles": {name: variable.get() for name, variable in self.toggle_vars.items()},
            }
        except ValueError as exc:
            raise ValueError(f"Invalid input value: {exc}") from exc

    def _compute(self):
        try:
            params = self._read_inputs()
            calc = ETCCalculator(fiber_length_m=params["fiber_length_m"])
            result = calc.get_SNR_from_spectrum(**params)
        except Exception as exc:
            messagebox.showerror("SNR error", str(exc))
            return

        meta = result["meta"]
        camera_model = meta["camera_model"]
        grating = meta["grating"]
        airmass = meta["airmass"]
        dispersion = meta["dispersion_nm_per_pix"]
        extraction_aperture = meta["extraction_aperture_pix"]
        extraction_fraction = meta["extraction_fraction"]
        read_noise = meta["read_noise_e"]
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"Camera model: {camera_model}\n")
        self.output.insert(tk.END, f"Grating: {grating}\n")
        self.output.insert(tk.END, f"Airmass: {airmass:.2f}\n")
        self.output.insert(tk.END, f"Dispersion: {dispersion:.4f} nm/pix\n")
        self.output.insert(tk.END, f"Extraction aperture: {extraction_aperture:.2f} pix ({extraction_fraction:.2%} of fiber profile)\n")
        self.output.insert(tk.END, f"Fiber coupling efficiency: {meta['fiber_coupling_efficiency']:.1f}\n")
        self.output.insert(tk.END, f"Sky background: {meta['sky_background']}\n")
        self.output.insert(tk.END, f"Detector temperature: {meta['detector_temperature_c']:.0f} C\n")
        self.output.insert(tk.END, f"Read noise: {read_noise:.2f} e-\n")
        if meta["target_magnitude"] is not None:
            target_magnitude = meta["target_magnitude"]
            magnitude_band = meta["magnitude_band"]
            spectrum_scale_factor = meta["spectrum_scale_factor"]
            self.output.insert(
                tk.END,
                (
                    f"Spectrum scaled to {target_magnitude:.3f} "
                    f"{magnitude_band}-band AB mag "
                    f"(scale={spectrum_scale_factor:.4g})\n"
                ),
            )
        self.output.insert(tk.END, "\n")

        for row in result["bins"]:
            component_text = ", ".join(
                f"{name}={value:.3f}"
                for name, value in row.component_averages.items()
            )
            self.output.insert(
                tk.END,
                (
                    f"{row.wave_center_nm:.1f} nm\n"
                    f"  Source counts: {row.source_counts:.3f}\n"
                    f"  Sky counts: {row.sky_counts:.3f}\n"
                    f"  SNR: {row.snr:.3f}\n"
                    f"  Mean throughput: {component_text}\n\n"
                ),
            )

    def _compute_limiting_magnitudes(self):
        try:
            params = self._read_inputs()
            params.pop("target_magnitude")
            params.pop("magnitude_band")
            calc = ETCCalculator(fiber_length_m=params["fiber_length_m"])
            result = calc.get_limiting_magnitudes_from_spectrum(
                **params,
                target_snr=float(self.limiting_snr.get()),
                magnitude_band=self.limiting_magnitude_band.get(),
            )
        except Exception as exc:
            messagebox.showerror("Limiting magnitude error", str(exc))
            return

        meta = result["meta"]
        target_snr = meta["target_snr"]
        magnitude_band = meta["limiting_magnitude_band"]
        extraction_aperture = meta["extraction_aperture_pix"]
        extraction_fraction = meta["extraction_fraction"]
        read_noise = meta["read_noise_e"]
        self.output.delete("1.0", tk.END)
        self.output.insert(
            tk.END,
            (
                f"Camera model: {meta['camera_model']}\n"
                f"Grating: {meta['grating']}\n"
                f"Airmass: {meta['airmass']:.2f}\n"
                f"Dispersion: {meta["dispersion_nm_per_pix"]:.4f} nm/pix\n"
                f"Extraction aperture: {extraction_aperture:.2f} pix ({extraction_fraction:.2%} of fiber profile)\n"
                f"Fiber coupling efficiency: {meta['fiber_coupling_efficiency']:.1f}\n"
                f"Sky background: {meta['sky_background']}\n"
                f"Detector temperature: {meta['detector_temperature_c']:.0f} C\n"
                f"Read noise: {read_noise:.2f} e-\n"
                f"Exposure: {meta['exp_time']:g} s\n\n"
                # f"In order to achieve S/N={target_snr:g} with a "
                # f"{meta['exp_time']:g}-sec exposure in the following "
                # f"wavelength\nbins, the {magnitude_band}-band limiting "
                # f"magnitude for a source with the given SED is:\n\n"
                f"Bin center (nm)   Bin width (nm)  Limiting {magnitude_band}"
                f"-band AB mag to achieve S/N={target_snr:g}\n"
            ),
        )
        for row in result["bins"]:
            self.output.insert(
                tk.END,
                (
                    f"{row.wave_center_nm:12.1f}      "
                    f"{params['binsize']:12.1f}           "
                    f"{row.limiting_magnitude:.2f}\n"
                ),
            )

    def _plot_throughput(self):
        try:
            params = self._read_inputs()
            calc = ETCCalculator(fiber_length_m=params["fiber_length_m"])
            spec = calc.load_spectrum(params["spectrum_file"])
            wave = spec["wave"]
            components = calc.get_throughput_components(
                wave,
                camera_model=params["camera_model"],
                grating_id=params["grating_id"],
                airmass=params["airmass"],
                fiber_length_m=params["fiber_length_m"],
                throughput_toggles=params["throughput_toggles"],
            )
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for index, name in enumerate(self.calc.THROUGHPUT_COMPONENTS):
            if params["throughput_toggles"].get(name, True):
                ax.plot(wave, components[name], label=name.capitalize(), color=cmap(index % 10), linestyle="--", alpha=0.9, linewidth=1.8)
        ax.plot(wave, components["total"], label="Total", linewidth=3.0, color="#0abd78")

        binsize = params["binsize"]
        for center in params["wave_centers"]:
            ax.axvline(center, color="#000000", linestyle="-", linewidth=1.2, alpha=0.9)
            ax.axvline(center - binsize / 2, color="#7f7f7f", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axvline(center + binsize / 2, color="#7f7f7f", linestyle="--", linewidth=0.9, alpha=0.7)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Throughput")
        ax.set_xlim(400, 900)
        ax.legend()
        fig.tight_layout()
        plt.show()

def run():
    app = ETCGui()
    app.mainloop()


if __name__ == "__main__":
    run()
