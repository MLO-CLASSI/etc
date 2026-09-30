from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import erf, sqrt
from pathlib import Path

import numpy as np
from astropy import constants as const, units as u
from astropy.table import Table
from shared_data import REFERENCE_SPECTRA
from simulator import (
    AtmosphericExtinction,
    InstrumentSimulator,
    SpectrographModel,
    f_lambda_to_photon_flux_density,
)
from simulator.components import (
    CANON_EF100_F2,
    CLASSI_FIBER,
    CLAUD_50INCH,
    E02_PICKOFF,
    FGL400S,
    FLI_AR571,
    FLI_KL400,
    NEWPORT_MASTER_1229,
    NEWPORT_MASTER_1294,
    QHY_268M,
    THORLABS_AC508_180_AB,
    THORLABS_GR50A_0305,
    DetectorModel,
    FiberModel,
    GratingModel,
    ThroughputCurve,
)
from simulator.core import FLUX_DENSITY_UNIT

DEFAULT_FIBER_LENGTH = CLASSI_FIBER.length
DEFAULT_AIRMASS = 1.3
DEFAULT_FIBER_COUPLING_EFFICIENCY = 1.0
FIBER_COUNT = 7
FIBER_PITCH = 250*u.micron
DETECTOR_TEMPERATURE_C = -20.0
DEFAULT_SKY_BACKGROUND = "dark"
SKY_SPECTRUM_RESOURCES = {
    "dark": "desi_sky_dark",
    "grey": "desi_sky_grey",
    "bright": "desi_sky_bright",
}

# Approximate LSST g/r/i photon-counting response curves. Wavelengths in Angstroms.
PHOTOMETRIC_BANDPASSES: dict[str, np.ndarray] = {
    "g": np.array(
        [
            [3850, 0.000],
            [3900, 0.065],
            [3950, 0.223],
            [4000, 0.408],
            [4050, 0.594],
            [4100, 0.780],
            [4150, 0.914],
            [4200, 0.928],
            [5350, 0.928],
            [5400, 0.878],
            [5450, 0.718],
            [5500, 0.534],
            [5550, 0.350],
            [5600, 0.166],
            [5650, 0.037],
            [5700, 0.000],
        ],
        dtype=float,
    ),
    "r": np.array(
        [
            [5350, 0.000],
            [5400, 0.057],
            [5450, 0.207],
            [5500, 0.396],
            [5550, 0.585],
            [5600, 0.773],
            [5650, 0.919],
            [5700, 0.943],
            [6750, 0.943],
            [6800, 0.882],
            [6850, 0.698],
            [6900, 0.509],
            [6950, 0.321],
            [7000, 0.132],
            [7050, 0.019],
            [7100, 0.000],
        ],
        dtype=float,
    ),
    "i": np.array(
        [
            [6750, 0.000],
            [6800, 0.075],
            [6850, 0.243],
            [6900, 0.429],
            [6950, 0.616],
            [7000, 0.802],
            [7050, 0.928],
            [7100, 0.933],
            [8000, 0.933],
            [8050, 0.910],
            [8100, 0.765],
            [8150, 0.579],
            [8200, 0.392],
            [8250, 0.205],
            [8300, 0.056],
            [8350, 0.000],
        ],
        dtype=float,
    ),
}

AB_ZERO_POINT_JY = 3631.0


@dataclass
class SNRBinResult:
    wave_center_nm: float
    source_counts: float
    sky_counts: float
    n_wave_pixels: float
    n_total_pixels: float
    read_noise_var: float
    dark_counts: float
    snr: float
    component_averages: dict[str, float]
    limiting_magnitude: float | None = None


class ETCCalculator:
    CAMERA_MODELS = {
        "Aurora": FLI_AR571,
        "Kepler": FLI_KL400,
        "QHY268": QHY_268M,
    }
    GRATING_MODELS = {
        1294: NEWPORT_MASTER_1294,
        1229: NEWPORT_MASTER_1229,
        "thorlabs": THORLABS_GR50A_0305,
    }

    THROUGHPUT_COMPONENTS = (
        "atmosphere",
        "pickoff",
        "order_filter",
        "fiber",
        "collimator",
        "grating",
        "camera_lens",
        "window",
        "detector",
    )

    def __init__(self, fiber_length_m: float = DEFAULT_FIBER_LENGTH.to_value(u.m)):
        self.fiber_length = float(fiber_length_m) * u.m
        if self.fiber_length < 0 * u.m:
            raise ValueError("Fiber length cannot be negative.")

    @property
    def available_gratings(self) -> list[int | str]:
        return list(self.GRATING_MODELS)

    @property
    def available_camera_models(self) -> list[str]:
        return list(self.CAMERA_MODELS)

    @property
    def available_magnitude_bands(self) -> list[str]:
        return list(PHOTOMETRIC_BANDPASSES)

    @property
    def available_sky_backgrounds(self) -> list[str]:
        return list(SKY_SPECTRUM_RESOURCES)

    @classmethod
    def detector_model(cls, camera_model: str) -> DetectorModel:
        try:
            return cls.CAMERA_MODELS[camera_model]
        except KeyError as exc:
            supported = ", ".join(cls.CAMERA_MODELS)
            raise ValueError(f"Unsupported camera model '{camera_model}'. Supported: {supported}") from exc

    @classmethod
    def grating_model(cls, grating_id: int | str) -> GratingModel:
        try:
            return cls.GRATING_MODELS[grating_id]
        except KeyError as exc:
            supported = ", ".join(map(str, cls.GRATING_MODELS))
            raise ValueError(
                f"Unsupported grating '{grating_id}'. Supported: {supported}"
            ) from exc

    def fiber_model(self, fiber_length_m: float | None = None) -> FiberModel:
        length = (
            self.fiber_length
            if fiber_length_m is None
            else float(fiber_length_m) * u.m
        )
        if length < 0 * u.m:
            raise ValueError("Fiber length cannot be negative.")
        return replace(CLASSI_FIBER, length=length)

    def spectrograph_model(
        self,
        camera_model: str,
        grating_id: int | str = 1294,
        fiber_length_m: float | None = None,
    ) -> SpectrographModel:
        return SpectrographModel.from_components(
            detector=self.detector_model(camera_model),
            grating=self.grating_model(grating_id),
            collimator=THORLABS_AC508_180_AB,
            camera_lens=CANON_EF100_F2,
            fiber=self.fiber_model(fiber_length_m),
            incidence_angle=32*u.deg,
            diffraction_angle=-20*u.deg,
            diffraction_order=1,
            fiber_count=FIBER_COUNT,
            fiber_pitch=FIBER_PITCH,
            optical_elements=(E02_PICKOFF, FGL400S),
        )

    def default_read_noise_for_camera(self, camera_model: str) -> float:
        return self.detector_model(camera_model).read_noise.to_value(u.electron)

    def dispersion_for_camera(
        self,
        camera_model: str,
        grating_id: int | str = 1294,
    ) -> float:
        return abs(
            self.spectrograph_model(camera_model, grating_id).dispersion.to_value(
                u.nm / u.pixel
            )
        )

    def spectral_pixel_count_for_bin(
        self,
        camera_model: str,
        wave_min_nm: float,
        wave_max_nm: float,
        grating_id: int | str = 1294,
    ) -> float:
        """Return the detector width of a wavelength bin in pixels."""
        if wave_max_nm <= wave_min_nm:
            raise ValueError("Wavelength-bin maximum must exceed its minimum.")
        x_edges = self.spectrograph_model(camera_model, grating_id).wavelength_to_x(
            np.array([wave_min_nm, wave_max_nm]) * u.nm
        )
        return float(abs(np.diff(x_edges.to_value(u.pixel))[0]))

    def extraction_aperture_for_camera(self, camera_model: str) -> float:
        """Return the full trace-to-trace spacing in detector pixels."""
        return self.spectrograph_model(camera_model).fiber_pitch_px.to_value(u.pixel)

    def extraction_fraction_for_camera(self, camera_model: str) -> float:
        """Fraction of a Gaussian fiber profile inside one fiber-pitch box."""
        spectrograph = self.spectrograph_model(camera_model)
        half_width = 0.5 * spectrograph.fiber_pitch_px.to_value(u.pixel)
        sigma = spectrograph.spatial_sigma_px.to_value(u.pixel)
        return float(erf(half_width / (sqrt(2) * sigma)))

    @property
    def fiber_sky_area(self) -> u.Quantity:
        angular_diameter = (
            CLASSI_FIBER.core_diameter / CLAUD_50INCH.focal_length
        ).decompose() * u.rad
        return (np.pi * (angular_diameter / 2) ** 2).to(u.arcsec**2)

    def instrument_simulator(
        self,
        camera_model: str,
        grating_id: int | str,
        airmass: float = DEFAULT_AIRMASS,
        fiber_length_m: float | None = None,
    ) -> InstrumentSimulator:
        if airmass <= 0:
            raise ValueError("Airmass must be positive.")
        return InstrumentSimulator(
            spectrograph=self.spectrograph_model(
                camera_model,
                grating_id,
                fiber_length_m,
            ),
            telescope=CLAUD_50INCH,
            atmosphere=AtmosphericExtinction(airmass=float(airmass)),
        )

    def throughput_curves(
        self,
        camera_model: str,
        grating_id: int | str,
        airmass: float = DEFAULT_AIRMASS,
        fiber_length_m: float | None = None,
    ) -> dict[str, ThroughputCurve]:
        simulator = self.instrument_simulator(
            camera_model,
            grating_id,
            airmass,
            fiber_length_m,
        )
        keys = [
            "atmosphere",
            "pickoff",
            "order_filter",
            "fiber",
            "collimator",
            "grating",
            "camera_lens",
        ]
        if simulator.detector.window_resource is not None:
            keys.append("window")
        if simulator.detector.qe_resource is not None:
            keys.append("detector")
        return dict(zip(keys, simulator.throughputs, strict=True))

    def get_throughput_components(
        self,
        wave_nm: np.ndarray,
        camera_model: str,
        grating_id: int | str,
        airmass: float = DEFAULT_AIRMASS,
        fiber_length_m: float | None = None,
        throughput_toggles: Mapping[str, bool] | None = None,
    ) -> dict[str, np.ndarray]:
        wavelength = np.asarray(wave_nm, dtype=float) * u.nm
        curves = self.throughput_curves(
            camera_model=camera_model,
            grating_id=grating_id,
            airmass=airmass,
            fiber_length_m=fiber_length_m,
        )
        toggles = {name: True for name in self.THROUGHPUT_COMPONENTS}
        if throughput_toggles:
            toggles.update(throughput_toggles)

        values = {
            name: (
                curves[name](wavelength)
                if name in curves
                else np.ones(wavelength.shape, dtype=float)
            )
            for name in self.THROUGHPUT_COMPONENTS
        }
        active_curves = [curve for name, curve in curves.items() if toggles.get(name, True)]
        simulator = InstrumentSimulator(
            spectrograph=self.spectrograph_model(
                camera_model,
                grating_id,
                fiber_length_m,
            ),
            telescope=CLAUD_50INCH,
            throughputs=active_curves,
        )
        values["total"] = simulator.combined_throughput(wavelength)
        return values

    @classmethod
    def get_dark_current(cls, camera_model: str) -> u.Quantity:
        return cls.detector_model(camera_model).dark_current

    @staticmethod
    def load_spectrum(spectrum_file: Path | str) -> np.ndarray:
        spec = np.genfromtxt(spectrum_file, dtype=[("wave", float), ("flux", float)])
        if spec.ndim == 0 or spec.size < 2:
            raise ValueError("Spectrum must contain at least two wavelength/flux rows.")
        spec = np.sort(spec, order="wave")
        spec["wave"] /= 10.0
        return spec

    @staticmethod
    def _samples_with_bin_boundaries(
        wavelength_nm: np.ndarray,
        flux_density: np.ndarray,
        wave_min_nm: float,
        wave_max_nm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select a bin and interpolate samples at both exact boundaries."""
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        flux_density = np.asarray(flux_density, dtype=float)
        if wave_min_nm < wavelength_nm[0] or wave_max_nm > wavelength_nm[-1]:
            raise ValueError("Spectrum does not cover the complete wavelength bin.")

        inside = (wavelength_nm > wave_min_nm) & (wavelength_nm < wave_max_nm)
        bin_wavelength = np.concatenate(
            ([wave_min_nm], wavelength_nm[inside], [wave_max_nm])
        )
        bin_flux = np.concatenate(
            (
                [np.interp(wave_min_nm, wavelength_nm, flux_density)],
                flux_density[inside],
                [np.interp(wave_max_nm, wavelength_nm, flux_density)],
            )
        )
        return bin_wavelength, bin_flux

    def get_band_flux_density_jy(self, spectrum: np.ndarray, magnitude_band: str) -> float:
        if magnitude_band not in PHOTOMETRIC_BANDPASSES:
            supported = ", ".join(self.available_magnitude_bands)
            raise ValueError(f"Unsupported magnitude band '{magnitude_band}'. Supported: {supported}")

        wave_angstrom = u.Quantity(spectrum["wave"], unit=u.nm).to_value(u.AA)
        flux_lambda = np.asarray(spectrum["flux"], dtype=float)
        order = np.argsort(wave_angstrom)
        wave_angstrom = wave_angstrom[order]
        flux_lambda = flux_lambda[order]

        bandpass = PHOTOMETRIC_BANDPASSES[magnitude_band]
        supported_wave = bandpass[bandpass[:, 1] > 0, 0]
        finite_wave = wave_angstrom[np.isfinite(wave_angstrom)]
        if (
            finite_wave.size < 2
            or finite_wave.min() > supported_wave.min()
            or finite_wave.max() < supported_wave.max()
        ):
            raise ValueError(f"Spectrum does not fully cover the LSST {magnitude_band} band.")

        response = np.interp(
            wave_angstrom,
            bandpass[:, 0],
            bandpass[:, 1],
            left=0.0,
            right=0.0,
        )
        valid = (
            np.isfinite(wave_angstrom)
            & np.isfinite(flux_lambda)
            & (wave_angstrom > 0)
            & (response > 0)
        )
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"Spectrum does not adequately sample the LSST {magnitude_band} band.")

        wave_valid = wave_angstrom[valid]
        response_valid = response[valid]
        flux_valid = flux_lambda[valid]
        denominator = np.trapezoid(response_valid / wave_valid, x=wave_valid)
        numerator = np.trapezoid(
            flux_valid * wave_valid * response_valid,
            x=wave_valid,
        )
        if denominator <= 0:
            raise ValueError(f"Could not integrate the LSST {magnitude_band} bandpass.")

        light_speed = const.c.to_value(u.AA / u.s)
        flux_nu_cgs = numerator / (light_speed * denominator)
        flux_nu_jy = float(flux_nu_cgs / 1e-23)
        if not np.isfinite(flux_nu_jy) or flux_nu_jy <= 0:
            raise ValueError(
                f"Spectrum has non-positive synthetic flux in the LSST {magnitude_band} band."
            )
        return flux_nu_jy

    def scale_spectrum_to_magnitude(
        self,
        spectrum: np.ndarray,
        target_magnitude: float,
        magnitude_band: str,
    ) -> tuple[np.ndarray, float]:
        current_flux_jy = self.get_band_flux_density_jy(spectrum, magnitude_band)
        target_flux_jy = AB_ZERO_POINT_JY * 10 ** (-0.4 * target_magnitude)
        scale_factor = float(target_flux_jy / current_flux_jy)
        scaled_spectrum = np.array(spectrum, copy=True)
        scaled_spectrum["flux"] *= scale_factor
        return scaled_spectrum, scale_factor

    def sky_spectrum(self, sky_background: str = DEFAULT_SKY_BACKGROUND) -> tuple[u.Quantity, u.Quantity]:
        """Return the selected line-resolved DESI sky spectrum.

        The flux-density values are numerically per square arcsecond. The solid
        angle is applied explicitly after integrating the spectral photon rate.
        """
        try:
            resource = SKY_SPECTRUM_RESOURCES[sky_background]
        except KeyError as exc:
            supported = ", ".join(self.available_sky_backgrounds)
            raise ValueError(
                f"Unsupported sky background '{sky_background}'. Supported: {supported}"
            ) from exc
        values = Table.read(REFERENCE_SPECTRA[resource])
        wavelength = values["wavelength"].quantity
        flux_density = values["flux"].quantity
        return wavelength, flux_density

    @staticmethod
    def _integrated_electron_rate(
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        throughput: np.ndarray,
    ) -> float:
        photon_flux_density = f_lambda_to_photon_flux_density(
            wavelength,
            flux_density,
            CLAUD_50INCH.collecting_area,
        )
        wavelength_angstrom = wavelength.to_value(u.AA)
        detected_rate_density = (
            photon_flux_density * np.asarray(throughput, dtype=float)
        ).to_value(1 / u.s / u.AA)
        return float(np.trapezoid(detected_rate_density, x=wavelength_angstrom))

    def get_SNR_from_spectrum(
        self,
        exp_time: float,
        spectrum_file: Path | str,
        wave_centers: Iterable[float],
        binsize: float,
        sky_background: str = DEFAULT_SKY_BACKGROUND,
        camera_model: str = "Aurora",
        grating_id: int | str = 1294,
        airmass: float = DEFAULT_AIRMASS,
        fiber_length_m: float | None = None,
        fiber_coupling_efficiency: float = DEFAULT_FIBER_COUPLING_EFFICIENCY,
        target_magnitude: float | None = None,
        magnitude_band: str | None = None,
        throughput_toggles: Mapping[str, bool] | None = None,
        dispersion: float | None = None,
        extraction_aperture: float | None = None,
        read_noise_e: float | None = None,
    ) -> dict[str, object]:
        wave_centers = np.asarray(list(wave_centers), dtype=float)
        if wave_centers.size == 0:
            raise ValueError("At least one wavelength-bin center is required.")
        if exp_time <= 0:
            raise ValueError("Exposure time must be positive.")
        if binsize <= 0:
            raise ValueError("Bin size must be positive.")
        if np.any((wave_centers - binsize / 2) < 400) or np.any(
            (wave_centers + binsize / 2) > 900
        ):
            raise ValueError(
                "One or more bins extend outside the ETC wavelength range (400-900 nm)."
            )
        if airmass <= 0:
            raise ValueError("Airmass must be positive.")
        if not 0 <= fiber_coupling_efficiency <= 1:
            raise ValueError("Fiber coupling efficiency must be between 0 and 1.")

        detector = self.detector_model(camera_model)
        spec = self.load_spectrum(spectrum_file)
        spectrum_scale_factor = 1.0
        if target_magnitude is not None:
            if not magnitude_band:
                raise ValueError("A magnitude band is required when scaling the spectrum.")
            spec, spectrum_scale_factor = self.scale_spectrum_to_magnitude(
                spec,
                target_magnitude=target_magnitude,
                magnitude_band=magnitude_band,
            )

        use_spectrograph_mapping = dispersion is None
        if use_spectrograph_mapping:
            dispersion = self.dispersion_for_camera(camera_model, grating_id)
        if extraction_aperture is None:
            extraction_aperture = self.extraction_aperture_for_camera(camera_model)
        if read_noise_e is None:
            read_noise_e = self.default_read_noise_for_camera(camera_model)
        if dispersion <= 0 or extraction_aperture <= 0 or read_noise_e < 0:
            raise ValueError("Detector extraction parameters must be non-negative and non-zero.")

        dark_current = self.get_dark_current(camera_model).to_value(u.electron/u.s)
        extraction_fraction = self.extraction_fraction_for_camera(camera_model)
        sky_wavelength, sky_surface_brightness = self.sky_spectrum(sky_background)
        sky_wave_nm = sky_wavelength.to_value(u.nm)
        sky_flux_density = sky_surface_brightness * self.fiber_sky_area
        sky_throughput_toggles = dict(throughput_toggles or {})
        # The sky spectrum is an at-observatory surface brightness, so applying
        # atmospheric extinction again would attenuate it twice.
        sky_throughput_toggles["atmosphere"] = False

        results: list[SNRBinResult] = []
        plot_wave = None
        plot_components = None

        for center in wave_centers:
            wave_min = center - binsize / 2
            wave_max = center + binsize / 2
            try:
                wave_bin_nm, source_flux_values = self._samples_with_bin_boundaries(
                    spec["wave"],
                    spec["flux"],
                    wave_min,
                    wave_max,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Spectrum does not fully cover the bin around {center} nm."
                ) from exc

            if use_spectrograph_mapping:
                n_wave = self.spectral_pixel_count_for_bin(
                    camera_model,
                    wave_min,
                    wave_max,
                    grating_id,
                )
            else:
                n_wave = binsize / dispersion
            n_total = n_wave * extraction_aperture
            read_noise_var = read_noise_e**2 * n_total
            dark_counts = dark_current * n_total * exp_time

            components = self.get_throughput_components(
                wave_bin_nm,
                camera_model=camera_model,
                grating_id=grating_id,
                airmass=airmass,
                fiber_length_m=fiber_length_m,
                throughput_toggles=throughput_toggles,
            )

            wavelength = wave_bin_nm * u.nm
            source_flux = source_flux_values * FLUX_DENSITY_UNIT
            source_rate = self._integrated_electron_rate(
                wavelength,
                source_flux,
                components["total"],
            )

            try:
                sky_wave_bin_nm, sky_flux_values = self._samples_with_bin_boundaries(
                    sky_wave_nm,
                    sky_flux_density.to_value(sky_flux_density.unit),
                    wave_min,
                    wave_max,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Sky spectrum does not fully cover the bin around {center} nm."
                ) from exc
            sky_wave_bin = sky_wave_bin_nm * u.nm
            sky_components = self.get_throughput_components(
                sky_wave_bin_nm,
                camera_model=camera_model,
                grating_id=grating_id,
                airmass=airmass,
                fiber_length_m=fiber_length_m,
                throughput_toggles=sky_throughput_toggles,
            )
            sky_rate = self._integrated_electron_rate(
                sky_wave_bin,
                sky_flux_values * sky_flux_density.unit,
                sky_components["total"],
            )

            source_counts = source_rate * exp_time * fiber_coupling_efficiency * extraction_fraction
            sky_counts = sky_rate * exp_time * extraction_fraction
            variance = source_counts + sky_counts + dark_counts + read_noise_var
            snr_bin = source_counts / np.sqrt(variance) if variance > 0 else 0.0

            results.append(
                SNRBinResult(
                    wave_center_nm=float(center),
                    source_counts=float(source_counts),
                    sky_counts=float(sky_counts),
                    n_wave_pixels=float(n_wave),
                    n_total_pixels=float(n_total),
                    read_noise_var=float(read_noise_var),
                    dark_counts=float(dark_counts),
                    snr=float(snr_bin),
                    component_averages={
                        name: float(np.mean(values))
                        for name, values in components.items()
                    },
                )
            )

            if plot_wave is None:
                plot_wave = wave_bin_nm
                plot_components = components

        return {
            "bins": results,
            "meta": {
                "exp_time": float(exp_time),
                "extraction_aperture_pix": float(extraction_aperture),
                "extraction_fraction": float(extraction_fraction),
                "dispersion_nm_per_pix": float(dispersion),
                "dark_current": float(dark_current),
                "detector_temperature_c": DETECTOR_TEMPERATURE_C,
                "fiber_sky_area_arcsec2": self.fiber_sky_area.to_value(u.arcsec**2),
                "fiber_coupling_efficiency": float(fiber_coupling_efficiency),
                "sky_background": sky_background,
                "sky_spectrum": SKY_SPECTRUM_RESOURCES[sky_background],
                "camera_model": camera_model,
                "read_noise_e": float(read_noise_e),
                "detector_binning": int(detector.binning),
                "grating": grating_id,
                "airmass": float(airmass),
                "spectrum_scale_factor": float(spectrum_scale_factor),
                "target_magnitude": target_magnitude,
                "magnitude_band": magnitude_band,
            },
            "throughput_plot": {
                "wave_nm": plot_wave,
                "components": plot_components,
            },
        }

    def get_limiting_magnitudes_from_spectrum(
        self,
        exp_time: float,
        spectrum_file: Path | str,
        wave_centers: Iterable[float],
        binsize: float,
        target_snr: float,
        magnitude_band: str,
        sky_background: str = DEFAULT_SKY_BACKGROUND,
        camera_model: str = "Aurora",
        grating_id: int | str = 1294,
        airmass: float = DEFAULT_AIRMASS,
        fiber_length_m: float | None = None,
        fiber_coupling_efficiency: float = DEFAULT_FIBER_COUPLING_EFFICIENCY,
        throughput_toggles: Mapping[str, bool] | None = None,
        dispersion: float | None = None,
        extraction_aperture: float | None = None,
        read_noise_e: float | None = None,
    ) -> dict[str, object]:
        """Return the AB magnitude that reaches ``target_snr`` in each bin."""
        if not np.isfinite(target_snr) or target_snr <= 0:
            raise ValueError("Target SNR must be positive.")

        spectrum = self.load_spectrum(spectrum_file)
        reference_flux_jy = self.get_band_flux_density_jy(
            spectrum,
            magnitude_band,
        )
        reference_magnitude = -2.5 * np.log10(
            reference_flux_jy / AB_ZERO_POINT_JY
        )
        result = self.get_SNR_from_spectrum(
            exp_time=exp_time,
            spectrum_file=spectrum_file,
            wave_centers=wave_centers,
            binsize=binsize,
            sky_background=sky_background,
            camera_model=camera_model,
            grating_id=grating_id,
            airmass=airmass,
            fiber_length_m=fiber_length_m,
            fiber_coupling_efficiency=fiber_coupling_efficiency,
            throughput_toggles=throughput_toggles,
            dispersion=dispersion,
            extraction_aperture=extraction_aperture,
            read_noise_e=read_noise_e,
        )

        limiting_bins = []
        for row in result["bins"]:
            background_variance = (
                row.sky_counts + row.dark_counts + row.read_noise_var
            )
            snr_squared = target_snr**2
            required_source_counts = 0.5 * (
                snr_squared
                + target_snr
                * np.sqrt(snr_squared + 4 * background_variance)
            )
            if row.source_counts > 0:
                source_scale = required_source_counts / row.source_counts
                limiting_magnitude = reference_magnitude - 2.5 * np.log10(
                    source_scale
                )
            else:
                limiting_magnitude = np.nan

            limiting_bins.append(
                replace(
                    row,
                    source_counts=float(required_source_counts),
                    snr=float(target_snr),
                    limiting_magnitude=float(limiting_magnitude),
                )
            )

        result["bins"] = limiting_bins
        result["meta"].update(
            {
                "target_snr": float(target_snr),
                "limiting_magnitude_band": magnitude_band,
                "reference_magnitude": float(reference_magnitude),
            }
        )
        return result


def get_default_spectrum_file():
    return REFERENCE_SPECTRA["SNIa_max_z0p05"]
