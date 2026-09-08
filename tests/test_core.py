from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from etc.core import AB_ZERO_POINT_JY, ETCCalculator
from simulator.core import FLUX_DENSITY_UNIT


def _write_flat_spectrum(path: Path, f_nu_jy: float = AB_ZERO_POINT_JY) -> Path:
    wavelength = np.linspace(3500.0, 9500.0, 3001) * u.AA
    flux_nu = f_nu_jy * u.Jy
    flux_lambda = flux_nu.to(
        FLUX_DENSITY_UNIT,
        equivalencies=u.spectral_density(wavelength),
    )
    np.savetxt(path, np.column_stack((wavelength.value, flux_lambda.value)))
    return path


def test_detector_sampling_is_derived_from_camera():
    calc = ETCCalculator()
    qhy_dispersion = calc.dispersion_for_camera("QHY268")
    kepler_dispersion = calc.dispersion_for_camera("Kepler")
    assert qhy_dispersion > 0
    assert kepler_dispersion > qhy_dispersion
    spectrograph = calc.spectrograph_model("QHY268")
    extraction_aperture = calc.extraction_aperture_for_camera("QHY268")
    assert np.isclose(
        extraction_aperture,
        spectrograph.fiber_pitch_px.to_value(u.pixel),
    )
    assert extraction_aperture > spectrograph.spatial_fwhm_px.to_value(u.pixel)
    assert 0 < calc.extraction_fraction_for_camera("QHY268") <= 1


def test_spectral_pixel_count_uses_spectrograph_mapping():
    calc = ETCCalculator()
    spectrograph = calc.spectrograph_model("QHY268")
    edges_nm = np.array([797.5, 802.5])
    expected = abs(
        np.diff(
            spectrograph.wavelength_to_x(edges_nm * u.nm).to_value(u.pixel)
        )[0]
    )
    measured = calc.spectral_pixel_count_for_bin(
        "QHY268",
        edges_nm[0],
        edges_nm[1],
    )
    linear_approximation = np.diff(edges_nm)[0] / calc.dispersion_for_camera("QHY268")
    assert np.isclose(measured, expected)
    assert not np.isclose(measured, linear_approximation, rtol=1e-3)


def test_invalid_camera_lists_supported_models():
    calc = ETCCalculator()
    with pytest.raises(ValueError, match="Unsupported camera model.*Kepler.*QHY268"):
        calc.detector_model("not-a-camera")


def test_load_spectrum_preserves_observed_wavelengths(tmp_path):
    spectrum_file = _write_flat_spectrum(tmp_path / "flat.txt")
    spectrum = ETCCalculator.load_spectrum(spectrum_file)
    assert np.isclose(spectrum["wave"][0], 350.0)
    assert np.isclose(spectrum["wave"][-1], 950.0)


def test_bin_samples_include_exact_interpolated_boundaries():
    wavelength = np.array([590.0, 598.0, 602.0, 610.0])
    flux = 2 * wavelength
    bin_wavelength, bin_flux = ETCCalculator._samples_with_bin_boundaries(
        wavelength,
        flux,
        596.0,
        606.0,
    )
    assert np.allclose(bin_wavelength, [596.0, 598.0, 602.0, 606.0])
    assert np.allclose(bin_flux, 2 * bin_wavelength)


def test_ab_magnitude_scaling(tmp_path):
    calc = ETCCalculator()
    spectrum_file = _write_flat_spectrum(tmp_path / "flat.txt")
    spectrum = calc.load_spectrum(spectrum_file)
    scaled, scale_factor = calc.scale_spectrum_to_magnitude(
        spectrum,
        target_magnitude=20.0,
        magnitude_band="r",
    )
    expected_jy = AB_ZERO_POINT_JY * 10 ** (-0.4 * 20.0)
    measured_jy = calc.get_band_flux_density_jy(scaled, "r")
    assert np.isclose(measured_jy, expected_jy, rtol=2e-3)
    assert scale_factor > 0


def test_total_throughput_uses_simulator_combination():
    calc = ETCCalculator()
    wavelength_nm = np.array([500.0, 600.0, 700.0])
    components = calc.get_throughput_components(
        wavelength_nm,
        camera_model="Kepler",
        grating_id=1294,
        airmass=1.3,
    )
    expected = np.ones_like(wavelength_nm)
    for name in calc.THROUGHPUT_COMPONENTS:
        expected *= components[name]
    assert np.allclose(components["total"], expected)


def test_line_resolved_sky_spectra_are_available():
    calc = ETCCalculator()
    assert calc.available_sky_backgrounds == ["dark", "grey", "bright"]
    for sky_background in calc.available_sky_backgrounds:
        wavelength, flux_density = calc.sky_spectrum(sky_background)
        assert wavelength.size == flux_density.size
        assert wavelength.size > 10_000
        assert wavelength.min() < 400 * u.nm
        assert wavelength.max() > 900 * u.nm
        assert np.nanmax(flux_density.value) > 10 * np.nanmedian(flux_density.value)


def test_snr_smoke(tmp_path):
    calc = ETCCalculator()
    spectrum_file = _write_flat_spectrum(tmp_path / "flat.txt", f_nu_jy=1e-4)
    result = calc.get_SNR_from_spectrum(
        exp_time=60.0,
        spectrum_file=spectrum_file,
        wave_centers=[600.0],
        binsize=5.0,
        camera_model="Kepler",
        grating_id=1294,
        airmass=1.3,
    )
    row = result["bins"][0]
    assert np.isfinite(row.snr)
    assert row.snr > 0
    assert result["meta"]["dispersion_nm_per_pix"] > 0
    assert result["meta"]["fiber_sky_area_arcsec2"] > 0
    assert result["meta"]["detector_temperature_c"] == -20.0
    assert np.isclose(
        row.n_wave_pixels,
        calc.spectral_pixel_count_for_bin("Kepler", 597.5, 602.5),
    )
    assert row.n_total_pixels > row.n_wave_pixels
    assert row.read_noise_var > 0
    assert row.dark_counts > 0


def test_limiting_magnitude_reaches_requested_snr(tmp_path):
    calc = ETCCalculator()
    spectrum_file = _write_flat_spectrum(tmp_path / "flat.txt", f_nu_jy=1e-4)
    limiting_result = calc.get_limiting_magnitudes_from_spectrum(
        exp_time=60.0,
        spectrum_file=spectrum_file,
        wave_centers=[600.0],
        binsize=5.0,
        target_snr=5.0,
        magnitude_band="r",
        camera_model="Kepler",
        grating_id=1294,
        airmass=1.3,
    )
    limiting_magnitude = limiting_result["bins"][0].limiting_magnitude
    forward_result = calc.get_SNR_from_spectrum(
        exp_time=60.0,
        spectrum_file=spectrum_file,
        wave_centers=[600.0],
        binsize=5.0,
        target_magnitude=limiting_magnitude,
        magnitude_band="r",
        camera_model="Kepler",
        grating_id=1294,
        airmass=1.3,
    )

    assert np.isclose(forward_result["bins"][0].snr, 5.0)
    assert limiting_result["meta"]["target_snr"] == 5.0
    assert limiting_result["meta"]["limiting_magnitude_band"] == "r"


def test_limiting_magnitude_does_not_depend_on_template_normalization(tmp_path):
    calc = ETCCalculator()
    bright_spectrum = _write_flat_spectrum(
        tmp_path / "bright.txt",
        f_nu_jy=1e-4,
    )
    faint_spectrum = _write_flat_spectrum(
        tmp_path / "faint.txt",
        f_nu_jy=1e-6,
    )
    common = {
        "exp_time": 60.0,
        "wave_centers": [600.0],
        "binsize": 5.0,
        "target_snr": 5.0,
        "magnitude_band": "r",
        "camera_model": "Kepler",
        "grating_id": 1294,
        "airmass": 1.3,
    }
    bright_limit = calc.get_limiting_magnitudes_from_spectrum(
        **common,
        spectrum_file=bright_spectrum,
    )
    faint_limit = calc.get_limiting_magnitudes_from_spectrum(
        **common,
        spectrum_file=faint_spectrum,
    )

    assert np.isclose(
        bright_limit["bins"][0].limiting_magnitude,
        faint_limit["bins"][0].limiting_magnitude,
    )


def test_limiting_magnitude_requires_positive_snr():
    calc = ETCCalculator()
    with pytest.raises(ValueError, match="Target SNR must be positive"):
        calc.get_limiting_magnitudes_from_spectrum(
            exp_time=60.0,
            spectrum_file="unused.txt",
            wave_centers=[600.0],
            binsize=5.0,
            target_snr=0.0,
            magnitude_band="r",
        )


def test_fiber_coupling_airmass_and_sky_background_affect_expected_terms(tmp_path):
    calc = ETCCalculator()
    spectrum_file = _write_flat_spectrum(tmp_path / "flat.txt", f_nu_jy=1e-4)
    common = {
        "exp_time": 60.0,
        "spectrum_file": spectrum_file,
        "wave_centers": [600.0],
        "binsize": 5.0,
        "camera_model": "Kepler",
        "grating_id": 1294,
        "airmass": 1.3,
    }
    full = calc.get_SNR_from_spectrum(**common, fiber_coupling_efficiency=1.0)
    half = calc.get_SNR_from_spectrum(**common, fiber_coupling_efficiency=0.5)
    high_airmass = calc.get_SNR_from_spectrum(
        **(common | {"airmass": 2.0}),
        fiber_coupling_efficiency=1.0,
    )
    grey = calc.get_SNR_from_spectrum(**common, sky_background="grey")
    bright = calc.get_SNR_from_spectrum(**common, sky_background="bright")
    full_bin = full["bins"][0]
    half_bin = half["bins"][0]
    high_airmass_bin = high_airmass["bins"][0]
    grey_bin = grey["bins"][0]
    bright_bin = bright["bins"][0]
    assert np.isclose(half_bin.source_counts, 0.5 * full_bin.source_counts)
    assert np.isclose(half_bin.sky_counts, full_bin.sky_counts)
    assert high_airmass_bin.source_counts < full_bin.source_counts
    assert np.isclose(high_airmass_bin.sky_counts, full_bin.sky_counts)
    assert np.isclose(grey_bin.source_counts, full_bin.source_counts)
    assert np.isclose(bright_bin.source_counts, full_bin.source_counts)
    assert full_bin.sky_counts < grey_bin.sky_counts < bright_bin.sky_counts
