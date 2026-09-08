# Exposure Time Calculator (ETC)

Computes signal-to-noise ratio (SNR) in one or more wavelength bins for an input spectrum and exposure time.

The application reuses the instrument models in `classi-sim` and the curves in `classi-shared-data`, so the ETC and detector simulator use the same telescope area, atmospheric extinction, detector geometry, grating/optics throughput representation, and photon-flux conversion.

## Installation and launch

Install the package and its dependencies, then launch the GUI with:

```bash
pip install .
classi-etc
```

The package is imported as `etc`.

## Spectrum input

Choose a two-column text spectrum containing:

1. wavelength in Angstrom
2. flux density in erg s^-1 cm^-2 Angstrom^-1

Input wavelengths, requested wave centers, and bin sizes are all interpreted in the observer frame. This calculator does not apply a redshift correction.

The packaged `SNIa_max_z0p05` spectrum from `shared-data` is offered as the initial reference location.

The optional **Flux scale magnitude** field rescales the entire input spectrum so that its synthetic g-, r-, or i-band AB magnitude matches the requested value. Leave the field blank to preserve the input spectrum's original flux normalization.

## Instrument options

- **Grating:** Newport 1229, Newport 1294, or the ThorLabs grating curve.
- **Airmass:** numerical airmass applied to the Palomar atmospheric-extinction curve from `spectrograph-sim`.
- **Camera model:** selects the detector QE, pixel size, detector dimensions, and read noise.

The throughput toggles expose the same component model used by the simulator: atmosphere, fiber, miscellaneous losses, collimator, grating, detector window, and detector QE.

Detector sampling parameters are not user inputs. Dispersion and fiber pitch are calculated from `SpectrographModel`, and read noise is taken from the selected camera configuration. Each extraction box is one full fiber pitch wide, extending halfway toward each adjacent trace.

## SNR inputs

- **Exposure Time** (s)
- **Wave Centers** (observer-frame nm), comma-separated
- **Bin Size** (nm)
- **Sky Background** (`dark`, `grey`, or `bright`; default `dark`)
- **Fiber Length** (m; default 10)
- **Fiber Coupling Efficiency** (fraction; default 1.0)

Fiber coupling represents point-source light lost before entering the fiber. It applies to source counts but not sky counts. The telescope diameter, pixel scale, lens throughput, detector read noise, dispersion, and spatial extraction width are not independent GUI inputs. Instrument geometry and throughput are taken from the shared simulator model instead.

The detector is assumed to operate at -20&deg;C. Each camera's fixed -20&deg;C dark-current value is used, and there is no temperature input.

## Sky and extraction model

The selectable dark, grey, and bright backgrounds use the corresponding line-resolved DESI benchmark sky spectra distributed with [desimodel](https://github.com/desihub/desimodel). They span 3500-10000 &#8491; in increments of 0.1 &#8491;, so narrow airglow features are integrated on the sky spectrum's own grid rather than on the potentially sparse source spectrum wavelength grid. The default is the "dark" spectrum, which equates to a sky brightness of approximately 20.6 msas in the $r$-band.

Sky background is integrated over the fiber's circular on-sky aperture. Source and sky counts are then multiplied by the same Gaussian-profile extraction fraction for the single fiber pitch extraction box. Fiber coupling is applied only to the source&mdash;sky flux enters the fiber no matter what. Dark-current and read-noise variance use the same extraction-box pixel count.

For each wavelength bin the ETC calculates

```text
source = integrated source electron rate * coupling * extraction fraction * time
sky    = integrated sky electron rate per arcsec^2 * fiber area * extraction fraction * time
SNR    = source / sqrt(source + sky + dark + read-noise variance)
```

## Python API

```python
from etc import ETCCalculator, get_default_spectrum_file

calc = ETCCalculator()
result = calc.get_SNR_from_spectrum(
    exp_time=1800,
    spectrum_file=get_default_spectrum_file(),
    wave_centers=[550, 650, 750],
    binsize=5,
    sky_background="grey",
    camera_model="QHY268",
    grating_id=1294,
    airmass=1.3,
    fiber_coupling_efficiency=0.75,
    target_magnitude=18.0,
    magnitude_band="g",
)
```
