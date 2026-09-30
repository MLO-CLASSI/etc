from etc.core import ETCCalculator
from etc.gui import ETCGui


class _Entry:
    def __init__(self, value: str, placeholder: str = ""):
        self._value = value
        self.placeholder = placeholder

    def value(self) -> str:
        return self._value


class _Variable:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Selector:
    def configure(self, **options):
        self.options = options


def test_optional_flux_scale_placeholder_is_not_parsed_as_float():
    gui = object.__new__(ETCGui)
    gui.entries = {
        "exp_time": _Entry("60"),
        "wave_centers_nm": _Entry("600, 700"),
        "binsize_nm": _Entry("5"),
        "fiber_length_m": _Entry("10"),
        "fiber_coupling_efficiency": _Entry("0.8"),
    }
    gui.magnitude_entry = _Entry("optional", placeholder="optional")
    gui.spectrum_path = _Variable("spectrum.txt")
    gui.camera_model = _Variable("QHY268")
    gui.binning = _Variable("2")
    gui.grating = _Variable("1229")
    gui.airmass = _Variable("1.3")
    gui.magnitude_band = _Variable("g")
    gui.sky_background = _Variable("grey")
    gui.toggle_vars = {}

    params = ETCGui._read_inputs(gui)

    assert params["target_magnitude"] is None
    assert params["fiber_coupling_efficiency"] == 0.8
    assert params["sky_background"] == "grey"
    assert params["binning"] == 2


def test_camera_selection_sets_valid_binning_choices_and_default():
    gui = object.__new__(ETCGui)
    gui.calc = ETCCalculator()
    gui.camera_model = _Variable("QHY268")
    gui.binning = _Variable("4")
    gui.binning_selector = _Selector()

    ETCGui._set_camera_binning_default(gui)

    assert gui.binning.get() == "1"
    assert gui.binning_selector.options["values"] == ["1", "2"]
