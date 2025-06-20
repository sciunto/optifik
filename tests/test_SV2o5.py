import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from optifik.scheludko import thickness_from_scheludko
from optifik.scheludko import get_default_start_stop_wavelengths
from optifik.io import load_spectrum
from optifik.analysis import smooth_intensities

import yaml


def load():
    FOLDER = os.path.join('tests', 'spectraVictor2', 'order5')

    yaml_file = os.path.join(FOLDER, 'known_value.yaml')
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    data = [(os.path.join(FOLDER, fn), val) for fn, val in thickness_dict.items()]
    return data


@pytest.mark.parametrize("spectrum_path, expected", load())
def test_SV2o5(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, lambda_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)

    r_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.02

    w_start, w_stop = get_default_start_stop_wavelengths(lambdas,
                                                         smoothed_intensities,
                                                         refractive_index=r_index,
                                                         min_peak_prominence=prominence,
                                                         plot=False)


    thickness_scheludko = thickness_from_scheludko(lambdas,
                                                   smoothed_intensities,
                                                   refractive_index=r_index,
                                                   wavelength_start=w_start,
                                                   wavelength_stop=w_stop,
                                                   interference_order=None,
                                                   plot=False)

    result = thickness_scheludko.thickness

    assert_allclose(result, expected, rtol=1e-1)

