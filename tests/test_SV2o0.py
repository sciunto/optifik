import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from optifik.scheludko import thickness_for_order0 
from optifik.io import load_spectrum
from optifik.analysis import smooth_intensities

import yaml


def load(filename):
    FOLDER = os.path.join('tests', 'spectraVictor2', 'order0')

    yaml_file = os.path.join(FOLDER, filename)
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    data = [(os.path.join(FOLDER, fn), val) for fn, val in thickness_dict.items()]
    return data


#@pytest.mark.skip('...')
@pytest.mark.parametrize("spectrum_path, expected", load('known_value.yaml'))
def test_SV2o0_small_tol(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, lambda_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)

    refractive_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.020


    result = thickness_for_order0(lambdas, smoothed_intensities,
                                               refractive_index=refractive_index,
                                               min_peak_prominence=prominence,
                                               plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)

@pytest.mark.parametrize("spectrum_path, expected", load('known_value_large_tol.yaml'))
def test_SV2o0_large_tol(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, lambda_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)

    refractive_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.020


    result = thickness_for_order0(lambdas, smoothed_intensities,
                                               refractive_index=refractive_index,
                                               min_peak_prominence=prominence,
                                               plot=False)

    assert_allclose(result.thickness, expected, rtol=2.5e-1)

