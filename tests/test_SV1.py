import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from optifik.minmax import thickness_from_minmax
from optifik.io import load_spectrum
from optifik.analysis import smooth_intensities
from optifik.analysis import finds_peak

import yaml


def load():
    FOLDER = os.path.join('tests', 'spectraVictor1')

    yaml_file = os.path.join(FOLDER, 'known_value.yaml')
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    data = [(os.path.join(FOLDER, fn), val) for fn, val in thickness_dict.items()]
    return data


@pytest.mark.parametrize("spectrum_path, expected", load())
def test_minmax(spectrum_path, expected):
    lambdas, raw_intensities = load_spectrum(spectrum_path, lambda_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)

    indice =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.02

    total_extrema, peaks_min, peaks_max = finds_peak(lambdas, smoothed_intensities,
                                                     min_peak_prominence=prominence,
                                                     plot=False)

    thickness_minmax = thickness_from_minmax(lambdas,
                                             smoothed_intensities,
                                             refractive_index=indice,
                                             min_peak_prominence=prominence)
    result = thickness_minmax.thickness

    assert_allclose(result, expected, rtol=1e-1)

