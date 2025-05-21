import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from optifik.minmax import thickness_from_minmax
from optifik.analysis import Data_Smoothed
from optifik.analysis import finds_peak
from optifik.io import load_spectrum

import yaml


def load():
    FOLDER = os.path.join('tests', 'spectraVictor1')

    yaml_file = os.path.join(FOLDER, 'known_value.yaml')
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    data = [(os.path.join(FOLDER, fn), val) for fn, val in thickness_dict.items()]
    return data


@pytest.mark.parametrize("spectrum, expected", load())
def test_minmax(spectrum, expected):
    raw_intensities = load_spectrum(spectrum)

    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectrum)

    indice =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.02

    total_extrema, smoothed_intensities, raw_intensities, lambdas, peaks_min, peaks_max = finds_peak(spectrum,
                                                                                                     min_peak_prominence=prominence)

    thickness_minmax = thickness_from_minmax(lambdas,
                                             smoothed_intensities,
                                             refractive_index=indice,
                                             min_peak_prominence=prominence)
    result = thickness_minmax.thickness

    assert_allclose(result, expected, rtol=1e-1)

