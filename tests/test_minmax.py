import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from optifik.minmax import thickness_from_minmax
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum


def test_minmax_ransac():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '000004310.xy'
    expected = 1338.35

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    lambdas, raw_intensities = load_spectrum(spectrum_path, lambda_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = 0.02

    result = thickness_from_minmax(lambdas,
                                   smoothed_intensities,
                                   refractive_index=r_index,
                                   min_peak_prominence=prominence,
                                   method='ransac',
                                   plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


def test_minmax_linreg():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '000004310.xy'
    expected = 1338.35

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    lambdas, raw_intensities = load_spectrum(spectrum_path, lambda_min=450)
    smoothed_intensities = smooth_intensities(raw_intensities)
    r_index =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = 0.02

    result = thickness_from_minmax(lambdas,
                                   smoothed_intensities,
                                   refractive_index=r_index,
                                   min_peak_prominence=prominence,
                                   method='linreg',
                                   plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


