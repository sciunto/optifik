import os.path
import numpy as np
from numpy.testing import assert_allclose
import pytest

from optifik.analysis import thickness_from_fft
from optifik.analysis import thickness_from_minmax
from optifik.analysis import thickness_from_scheludko
from optifik.analysis import thickness_for_order0
from optifik.analysis import plot_xy
from optifik.analysis import Data_Smoothed
from optifik.analysis import Prominence_from_fft
from optifik.analysis import finds_peak
from optifik.io import load_spectrum


def test_FFT():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '003582.xy'
    expected = 3524.51

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    raw_intensities = load_spectrum(spectrum_path)
    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectrum_path)
    indice =  1.324188 + 3102.060378 / (lambdas**2)

    thickness_FFT = thickness_from_fft(lambdas,
                                       smoothed_intensities,
                                       refractive_index=indice)
    result = thickness_FFT.thickness

    assert_allclose(result, expected, rtol=1e-1)

def test_minmax_ransac():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '000004310.xy'
    expected = 1338.35

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    raw_intensities = load_spectrum(spectrum_path)
    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectrum_path)
    indice =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = Prominence_from_fft(lambdas=lambdas,
                                     intensities=smoothed_intensities,
                                     refractive_index=indice,
                                     plot=False)


    thickness_minmax = thickness_from_minmax(lambdas,
                                             smoothed_intensities,
                                             refractive_index=indice,
                                             min_peak_prominence=prominence,
                                             method='ransac',
                                             plot=False)
    result = thickness_minmax.thickness

    assert_allclose(result, expected, rtol=1e-1)


def test_scheludko_4peaks():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '000005253.xy'
    expected = 777.07

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    raw_intensities = load_spectrum(spectrum_path)
    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectrum_path)
    indice =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = Prominence_from_fft(lambdas=lambdas, intensities=smoothed_intensities, refractive_index=indice, plot=False)

    total_extrema, smoothed_intensities, raw_intensities, lambdas, peaks_min, peaks_max = finds_peak(spectrum_path,
                                                                                                     min_peak_prominence=prominence)


    result = thickness_from_scheludko(lambdas, smoothed_intensities,
                                             peaks_min, peaks_max,
                                             refractive_index=indice,
                                             plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)



def test_scheludko_2peaks():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '000006544.xy'
    expected = 495.69

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    raw_intensities = load_spectrum(spectrum_path)
    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectrum_path)
    indice =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = 0.03

    total_extrema, smoothed_intensities, raw_intensities, lambdas, peaks_min, peaks_max = finds_peak(spectrum_path,
                                                                                                     min_peak_prominence=prominence)


    result = thickness_from_scheludko(lambdas, smoothed_intensities,
                                             peaks_min, peaks_max,
                                             refractive_index=indice,
                                             plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)





def test_order0():
    FOLDER = os.path.join('tests', 'basic')
    FILE_NAME = '000018918.xy'
    expected = 115.33

    spectrum_path = os.path.join(FOLDER, FILE_NAME)
    raw_intensities = load_spectrum(spectrum_path)
    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectrum_path)
    indice =  1.324188 + 3102.060378 / (lambdas**2)

    prominence = 0.03

    total_extrema, smoothed_intensities, raw_intensities, lambdas, peaks_min, peaks_max = finds_peak(spectrum_path,
                                                                                                     min_peak_prominence=prominence)


    result = thickness_for_order0(lambdas, smoothed_intensities,
                                             peaks_min, peaks_max,
                                             refractive_index=indice,
                                             plot=False)

    assert_allclose(result.thickness, expected, rtol=1e-1)


