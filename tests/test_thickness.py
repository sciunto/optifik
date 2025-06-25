import pytest
import yaml
from pathlib import Path
import os.path

import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from optifik.minmax import thickness_from_minmax
from optifik.analysis import smooth_intensities
from optifik.io import load_spectrum


def load():
    test_data_dir = Path(__file__).parent.parent / 'data'
    FOLDER = test_data_dir / 'spectraLorene' / 'sample1'

    yaml_file = os.path.join(FOLDER, 'sample1.yaml')
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    #print(thickness_dict)
    data = [(os.path.join(FOLDER, fn), val) for fn, val in thickness_dict['known_thicknesses'].items()]
    return data


def test_minmax_sample1():
    min_peak_prominence = 0.02
    min_peak_distance = 10
    skipped = ('011137.xy',
               '012426.xy',
               '012795.xy',
               '012979.xy',
               '011321.xy', #Insufficient number of data points
               )

    for path, expected in load():
        file = os.path.split(path)[-1]
        if file not in skipped:
            lambdas, raw_intensities = load_spectrum(path, wavelength_min=450)
            smoothed_intensities = smooth_intensities(raw_intensities)
            r_index = 1.33 #TODO

            prominence = 0.02
            distance = 10

            result = thickness_from_minmax(lambdas,
                                           smoothed_intensities,
                                           refractive_index=r_index,
                                           min_peak_prominence=prominence,
                                           min_peak_distance=distance,
                                           method='ransac',
                                           plot=False)

            assert_allclose(result.thickness, expected, rtol=2.3e-1)

#def test_fft_sample1():
#    for name, value in known_values.sample1['known_thicknesses'].items():
#        index = known_values.sample1['refractive_index']
#        # this method works only for large thicknesses
#        if value < 3000:
#            continue
#        path = os.path.join('spectra', name)
#        lambdas, intensities = io.load_spectrum(path, lambda_min=450)
#        result = thickness_from_fft(lambdas, intensities, refractive_index=index,)
#        error = 100 * np.abs(result.thickness - value) / value
#        assert(error < 5)
