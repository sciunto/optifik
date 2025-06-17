from scipy.signal import savgol_filter
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({
    'axes.labelsize': 26,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 23,
})

from .utils import setup_matplotlib

def plot_spectrum(wavelengths, intensities, title=''):
    setup_matplotlib()
    plt.figure()
    plt.plot(wavelengths, intensities, 'o-', markersize=2)
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel(r'$I^*$')
    plt.title(title)
    plt.tight_layout()
    plt.show()



def finds_peak(wavelengths, intensities, min_peak_prominence, min_peak_distance=10, plot=None):
    """
    Detect minima and maxima.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    min_peak_prominence : float
        min prominence for scipy find_peak.
    min_peak_distance : int, optional
        min peak distance for scipy find_peak. The default is 10.
    plot : bool, optional
        Display a curve, useful for checking or debuging. The default is None.

    Returns
    -------
    (peaks_min, peaks_max)

    """
    if plot:
        setup_matplotlib()

    peaks_max, _ = find_peaks(intensities, prominence=min_peak_prominence, distance=min_peak_distance)
    peaks_min, _ = find_peaks(-intensities, prominence=min_peak_prominence, distance=min_peak_distance)

    if plot:
        plt.figure()
        plt.plot(wavelengths, intensities, 'o-', markersize=2, label="Smoothed data")
        plt.plot(wavelengths[peaks_max], intensities[peaks_max], 'ro')
        plt.plot(wavelengths[peaks_min], intensities[peaks_min], 'ro')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$I^*$')
        plt.legend()
        import inspect
        plt.title(inspect.currentframe().f_code.co_name)
        plt.tight_layout()
        plt.show()

    return peaks_min, peaks_max



def smooth_intensities(intensities, window_size=11):
    """
    Return a smoothed intensities array with a Savitzky-Golay filter.

    Parameters
    ----------
    intensities : ndarray
        Intensity values
    window_size : int, optional
        The length of the filter window. The default is 11.

    Returns
    -------
    smoothed_intensities

    """
    polynom_order = 3
    smoothed_intensities = savgol_filter(intensities, window_size, 3)
    return smoothed_intensities

