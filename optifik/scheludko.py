import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from .io import load_spectrum
from .utils import OptimizeResult, setup_matplotlib
from .analysis import finds_peak


def thickness_scheludko_at_order(wavelengths,
                                 intensities,
                                 interference_order,
                                 refractive_index,
                                 Imin=None):
    """
    Compute the film thickness for a given interference order.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    interference_order : int
        Interference order.
    refractive_index : array_like (or float)
        Refractive index.
    Imin : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    thickness : TYPE
        DESCRIPTION.

    """
    if Imin is None:
        Imin = np.min(intensities)

    n = refractive_index
    m = interference_order
    I = (np.asarray(intensities) - Imin) / (np.max(intensities) - Imin)


    prefactor = wavelengths / (2 * np.pi * n)
    argument = np.sqrt(I / (1 + (1 - I) * (n**2 - 1)**2 / (4 * n**2)))

    if m % 2 == 0:
        term1 = (m / 2) * np.pi
    else:
        term1 = ((m+1) / 2) * np.pi

    term2 = (-1)**m * np.arcsin(argument)

    return prefactor * (term1 + term2)


def Delta(wavelengths, thickness, interference_order, refractive_index):
    """
    Compute the Delta values.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    thickness : array_like (or float)
        Film thickness.
    interference_order : int
        Interference order.
    refractive_index : array_like (or float)
        Refractive index.

    Returns
    -------
    ndarray
        Delta values.

    """

    # ensure that the entries are numpy arrays
    wavelengths = np.asarray(wavelengths)
    h = np.asarray(thickness)
    n = np.asarray(refractive_index)
    m = interference_order

    # Calculation of p as a function of the parity of m
    if m % 2 == 0:
        p = m / 2
    else:
        p = (m + 1) / 2

    # Calculation of alpha
    alpha = ((n**2 - 1)**2) / (4 * n**2)

    # Argument of sinus
    angle = (2 * np.pi * n * h / wavelengths) - p * np.pi

    # A = sin²(argument)
    A = np.sin(angle)**2

    # Final calcuation of Delta
    return (A * (1 + alpha)) / (1 + A * alpha)


def Delta_fit(xdata, thickness, interference_order):
    """
    Wrapper on Delta() for curve_fit.

    Parameters
    ----------
    xdata : tuple
        (wavelengths, refractive_index)
    thickness : array_like (or float)
        Film thickness.
    interference_order : int
        Interference order.

    Returns
    -------
    ndarray
        Delta values.

    """
    lambdas, r_index = xdata
    return Delta(lambdas, thickness, interference_order, r_index)




def thickness_from_scheludko(wavelengths,
                             intensities,
                             refractive_index,
                             min_peak_prominence,
                             plot=None):
    """


    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar, optional
        Value of the refractive index of the medium.
    plot : bool, optional
        Display a curve, useful for checking or debuging. The default is None.

    Returns
    -------
    thickness : TYPE
        DESCRIPTION.

    """
    if plot:
        setup_matplotlib()

    max_tested_order = 12
    r_index = refractive_index

    peaks_min, peaks_max = finds_peak(wavelengths, intensities,
                                      min_peak_prominence=min_peak_prominence,
                                      plot=plot)

    failure, message = False, ''
    if len(peaks_min) == 0:
        message += 'Failed to detect at least one minimum. '
        failure = True
    if len(peaks_max) == 0:
        message += 'Failed to detect at least one maximum. '
        failure = True
    if failure:
        raise RuntimeError(message)

    # Get the last oscillation peaks
    lambda_min = wavelengths[peaks_min[-1]]
    lambda_max = wavelengths[peaks_max[-1]]

    # Order them
    lambda_start = min(lambda_min, lambda_max)
    lambda_stop = max(lambda_min, lambda_max)

    # mask input data
    mask = (wavelengths >= lambda_start) & (wavelengths <= lambda_stop)
    wavelengths_masked = wavelengths[mask]
    r_index_masked = r_index[mask]
    intensities_masked = intensities[mask]

    min_difference = np.inf
    best_m = None
    best_h_values = None

    if plot:
        plt.figure()
        plt.ylabel(r'$h$ ($\mathrm{{nm}}$)')
        plt.xlabel(r'$\lambda$ ($ \mathrm{nm} $)')

    for m in range(0, max_tested_order+1):
        h_values = thickness_scheludko_at_order(wavelengths_masked,
                                                intensities_masked,
                                                m, r_index_masked)

        difference = np.max(h_values) - np.min(h_values)

        print(f"h-difference for m={m}: {difference:.1f} nm")

        if difference < min_difference:
            min_difference = difference
            best_m = m
            best_h_values = h_values

        if plot:
            plt.plot(wavelengths_masked, h_values,'.', markersize=3, label=f"Épaisseur du film (Scheludko, m={m})")


    print(f"Optimized: m={best_m}")

    # Delta
    num = intensities_masked - np.min(intensities_masked)
    denom = np.max(intensities_masked) - np.min(intensities_masked)
    DeltaVrai = num / denom

    # DeltaVrai = (intensities_masked -np.min(intensities_masked))/(np.max(intensities_masked) -np.min(intensities_masked))
    # DeltaVrai = (intensities_raw_masked -np.min(intensities_raw_masked))/(np.max(intensities_raw_masked) -np.min(intensities_raw_masked))

    DeltaScheludko = Delta(wavelengths_masked,
                           np.mean(best_h_values),
                           best_m,
                           r_index_masked)


    xdata = (wavelengths_masked, r_index_masked)
    popt, pcov = curve_fit(lambda x, h: Delta_fit(x, h, m), xdata, DeltaVrai, p0=[np.mean(best_h_values)])
    fitted_h = popt[0]
    std_err = np.sqrt(pcov[0][0])

    if plot:
        Delta_values = Delta(wavelengths_masked, fitted_h, best_m, r_index_masked)

        plt.figure()
        plt.plot(wavelengths_masked, DeltaVrai,
                 'bo-', markersize=2, label=r'$\mathrm{{Smoothed}}\ \mathrm{{Data}}$')

        # Scheludko
        label = rf'$\mathrm{{Scheludko}}\ (h = {np.mean(best_h_values):.1f} \pm {np.std(best_h_values):.1f}\ \mathrm{{nm}})$'
        plt.plot(wavelengths_masked, DeltaScheludko,
                 'go-', markersize=2, label=label)
        # Fit
        label = rf'$\mathrm{{Fit}}\ (h = {fitted_h:.1f}\pm {std_err:.1f} \ \mathrm{{nm}})$'
        plt.plot(wavelengths_masked,  Delta_values, 'ro-', markersize=2, label=label)

        plt.legend()
        plt.ylabel(r'$\Delta$')
        plt.xlabel(r'$\lambda$ ($ \mathrm{nm} $)')
        import inspect
        plt.title(inspect.currentframe().f_code.co_name)

    return OptimizeResult(thickness=fitted_h, stderr=std_err)


def thickness_for_order0(wavelengths,
                         intensities,
                         refractive_index,
                         min_peak_prominence,
                         plot=None):
    if plot:
        setup_matplotlib()

    # TODO :
    # Load "trou"
    File_I_min = 'tests/spectre_trou/000043641.xy'
    wavelengths_I_min, intensities_I_min = load_spectrum(File_I_min, lambda_min=450)

    r_index = refractive_index

    peaks_min, peaks_max = finds_peak(wavelengths, intensities,
                                      min_peak_prominence=min_peak_prominence,
                                      plot=plot)


    if len(peaks_max) != 1:
        raise RuntimeError('Failed to detect a single maximum peak.')

    lambda_unique = wavelengths[peaks_max[0]]


    # On crée le masque pour ne garder que les wavelengths superieures a wavelengths unique
    mask = wavelengths >= lambda_unique
    wavelengths_masked = wavelengths[mask]
    r_index_masked = r_index[mask]
    intensities_masked = intensities[mask]
    intensities_I_min_masked =intensities_I_min[mask]

    # We assume to be at order zero.
    best_m = 0
    best_h_values = thickness_scheludko_at_order(wavelengths_masked,
                                                 intensities_masked,
                                                 best_m,
                                                 r_index_masked,
                                                 Imin=intensities_I_min_masked)

    if plot:
        plt.figure()
        plt.plot(wavelengths_masked, best_h_values, label=r"Épaisseur du film (Scheludko, m=0)")
        plt.ylabel(r'$h$ ($\mathrm{{nm}}$)')
        plt.xlabel(r'$\lambda$ (nm)')
        import inspect
        plt.title(inspect.currentframe().f_code.co_name)

    # Delta
    num = intensities_masked - np.min(intensities_I_min_masked)
    denom = np.max(intensities_masked) - np.min(intensities_I_min_masked)
    DeltaVrai = num / denom

    DeltaScheludko = Delta(wavelengths_masked, np.mean(best_h_values), best_m, r_index_masked)
    #print(np.mean(best_h_values),np.std(best_h_values))

    xdata = (wavelengths_masked, r_index_masked)
    popt, pcov = curve_fit(lambda x, h: Delta_fit(x, h, best_m), xdata, DeltaVrai, p0=[np.mean(best_h_values)])
    fitted_h = popt[0]
    std_err = np.sqrt(pcov[0][0])

    if plot:
        Delta_values = Delta(wavelengths_masked, fitted_h, best_m, r_index_masked)

        plt.figure()
        plt.plot(wavelengths_masked, DeltaVrai,
                 'bo-', markersize=2, label=r'$\mathrm{{Smoothed}}\ \mathrm{{Data}}$')

        # Scheludko
        label = rf'$\mathrm{{Scheludko}}\ (h = {np.mean(best_h_values):.1f} \pm {np.std(best_h_values):.1f}\ \mathrm{{nm}})$'
        plt.plot(wavelengths_masked, DeltaScheludko,
                 'go-', markersize=2, label=label)
        # Fit
        label = rf'$\mathrm{{Fit}}\ (h = {fitted_h:.1f}\pm {std_err:.1f} \ \mathrm{{nm}})$'
        plt.plot(wavelengths_masked,  Delta_values, 'ro-', markersize=2, label=label)

        plt.legend()
        plt.ylabel(r'$\Delta$')
        plt.xlabel(r'$\lambda$ ($ \mathrm{nm} $)')
        import inspect
        plt.title(inspect.currentframe().f_code.co_name)

    return OptimizeResult(thickness=fitted_h, stderr=std_err)
