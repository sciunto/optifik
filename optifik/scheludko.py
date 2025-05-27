import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({
    'axes.labelsize': 26,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 23,
})

from .io import load_spectrum
from .utils import OptimizeResult
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
    interference_order: TYPE
        DESCRIPTION.
    refractive_index : TYPE
        DESCRIPTION.
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



    """
    Calculates the Delta value for arrays of wavelengths, thicknesses h and r_indexs n.

    Parameters:
    - wavelengths: array_like (or float), wavelengths λ
    - thickness : array_like (or float), thicknesses h
    - interference_order : int, interference order
    - refractive_index : array_like (or float), refractive r_indexs n

    Returns:
    - delta: ndarray of corresponding Δ values
    """


def Delta(wavelengths, thickness, interference_order, refractive_index):
    """


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
    TYPE
        DESCRIPTION.

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
        (wavelengths, n)
    thickness : array_like (or float)
        Film thickness.
    interference_order : int
        Interference order.

    Returns
    -------
    ndarray
        Delta values.

    """
    lambdas, n = xdata
    return Delta(lambdas, thickness, interference_order, n)




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
    max_tested_order = 12
    r_index = refractive_index

    peaks_min, peaks_max = finds_peak(wavelengths, intensities,
                                      min_peak_prominence=min_peak_prominence,
                                      plot=False)

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


    min_ecart = np.inf
    best_m = None
    meilleure_h = None

    if plot:
        plt.figure(figsize=(10, 6),dpi =600)
        plt.ylabel(r'$h$ ($\mathrm{{nm}}$)')
        plt.xlabel(r'$\lambda$ ($ \mathrm{nm} $)')


    for m in range(0, max_tested_order+1):
        h_values = thickness_scheludko_at_order(wavelengths_masked, intensities_masked, m, r_index_masked)

        if plot:
            plt.plot(wavelengths_masked, h_values,'.', markersize=3, label=f"Épaisseur du film (Scheludko, m={m})")
        ecart = np.max(h_values)-np.min(h_values)

        print(f"Écart pour m={m} : {ecart:.3f} nm")

        if ecart < min_ecart:
            min_ecart = ecart
            best_m = m
            meilleure_h = h_values


    DeltaVrai = (intensities_masked -np.min(intensities_masked))/(np.max(intensities_masked) -np.min(intensities_masked))
    #DeltaVrai = (intensities_raw_masked -np.min(intensities_raw_masked))/(np.max(intensities_raw_masked) -np.min(intensities_raw_masked))

    DeltaScheludko = Delta(wavelengths_masked, np.mean(meilleure_h), best_m, r_index_masked)
    #print(np.mean(meilleure_h),np.std(meilleure_h))

    if plot:
        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(wavelengths_masked, DeltaVrai,
                 'bo-', markersize=2, label=r'$\mathrm{{Smoothed}}\ \mathrm{{Data}}$')
        plt.plot(wavelengths_masked, DeltaScheludko,
                 'go-', markersize=2, label = rf'$\mathrm{{Scheludko}}\ (h = {np.mean(meilleure_h):.1f} \pm {np.std(meilleure_h):.1f}\ \mathrm{{nm}})$')


    xdata = (wavelengths_masked, r_index_masked)
    popt, pcov = curve_fit(lambda x, h: Delta_fit(x, h, m), xdata, DeltaVrai, p0=[np.mean(meilleure_h)])
    fitted_h = popt[0]


    if plot:
        plt.plot(wavelengths_masked, Delta(wavelengths_masked, fitted_h, best_m, r_index_masked ), 'ro-',markersize=2, label=rf'$\mathrm{{Fit}}\ (h = {fitted_h:.1f}\pm {np.sqrt(pcov[0][0]):.1f} \ \mathrm{{nm}})$')
        plt.legend()
        plt.ylabel(r'$\Delta$')
        plt.xlabel(r'$\lambda$ ($ \mathrm{nm} $)')


    return OptimizeResult(thickness=fitted_h ,)


def thickness_for_order0(wavelengths,
                         intensities,
                         refractive_index,
                         min_peak_prominence,
                         plot=None):


    File_I_min = 'tests/spectre_trou/000043641.xy'
    r_index = refractive_index

    peaks_min, peaks_max = finds_peak(wavelengths, intensities,
                                                     min_peak_prominence=min_peak_prominence,
                                                     plot=False)




    wavelengths_I_min, intensities_I_min = load_spectrum(File_I_min, lambda_min=450)

    lambda_unique = wavelengths[peaks_max[0]]


    # On crée le masque pour ne garder que les wavelengths superieures a wavelengths unique
    mask = wavelengths >= lambda_unique
    wavelengths_masked = wavelengths[mask]
    r_index_masked = r_index[mask]
    intensities_masked = intensities[mask]
    intensities_I_min_masked =intensities_I_min[mask]

    min_ecart = np.inf
    best_m = None
    meilleure_h = None


    m = 0
    h_values = thickness_scheludko_at_order(wavelengths_masked,
                                           intensities_masked,
                                           0,
                                           r_index_masked,
                                           Imin=intensities_I_min_masked)

    if plot:
        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(wavelengths_masked, h_values, label=r"Épaisseur du film (Scheludko, m=0)")

    ecart = np.max(h_values) - np.min(h_values)
    best_m = m
    meilleure_h = h_values



    DeltaVrai = (intensities_masked -np.min(intensities_I_min_masked))/(np.max(intensities_masked) -np.min(intensities_I_min_masked))

    #DeltaVrai = (intensities_masked -np.min(intensities_masked))/(np.max(intensities_masked) -np.min(intensities_masked))

    DeltaScheludko = Delta(wavelengths_masked, np.mean(meilleure_h), best_m, r_index_masked)
    #print(np.mean(meilleure_h),np.std(meilleure_h))


    if plot:
        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(wavelengths_masked,DeltaVrai,'bo-', markersize=2,label=r'$\mathrm{{Raw}}\ \mathrm{{Data}}$')
        plt.plot(wavelengths_masked,DeltaScheludko,'ro-', markersize=2,label = rf'$\mathrm{{Scheludko}}\ (h = {np.mean(meilleure_h):.1f} \pm {np.std(meilleure_h):.1f}\ \mathrm{{nm}})$')


    xdata = (wavelengths_masked, r_index_masked)
    popt, pcov = curve_fit(lambda x, h: Delta_fit(x, h, m), xdata, DeltaVrai, p0=[np.mean(meilleure_h)])
    fitted_h = popt[0]

    if plot:
        plt.plot(wavelengths_masked, Delta(wavelengths_masked, fitted_h, best_m, r_index_masked ), 'go-',markersize=2, label=rf'$\mathrm{{Fit}}\ (h = {fitted_h:.1f}\pm {np.sqrt(pcov[0][0]):.1f} \ \mathrm{{nm}})$')
        plt.legend()
        plt.ylabel(r'$\Delta$')
        plt.xlabel(r'$\lambda$ (nm)')

    return OptimizeResult(thickness=fitted_h ,)
