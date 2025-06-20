import numpy as np


def load_spectrum(spectrum_path,
                  wavelength_min=0,
                  wavelength_max=np.inf,
                  delimiter=','):
    """
    Load a spectrum file.


    TODO : describe expected format

    Parameters
    ----------
    spectrum_path : string
        File path.
    wavelength_min : scalar, optional
        Cut the data at this minimum wavelength (included).
    wavelength_max : scalar, optional
        Cut the data at this maximum wavelength (included).
    delimiter : string, optional
        Delimiter between columns in the datafile.

    Returns
    -------
    values : arrays
        (lamdbas, intensities)
    """
    data = np.loadtxt(spectrum_path, delimiter=delimiter)
    lambdas, intensities = np.column_stack(data)

    mask = (lambdas >= wavelength_min) & (lambdas <= wavelength_max)
    return lambdas[mask], intensities[mask]
