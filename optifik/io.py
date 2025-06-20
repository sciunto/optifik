import numpy as np


def load_spectrum(spectrum_path,
                  lambda_min=0,
                  lambda_max=np.inf,
                  delimiter=','):
    """
    Load a spectrum file.


    TODO : describe expected format

    Parameters
    ----------
    spectrum_path : string
        File path.
    lambda_min : scalar, optional
        Cut the data at this minimum wavelength in nm.
    lambda_max : scalar, optional
        Cut the data at this maximum wavelength in nm.
    delimiter : string, optional
        Delimiter between columns in the datafile.

    Returns
    -------
    values : arrays
        (lamdbas, intensities)
    """
    data = np.loadtxt(spectrum_path, delimiter=delimiter)
    lambdas, intensities = np.column_stack(data)

    mask = (lambdas > lambda_min) & (lambdas < lambda_max)
    return lambdas[mask], intensities[mask]
