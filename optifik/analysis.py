
import pandas as pd

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

from .io import load_spectrum
from .fft import *
from .scheludko import *
from .minmax import *






def plot_xy(file_path, plot=True):
    try:
        # Lecture du fichier .xy en utilisant pandas
        data = pd.read_csv(file_path, delimiter=',', header=None, names=["x", "y"])

        # Extraction des colonnes
        x = data["x"]
        y = data["y"]

        # Tracer la deuxième colonne en fonction de la première
        plt.figure(figsize=(10, 6),dpi = 600)
        plt.plot(x, y, 'o-', markersize=2, label="Raw data")

        # Ajout des labels et du titre
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$I^*$')
        plt.legend()

    except FileNotFoundError:
        print(f"Erreur : le fichier '{file_path}' est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")



def finds_peak(lambdas, intensities, min_peak_prominence, min_peak_distance=10, plot=None):
    """
    Charge un fichier .xy et affiche les données avec les extrema détectés (minima et maxima).

    Parameters
    ----------
    filename : str
        Chemin vers le fichier .xy (2 colonnes : lambda et intensité).
    min_peak_prominence : float
        Importance minimale des pics.
    min_peak_distance : float
        Distance minimale entre les pics.
    """
 
    smoothed_intensities =  intensities
 
    
 
    # Trouver les maxima et minima sur le signal lissé
    peaks_max, _ = find_peaks(smoothed_intensities, prominence=min_peak_prominence, distance=min_peak_distance)
    peaks_min, _ = find_peaks(-smoothed_intensities, prominence=min_peak_prominence, distance=min_peak_distance)
    
    if plot:
        plt.figure(figsize=(10, 6),dpi =600)
        plt.plot(lambdas, smoothed_intensities, 'o-', markersize=2, label="Smoothed data")
        plt.plot(lambdas[peaks_max], smoothed_intensities[peaks_max], 'ro')
        plt.plot(lambdas[peaks_min], smoothed_intensities[peaks_min], 'ro')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$I^*$')
        plt.legend()
        plt.tight_layout() 
        plt.show()

    # Nombre total d’extremums
    total_extrema = len(peaks_max) + len(peaks_min)
    if total_extrema >= 15:
        print('Number of extrema', total_extrema,'.')
        print('FFT method')

    if total_extrema <= 15 and total_extrema > 4:
        print('Number of extrema', total_extrema,'.')
        print('OOSpectro method')

    if total_extrema <= 4:
        print('Number of extrema', total_extrema,'.')
        print('Scheludko method')

    return total_extrema, peaks_min, peaks_max



def smooth_intensities(intensities):
    WIN_SIZE = 11
    smoothed_intensities = savgol_filter(intensities, WIN_SIZE, 3)
    return smoothed_intensities

