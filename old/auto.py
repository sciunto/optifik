import os.path

from .analysis import *
from .io import load_spectrum

def auto(spectrum_file, plot=None):


    spectre_file = spectrum_file

    ##### Affichage du spectre brut et récupération des Intesités brutes#####

    lambdas, raw_intensities = load_spectrum(spectre_file, lambda_min=450)

    ##### Affichage du spectre lissé #####

    #smoothed_intensities, intensities, lambdas = Data_Smoothed(spectre_file)

    smoothed_intensities = smooth_intensities(raw_intensities)

    ##### Indice Optique en fonction de Lambda #####

    indice =  1.324188 + 3102.060378 / (lambdas**2)

    ##### Determination de la prominence associé #####

    prominence, signal, wavelength = Prominence_from_fft(lambdas,
                                                        smoothed_intensities,
                                                        refractive_index=indice,
                                                        plot=plot)

    prominence = 0.03
    ##### Find Peak #####

    peaks_min, peaks_max = finds_peak(lambdas, smoothed_intensities,
                                                     min_peak_prominence=prominence,
                                                     plot=False)

    ##### Epaisseur selon la methode #####

    #thickness_FFT = thickness_from_fft(lambdas,smoothed_intensities,refractive_index=1.33)


    total_extrema = len(peaks_max) + len(peaks_min)

    if total_extrema > 15 and total_extrema > 4:
        print('Apply method FFT')
        result = thickness_from_fft(lambdas, smoothed_intensities,
                                           refractive_index=indice,
                                           plot=plot)

        print(f'thickness: {result.thickness:.2f} nm')


    if total_extrema <= 15 and total_extrema > 4:
        print('Apply method minmax')
        result = thickness_from_minmax(lambdas, smoothed_intensities,
                                                 refractive_index=indice,
                                                 min_peak_prominence=prominence,
                                                 plot=plot)

        print(f'thickness: {result.thickness:.2f} nm')

    if total_extrema <= 4 and total_extrema >= 2:  #& 2peak minimum:
        print('Apply method Scheludko')
        result = thickness_from_scheludko(lambdas, smoothed_intensities,
                                             refractive_index=indice,
                                             min_peak_prominence=prominence,
                                             plot=plot)

        print(f'thickness: {result.thickness:.2f} nm')

    if total_extrema <= 4 and len(peaks_max) == 1 and len(peaks_min) == 0 : #dans l'ordre zéro !
        print('Apply method ordre0')

        result = thickness_for_order0(lambdas, smoothed_intensities,
                                             refractive_index=indice,
                                             min_peak_prominence=prominence,
                                             plot=plot)

        print(f'thickness: {result.thickness:.2f} nm')

    if total_extrema <= 4 and len(peaks_max) == 0 and (len(peaks_min) == 1 or  len(peaks_min) == 0):
        #& 1peak min ou zéro:
        thickness = None
        print('Zone Ombre')


