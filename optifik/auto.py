import os.path

from .analysis import *
from .io import load_spectrum

def auto(DATA_FOLDER, FILE_NAME, plot=None):


    spectre_file = os.path.join(DATA_FOLDER, FILE_NAME)

    ##### Affichage du spectre brut et récupération des Intesités brutes#####

    raw_intensities = load_spectrum(spectre_file)

    ##### Affichage du spectre lissé #####

    smoothed_intensities, intensities, lambdas = Data_Smoothed(spectre_file)

    ##### Indice Optique en fonction de Lambda #####

    indice =  1.324188 + 3102.060378 / (lambdas**2)

    ##### Determination de la prominence associé #####

    prominence = Prominence_from_fft(lambdas=lambdas, 
                                     intensities=smoothed_intensities, 
                                     refractive_index=indice,
                                     plot=plot)

    prominence = 0.03
    ##### Find Peak #####

    total_extrema, smoothed_intensities, raw_intensities, lambdas, peaks_min, peaks_max = finds_peak(spectre_file,
                                                                                                     min_peak_prominence=prominence,
                                                                                                     plot=plot)

    ##### Epaisseur selon la methode #####

    #thickness_FFT = thickness_from_fft(lambdas,smoothed_intensities,refractive_index=1.33)

    if total_extrema > 15 and total_extrema > 4:
        print('Apply method FFT')
        thickness_FFT = thickness_from_fft(lambdas,smoothed_intensities,
                                           refractive_index=indice,
                                           plot=plot)
        thickness = thickness_FFT.thickness
        print(f'thickness: {thickness:.2f} nm')


    if total_extrema <= 15 and total_extrema > 4:
        print('Apply method minmax')
        thickness_minmax = thickness_from_minmax(lambdas,smoothed_intensities,
                                                 refractive_index=indice,
                                                 min_peak_prominence=prominence,
                                                 plot=plot)
        thickness = thickness_minmax.thickness
        print(f'thickness: {thickness:.2f} nm')

    if total_extrema <= 4 and total_extrema >= 2:  #& 2peak minimum:
        print('Apply method Scheludko')
        thickness = thickness_from_scheludko(lambdas, raw_intensities, smoothed_intensities,
                                             peaks_min, peaks_max,
                                             refractive_index=indice,
                                             plot=plot)
        print(f'thickness: {thickness:.2f} nm')

    if total_extrema <= 4 and len(peaks_max) == 1 and len(peaks_min) == 0 : #dans l'ordre zéro !
        print('Apply method ordre0')

        thickness = thickness_for_order0(lambdas, raw_intensities, smoothed_intensities,
                                             peaks_min, peaks_max,
                                             refractive_index=indice,
                                             plot=plot)

        print(f'thickness: {thickness:.2f} nm')

    if total_extrema <= 4 and len(peaks_max) == 0 and (len(peaks_min) == 1 or  len(peaks_min) == 0):
        #& 1peak min ou zéro:
        thickness = None
        print('Zone Ombre')


