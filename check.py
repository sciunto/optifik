

import os
import matplotlib.pyplot as plt




from optifik.analysis import *
from optifik import io

plt.rc('text', usetex=True)
plt.rcParams.update({
    'axes.labelsize': 26,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 23,
})


def play_oder1():
    ##### Chemin du dossier contenant le spectre #####
    from optifik.scheludko import thickness_from_scheludko

    DATA_FOLDER = os.path.abspath(os.path.join(os.path.curdir, 'tests', 'basic'))

    #SAVE_FOLDER = DATA_FOLDER

    # FILE_NAME = '003582.xy' #FFT Exemple -> FFT 3524.51
    # FILE_NAME = '000004310.xy' #OOspectro Exemple -> minmax 1338.35
    # FILE_NAME = '000005253.xy'#Scheludko 4 pics Exemple -> scheludko ²
    # FILE_NAME = '000006544.xy'#Scheludko 2 pics Exemple -> ombre ## Diviser prominence FFT par 2
    # FILE_NAME = '000018918.xy' #Scheludko 1 pic max Exemple -> ombre ## Diviser prominence FFT par 2

    FILE_NAME = '000004310.xy' #TEST#
    spectrum_file = os.path.join(DATA_FOLDER, FILE_NAME)

 

    lambdas, intensities = io.load_spectrum(spectrum_file)
    plot_spectrum(lambdas, intensities, title='Raw')

    lambdas, intensities = io.load_spectrum(spectrum_file, lambda_min=450)
    plot_spectrum(lambdas, intensities, title='Raw, cropped')

    smoothed_intensities = smooth_intensities(intensities)
    plot_spectrum(lambdas, smoothed_intensities, title='Smoothed')

    refractive_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.025
    peaks_min, peaks_max = finds_peak(lambdas, smoothed_intensities,
                                                     min_peak_prominence=prominence,
                                                     plot=True)
 


def play_order1():
    ##### Chemin du dossier contenant le spectre #####
    from optifik.scheludko import thickness_from_scheludko

    DATA_FOLDER = os.path.abspath(os.path.join(os.path.curdir, 'tests', 'basic'))

    #SAVE_FOLDER = DATA_FOLDER

    # FILE_NAME = '003582.xy' #FFT Exemple -> FFT 3524.51
    # FILE_NAME = '000004310.xy' #OOspectro Exemple -> minmax 1338.35
    # FILE_NAME = '000005253.xy'#Scheludko 4 pics Exemple -> scheludko ²
    # FILE_NAME = '000006544.xy'#Scheludko 2 pics Exemple -> ombre ## Diviser prominence FFT par 2
    # FILE_NAME = '000018918.xy' #Scheludko 1 pic max Exemple -> ombre ## Diviser prominence FFT par 2

    FILE_NAME = '000004310.xy' #TEST#
    spectrum_file = os.path.join(DATA_FOLDER, FILE_NAME)


    spectrum_file = 'tests/spectraVictor2/order1/T10219.xy' #TEST#

    lambdas, intensities = io.load_spectrum(spectrum_file)
    plot_spectrum(lambdas, intensities, title='Raw')

    lambdas, intensities = io.load_spectrum(spectrum_file, lambda_min=450)
    plot_spectrum(lambdas, intensities, title='Raw, cropped')

    smoothed_intensities = smooth_intensities(intensities)
    plot_spectrum(lambdas, smoothed_intensities, title='Smoothed')

    refractive_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = 0.025
    peaks_min, peaks_max = finds_peak(lambdas, smoothed_intensities,
                                                     min_peak_prominence=prominence,
                                                     plot=True)

    result = thickness_from_scheludko(lambdas, smoothed_intensities,
                                      refractive_index=refractive_index,
                                      min_peak_prominence=prominence,
                                      plot=True)

def play_order0():
    ##### Chemin du dossier contenant le spectre #####
    from optifik.scheludko import thickness_for_order0

    DATA_FOLDER = os.path.abspath(os.path.join(os.path.curdir, 'tests', 'basic'))

    #SAVE_FOLDER = DATA_FOLDER

    # FILE_NAME = '003582.xy' #FFT Exemple -> FFT 3524.51
    # FILE_NAME = '000004310.xy' #OOspectro Exemple -> minmax 1338.35
    # FILE_NAME = '000005253.xy'#Scheludko 4 pics Exemple -> scheludko ²
    # FILE_NAME = '000006544.xy'#Scheludko 2 pics Exemple -> ombre ## Diviser prominence FFT par 2
    # FILE_NAME = '000018918.xy' #Scheludko 1 pic max Exemple -> ombre ## Diviser prominence FFT par 2

    FILE_NAME = '000004310.xy' #TEST#
    spectrum_file = os.path.join(DATA_FOLDER, FILE_NAME)


    spectrum_file = 'tests/spectraVictor2/order0/T14787.xy' #TEST#

    lambdas, intensities = io.load_spectrum(spectrum_file)
    plot_spectrum(lambdas, intensities, title='Raw')

    lambdas, intensities = io.load_spectrum(spectrum_file, lambda_min=450)
    plot_spectrum(lambdas, intensities, title='Raw, cropped')

    smoothed_intensities = smooth_intensities(intensities)
    plot_spectrum(lambdas, smoothed_intensities, title='Smoothed')

    refractive_index =  1.324188 + 3102.060378 / (lambdas**2)
    prominence = .02
    peaks_min, peaks_max = finds_peak(lambdas, smoothed_intensities,
                                                     min_peak_prominence=prominence,
                                                     plot=True)

    result = thickness_for_order0(lambdas, smoothed_intensities,
                                      refractive_index=refractive_index,
                                      min_peak_prominence=prominence,
                                      plot=True)

 

def check_basic():

    ##### Chemin du dossier contenant le spectre #####

    DATA_FOLDER = os.path.abspath(os.path.join(os.path.curdir, 'tests', 'basic'))

    #SAVE_FOLDER = DATA_FOLDER

    # FILE_NAME = '003582.xy' #FFT Exemple -> FFT 3524.51
    # FILE_NAME = '000004310.xy' #OOspectro Exemple -> minmax 1338.35
    # FILE_NAME = '000005253.xy'#Scheludko 4 pics Exemple -> scheludko ²
    # FILE_NAME = '000006544.xy'#Scheludko 2 pics Exemple -> ombre ## Diviser prominence FFT par 2
    # FILE_NAME = '000018918.xy' #Scheludko 1 pic max Exemple -> ombre ## Diviser prominence FFT par 2

    FILE_NAME = '000004310.xy' #TEST#

    spectrum_file = os.path.join(DATA_FOLDER, FILE_NAME)
    auto(spectrum_file, plot=False)


def check_SV1():


    DATA_FOLDER = os.path.join('tests', 'spectraVictor1')

    import yaml

    yaml_file = os.path.join(DATA_FOLDER, 'known_value.yaml')
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)


    for fn, val in thickness_dict.items():
        #auto(DATA_FOLDER, fn)

        spectre_file = os.path.join(DATA_FOLDER, fn)

        lambdas, raw_intensities = load_spectrum(spectre_file, lambda_min=450)

        ##### Affichage du spectre lissé #####

        #smoothed_intensities, intensities, lambdas = Data_Smoothed(spectre_file)

        smoothed_intensities = smooth_intensities(raw_intensities)


#        smoothed_intensities, intensities, lambdas = Data_Smoothed(spectre_file)

        ##### Indice Optique en fonction de Lambda #####

        indice =  1.324188 + 3102.060378 / (lambdas**2)

        prominence = 0.02

        ##### Find Peak #####

        peaks_min, peaks_max = finds_peak(lambdas, smoothed_intensities,
                                                     min_peak_prominence=prominence,
                                                     plot=False)

        result = thickness_from_minmax(lambdas,
                                                 smoothed_intensities,
                                                 refractive_index=indice,
                                                 min_peak_prominence=prominence)
        print(f'thickness: {result.thickness:.2f} nm')


        print(f'expected: {val}')
        print('#-' * 10)




if __name__ == '__main__':

    #check_basic()
    #check_SV1()
    #play()
    play_order0()
