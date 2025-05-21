# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:34:02 2025

@author: ziapkoff
"""

import os

import matplotlib.pyplot as plt




from optifik.analysis import *
from optifik.auto import auto

plt.rc('text', usetex=True)
plt.rcParams.update({
    'axes.labelsize': 26,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 23,
})



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

    auto(DATA_FOLDER, FILE_NAME, plot=False)


def check_SV1():
    
    
    DATA_FOLDER = os.path.join('tests', 'spectraVictor1')
    
    import yaml

    yaml_file = os.path.join(DATA_FOLDER, 'known_value.yaml')
    with open(yaml_file, "r") as yaml_file:
        thickness_dict = yaml.safe_load(yaml_file)
    
    
    for fn, val in thickness_dict.items():
        #auto(DATA_FOLDER, fn)
        
        spectre_file = os.path.join(DATA_FOLDER, fn)

        ##### Affichage du spectre brut et récupération des Intesités brutes#####

        raw_intensities = plot_xy(spectre_file)

        ##### Affichage du spectre lissé #####

        smoothed_intensities, intensities, lambdas = Data_Smoothed(spectre_file)

        ##### Indice Optique en fonction de Lambda #####

        indice =  1.324188 + 3102.060378 / (lambdas**2)

        prominence = 0.02

        ##### Find Peak #####

        total_extrema, smoothed_intensities, raw_intensities, lambdas, peaks_min, peaks_max = finds_peak(spectre_file,
                                                                                                         min_peak_prominence=prominence)
        
        thickness_minmax = thickness_from_minmax(lambdas,
                                                 smoothed_intensities,
                                                 refractive_index=indice,
                                                 min_peak_prominence=prominence)
        thickness = thickness_minmax.thickness
        print(f'thickness: {thickness:.2f} nm')
        
        
        print(f'expected: {val}')
        print('#-' * 10)


    

if __name__ == '__main__':

    check_basic()
    #check_SV1()
