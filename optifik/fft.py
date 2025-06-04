import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, fftfreq


import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({
    'axes.labelsize': 26,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 23,
})


from .utils import OptimizeResult



def thickness_from_fft(wavelengths, intensities,
                       refractive_index,
                       num_half_space=None,
                       plot=None):
    """
    Determine the tickness by Fast Fourier Transform.

    Parameters
    ----------
    wavelengths : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar, optional
        Value of the refractive index of the medium.
    num_half_space : scalar, optional
        Number of points to compute FFT's half space.
        If `None`, default corresponds to `10*len(wavelengths)`.
    debug : boolean, optional
        Show plot of the transformed signal and the peak detection.

    Returns
    -------
    results : Instance of `OptimizeResult` class.
        The attribute `thickness` gives the thickness value in nm.
    """
    if num_half_space is None:
        num_half_space = 10 * len(wavelengths)

    # FFT requires evenly spaced data.
    # So, we interpolate the signal.
    # Interpolate to get a linear increase of 1 / wavelengths.
    inverse_wavelengths_times_n = refractive_index / wavelengths
    f = interp1d(inverse_wavelengths_times_n, intensities)

    inverse_wavelengths_linspace = np.linspace(inverse_wavelengths_times_n[0],
                                           inverse_wavelengths_times_n[-1],
                                           2*num_half_space)
    intensities_linspace = f(inverse_wavelengths_linspace)


    # Perform FFT
    density = (inverse_wavelengths_times_n[-1]-inverse_wavelengths_times_n[0]) / (2*num_half_space)
    inverse_wavelengths_fft = fftfreq(2*num_half_space, density)
    intensities_fft = fft(intensities_linspace)

    # The FFT is symetrical over [0:N] and [N:2N].
    # Keep over [N:2N], ie for positive freq.
    intensities_fft = intensities_fft[num_half_space:2*num_half_space]
    inverse_wavelengths_fft = inverse_wavelengths_fft[num_half_space:2*num_half_space]

    idx_max_fft = np.argmax(abs(intensities_fft))
    freq_max = inverse_wavelengths_fft[idx_max_fft]

    thickness_fft = freq_max / 2.

    if plot:
        plt.figure(figsize=(10, 6),dpi =600)
        plt.loglog(inverse_wavelengths_fft, np.abs(intensities_fft))
        plt.loglog(freq_max, np.abs(intensities_fft[idx_max_fft]), 'o')
        plt.xlabel('Frequency')
        plt.ylabel(r'FFT($I^*$)')
        plt.title(f'Thickness={thickness_fft:.2f}')

    return OptimizeResult(thickness=thickness_fft,)


#def Prominence_from_fft(wavelengths, intensities, refractive_index,
#                        num_half_space=None, plot=None):
#    if num_half_space is None:
#        num_half_space = len(wavelengths)
#
#    # # # 1. Spectre original
#    # if plot:
#    #     plt.figure(figsize=(10, 6), dpi=150)
#    #     plt.plot(1/wavelengths, intensities, label='Spectre original')
#    #     plt.xlabel('1/Longueur d\'onde (nm-1)')
#    #     plt.ylabel('Intensité')
#    #     plt.legend()
#    #     plt.show()
#
#
#    # 2. Conversion lambda → k = n(λ) / λ
#    k_vals = refractive_index / wavelengths
#    f_interp = interp1d(k_vals, intensities, kind='linear', fill_value="extrapolate")
#
#    # 3. Axe k uniforme + interpolation
#    k_linspace = np.linspace(k_vals[0], k_vals[-1], 2 * num_half_space)
#    intensities_k = f_interp(k_linspace)
#
#    # 4. FFT
#    delta_k = (k_vals[-1] - k_vals[0]) / (2 * num_half_space)
#    fft_vals = fft(intensities_k)
#    freqs = fftfreq(2 * num_half_space, delta_k)
#
#    # 5. Pic FFT
#    freqs_pos = freqs[freqs > 0]
#    abs_fft_pos = np.abs(fft_vals[freqs > 0])
#    idx_max = np.argmax(abs_fft_pos)
#    F_max = freqs_pos[idx_max]
#
#    if plot:
#        plt.figure(figsize=(10, 6), dpi=150)
#        plt.plot(freqs_pos, abs_fft_pos, label='|FFT|')
#        plt.axvline(F_max, color='r', linestyle='--', label='Pic principal')
#        plt.xlabel('Distance optique [nm]')
#        plt.ylabel(r'FFT($I^*$)')
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.legend()
#        plt.show()
#
#    # 6. Filtrage (garde hautes fréquences)
#    cutoff_HF = 2 * F_max
#
#    mask_HF = np.abs(freqs) >= cutoff_HF
#    fft_filtered_HF = np.zeros_like(fft_vals, dtype=complex)
#    fft_filtered_HF[mask_HF] = fft_vals[mask_HF]
#
#    # 7. Filtrage (garde basses fréquences)
#    cutoff_BF = 10 * F_max
#    mask_BF = np.abs(freqs) <= cutoff_BF
#    fft_filtered_BF = np.zeros_like(fft_vals, dtype=complex)
#    fft_filtered_BF[mask_BF] = fft_vals[mask_BF]
#
#
#    # 8. Reconstruction
#    signal_filtered_HF = np.real(ifft(fft_filtered_HF))
#    signal_filtered_BF = np.real(ifft(fft_filtered_BF))
#    lambda_reconstructed = np.interp(k_linspace, k_vals[::-1], wavelengths[::-1])
#
#    # Masque dans la plage [550, 700] nm
#    mask_Cam_Sensitivity = (lambda_reconstructed >= 550) & (lambda_reconstructed <= 700)
#
#    # 9. Affichage reconstruction
#    if plot:
#        plt.figure(figsize=(10, 6), dpi=150)
#        plt.plot(lambda_reconstructed, intensities_k, '--', label='Original interpolé')
#        plt.plot(lambda_reconstructed, signal_filtered_HF,'--', color='gray')
#
#        plt.plot(lambda_reconstructed[mask_Cam_Sensitivity], signal_filtered_HF[mask_Cam_Sensitivity],
#                 color='orange', label='Spectre filtré HF')
#        plt.plot(lambda_reconstructed, signal_filtered_BF,
#                 color='red', label='Spectre filtré BF')
#
#        plt.xlabel('Wavelength (nm)')
#        plt.ylabel('Intensity')
#        plt.legend()
#        plt.show()
#
#    max_amplitude = np.max(np.abs(signal_filtered_HF[mask_Cam_Sensitivity]))
#
#    return max_amplitude, signal_filtered_BF, lambda_reconstructed
#
