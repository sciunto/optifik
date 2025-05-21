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



def thickness_from_fft(lambdas, intensities,
                       refractive_index,
                       num_half_space=None,
                       plot=None):
    """
    Determine the tickness by Fast Fourier Transform.

    Parameters
    ----------
    lambdas : array
        Wavelength values in nm.
    intensities : array
        Intensity values.
    refractive_index : scalar, optional
        Value of the refractive index of the medium.
    num_half_space : scalar, optional
        Number of points to compute FFT's half space.
        If `None`, default corresponds to `10*len(lambdas)`.
    debug : boolean, optional
        Show plot of the transformed signal and the peak detection.

    Returns
    -------
    results : Instance of `OptimizeResult` class.
        The attribute `thickness` gives the thickness value in nm.
    """
    if num_half_space is None:
        num_half_space = 10 * len(lambdas)

    # FFT requires evenly spaced data.
    # So, we interpolate the signal.
    # Interpolate to get a linear increase of 1 / lambdas.
    inverse_lambdas_times_n = refractive_index / lambdas
    f = interp1d(inverse_lambdas_times_n, intensities)

    inverse_lambdas_linspace = np.linspace(inverse_lambdas_times_n[0],
                                           inverse_lambdas_times_n[-1],
                                           2*num_half_space)
    intensities_linspace = f(inverse_lambdas_linspace)


    # Perform FFT
    density = (inverse_lambdas_times_n[-1]-inverse_lambdas_times_n[0]) / (2*num_half_space)
    inverse_lambdas_fft = fftfreq(2*num_half_space, density)
    intensities_fft = fft(intensities_linspace)

    # The FFT is symetrical over [0:N] and [N:2N].
    # Keep over [N:2N], ie for positive freq.
    intensities_fft = intensities_fft[num_half_space:2*num_half_space]
    inverse_lambdas_fft = inverse_lambdas_fft[num_half_space:2*num_half_space]

    idx_max_fft = np.argmax(abs(intensities_fft))
    freq_max = inverse_lambdas_fft[idx_max_fft]


    thickness_fft = freq_max / 2.


    plt.figure(figsize=(10, 6),dpi =600)
    if plot:
        plt.loglog(inverse_lambdas_fft, np.abs(intensities_fft))
        plt.loglog(freq_max, np.abs(intensities_fft[idx_max_fft]), 'o')
        plt.xlabel('Frequency')
        plt.ylabel(r'FFT($I^*$)')
        plt.title(f'Thickness={thickness_fft:.2f}')

    return OptimizeResult(thickness=thickness_fft,)


def Prominence_from_fft(lambdas, intensities, refractive_index, num_half_space=None, plot=True):
    if num_half_space is None:
        num_half_space = 10 * len(lambdas)

    # Interpolation pour que les données soient uniformément espacées
    inverse_lambdas_times_n = refractive_index / lambdas
    f = interp1d(inverse_lambdas_times_n, intensities)

    inverse_lambdas_linspace = np.linspace(inverse_lambdas_times_n[0],
                                           inverse_lambdas_times_n[-1],
                                           2*num_half_space)
    intensities_linspace = f(inverse_lambdas_linspace)

    # FFT
    density = (inverse_lambdas_times_n[-1] - inverse_lambdas_times_n[0]) / (2*num_half_space)
    freqs = fftfreq(2*num_half_space, density)
    fft_vals = fft(intensities_linspace)

    # On conserve uniquement les fréquences positives
    freqs = freqs[num_half_space:]
    fft_vals = fft_vals[num_half_space:]

    # Trouver le pic principal
    abs_fft = np.abs(fft_vals)
    idx_max = np.argmax(abs_fft)
    F_max = freqs[idx_max]

    if plot:
        print(f"F_max detected at: {F_max:.4f}")
        plt.figure(figsize=(10, 6),dpi = 600)
        plt.plot(freqs, abs_fft, label='|FFT|')
        plt.axvline(F_max, color='r', linestyle='--', label='F_max')
        plt.xlabel('Fréquence')
        plt.ylabel('Amplitude FFT')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()

    # Filtrage : on garde les composantes au-dessus de 10 * F_max
    cutoff = 10 * F_max
    mask = freqs >= cutoff
    fft_filtered = np.zeros_like(fft_vals)
    fft_filtered[mask] = fft_vals[mask]

    fft_full = np.zeros(2 * num_half_space, dtype=complex)
    fft_full[num_half_space:] = fft_filtered                      # fréquences positives
    fft_full[:num_half_space] = np.conj(fft_filtered[::-1])
    # IFFT
    signal_filtered = np.real(ifft(fft_full))

    # Max amplitude après filtrage
    max_amplitude = np.max(np.abs(signal_filtered))

    if plot:
        plt.figure(figsize=(10, 6),dpi = 600)
        plt.plot(signal_filtered, label='Signal filtered')
        plt.xlabel('Échantillons')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
        print(f"Amplitude Mal filtered : {max_amplitude:.4f}")

    return max_amplitude
