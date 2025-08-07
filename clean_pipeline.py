import numpy as np
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tifffile import tifffile

from globals import *


def high_pass_filter(signal, low_freq_to_filter):
    """ High pass filter a signal
        Firstly get the low-pass filtered signal (up to low_freq_to_filter)
        Then subtract from the original signal
    """
    smooth_time = 1 / low_freq_to_filter
    window_length = int(smooth_time * SAMPLING_RATE)
    window_length = int(window_length if window_length % 2 == 1 else window_length + 1)
    signal_smoothed = savgol_filter(signal, window_length=window_length, polyorder=2)
    signal_high_pass = signal - signal_smoothed

    return signal_high_pass


def _compute_intensity(movie):
    """ Compute mean intensity of each frame throughout the movie """

    frames = movie.shape[0]
    intensity = np.mean(movie, axis=(1, 2))  # mean across height and width
    assert frames == len(intensity), "Intensity array does not match number of frames"

    return intensity


def _detect_intensity_drops(intensity, window_length=11, polyorder=3, bad_frames_threshold=-3.5):
    """ Detect sudden drops in brightness using a highpass filter on the intensity array """

    smoothed_intensity = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)
    high_pass_intensity = intensity - smoothed_intensity
    bad_frames = np.where(high_pass_intensity < bad_frames_threshold)[0]

    return bad_frames


def clean_intensity_noise(movie, initial_threshold=-3.5):
    """Interactively adjust the intensity drop threshold for bad frame detection using a slider."""
    intensity = _compute_intensity(movie)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_bad_frames = axs[0]
    ax_cleaned = axs[1]

    t_original = np.arange(len(intensity))

    # Initial detection and plot
    bad_frames = _detect_intensity_drops(intensity, bad_frames_threshold=initial_threshold)
    l1, = ax_bad_frames.plot(t_original, intensity, 'k-', label="Intensity")
    bad_markers1, = ax_bad_frames.plot(bad_frames, intensity[bad_frames], 'r*', label="Bad Frames")

    movie_clean = np.delete(movie, bad_frames, axis=0)
    intensity_clean = _compute_intensity(movie_clean)
    t_clean = np.arange(len(intensity_clean))
    bad_frames_clean = _detect_intensity_drops(intensity_clean, bad_frames_threshold=initial_threshold)
    l2, = ax_cleaned.plot(t_clean, intensity_clean, 'k-', label="Intensity (Cleaned)")
    bad_markers2, = ax_cleaned.plot(bad_frames_clean, intensity_clean[bad_frames_clean], 'r*', label="Bad Frames")

    ax_bad_frames.set_title('Bad Frames Detection')
    ax_bad_frames.set_xlabel('Frame')
    ax_bad_frames.set_ylabel('Mean Intensity')
    ax_bad_frames.legend()

    ax_cleaned.set_title('Post-Cleaning Bad Frames Detection')
    ax_cleaned.set_xlabel('Frame')
    ax_cleaned.set_ylabel('Mean Intensity')
    ax_cleaned.legend()

    # Slider
    ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
    slider = Slider(ax_slider, 'Detection Threshold', -10.0, 0.0, valinit=initial_threshold, valstep=0.1)

    def update(val):
        threshold = slider.val

        bad = _detect_intensity_drops(intensity, bad_frames_threshold=threshold)
        bad_markers1.set_data(bad, intensity[bad])

        # Recompute cleaned intensity without updating movie
        mask = np.ones(len(intensity), dtype=bool)
        mask[bad] = False
        intensity_cleaned = intensity[mask]
        t_cleaned = np.arange(len(intensity_cleaned))
        bad_cleaned = _detect_intensity_drops(intensity_cleaned, bad_frames_threshold=threshold)

        # Update plot lines
        l2.set_data(t_cleaned, intensity_cleaned)
        bad_markers2.set_data(bad_cleaned, intensity_cleaned[bad_cleaned])

        ax_cleaned.set_xlim(0, len(intensity_cleaned))
        ax_cleaned.set_ylim(np.min(intensity_cleaned), np.max(intensity_cleaned))

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    final_threshold = slider.val
    final_bad = _detect_intensity_drops(intensity, bad_frames_threshold=final_threshold)
    final_cleaned_movie = np.delete(movie, final_bad, axis=0)

    return final_cleaned_movie



def _compute_psd(signal):
    """ Compute PSD of signal """
    signal_fft = np.fft.fft(signal)
    signal_psd = np.abs(signal_fft) ** 2
    return signal_psd

def _regress_out_poly2(movie, intensity=None):
    """ Regresses out global intensity and low-order polynomial trends from each pixel's time trace
        by fitting each pixel's fluorescence trace to a set of regressors
        (global intensity, linear and quadratic trends) and subtracting the fitted signal.
        Optionally visualizes the spatial weights (scale images) of each regressor.
    """

    n_frames, n_row, n_col = movie.shape
    t = np.arange(n_frames)

    # Create regressors
    t_centered = t - np.mean(t)
    t_quad = t_centered ** 2
    t_quad = t_quad - np.mean(t_quad)

    regressors = [t_centered, t_quad] if intensity is None else [intensity, t_centered, t_quad]
    X = np.vstack(regressors).T  # ← shape: (n_frames, 2 or 3), matching MATLAB regressor matrix
    flat = movie.reshape(n_frames, -1)  # each column is a pixel’s trace
    beta = np.linalg.pinv(X) @ flat
    residuals = flat - X @ beta
    movie_clean = residuals.reshape(movie.shape)

    n_regressors = X.shape[1]
    scale_images = beta.reshape(n_regressors, n_row, n_col)

    if PLOT_CLEANING_STEPS:
        fig, axes = plt.subplots(1, n_regressors, figsize=(12, 4))
        for i in range(n_regressors):
            ax = axes[i]
            im = ax.imshow(scale_images[i], cmap='gray')
            ax.set_title(f"Regressor {i + 1}")
            fig.colorbar(im, ax=ax)

        fig.suptitle("Scale Images")
        plt.show()

    return movie_clean


def _choose_low_freq_to_filter(movie, initial_cutoff=3.0):
    """ Interactively adjust the low cutoff frequency to filter """

    intensity = _compute_intensity(movie)
    t = np.arange(len(intensity))
    freq = np.fft.fftfreq(len(intensity), d=1/SAMPLING_RATE)
    freq_half = freq[:len(freq)//2]

    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_psd = axs[0]
    ax_time = axs[1]

    # Initial filtered signal and PSD
    intensity_high_pass = high_pass_filter(intensity, low_freq_to_filter=initial_cutoff)
    intensity_psd = _compute_psd(intensity)
    noise_psd = _compute_psd(intensity_high_pass)

    # Plot PSD
    l1, = ax_psd.semilogy(freq_half, intensity_psd[:len(freq)//2], label="Raw")
    l2, = ax_psd.semilogy(freq_half, noise_psd[:len(freq)//2], label=f"Filtered @ {initial_cutoff}Hz")
    ax_psd.set_title('Intensity Power Spectrum')
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power")
    ax_psd.legend()

    # Plot intensity traces
    l3, = ax_time.plot(t, intensity_high_pass, 'b-', label=f'High Pass Intensity')
    ax_time.set_title('Mean Intensity over time')
    ax_time.set_xlabel('Frame')
    ax_time.set_ylabel('Mean Intensity')
    ax_time.legend()

    # Slider setup
    ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
    slider = Slider(ax_slider, 'Low freq cutoff (Hz)', 0.1, 400.0, valinit=initial_cutoff, valstep=0.1)

    def update(val):
        low_freq = slider.val
        filtered = high_pass_filter(intensity, low_freq_to_filter=low_freq)
        new_psd = _compute_psd(filtered)

        l2.set_ydata(new_psd[:len(freq)//2])
        l2.set_label(f"Filtered @ {low_freq:.1f}Hz")
        ax_psd.legend()

        l3.set_ydata(filtered)
        l3.set_label(f'High Pass Intensity @ {low_freq:.1f}Hz')
        ax_time.legend()

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return slider.val


def clean_power_spectrum_noise(movie):
    """ Cleans the movie by removing global intensity fluctuations and slow temporal drifts from each pixel's time series.

        Steps:
        1. Interactively adjusts the low cutoff frequency to filter
        2. Applies high-pass filtering to remove slow trends.
        3. Drops initial frames to account for filter edge effects.
        3. Regresses out high-pass filtered intensity, linear, and quadratic time trends
           from each pixel using second-order polynomial regression.

        Returns cleaned movie of shape (n_frames - drop_frames, height, width).
    """

    low_freq_cutoff = _choose_low_freq_to_filter(movie)
    intensity = _compute_intensity(movie)
    intensity_high_pass = high_pass_filter(intensity, low_freq_to_filter=low_freq_cutoff)

    # Remove the first few frames where the filter doesn't work
    drop_frames = 10
    movie_clean = movie[drop_frames:]
    intensity_high_pass = intensity_high_pass[drop_frames:]

    # Regress out intensity, linear drift, and quadratic drift from each pixel's time trace
    movie_clean = _regress_out_poly2(movie_clean, intensity=intensity_high_pass)

    return movie_clean


def clean_movie_pipeline(movie_original, save_clean=False):
    """ Pipeline of all cleaning functions - return the clean movie """

    print("Cleaning the data...")
    movie_clean = clean_power_spectrum_noise(clean_intensity_noise(movie_original))

    intensity_raw = _compute_intensity(movie_original)
    intensity_clean = _compute_intensity(movie_clean)

    if PLOT_CLEANING_STEPS:
        plt.figure(figsize=(12, 4))
        plt.plot(intensity_raw, label="Raw")
        plt.plot(intensity_clean, label="Drift Removed")
        plt.title("Drift Removal from Mean Intensity")
        plt.xlabel("Frame")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Standard deviation of corrected trace: {np.std(intensity_clean):.4f}")

    if save_clean:
        tifffile.imwrite(CLEAN_PATH, movie_clean)

    return movie_clean