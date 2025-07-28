from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from experiment_constants import *


def low_pass_filter(signal, low_freq_to_filter):
    smooth_time = 1 / low_freq_to_filter
    window_length = int(smooth_time * SAMPLING_RATE)
    window_length = int(window_length if window_length % 2 == 1 else window_length + 1)
    signal_smoothed = savgol_filter(signal, window_length=window_length, polyorder=2)
    signal_high_pass = signal - signal_smoothed

    return signal_high_pass

def compute_intensity(movie):
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


def clean_intensity_noise(movie):
    """ Remove noise and filter the data """
    fig, ax = plt.subplots(1, 3, figsize=(20, 4))

    # Compute mean intensity of each frame throughout the movie
    intensity = compute_intensity(movie)
    t = np.arange(len(intensity))

    ax[0].plot(t, intensity, 'k-')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Mean intensity')
    ax[0].set_title('Mean intensity over time')

    # Sudden drops in brightness --> noise
    initial_bad_frames = True

    while len(bad_frames) > 0 or initial_bad_frames:
        bad_frames = _detect_intensity_drops(intensity)
        movie_clean = np.delete(movie, bad_frames, axis=0)

        if initial_bad_frames:
            ax[1].plot(t, intensity, 'k-')
            ax[1].plot(bad_frames, intensity[bad_frames], 'r*', label='Bad frames')
            ax[1].set_xlabel('Frame')
            ax[1].set_ylabel('Mean intensity')
            ax[1].set_title("Bad frame detection")
            ax[1].legend()

        initial_bad_frames = False

    # Recheck noise
    intensity = compute_intensity(movie_clean)
    t = np.arange(len(intensity))
    bad_frames = _detect_intensity_drops(intensity)

    ax[2].plot(t, intensity, 'k-')
    ax[2].plot(bad_frames, intensity[bad_frames], 'r*', label='Bad frames')
    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('Mean intensity')
    ax[2].set_title('Mean intensity over time')

    plt.tight_layout()
    plt.show()

    return movie_clean


def _compute_psd(signal):
    signal_fft = np.fft.fft(signal)
    signal_psd = np.abs(signal_fft) ** 2

    return signal_psd

def regress_out_poly2(movie, intensity=None, scale_images=False):
    """" Regress out linear drift, and quadratic drift from each pixel's time trace
         Optional: Regress out intensity as well
    """
    nframes, nrow, ncol = movie.shape
    t = np.arange(nframes)

    # Create regressors
    t_centered = (t - np.mean(t)) / np.std(t)
    t_quad = t_centered ** 2
    t_quad = (t_quad - np.mean(t_quad)) / np.std(t_quad)

    regressors = [t_centered, t_quad] if intensity is None else [intensity, t_centered, t_quad]
    X = np.vstack(regressors).T  # ← shape: (nframes, 2 or 3), matching MATLAB regressor matrix
    flat = movie.reshape(nframes, -1)  # each column is a pixel’s trace
    beta = np.linalg.pinv(X) @ flat
    residuals = flat - X @ beta
    movie_clean = residuals.reshape(movie.shape)

    if scale_images:
        n_regressors = X.shape[1]
        scale_images = beta.reshape(n_regressors, nrow, ncol)

        fig, axes = plt.subplots(1, n_regressors, figsize=(12, 4))
        for i in range(n_regressors):
            ax = axes[i]
            im = ax.imshow(scale_images[i], cmap='gray')
            ax.set_title(f"Regressor {i + 1}")
            fig.colorbar(im, ax=ax)

        fig.suptitle("Scale Images")
        plt.show()

    return movie_clean


def choose_low_freq_to_filter(movie, initial_cutoff=3.0):
    intensity = compute_intensity(movie)
    t = np.arange(len(intensity))
    freq = np.fft.fftfreq(len(intensity), d=1/SAMPLING_RATE)
    freq_half = freq[:len(freq)//2]

    # Set up figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_psd = axs[0]
    ax_time = axs[1]

    # Initial filtered signal and PSD
    intensity_high_pass = low_pass_filter(intensity, low_freq_to_filter=initial_cutoff)
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
    l3, = ax_time.plot(t, intensity, 'k-', label='Original Intensity')
    l4, = ax_time.plot(t, intensity_high_pass, 'b-', label=f'High Pass Intensity')
    ax_time.set_title('Mean Intensity over time')
    ax_time.set_xlabel('Frame')
    ax_time.set_ylabel('Mean Intensity')
    ax_time.legend()

    # Slider setup
    ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
    slider = Slider(ax_slider, 'Low freq cutoff (Hz)', 0.1, 400.0, valinit=initial_cutoff, valstep=0.1)

    def update(val):
        low_freq = slider.val
        filtered = low_pass_filter(intensity, low_freq_to_filter=low_freq)
        new_psd = _compute_psd(filtered)

        l2.set_ydata(new_psd[:len(freq)//2])
        l2.set_label(f"Filtered @ {low_freq:.1f}Hz")
        ax_psd.legend()

        l4.set_ydata(filtered)
        l4.set_label(f'High Pass Intensity @ {low_freq:.1f}Hz')
        ax_time.legend()

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return slider.val


def clean_power_spectrum_noise(movie):
    low_freq_cutoff = choose_low_freq_to_filter(movie)
    intensity = compute_intensity(movie)
    intensity_high_pass = low_pass_filter(intensity, low_freq_to_filter=low_freq_cutoff)

    # Remove the first few frames where the filter doesn't work
    drop_frames = 10
    movie_clean = movie[drop_frames:]
    intensity_high_pass = intensity_high_pass[drop_frames:]

    # Regress out intensity, linear drift, and quadratic drift from each pixel's time trace
    movie_clean = regress_out_poly2(movie_clean, intensity=intensity_high_pass, scale_images=False)

    return movie_clean


def clean_movie_pipeline(movie_raw):
    movie_clean = clean_intensity_noise(movie_raw)
    movie_clean = clean_power_spectrum_noise(movie_clean)

    intensity_raw = compute_intensity(movie_raw)
    intensity_clean = compute_intensity(movie_clean)

    plt.figure(figsize=(12, 4))
    plt.plot(intensity_raw, label="Raw")
    plt.plot(intensity_clean, label="Drift Removed")
    plt.title("Drift Removal from Mean Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Mean Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Step 5: Estimate noise level ---
    std_intensity = np.std(intensity_clean)
    print(f"Standard deviation of corrected trace: {std_intensity:.4f}")

    return movie_clean