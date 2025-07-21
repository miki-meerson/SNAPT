from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from experiment_constants import *


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
    bad_frames = _detect_intensity_drops(intensity)

    ax[1].plot(t, intensity, 'k-', label='Original intensity')
    ax[1].plot(bad_frames, intensity[bad_frames], 'r*', label='Bad frames')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Mean intensity')
    ax[1].set_title("Bad frame detection")
    ax[1].legend()

    movie_clean = np.delete(movie, bad_frames, axis=0)

    # Recheck noise
    intensity = compute_intensity(movie_clean)
    t = np.arange(len(intensity))

    ax[2].plot(t, intensity, 'k-')
    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('Mean intensity')
    ax[2].set_title('Mean intensity over time')

    plt.tight_layout()
    plt.show()

    # TODO consider removing bad frames in a loop if possible or play with -3.5 threshold
    bad_frames = _detect_intensity_drops(intensity)
    if len(bad_frames) > 0:
        print(f"Warning: {len(bad_frames)} bad frames remain after cleaning")

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
    t_centered = t - np.mean(t)

    to_regress = [t_centered, t_centered ** 2] if intensity is None else [intensity, t_centered, t_centered ** 2]
    X = np.vstack(to_regress).T

    flat = movie.reshape(nframes, -1)
    model = LinearRegression().fit(X, flat)
    residuals = flat - model.predict(X)
    movie_clean = residuals.reshape(movie.shape)

    if scale_images:
        scale_images = model.coef_.T.reshape(3, nrow, ncol)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(3):
            ax = axes[i]
            im = ax.imshow(scale_images[i], cmap='gray')
            ax.set_title(f"Regressor {i + 1}")
            fig.colorbar(im, ax=ax)

        fig.suptitle("Scale Images")
        plt.show()

    return movie_clean


def clean_power_spectrum_noise(movie):
    intensity = compute_intensity(movie)
    t = np.arange(len(intensity))

    intensity_psd = _compute_psd(intensity)
    freq = np.fft.fftfreq(len(intensity), d=1/SAMPLING_RATE)

    fig, ax = plt.subplots(1, 2, figsize=(20, 4))

    # TODO get exact frequency to filter
    # Remove low-freq noise ~60Hz --> ~15ms window
    low_freq_to_filter = 60
    smooth_time = 1/low_freq_to_filter
    window_length = int(smooth_time if smooth_time % 2 == 1 else smooth_time + 1)
    intensity_smoothed = savgol_filter(intensity, window_length=window_length, polyorder=2)
    intensity_high_pass = intensity - intensity_smoothed

    noise_psd = _compute_psd(intensity_high_pass)

    ax[0].semilogy(freq[:len(freq)//2], intensity_psd[:len(freq)//2], label="Raw")
    ax[0].semilogy(freq[:len(freq)//2], noise_psd[:len(freq)//2], label="Filtered")
    ax[0].set_title('Intensity Power Spectrum')
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Power")
    ax[0].legend()

    ax[1].plot(t, intensity, 'k-', label='Original Intensity')
    ax[1].plot(t, intensity_high_pass, 'k-', label='High Pass Intensity')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Mean Intensity')
    ax[1].set_title('Mean Intensity over time')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # TODO calculate the exact number of frames to remove
    # Remove the first few frames where the filter doesn't work
    movie_clean = movie[10:]
    intensity_high_pass = intensity_high_pass[10:]
    t_clean = np.arange(len(intensity_high_pass))

    # Regress out intensity, linear drift, and quadratic drift from each pixel's time trace
    movie_clean = regress_out_poly2(movie_clean, intensity=intensity_high_pass, scale_images=True)
    return movie_clean


def clean_movie_pipeline(movie_to_clean):
    movie_to_clean = clean_intensity_noise(movie_to_clean)
    movie_to_clean = clean_power_spectrum_noise(movie_to_clean)

    return movie_to_clean