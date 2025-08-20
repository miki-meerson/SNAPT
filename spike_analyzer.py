import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from constants import *
from clean_pipeline import hp_filter
from roi_analyzer import get_average_image
from scipy.signal import find_peaks


def get_bright_pixel_mask(movie):
    avg_img = get_average_image(movie)
    chosen_percentile = 25

    # Initial mask
    if IS_NEGATIVE_GEVI:
        mask = avg_img <= np.percentile(avg_img, chosen_percentile)
    else:
        mask = avg_img >= np.percentile(avg_img, chosen_percentile)

    masked_img = np.where(mask, avg_img, 0)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)
    ax_avg_img, ax_mask, ax_masked_img = axs

    im1 = ax_avg_img.imshow(avg_img, cmap='gray')
    ax_avg_img.set_title('Average Image')
    ax_avg_img.axis('off')

    im2 = ax_mask.imshow(mask, cmap='gray')
    ax_mask.set_title(f'Mask ≤ {chosen_percentile}th Percentile')
    ax_mask.axis('off')

    im3 = ax_masked_img.imshow(masked_img, cmap='gray')
    ax_masked_img.set_title('Masked Average Image')
    ax_masked_img.axis('off')

    # Slider
    ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
    slider = Slider(ax_slider, 'Mask Percentile', 1, 100, valinit=chosen_percentile, valstep=1)

    def update(val):
        percentile = slider.val

        if IS_NEGATIVE_GEVI:
            new_mask = avg_img <= np.percentile(avg_img, percentile)
        else:
            new_mask = avg_img >= np.percentile(avg_img, percentile)
        new_masked_img = np.where(new_mask, avg_img, 0)

        im2.set_data(new_mask)
        ax_mask.set_title(f'Mask ≤ {percentile:.0f}th Percentile')

        im3.set_data(new_masked_img)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.suptitle('Visualization of Signal-Rich Pixels')
    plt.show()

    return mask


def detect_spikes(trace, movmean_window=40, min_distance=10):
    spikes = []
    trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

    trace_highpass = trace - uniform_filter1d(trace, size=movmean_window)
    trace_highpass_shifted = trace_highpass + 1 # Shift trace to be positive

    assert trace_highpass_shifted.min() > 0, f"Values under 0 exist in high-pass filtered trace: smallest value is {trace_highpass_shifted.min()}"

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.1])
    ax_trace = fig.add_subplot(gs[0])
    ax_count = fig.add_subplot(gs[1])
    ax_slider = fig.add_subplot(gs[2])

    ax_trace.plot(trace, label='Normalized ROI trace')
    ax_trace.plot(trace_highpass, label='High-pass ROI trace')
    spike_dots, = ax_trace.plot([], [], 'ro', label='Spikes')
    ax_trace.legend()
    ax_trace.set_title("Spike detection with adjustable threshold")

    bar = ax_count.bar(['Detected Spikes'], [0])
    ax_count.set_ylim(0, 500)
    ax_count.set_ylabel('Spike count')

    slider = Slider(ax_slider, 'Threshold', 1, 1.5, valinit=1.1, valstep=0.01)

    def _update_threshold(val):
        nonlocal spikes
        threshold = val
        peaks, _ = find_peaks(trace_highpass_shifted, height=threshold, distance=min_distance)
        spikes = peaks

        spike_dots.set_data(peaks, trace[peaks])
        ax_trace.relim()
        ax_trace.autoscale_view()

        ax_count.set_title(f"{len(peaks)} spikes detected")

        fig.canvas.draw_idle()

    slider.on_changed(_update_threshold)
    _update_threshold(slider.val)
    plt.show()

    return spikes


def extract_sta(movie, roi_traces):
    n_frames, n_row, n_col = movie.shape
    n_rois = roi_traces.shape[1]
    first_trace = roi_traces[:, 0]
    spike_indices = detect_spikes(first_trace)

    window_size = 2 * PULSE_FRAMES + 1
    spikes = np.full((window_size, n_rois, len(spike_indices)), np.nan)
    spike_movie = np.zeros((window_size, n_row, n_col))

    c = 0  # spike counter
    for i, spike_time in enumerate(spike_indices):
        if PULSE_FRAMES < spike_time < (n_frames - PULSE_FRAMES):
            # ROI trace segment
            fs = roi_traces[(spike_time - PULSE_FRAMES):(spike_time + PULSE_FRAMES + 1), :]
            f0 = np.mean(fs[:10, :], axis=0)
            spikes[:, :, c] = fs - f0

            # Movie segment
            segment = movie[(spike_time - PULSE_FRAMES):(spike_time + PULSE_FRAMES + 1), :, :]
            spike_movie += segment
            c += 1

    if c == 0:
        print("No spikes in valid window range.")
        return None, None, spike_indices

    spike_movie_avg = spike_movie / c
    return spike_movie_avg, spikes, spike_indices


def build_spike_triggered_movie(spike_indices, movie, window_size, mask=None):
    sta_frames = []
    n_frames = movie.shape[0]

    for spike_index in spike_indices:
        if window_size < spike_index < n_frames - window_size:
            window = movie[spike_index - window_size:spike_index + window_size + 1].copy()

            if mask is not None:
                window[:, ~mask] = 0

            sta_frames.append(window)

    if sta_frames:
        sta_frames = np.array(sta_frames)
        sta_movie = np.mean(sta_frames, axis=0)
        assert sta_movie.shape[0] == 2 * window_size + 1, f"STA movie has wrong length: {sta_movie.shape[0]} vs expected {2 * window_size + 1}"
        return sta_movie
    else:
        return np.zeros((2 * window_size + 1, movie.shape[1], movie.shape[2]))


