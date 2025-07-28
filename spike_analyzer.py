import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.signal import find_peaks

from clean_pipeline import low_pass_filter, choose_low_freq_to_filter
from experiment_constants import SAMPLING_RATE


def detect_spikes_in_roi(trace):
    spikes = []
    trace_high_pass = low_pass_filter(trace, low_freq_to_filter=3)
    trace_high_pass_shifted = trace_high_pass - trace_high_pass.min() + 1

    assert trace_high_pass_shifted.min() > 0, f"Values under 0 exist in highpass filtered trace: smallest value is {trace_high_pass_shifted.min()}"

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.1])
    ax_trace = fig.add_subplot(gs[0])
    ax_movie = fig.add_subplot(gs[1])
    ax_slider = fig.add_subplot(gs[2])

    ax_trace.plot(trace, label='Normalized ROI trace')
    ax_trace.plot(trace_high_pass, label='High-pass ROI trace')
    spike_dots, = ax_trace.plot([], [], 'ro', label='Spikes')
    ax_trace.legend()
    ax_trace.set_title("Spike detection with adjustable threshold")

    slider = Slider(ax_slider, 'Threshold', 1, 2, valinit=1.5, valstep=0.01)

    def _update_threshold(val):
        nonlocal spikes
        threshold = val
        peaks, _ = find_peaks(trace_high_pass_shifted, height=threshold, distance=10)
        spikes = peaks
        spike_dots.set_data(peaks, trace[peaks])
        ax_trace.relim()
        ax_trace.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(_update_threshold)
    _update_threshold(slider.val)
    plt.show()

    return spikes

def extract_spike_triggered_average(roi_traces, pulse=50):
    _, n_rois = roi_traces.shape

    f = roi_traces[:, 0]  # <- assuming you want the first ROI
    f_norm = (f - f.min()) / (f.max() - f.min())

    spike_indices = detect_spikes_in_roi(f_norm)
    return spike_indices


def play_movie(movie, interval=100):
    fig, ax = plt.subplots()
    img = ax.imshow(movie[0], cmap='gray', vmin=movie.min(), vmax=movie.max())
    ax.set_title("Spike-triggered average movie")

    def update(i):
        img.set_data(movie[i])
        ax.set_title(f"Frame {i}")
        return [img]

    ani = FuncAnimation(fig, update, frames=len(movie), interval=interval, blit=False)
    plt.show()
    return ani


def extract_peri_spike_movies(spike_frames, movie, roi_traces, pre_frames=50, post_frames=50):
    peri_movies = []
    peri_traces = []
    n_frames, n_row, n_col = movie.shape
    _, n_rois = roi_traces.shape

    # Remove spikes too close to beginning or end
    spike_frames = spike_frames[(spike_frames > pre_frames) & (spike_frames < n_frames - post_frames)]
    n_spikes = len(spike_frames)
    peri_movie_length = pre_frames + post_frames + 1

    for spk_frame in spike_frames:
        peri_movies.append(movie[spk_frame - pre_frames: spk_frame + post_frames + 1])
        peri_traces.append(roi_traces[spk_frame - pre_frames: spk_frame + post_frames + 1, :])

    peri_movies = np.stack(peri_movies)
    assert peri_movies.shape == (n_spikes, peri_movie_length, n_row, n_col), f"Peri movies shape is wrong: {peri_movies.shape}"

    # peri_traces = np.stack(peri_traces)
    # assert peri_traces.shape == (n_spikes, peri_movie_length, n_rois), "Peri traces shape is wrong"

    # Compute spike-triggered average
    avg_spike_movie = peri_movies.mean(axis=0)  # shape: (t, H, W)
    # avg_spike_trace = peri_traces.mean(axis=0)  # shape: (t, n_rois)

    # ani = play_movie(avg_spike_movie)
    return avg_spike_movie


def click_and_plot_traces(movie, n_traces=3):
    fig, ax = plt.subplots()
    ax.imshow(movie[movie.shape[0] // 2], cmap='gray')
    ax.set_title("Click to select pixels (close window when done)")
    points = plt.ginput(n_traces, timeout=0)
    plt.close(fig)

    points = [(int(y), int(x)) for x, y in points]
    fig, ax = plt.subplots()
    for (y, x) in points:
        trace = movie[:, y, x]
        trace_norm = (trace - trace.min()) / (trace.max() - trace.min())
        ax.plot(trace_norm, label=f'({x},{y})')
    ax.set_title('Normalized intensity profiles of selected pixels')
    ax.legend()
    plt.show()

