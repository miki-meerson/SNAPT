import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.widgets import Slider
from scipy.signal import find_peaks

from clean_pipeline import low_pass_filter
from experiment_constants import SAMPLING_RATE


def detect_spikes_in_roi(trace):
    spikes = []
    trace_high_pass = low_pass_filter(trace, low_freq_to_filter=3)
    trace_high_pass_shifted = trace_high_pass - trace_high_pass.min() + 1

    assert trace_high_pass_shifted.min() > 0, f"Values under 0 exist in high-pass filtered trace: smallest value is {trace_high_pass_shifted.min()}"

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

def extract_roi_sta(roi_trace, window_size):
    roi_trace_norm = (roi_trace - roi_trace.min()) / (roi_trace.max() - roi_trace.min())
    spike_indices = detect_spikes_in_roi(roi_trace_norm)

    peri_spike_traces = []
    n_frames = len(roi_trace)

    for idx in spike_indices:
        if window_size < idx < n_frames - window_size:
            trace_snippet = roi_trace_norm[idx - window_size: idx + window_size + 1]
            peri_spike_traces.append(trace_snippet)


    if not peri_spike_traces:
        print("No valid spikes found for spike-triggered averaging.")
        return None, None, spike_indices

    peri_spike_traces = np.array(peri_spike_traces)
    sta = np.mean(peri_spike_traces, axis=0)
    assert len(sta) == 2*window_size+1, f"STA trace length is wrong: should be {2*window_size+1} but is {len(sta)}"
    return sta, spike_indices


def spike_triggered_average_analysis(roi_traces):
    n_rois = roi_traces.shape[0]

    sta_per_roi = []
    spike_indices_per_roi = []
    window_size_ms = 25
    window_size = int(window_size_ms * SAMPLING_RATE / 1000)  # in frames

    for roi_trace in roi_traces:
        sta, spike_indices = extract_roi_sta(roi_trace, window_size)

        sta_per_roi.append(sta)
        spike_indices_per_roi.append(spike_indices)

    fig, ax = plt.subplots(n_rois, 1, figsize=(20, 20))
    fig.tight_layout()

    for roi_index in range(n_rois):
        t = np.arange(-window_size, window_size + 1)
        ax[roi_index].plot(t, sta_per_roi[roi_index], label=f'Spike-Triggered Average - ROI {roi_index + 1}')
        ax[roi_index].set_xlabel("Frames relative to spike")
        ax[roi_index].set_ylabel("Normalized fluorescence")
        ax[roi_index].axvline(0, color='gray', linestyle='--', label='Spike time')
        ax[roi_index].legend()

    plt.tight_layout()
    plt.show()

    return spike_indices_per_roi, sta_per_roi


def build_spike_triggered_movie(spike_indices, movie, roi_mask, window_size):
    sta_frames = []
    n_frames = movie.shape[0]

    for spike_time in spike_indices:
        if window_size < spike_time < n_frames - window_size:

            # Broadcast mask to all frames
            window = movie[spike_time - window_size:spike_time + window_size + 1].copy()

            if roi_mask:
                window[:, ~roi_mask] = 0

            sta_frames.append(window)

    if sta_frames:
        sta_frames = np.array(sta_frames)
        sta_movie = np.mean(sta_frames, axis=0)
        assert sta_movie.shape[0] == 2 * window_size + 1, f"STA movie has wrong length: {sta_movie.shape[0]} vs expected {2 * window_size + 1}"
        return sta_movie
    else:
        return np.zeros((2 * window_size + 1, movie.shape[1], movie.shape[2]))