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

def extract_spike_triggered_average(roi_trace):
    roi_trace_norm = (roi_trace - roi_trace.min()) / (roi_trace.max() - roi_trace.min())

    spike_indices = detect_spikes_in_roi(roi_trace_norm)
    return spike_indices


