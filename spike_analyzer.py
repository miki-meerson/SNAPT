import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from clean_pipeline import low_pass_filter
from experiment_constants import SAMPLING_RATE


def detect_spikes(roi_traces, movie):
    n_frames, n_rois = roi_traces.shape

    f = roi_traces[:, 1] # first ROI should be soma
    f_norm = (f - f.min()) / (f.max() - f.min())
    f_high_pass = low_pass_filter(f_norm, low_freq_to_filter=2)
    f_high_pass_shifted = f_high_pass + 1 # Avoid negative values

    peaks, _ = find_peaks(f_high_pass_shifted, height=1.05, distance=10)
    print(f"Number of spikes detected: {len(peaks)}")

    _, nrow, ncol = movie.shape
    spike_window_ms = 50  # ms
    spike_window_frames = int(spike_window_ms * SAMPLING_RATE / 1000)
    # assert spike_window_frames == 50, "ms to frames conversion is wrong"

    spikes = np.full((2 * spike_window_frames + 1, n_rois, len(peaks)), np.nan)

    spike_movie = np.zeros((2 * spike_window_frames + 1, nrow, ncol))
    c = 0  # spike counter for averaging

    for i, spk in enumerate(peaks):
        if spike_window_frames < spk < n_frames - spike_window_frames:
            fs = roi_traces[(spk - spike_window_frames):(spk + spike_window_frames + 1), :]  # shape (2*Pulse+1, n_rois)
            f0 = fs[:10, :].mean(axis=0)  # baseline is mean of first 10 frames in window
            spikes[:, :, i] = fs - f0[np.newaxis, :]
            spike_movie += movie[(spk - spike_window_frames):(spk + spike_window_frames + 1), :, :]
            c += 1

    spike_movie_avg = spike_movie / c

    # Plot an example ROI trace and spikes
    plt.figure()
    plt.plot(f_norm, label='Normalized ROI1 trace')
    plt.plot(peaks, f_norm[peaks], 'ro', label='Detected spikes')
    plt.legend()
    plt.title('Spike detection on ROI1 trace')
    plt.show()

    start_frame = spike_window_frames - 10
    end_frame = spike_window_frames + 10

    fig, axes = plt.subplots(1, end_frame - start_frame + 1, figsize=(15, 2))
    for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        axes[i].imshow(spike_movie_avg[:, :, frame_idx], cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Average spike-triggered movie frames')
    plt.show()


