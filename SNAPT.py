import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from roi_analyzer import get_average_image
from spike_analyzer import play_movie


# ---- clicky3 replacement ----
def clicky3(movie, avg_img):

    fig, ax = plt.subplots()
    ax.imshow(avg_img, cmap='gray')
    coords = []

    def onclick(event):
        if event.inaxes == ax:
            coords.append((int(event.ydata), int(event.xdata)))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('Click to select 3 pixels (Soma & Dendrites), then close window')
    plt.show()

    fig.canvas.mpl_disconnect(cid)

    time_traces = []
    for y, x in coords[:3]:
        trace = movie[:, y, x]
        time_traces.append(trace)

    return np.array(coords[:3]), np.stack(time_traces, axis=1)

# ---- extractV function (cross-correlation approach) ----
def extract_v(movie, kernel):
    t, h, w = movie.shape
    kernel = kernel - np.mean(kernel)
    kernel = kernel / np.linalg.norm(kernel)

    v_out = np.zeros((h, w))
    corr_img = np.zeros((h, w))
    weight_img = np.zeros((h, w))
    offset_img = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            ts = movie[:, i, j]
            ts = ts - np.mean(ts)
            offset_img[i, j] = np.mean(ts)
            norm = np.linalg.norm(ts)
            if norm == 0:
                continue
            ts_norm = ts / norm
            corr = np.dot(ts_norm, kernel)
            corr_img[i, j] = corr
            v_out[i, j] = corr * norm
            weight_img[i, j] = norm

    return v_out, corr_img, weight_img, offset_img


def main_analysis(spike_movie, original_movie):
    avg_img = get_average_image(original_movie)

    # Get intensity time traces by clicking on soma and dendrites
    coords, tmp = clicky3(spike_movie, avg_img)

    plt.figure()
    for i in range(tmp.shape[1]):  # loop over pixels
        trace = (tmp[:, i] - tmp[:, i].min()) / (np.ptp(tmp[:, i]))
        plt.plot(trace, label=f'Pixel {i + 1}')
    plt.title("Normalized Intensity Profiles")
    plt.legend()
    plt.show()

    # Kernel extraction
    pulse = spike_movie.shape[0] // 2
    kernel = np.mean(tmp, axis=1)  # average of soma + dendrite trace
    kernel = kernel[(pulse-13):(pulse+17)] # TODO find correct kernel length
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # mat2gray
    assert kernel.shape == (30,), f"Kernel shape is incorrect: {kernel.shape}"

    # Filter peri-spike movie and normalize
    spike_mov_fit = spike_movie[(pulse - 13):(pulse + 17), :, :]
    spike_mov_fit = gaussian_filter(spike_mov_fit, sigma=(1, 1, 0))
    spike_mov_fit = (spike_mov_fit - np.min(spike_mov_fit)) / (np.max(spike_mov_fit) - np.min(spike_mov_fit))

    # Fit using extractV
    v_out, corr_img, weight_img, offset_img = extract_v(spike_mov_fit, kernel)

    # Show results
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(kernel, label="Kernel")
    plt.plot(spike_mov_fit[:, coords[0][0], coords[0][1]], label="Soma Pixel")
    plt.title('Kernel vs. Pixel Trace')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.imshow(corr_img, cmap='hot')
    plt.title("Correlation Map")

    plt.subplot(2, 2, 3)
    plt.imshow(weight_img, cmap='hot')
    plt.title("Weight Map")

    plt.subplot(2, 2, 4)
    plt.imshow(offset_img, cmap='hot')
    plt.title("Offset Map")

    plt.tight_layout()
    plt.show()

    # Optional: play the movie
    play_movie(spike_mov_fit, interval=100)