import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm

from pca_utils import get_n_eigenimages, get_movie_pca, get_movie_eigenvectors, flatten_movie
from roi_analyzer import get_average_image

def _mat2gray(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)


def compute_dff_from_pca(peri_spike_movie, sigma=1.5):
    """ Computes a dF/F image by dividing the first PCA image by a smoothed average image """
    avg_img = get_average_image(peri_spike_movie)
    eig_img_1 = get_n_eigenimages(peri_spike_movie, n_components=1)

    avg_img_smooth = gaussian_filter(avg_img, sigma=sigma)
    eig_img_1 = np.squeeze(eig_img_1)
    dff_img = eig_img_1 / (avg_img_smooth + 1e-8)  # Avoid divide-by-zero

    # Normalize for visualization
    dff_norm = (dff_img - np.min(dff_img)) / (np.max(dff_img) - np.min(dff_img))

    jet = plt.get_cmap('jet')
    dff_colored_modulated = jet(dff_norm)[..., :3]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(dff_img, cmap='gray')
    plt.title('Raw dF/F')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dff_colored_modulated)
    plt.title('Colored dF/F')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return dff_colored_modulated


def compute_spike_amp_image(peri_spike_movie):
    spike_movie_pca = get_movie_pca(peri_spike_movie, is_plot=False)
    pulse_frame = peri_spike_movie.shape[0] // 2

    post_mean = spike_movie_pca[pulse_frame + 1:pulse_frame + 3, :, :].mean(axis=0)
    pre_mean = spike_movie_pca[pulse_frame - 3:pulse_frame, :, :].mean(axis=0)
    baseline = spike_movie_pca[0:(pulse_frame // 2), :, :].mean(axis=0)

    spike_amp_img = (post_mean - pre_mean) / (pre_mean - baseline + 1e-8) # Avoid divide-by-zero

    # Normalize and color
    spike_amp_norm = (spike_amp_img - np.min(spike_amp_img)) / (np.max(spike_amp_img) - np.min(spike_amp_img))

    jet = cm.get_cmap('jet')
    spike_amp_color_modulated = jet(spike_amp_norm)[..., :3]

    plt.figure(figsize=(6, 6))
    plt.imshow(spike_amp_color_modulated)
    plt.title("Spike Amplitude Map (modulated by avgImg)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return spike_amp_color_modulated


def compute_pc_ratios(eigenvectors):
    traces_pc2 = []
    traces_pc3 = []

    # Influence of 2nd PC on 1st
    plt.figure(figsize=(8, 5))
    for j in np.arange(-2, 2.2, 0.2):  # from -2 to 2 with step 0.2
        trace = eigenvectors[:, 0] + j * eigenvectors[:, 1] + 3 * j  # offset vertically
        traces_pc2.append(trace)
        plt.plot(trace, label=f"j={j:.1f}")
    plt.title("PC1 + j·PC2 (with vertical offset)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Influence of 3rd PC on 1st
    plt.figure(figsize=(8, 5))
    for j in np.arange(-0.5, 0.35, 0.05):  # from -0.5 to 0.3
        trace = eigenvectors[:, 0] + j * eigenvectors[:, 2] + 3 * j
        traces_pc3.append(trace)
        plt.plot(trace, label=f"j={j:.2f}")
    plt.title("PC1 + j·PC3 (with vertical offset)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return traces_pc2, traces_pc3


def compute_spike_shift_image(eigenvectors, eigenimages, pulse_frame):
    """
    eigImgs: shape [H, W, 3], PCA spatial components
    V: shape [T, 3], temporal eigenvectors
    pulse_frame: int, center frame index
    """
    pc1 = eigenimages[:, :, 0]
    pc2 = eigenimages[:, :, 1]

    spike_shift_img = -pc2 / (pc1 + 1e-8)  # avoid division by zero

    # Normalize and color map
    spike_shift_norm = np.clip((spike_shift_img + 0.5) / 1.0, 0, 1)  # map from [-0.5, 0.5] to [0, 1]
    jet = cm.get_cmap('jet')
    spike_shift_colored = jet(spike_shift_norm)[..., :3]  # drop alpha channel

    # Brightness modulation using PC1
    brightness = _mat2gray(pc1) * 1.3  # optional boost
    spike_shift_color_modulated = spike_shift_colored * brightness[..., np.newaxis]

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    t_range = np.arange(-pulse_frame, len(eigenvectors) - pulse_frame)
    axs[0, 0].plot(t_range, eigenvectors[:, :2])
    axs[0, 0].set_xlim([-50, 50])
    axs[0, 0].set_title("Temporal PCs")
    axs[0, 0].legend(["PC1", "PC2"])

    axs[1, 0].imshow(spike_shift_color_modulated)
    axs[1, 0].set_title("PC2 / PC1 (Color modulated)")
    axs[1, 0].axis("off")

    axs[0, 1].axis('off')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return spike_shift_img, spike_shift_color_modulated


def compute_spike_width_image(eigenvectors, eigenimages, pulse_frame):
    pc1 = eigenimages[:, :, 0]
    pc3 = eigenimages[:, :, 2]

    # Compute spike width image
    spike_width_img = -pc3 / (pc1 + 1e-8)

    # Normalize and apply colormap
    width_norm = np.clip((spike_width_img + 0.5) / 1.0, 0, 1)
    jet = cm.get_cmap('jet')
    spike_width_col = jet(width_norm)[..., :3]

    # Brightness modulation using PC1
    brightness = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    brightness *= 1.3
    spike_width_color_modulated = spike_width_col * brightness[..., np.newaxis]

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    t_range = np.arange(-pulse_frame, len(eigenvectors) - pulse_frame)
    axs[0, 1].plot(t_range, eigenvectors[:, [0, 2]])
    axs[0, 1].set_xlim([-50, 50])
    axs[0, 1].set_title("Temporal PCs: PC1 & PC3")
    axs[0, 1].legend(["PC1", "PC3"])

    axs[1, 1].imshow(spike_width_color_modulated)
    axs[1, 1].set_title("PC3 / PC1 (Spike Width Image)")
    axs[1, 1].axis("off")

    axs[0, 0].axis('off')
    axs[1, 0].axis('off')

    plt.tight_layout()
    plt.show()

    return spike_width_img, spike_width_color_modulated


def validations_pipeline(peri_spike_movie):
    eigenvectors = get_movie_eigenvectors(peri_spike_movie)
    eigenimages = get_n_eigenimages(peri_spike_movie, n_components=3)
    pulse_frame = peri_spike_movie.shape[0] // 2

    dff_color_modulated = compute_dff_from_pca(peri_spike_movie)
    spike_amp_color_modulated = compute_spike_amp_image(peri_spike_movie)
    traces_pc2, traces_pc3 = compute_pc_ratios(eigenvectors)

    spike_shift_img, spike_shift_col_mod = compute_spike_shift_image(eigenvectors, eigenimages, pulse_frame)
    spike_width_img, spike_width_color_modulated = compute_spike_width_image(eigenvectors, eigenimages, pulse_frame)