import os
import numpy as np
from matplotlib import cm, pyplot as plt

from basic_utils import save_movie_as_mp4
from constants import *


def flatten_movie(movie):
    n_frames, n_row, n_col = movie.shape  # T, H, W
    movie_mean = np.mean(movie, axis=0)
    movie_centered = movie - movie_mean

    movie_flat = movie_centered.reshape(n_frames, -1)  # [T, H*W]
    return movie_flat


def get_movie_eigenvectors(movie, is_flat=False):
    movie_flat = movie if is_flat else flatten_movie(movie)
    cov = movie_flat @ movie_flat.T  # shape [T, T]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    eigenvectors = eigenvectors[:, ::-1]
    eigenvectors[:, 0] *= -1
    eigenvectors[:, 2] *= -1

    return eigenvectors


def get_n_eigenimages(movie, n_components):
    n_frames, n_row, n_col = movie.shape
    movie_flat = flatten_movie(movie)
    eigenvectors = get_movie_eigenvectors(movie_flat, is_flat=True)
    proj_n = movie_flat.T @ eigenvectors[:, :n_components]
    eigenimages_n = proj_n.reshape(n_row, n_col, n_components)

    return eigenimages_n


def visualize_eigenimages(eigenimages):
    n_components = eigenimages.shape[-1]
    n_cols = int(np.ceil(np.sqrt(n_components)))
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Flatten axs to 1D list for easy indexing
    axs = np.array(axs).reshape(-1)

    for i in range(n_rows * n_cols):
        ax = axs[i]
        if i < n_components:
            ax.imshow(eigenimages[:, :, i], cmap='gray')
            ax.set_title(f"Eigen-image {i + 1}")
        ax.axis('off')

    plt.suptitle("Top PCA Spatial Components")
    plt.tight_layout()
    plt.show()


def get_movie_pca(peri_spike_movie, is_plot=True, save_dir=PCA_RESULTS_DIR):
    n_frames, n_row, n_col = peri_spike_movie.shape  # T, H, W
    movie_flat = flatten_movie(peri_spike_movie)
    eigenvectors = get_movie_eigenvectors(movie_flat, is_flat=True)

    # Visualize top 9 components
    proj_9 = movie_flat.T @ eigenvectors[:, :9]
    eigenimages_9 = proj_9.reshape(n_row, n_col, 9)
    visualize_eigenimages(eigenimages_9)

    proj = movie_flat.T @ eigenvectors[:, :3]  # shape: [H*W, 3]
    movie_pca_flat = proj @ eigenvectors[:, :3].T  # [H*W, T]
    movie_pca = movie_pca_flat.T.reshape(peri_spike_movie.shape)

    movie_pca -= movie_pca[0]
    movie_pca /= np.max(np.abs(movie_pca))

    if is_plot:
        amp_img = np.std(movie_pca, axis=0)  # [H, W]
        amp_img_norm = (amp_img - amp_img.min()) / (amp_img.max() - amp_img.min())

        color_movie = np.zeros((n_frames, n_row, n_col, 3))  # RGB movie

        for t in range(n_frames):
            jet = cm.get_cmap('jet')
            color_frame = jet((movie_pca[t] - 0.0) / 0.4)[..., :3]  # normalize to 0–0.4
            base_gray = 0.5 * amp_img_norm[..., np.newaxis]
            color_movie[t] = base_gray + 3 * amp_img_norm[..., np.newaxis] * color_frame
            color_movie[t] = np.clip(color_movie[t], 0, 1)

        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        for i in range(6):
            axs[i // 3, i % 3].imshow(color_movie[i + 48])
            axs[i // 3, i % 3].axis('off')
        plt.suptitle("PCA-filtered response montage")
        plt.show()

        fig, ax = plt.subplots()
        im = ax.imshow(color_movie[0])
        title = ax.set_title("0 ms")
        ax.axis('off')

        for i in range(1, n_frames):
            im.set_data(color_movie[i])
            title.set_text(f"{i} ms")
            plt.pause(0.05)

        save_path = os.path.join(save_dir, "pca_movie.mp4")
        save_movie_as_mp4(color_movie, filename=save_path, fps=10)

    return movie_pca
