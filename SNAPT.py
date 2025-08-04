import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings

from globals import *


def normalize_to_unit_vector(vector):
    offset = np.mean(vector)
    vector_centered = vector - offset
    norm = np.linalg.norm(vector_centered)

    if norm == 0:
        return np.zeros_like(vector), offset, norm

    unit_vector = vector_centered / norm
    return unit_vector, offset, norm


def compute_kernel_projection_maps(movie, kernel):
    n_frames, n_row, n_col = movie.shape
    projection_img = np.zeros((n_row, n_col))
    corr_img = np.zeros((n_row, n_col))
    weight_img = np.zeros((n_row, n_col))
    offset_img = np.zeros((n_row, n_col))

    # Normalize the kernel to be a unit vector
    kernel_unit_vec, _, norm = normalize_to_unit_vector(kernel)
    assert norm != 0, "Kernel has zero norm after mean subtraction (flat or empty kernel)."

    for y in range(n_row):
        for x in range(n_col):
            pixel_trace = movie[:, y, x]

            # Normalize the pixel trace to be a unit vector
            pixel_trace_unit_vec, offset, norm = normalize_to_unit_vector(pixel_trace)
            offset_img[y, x] = offset

            if norm == 0:
                corr_img[y, x] = np.nan
                projection_img[y, x] = np.nan
                weight_img[y, x] = np.nan
            else:
                corr = np.dot(pixel_trace_unit_vec, kernel_unit_vec)
                corr_img[y, x] = corr
                projection_img[y, x] = corr * norm
                weight_img[y, x] = norm

    return projection_img, corr_img, weight_img, offset_img


def fit_pixel_trace(kernel, trace, t, p0=[1, 0, 1, 0]):
    def model(t, a, b, c, dt):
        shifted_t = (t - dt) * c
        f = interp1d(t, kernel, bounds_error=False, fill_value='extrapolate')
        return a * f(shifted_t) + b

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            [fitted_params, fitted_params_cov] = curve_fit(model, t, trace, p0=p0, maxfev=2000)
            residuals = trace - model(t, *fitted_params)


            ssr = np.sum(residuals**2)  # sum of squared residuals (total error between the trace and the model)
            sst = np.sum((trace - np.mean(trace))**2)  # total sum of squares (total variance in the trace)
            r_squared = 1 - ssr / sst if sst != 0 else 0

            return fitted_params, fitted_params_cov, residuals, r_squared, True

    except Exception as e:
        print("Fit failed:", e)
        return [np.nan]*4, None, None, 0, False


def get_temporal_kernel(roi_sta_movie):

    avg_trace = np.mean(roi_sta_movie, axis=(1, 2))
    center_frame = roi_sta_movie.shape[0] // 2
    kernel = avg_trace[(center_frame - KERNEL_PRE_FRAMES):(center_frame + KERNEL_POST_FRAMES)]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # Normalize to [0,1]

    return kernel


def get_peri_spike_movie(roi_sta_movie):
    center_frame = roi_sta_movie.shape[0] // 2

    # Extract peri-spike movie snippet aligned with kernel length
    peri_spike_movie = roi_sta_movie[(center_frame - KERNEL_PRE_FRAMES):(center_frame + KERNEL_POST_FRAMES), :, :]
    peri_spike_movie_smooth = gaussian_filter(peri_spike_movie, sigma=(0, 1, 1))
    peri_spike_movie_norm = (peri_spike_movie_smooth - np.min(peri_spike_movie_smooth)) / (np.max(peri_spike_movie_smooth) - np.min(peri_spike_movie_smooth))

    return peri_spike_movie_norm


def analyze_spike_triggered_movie(roi_mask, kernel, peri_spike_movie_norm, save_dir):

    # Compute spatial maps of correlation and related metrics
    v_out, corr_img, weight_img, offset_img = compute_kernel_projection_maps(peri_spike_movie_norm, kernel)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(kernel, label="Temporal Kernel (Avg ROI Trace)")

    # Plot example pixels traces
    ys, xs = np.where(roi_mask)
    indices = np.random.choice(len(xs), min(30, len(xs)), replace=False)
    random_pixels = list(zip(xs[indices], ys[indices]))

    for (x, y) in random_pixels:
        plt.plot(peri_spike_movie_norm[:, int(y), int(x)], alpha=0.3)

    plt.legend()
    plt.title('Kernel vs. ROI Pixel Traces')

    plt.subplot(2, 2, 2)
    plt.imshow(corr_img, cmap='seismic')
    plt.colorbar()
    plt.title("Correlation Map")

    plt.subplot(2, 2, 3)
    plt.imshow(weight_img, cmap='hot')
    plt.colorbar()
    plt.title("Weight (Signal Strength) Map")

    plt.subplot(2   , 2, 4)
    plt.imshow(offset_img, cmap='hot')
    plt.colorbar()
    plt.title("Offset (Baseline Fluorescence) Map")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_summary.png"))  # full panel
    plt.show()

    return kernel, peri_spike_movie_norm


def snapt_algorithm(kernel, sta_movie_norm, save_dir):
    t = np.arange(kernel.shape[0])
    n_frames, n_row, n_col = sta_movie_norm.shape

    beta_mat = np.zeros((n_row, n_col, 4))
    rsq_mat = np.zeros((n_row, n_col))
    good_fit = np.zeros((n_row, n_col), dtype=bool)

    for y in range(n_row):
        for x in range(n_col):
            trace = sta_movie_norm[:, y, x]
            popt, pcov, residuals, rsq, success = fit_pixel_trace(kernel, trace, t)

            beta_mat[y, x, :] = popt
            rsq_mat[y, x] = rsq
            good_fit[y, x] = success
        print(f"Completed row {y+1}/{n_row}")

    amp_img = beta_mat[:, :, 0]
    width_img = 1.0 / beta_mat[:, :, 2]
    dt_img = beta_mat[:, :, 3]

    # Constraints
    min_a = 0
    min_c = 0.2
    max_c = 4
    min_dt = -4
    max_dt = 4

    # Apply constraints
    good_pix = (
            (amp_img > min_a) &
            (width_img > min_c) & (width_img < max_c) &
            (dt_img > min_dt) & (dt_img < max_dt)
    )

    amp_img = amp_img * good_pix
    width_img = np.clip(width_img, min_c, max_c)
    dt_img = np.clip(dt_img, min_dt, max_dt)

    plt.figure(figsize=(16, 10))

    plt.subplot(3, 2, 1)
    plt.imshow(good_fit, cmap='gray')
    plt.colorbar()
    plt.title('Fit success')

    plt.subplot(3, 2, 2)
    plt.imshow(amp_img, cmap='hot')
    plt.colorbar()
    plt.title('Amplitude image')

    plt.subplot(3, 2, 3)
    plt.imshow(width_img, cmap='jet')
    plt.colorbar()
    plt.title('Spike width image')

    plt.subplot(3, 2, 4)
    plt.imshow(dt_img, cmap='jet')
    plt.colorbar()
    plt.title('Spike delay image')


    plt.subplot(3, 2, 5)
    plt.imshow(rsq_mat, cmap='plasma', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R² map')
    plt.savefig(os.path.join(save_dir, "r2_map.png"))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "snapt_summary.png"))  # full panel
    plt.show()

    return beta_mat, rsq_mat, good_fit


def get_movie_pca(peri_spike_movie):
    n_frames, n_row, n_col = peri_spike_movie.shape # T, H, W
    movie_mean = np.mean(peri_spike_movie, axis=0)
    movie_centered = peri_spike_movie - movie_mean

    movie_flat = movie_centered.reshape(n_frames, -1)  # [T, H*W]

    # Covariance matrix
    cov = movie_flat @ movie_flat.T  # shape [T, T]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    eigenvectors = eigenvectors[:, ::-1]
    eigenvectors[:, 0] *= -1
    eigenvectors[:, 2] *= -1

    proj = movie_flat.T @ eigenvectors[:, :3]  # shape: [H*W, 3]
    eigen_images = proj.reshape(n_row, n_col, 3)  # 3 eigen images

    # Visualize top 9 components
    proj_9 = movie_flat.T @ eigenvectors[:, :9]
    eigen_images_9 = proj_9.reshape(n_row, n_col, 9)
    # TODO Add eigenimage display for debugging in Python.

    movie_pca_flat = proj @ eigenvectors[:, :3].T  # [H*W, T]
    movie_pca = movie_pca_flat.T.reshape(peri_spike_movie.shape)

    movie_pca -= movie_pca[0]
    movie_pca /= np.max(np.abs(movie_pca))

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

    for i in range(n_frames):
        plt.imshow(color_movie[i])
        plt.title(f"{i} ms")
        plt.axis('off')
        plt.pause(0.05)

    return movie_pca


def snapt_pipeline(sta_movie, roi_mask, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    kernel = get_temporal_kernel(sta_movie)
    peri_spike_movie_norm = get_peri_spike_movie(sta_movie)

    # Kernel visualization
    analyze_spike_triggered_movie(roi_mask, kernel, peri_spike_movie_norm, save_dir=results_dir)

    # run SNAPT
    snapt_algorithm(kernel, peri_spike_movie_norm, save_dir=results_dir)
