import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings
import time


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


def shift_kernel(beta, kernel, t):
    a, b, c, dt = beta
    shifted_t = (t - dt) * c
    f = interp1d(t, kernel, bounds_error=False, fill_value='extrapolate')
    return a * f(shifted_t) + b


def fit_pixel_trace(kernel, trace, t, p0=[1, 0, 1, 0]):
    def model(t, a, b, c, dt):
        return shift_kernel([a, b, c, dt], kernel, t)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            [popt, pcov] = curve_fit(model, t, trace, p0=p0, maxfev=1000)
            residuals = trace - model(t, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((trace - np.mean(trace))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            return popt, pcov, residuals, r_squared, True
    except Exception:
        return [np.nan]*4, None, None, 0, False



def get_temporal_kernel(roi_sta_movie):

    avg_trace = np.mean(roi_sta_movie, axis=(1, 2))
    center_frame = roi_sta_movie.shape[0] // 2
    kernel = avg_trace[(center_frame - 13):(center_frame + 17)]  # 30-frame window; adjust as needed
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # Normalize to [0,1]
    assert kernel.shape == (30,), f"Kernel shape is incorrect: {kernel.shape}"

    return kernel

def get_peri_spike_movie(roi_sta_movie):
    center_frame = roi_sta_movie.shape[0] // 2

    # Extract peri-spike movie snippet aligned with kernel length
    peri_spike_movie = roi_sta_movie[(center_frame - 13):(center_frame + 17), :, :]
    peri_spike_movie_smooth = gaussian_filter(peri_spike_movie, sigma=(1, 1, 0))
    peri_spike_movie_norm = (peri_spike_movie_smooth - np.min(peri_spike_movie_smooth)) / (np.max(peri_spike_movie_smooth) - np.min(peri_spike_movie_smooth))

    return peri_spike_movie_norm


def analyze_spike_triggered_movie(roi_vertices, kernel, peri_spike_movie_norm):

    # Compute spatial maps of correlation and related metrics
    v_out, corr_img, weight_img, offset_img = extract_v(peri_spike_movie_norm, kernel)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(kernel, label="Temporal Kernel (Avg ROI Trace)")

    # Plot example pixels traces
    random_indices = np.random.choice(len(roi_vertices), 30, replace=False)
    random_pixels = [roi_vertices[i] for i in random_indices]

    for (x, y) in random_pixels:
        plt.plot(peri_spike_movie_norm[:, int(y), int(x)], alpha=0.3)

    plt.title('Kernel vs. ROI Pixel Traces')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.imshow(corr_img, cmap='hot')
    plt.colorbar()
    plt.title("Correlation Map")

    plt.subplot(2, 2, 3)
    plt.imshow(weight_img, cmap='hot')
    plt.colorbar()
    plt.title("Weight (Signal Strength) Map")

    plt.subplot(2, 2, 4)
    plt.imshow(offset_img, cmap='hot')
    plt.colorbar()
    plt.title("Offset (Baseline Fluorescence) Map")

    plt.tight_layout()
    plt.show()

    return kernel, peri_spike_movie_norm


def snapt_algorithm(kernel, sta_movie_norm):
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

        print(f"Completed row {y+1} of {n_row}")

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

    plt.figure()
    plt.imshow(good_fit, cmap='gray')
    plt.colorbar()
    plt.title('Fit success')

    plt.figure()
    plt.imshow(amp_img, cmap='hot')
    plt.colorbar()
    plt.title('Amplitude image')

    plt.figure()
    plt.imshow(width_img, cmap='jet')
    plt.colorbar()
    plt.title('Spike width image')

    plt.figure()
    plt.imshow(dt_img, cmap='jet')
    plt.colorbar()
    plt.title('Spike delay image')

    plt.figure()
    plt.imshow(rsq_mat, cmap='plasma', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R² map')

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