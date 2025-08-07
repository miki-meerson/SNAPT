import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings

from globals import *


def compute_kernel_projection_maps(movie, kernel):
    n_frames, n_row, n_col = movie.shape
    avg_img = np.mean(movie, axis=0)
    avg_v = np.mean(kernel)
    voltage_deviation = kernel - avg_v

    # Subtract baseline image
    movie_centered = movie - avg_img[np.newaxis, :, :]

    # Compute correlation map (regression slope)
    voltage_deviation_reshaped = voltage_deviation[:, np.newaxis, np.newaxis]  # shape (T, 1, 1)
    voltage_deviation_mat = np.broadcast_to(voltage_deviation_reshaped, (n_frames, n_row, n_col))
    corr_img = np.mean(voltage_deviation_mat * movie_centered, axis=0) / np.mean(voltage_deviation ** 2)

    # Avoid division by zero in later step
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_img_safe = np.where(corr_img == 0, np.nan, corr_img)

        # Estimate dV at each pixel
        projection_img = movie_centered / corr_img_safe[np.newaxis, :, :]  # shape (T, H, W)

        # Residual variance (noise image)
        residual = projection_img - voltage_deviation_mat
        sigma_img = np.mean(residual ** 2, axis=0)

        # Weight image = 1 / variance
        weight_img = np.where(np.isnan(sigma_img), 0, 1.0 / sigma_img)
        weight_img /= np.mean(weight_img[weight_img > 0])  # normalize to mean 1

        # Set invalid estimates to 0
        projection_img = np.nan_to_num(projection_img)

    # Compute final voltage estimate (Vout): weighted average across all pixels
    weighted_projection = projection_img * weight_img[np.newaxis, :, :]
    v_out = np.sum(weighted_projection, axis=(1, 2)) / np.sum(weight_img)

    # Offset image
    offset_img = avg_img - avg_v * corr_img

    return corr_img, weight_img, offset_img, projection_img, v_out


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


def get_temporal_kernel(roi_trace, spike_indices):
    kernel_segments = []
    for spike_time in spike_indices:
        if PULSE_FRAMES < spike_time < len(roi_trace) - PULSE_FRAMES:
            segment = roi_trace[spike_time - PULSE_FRAMES : spike_time + PULSE_FRAMES + 1]
            kernel_segments.append(segment)

    kernel = np.mean(kernel_segments, axis=0)
    kernel = kernel[(PULSE_FRAMES - KERNEL_PRE_FRAMES):(PULSE_FRAMES + KERNEL_POST_FRAMES + 1)]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    return kernel


def get_peri_spike_movie(spike_movie, smooth_sigma=1.5):
    center = PULSE_FRAMES

    # Extract peri-spike movie snippet aligned with kernel length
    peri_spike_movie = spike_movie[(center - KERNEL_PRE_FRAMES):(center + KERNEL_POST_FRAMES + 1), :, :]
    peri_spike_movie_smooth = gaussian_filter(peri_spike_movie, sigma=(0, 1.5, 1.5))
    peri_spike_movie_norm = (peri_spike_movie_smooth - np.min(peri_spike_movie_smooth)) / (np.max(peri_spike_movie_smooth) - np.min(peri_spike_movie_smooth))

    return peri_spike_movie_norm


def analyze_spike_triggered_movie(kernel, peri_spike_movie_norm, save_dir):

    # Compute spatial maps of correlation and related metrics
    corr_img, weight_img, offset_img, projection_img, v_out = compute_kernel_projection_maps(peri_spike_movie_norm, kernel)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(kernel, label="Temporal Kernel (Avg ROI Trace)")

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


def snapt_algorithm(kernel, peri_spike_movie, save_dir):
    t = np.arange(kernel.shape[0])
    n_frames, n_row, n_col = peri_spike_movie.shape

    beta_mat = np.zeros((n_row, n_col, 4))
    rsq_mat = np.zeros((n_row, n_col))
    good_fit = np.zeros((n_row, n_col), dtype=bool)

    for y in range(n_row):
        for x in range(n_col):
            trace = peri_spike_movie[:, y, x]
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
    plt.imshow(rsq_mat, cmap='plasma')
    plt.colorbar()
    plt.title('R² map')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "snapt_summary.png"))  # full panel
    plt.show()

    def weighted_color_map(image, weight, cmap='jet', vmin=0, vmax=1):
        norm_img = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        colormap = cm.get_cmap(cmap)
        color_img = colormap(norm_img)[..., :3]
        weight_scaled = (weight - np.nanmin(weight)) / (np.nanmax(weight) - np.nanmin(weight))
        amp_scale = 3 * weight_scaled - 0.1
        amp_scale = np.clip(amp_scale, 0, 1)
        return color_img * amp_scale[..., np.newaxis]

    maps_to_weight = [dt_img, width_img]
    display_names = ["Delay", "Width"]

    for i, map_to_weight in enumerate(maps_to_weight):
        weighted_map = weighted_color_map(map_to_weight, amp_img, vmin=0, vmax=1)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(weighted_map)
        ax.set_title(f'{display_names[i]} map (amplitude-weighted, jet colormap)')
        ax.axis('off')

        norm = Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=f'{display_names[i]} map')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"weighted {display_names[i]}.png"))
        plt.show()

    return beta_mat, rsq_mat, good_fit


def snapt_pipeline(spike_movie_avg, soma_trace, spike_indices, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    kernel = get_temporal_kernel(soma_trace, spike_indices)
    peri_spike_movie = get_peri_spike_movie(spike_movie_avg)

    # Kernel visualization
    analyze_spike_triggered_movie(kernel, peri_spike_movie, save_dir=results_dir)

    # run SNAPT
    snapt_algorithm(kernel, peri_spike_movie, save_dir=results_dir)
