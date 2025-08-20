import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import LassoSelector
from skimage.draw import polygon2mask
from matplotlib.path import Path

from constants import PLOT_CLEANING_STEPS


def get_average_image(movie, plot=True):
    """ Present a 2D image of mean value for each pixel across time """
    average_image = np.mean(movie, axis=0)

    # Baseline correction - suppress low intensity background
    baseline_image = np.percentile(average_image, 20)
    average_image -= baseline_image

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(average_image, cmap='gray')
        plt.colorbar()
        plt.title('Average image')
        plt.xlabel('X')
        plt.ylabel('Y')

    return average_image


class ROISelector:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(image, cmap='gray')
        self.canvas = self.ax.figure.canvas

        self.roi_masks = []
        self.roi_vertices = []

        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        plt.title("Draw ROI with mouse. Close window when done.")
        plt.show(block=True)

    def on_select(self, vertices):
        # Save vertices
        self.roi_vertices.append(vertices)

        # Convert polygon to binary mask
        mask = polygon2mask(self.image.shape, vertices)
        self.roi_masks.append(mask)

        # Show the selected ROI on the image
        patch = plt.Polygon(vertices, closed=True, fill=False, edgecolor='red', linewidth=1.5)
        self.ax.add_patch(patch)
        self.canvas.draw_idle()

    def get_roi_data(self):
        return self.roi_masks, self.roi_vertices


def extract_traces(roi_masks, movie, avg_img, plot=True):
    """ Extract mean ROI fluorescence traces given boolean masks """
    n_frames, height, width = movie.shape
    n_rois = len(roi_masks)
    roi_traces = np.zeros((n_frames, len(roi_masks)))

    for i, mask in enumerate(roi_masks):
        roi_pixels = movie[:, mask]  # shape (n_frames, n_pixels_in_roi)
        roi_traces[:, i] = roi_pixels.mean(axis=1)

    # ΔF/F0 normalization
    f0 = np.array([avg_img[mask].mean() for mask in roi_masks])
    dff_traces = roi_traces / (f0[np.newaxis, :] + 1e-8)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Show average image with ROIs
        axes[0].imshow(avg_img, cmap="gray")

        for i, mask in enumerate(roi_masks):
            y, x = np.nonzero(mask)
            axes[0].plot(y, x, '.', markersize=0.5, label=f"ROI {i + 1}")

        axes[0].set_title("ROIs on Avg Image")

        # Plot traces
        for i in range(len(roi_masks)):
            axes[1].plot(roi_traces[:, i], label=f"ROI {i + 1}")
        axes[1].set_title("ROI Fluorescence Traces")
        axes[1].legend()

        axes[2].plot(dff_traces)
        axes[2].set_title("ΔF/F0 Traces")

        plt.tight_layout()
        plt.show()

    return roi_traces, f0, dff_traces


def roi_analysis(movie):
    avg_img = get_average_image(movie)
    selector = ROISelector(avg_img)
    roi_masks, roi_vertices = selector.get_roi_data()
    roi_traces, f0, roi_dff_traces = extract_traces(roi_masks, movie, avg_img)


    return roi_masks, roi_vertices, roi_dff_traces