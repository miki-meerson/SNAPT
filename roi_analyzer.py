import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.widgets import LassoSelector
from skimage.draw import polygon2mask

from experiment_constants import SAMPLING_RATE


def get_average_image(movie, plot=False):
    """ Present a 2D image of mean value for each pixel across time """
    average_image = np.mean(movie, axis=0)

    if plot:

        plt.figure(figsize=(10, 10))
        im = plt.imshow(average_image, cmap='gray')
        plt.colorbar()
        plt.title('Average image')
        plt.xlabel('X')
        plt.ylabel('Y')

    # Baseline correction - suppress low intensity background
    baseline_image = np.percentile(average_image, 20)
    average_image -= baseline_image

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


def extract_traces(movie, masks, avg_img):
    n_frames = movie.shape[0]
    n_rois = len(masks)
    traces = np.zeros((n_rois, n_frames))
    f0 = np.zeros(n_rois)

    for i, mask in enumerate(masks):
        # Average intensity within mask for each frame
        roi_pixels = movie[:, mask]
        traces[i, :] = roi_pixels.mean(axis=1)
        f0[i] = avg_img[mask].mean() # Get baseline per ROI

    # ΔF/F calculation
    dff_traces = traces / f0[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_rois):
        ax.plot(dff_traces[i, :], label=f"ROI {i + 1}")
    ax.set_title("ΔF/F traces")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔF/F")
    ax.legend()
    plt.tight_layout()

    return dff_traces


def roi_analysis(movie):
    avg_img = get_average_image(movie)
    selector = ROISelector(avg_img)
    roi_masks, roi_vertices = selector.get_roi_data()
    roi_traces = extract_traces(movie, roi_masks, avg_img)

    return roi_masks, roi_vertices, roi_traces