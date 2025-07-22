import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import LassoSelector
from skimage.draw import polygon2mask

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
        plt.show()

    def on_select(self, verts):
        # Save vertices
        self.roi_vertices.append(verts)

        # Convert polygon to binary mask
        mask = polygon2mask(self.image.shape, verts)
        self.roi_masks.append(mask)

        # Show the selected ROI on the image
        patch = plt.Polygon(verts, closed=True, fill=False, edgecolor='red', linewidth=1.5)
        self.ax.add_patch(patch)
        self.canvas.draw_idle()

    def get_masks(self):
        return self.roi_masks, self.roi_vertices


def apply_clicky(roi_masks, movie):
    n_rois = len(roi_masks)
    n_frames = movie.shape[0]
    traces = []

    for mask in roi_masks:
        # Apply mask to each frame and compute mean trace
        masked_values = movie[:, mask]
        mean_trace = np.mean(masked_values, axis=1)
        traces.append(mean_trace)

    traces = np.array(traces)  # shape: (n_rois, n_frames)
    assert traces.shape == (n_rois, n_frames), "Traces shape is incorrect"

    # Plot all ROI traces
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(n_rois, 1, hspace=0.3)
    for i in range(n_rois):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(traces[i])
        ax.set_title(f"ROI {i+1}")
        ax.set_ylabel("dF/F")
        if i == n_rois - 1:
            ax.set_xlabel("Frame")

    plt.tight_layout()
    plt.savefig("ROI analysis.png", dpi=300)
    with PdfPages("ROI analysis.pdf") as pdf:
        pdf.savefig(fig)
    print("Saved ROI traces as PNG and PDF.")
    plt.show()