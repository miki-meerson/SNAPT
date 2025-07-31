import numpy as np
import tifffile
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

from experiment_constants import SAMPLING_RATE
from SNAPT import analyze_spike_triggered_movie, snapt_algorithm, get_temporal_kernel, get_peri_spike_movie, \
    get_movie_pca
from clean_pipeline import clean_movie_pipeline
from spike_analyzer import extract_roi_sta, spike_triggered_average_analysis, build_spike_triggered_movie
from roi_analyzer import roi_analysis


if __name__ == '__main__':
    # path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/28-10-2024-acax3-s4-awake/fov1/vol_001/vol.tif"
    path = "vol.tif"
    clean_path = "clean_movie.tif"
    #
    # print("Loading data...")
    # movie = tifffile.imread(path)
    # movie = -movie  # Flip polarity for negatively going GEVI
    #
    # print("Cleaning the data...")
    # movie = clean_movie_pipeline(movie)
    # tifffile.imwrite(clean_path, movie)

    # To skip cleanup process - load clean movie
    print("Loading clean data...")
    movie = tifffile.imread(clean_path)

    # Apply Clicky and get traces per ROI
    roi_masks, roi_vertices, roi_traces = roi_analysis(movie)

    # Analyze first ROI (soma)
    first_roi_trace = roi_traces[0]
    window_size_ms = 50
    window_size = int(window_size_ms * SAMPLING_RATE / 1000)
    sta, spike_indices = extract_roi_sta(first_roi_trace, window_size)
    sta_movie = build_spike_triggered_movie(spike_indices, movie, roi_mask=None, window_size=window_size)

    kernel = get_temporal_kernel(sta_movie)
    peri_spike_movie_norm = get_peri_spike_movie(sta_movie)

    # Kernel visualization
    analyze_spike_triggered_movie(roi_vertices[0], kernel, peri_spike_movie_norm)

    # SNAPT initial visualization
    snapt_algorithm(kernel, peri_spike_movie_norm)

    # TODO check if working:
    # Try SNAPT again on the PCA movie
    pca_movie = get_movie_pca(sta_movie)
    pca_sta_movie = build_spike_triggered_movie(spike_indices, pca_movie, roi_mask=None, window_size=window_size)
    pca_kernel = get_temporal_kernel(pca_sta_movie)
    pca_peri_spike_movie_norm = get_peri_spike_movie(pca_sta_movie)
    analyze_spike_triggered_movie(roi_vertices[0], pca_kernel, pca_peri_spike_movie_norm)
    snapt_algorithm(kernel, pca_peri_spike_movie_norm)

    input("Press Enter to exit...")






