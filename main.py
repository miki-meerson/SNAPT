import matplotlib

from basic_utils import read_movie

matplotlib.use('TkAgg')

from globals import *
from clean_pipeline import clean_movie_pipeline
from roi_analyzer import roi_analysis
from spike_analyzer import extract_roi_sta, build_spike_triggered_movie, get_bright_pixel_mask
from SNAPT import snapt_pipeline, get_movie_pca





if __name__ == '__main__':
    # path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/28-10-2024-acax3-s4-awake/fov1/vol_001/vol.tif"
    # path = "Z:/Adam-Lab-Shared/Data/Efrat_Sheinbach/Imaging_Data/2023-07-09_088R1/R-FOV1/spont_1000hz_15sec/Image_001_001.raw"
    path = "./vol.tif"
    clean_path = "clean_movie.tif"

    movie = read_movie(path)
    movie = clean_movie_pipeline(movie)

    # Apply Clicky and get traces per ROI
    roi_masks, roi_vertices, roi_traces = roi_analysis(movie)

    # Analyze first ROI (soma)
    first_roi_trace = roi_traces[0]
    window_size_ms = 50
    window_size = int(window_size_ms * SAMPLING_RATE / 1000)
    sta, spike_indices = extract_roi_sta(first_roi_trace, window_size)

    bright_pixel_mask = get_bright_pixel_mask(movie)
    sta_movie = build_spike_triggered_movie(spike_indices, movie, window_size, mask=bright_pixel_mask)
    snapt_pipeline(sta_movie, roi_masks[0], results_dir=RAW_RESULTS_DIR)

    # Try SNAPT again on the PCA movie
    pca_movie = get_movie_pca(sta_movie)
    pca_movie[:, ~bright_pixel_mask] = 0
    snapt_pipeline(pca_movie, roi_masks[0], results_dir=PCA_RESULTS_DIR)

    input("Press Enter to exit...")






