import matplotlib

from basic_utils import read_movie, basic_movie_preprocessing
from clean_pipeline import clean_movie_pipeline
from validations import validations_pipeline

matplotlib.use('TkAgg')

from constants import *
from roi_analyzer import roi_analysis
from spike_analyzer import build_spike_triggered_movie, get_bright_pixel_mask, extract_sta
from pca_utils import get_movie_pca
from SNAPT import snapt_pipeline


if __name__ == '__main__':
    # path = "Z:/Adam-Lab-Shared/Data/Michal_Rubin/Dendrites/AceM-neon/AcAx3/28-10-2024-acax3-s4-awake/fov1/vol_001/vol.tif"
    path = "Z:/Adam-Lab-Shared/Data/Efrat_Sheinbach/Imaging_Data/2023-07-09_088R1/R-FOV1/spont_1000hz_15sec/Image_001_001.raw"
    # path = "./vol.tif"
    # clean_path = "./clean_movie.tif"

    movie = read_movie(path)
    movie = basic_movie_preprocessing(movie)
    movie = clean_movie_pipeline(movie)

    # Apply Clicky and get traces per ROI
    roi_masks, roi_vertices, roi_traces = roi_analysis(movie)
    soma_trace = roi_traces[0]

    # Detect spikes and get STA
    spike_movie_avg, spikes, spike_indices = extract_sta(movie, roi_traces)

    snapt_pipeline(spike_movie_avg, soma_trace, spike_indices, results_dir=RAW_RESULTS_DIR)

    # Try SNAPT again on the PCA movie
    pca_movie = get_movie_pca(spike_movie_avg)
    snapt_pipeline(pca_movie, soma_trace, spike_indices, results_dir=PCA_RESULTS_DIR)

    # validations_pipeline(sta_movie)
    input("Press Enter to exit...")






